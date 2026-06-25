"""Per-frame render orchestration: the BalaGAN engine."""

import logging
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from balagan.config import EngineConfig
from balagan.core.canonical_mapping import load_canonical_mapping, load_network_pkl
from balagan.core.interpolator import Interpolator
from balagan.core.latent_navigator import LatentNavigator
from balagan.core.runtime_state import RuntimeState
from balagan.core.snapshot_manager import SnapshotManager
from balagan.core.weight_blender import WeightBlender

logger = logging.getLogger(__name__)


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """Convert a [C, H, W] synthesis output in [-1, 1] to a uint8 HWC RGB array."""
    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return image.permute(1, 2, 0).contiguous().cpu().numpy()


def _draw_debug_overlay(frame: np.ndarray, status: str, frame_count: int) -> np.ndarray:
    """Bake the engine status into the frame's bottom-left corner, one metric
    per line, as white text over a transparent background.

    The frame counter is drawn on its own line and advances every frame (unlike
    the per-second status), so it can be read off both the GUI window and the web
    stream to gauge end-to-end rendering delay.
    """
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, frame.shape[0] // 40))
    lines = [f"frame {frame_count}"]
    if status:
        lines.extend(status.split(" | "))
    text = "\n".join(lines)
    margin = 6
    box = draw.multiline_textbbox((0, 0), text, font=font)
    y = frame.shape[0] - margin - box[3]
    draw.multiline_text((margin, y), text, fill=(255, 255, 255), font=font)
    return np.array(image)


class Engine:
    """Owns the component graph and produces one rendered frame per call.

    Each frame snapshots the runtime state, advances the latent walk if
    animating, maps the audience position to a snapshot pair, blends their
    weights, synthesizes, and returns a uint8 HWC RGB frame.
    """

    def __init__(
        self,
        *,
        interpolator: Interpolator,
        latent_navigator: LatentNavigator,
        weight_blender: WeightBlender,
        snapshot_manager: SnapshotManager,
        runtime_state: RuntimeState,
        snapshot_names: tuple[str, ...] = (),
    ) -> None:
        self._interpolator = interpolator
        self._latent_navigator = latent_navigator
        self._weight_blender = weight_blender
        self._snapshot_manager = snapshot_manager
        self._runtime_state = runtime_state
        self._snapshot_names = snapshot_names

        self._blender_cached: set[int] = set()
        self._last_frame_start: float | None = None
        self._next_deadline: float | None = None
        self._last_log_time = time.perf_counter()
        self._frames_since_log = 0
        self._frame_count = 0
        self._last_status = ""

    @property
    def runtime_state(self) -> RuntimeState:
        """The shared runtime state, exposed for OSC and GUI controls to mutate."""
        return self._runtime_state

    def status(self) -> str:
        """The most recent per-second status line, for GUI and log display."""
        return self._last_status

    def prime(self) -> None:
        """Synchronously load the initial window for the current position."""
        position = self._runtime_state.snapshot().position
        index_a, index_b, _ = self._interpolator(position)
        self._snapshot_manager.prime(index_a, index_b)

    def start(self) -> None:
        """Start the snapshot manager's background loader thread."""
        self._snapshot_manager.start()

    def stop(self) -> None:
        """Stop the snapshot manager's background loader thread."""
        self._snapshot_manager.stop()

    def render_frame(self) -> np.ndarray:
        """Produce one frame as a uint8 HWC RGB array."""
        frame_start = time.perf_counter()
        self._frame_count += 1
        delta = (
            frame_start - self._last_frame_start
            if self._last_frame_start is not None
            else 0.0
        )
        self._last_frame_start = frame_start

        state = self._runtime_state.snapshot()
        latent_x = state.latent_x
        latent_y = state.latent_y
        if state.anim_playing:
            latent_x = state.latent_x + state.anim_speed_x * delta
            latent_y = state.latent_y + state.anim_speed_y * delta
            self._runtime_state.update(latent_x=latent_x, latent_y=latent_y)

        index_a, index_b, alpha = self._interpolator(state.position)
        self._snapshot_manager.set_active_pair(index_a, index_b)

        # One consistent view of the loaded snapshots drives the whole frame:
        # the blender cache, the pair choice, and the blend. Re-reading the
        # snapshot manager would race the loader thread, which can evict a
        # snapshot between the pair choice and the blend.
        loaded = self._snapshot_manager.loaded_networks()
        self._sync_blender_cache(loaded)

        render_a, render_b = index_a, index_b
        if index_a not in loaded or index_b not in loaded:
            render_a, render_b = self._nearest_loaded_pair(loaded, index_a, index_b)
            logger.warning(
                "Snapshot pair (%d, %d) not ready; rendering nearest loaded (%d, %d)",
                index_a,
                index_b,
                render_a,
                render_b,
            )

        with torch.no_grad():
            ws = self._latent_navigator(latent_x, latent_y, state.truncation_psi)
            synthesis = self._weight_blender(render_a, render_b, alpha)
            image = synthesis(ws=ws.unsqueeze(0), noise_mode="const")[0]
            frame = _to_uint8_hwc(image)

        self._report(state.position, index_a, index_b, alpha)
        if state.debug:
            frame = _draw_debug_overlay(frame, self._last_status, self._frame_count)
        self._limit_framerate(state.fps_cap, frame_start)
        return frame

    def _nearest_loaded_pair(
        self, loaded: dict[int, torch.nn.Module], index_a: int, index_b: int
    ) -> tuple[int, int]:
        indices = loaded.keys()
        nearest_a = min(indices, key=lambda i: abs(i - index_a))
        nearest_b = min(indices, key=lambda i: abs(i - index_b))
        return nearest_a, nearest_b

    def _sync_blender_cache(self, loaded: dict[int, torch.nn.Module]) -> None:
        """Mirror the weight blender's cache onto a single loaded-snapshot view.

        Caching from the same ``loaded`` dict the frame selects its pair from
        guarantees every snapshot the frame goes on to blend is resident in
        the blender.
        """
        for index, network in loaded.items():
            if index not in self._blender_cached:
                self._weight_blender.cache_snapshot(index, network)
                self._blender_cached.add(index)
        for index in self._blender_cached - loaded.keys():
            self._weight_blender.evict_snapshot(index)
            self._blender_cached.discard(index)

    def _limit_framerate(self, fps_cap: int, frame_start: float) -> None:
        # Schedule against an absolute deadline rather than each frame's own
        # start. A relative budget only compensates for time spent inside
        # render_frame, so the caller's post-render work (Spout send, the GUI's
        # QImage copy) stacks on top of the cap every frame and the engine
        # never reaches it. Anchoring to a fixed timeline lets that overhead be
        # absorbed into the next frame's shorter sleep.
        if fps_cap <= 0:
            self._next_deadline = None  # reset so re-enabling re-anchors cleanly
            return
        period = 1.0 / fps_cap
        if self._next_deadline is None:
            self._next_deadline = frame_start + period
        remaining = self._next_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        self._next_deadline += period
        now = time.perf_counter()
        if self._next_deadline < now:
            # Fell a full period behind (stall, or work > budget): resync rather
            # than fast-forwarding through frames with zero-length sleeps.
            self._next_deadline = now + period

    def _report(self, position: float, index_a: int, index_b: int, alpha: float) -> None:
        self._frames_since_log += 1
        elapsed = time.perf_counter() - self._last_log_time
        if elapsed < 1.0:
            return
        fps = f"{self._frames_since_log / elapsed:.1f} fps"
        name_a = self._snapshot_names[index_a] if self._snapshot_names else str(index_a)
        name_b = self._snapshot_names[index_b] if self._snapshot_names else str(index_b)
        pct_a = round((1.0 - alpha) * 100)
        pct_b = round(alpha * 100)
        self._last_status = f"{fps} | {name_a} ({pct_a}%) | {name_b} ({pct_b}%)"
        self._runtime_state.update(status=self._last_status)
        logger.info("%s | t=%.3f | %s (%d%%) -> %s (%d%%)", fps, position, name_a, pct_a, name_b, pct_b)
        self._frames_since_log = 0
        self._last_log_time = time.perf_counter()


def build_engine(
    config: EngineConfig,
    device: str | torch.device,
    window_size: int = 32,
    runtime_state: RuntimeState | None = None,
) -> Engine:
    """Construct the full engine component graph from a validated config.

    Loads the canonical mapping network and wires a snapshot loader that
    extracts each snapshot's synthesis network. Submodule imports happen lazily
    inside the loaders, so ``stylegan3/`` must be on ``sys.path`` before the
    returned engine is primed.

    Pass ``runtime_state`` to share an existing state across engine rebuilds, so
    GUI control values and widget bindings survive a swap; when omitted a fresh
    one is created.
    """
    canonical_pkl = config.snapshots[config.canonical_index].pkl_path
    canonical_mapping = load_canonical_mapping(canonical_pkl, device)

    def snapshot_loader(pkl_path) -> torch.nn.Module:
        return load_network_pkl(pkl_path)["G_ema"].synthesis.to(device)

    engine = Engine(
        interpolator=Interpolator(config.snapshots),
        latent_navigator=LatentNavigator(canonical_mapping, z_dim=canonical_mapping.z_dim),
        weight_blender=WeightBlender(),
        snapshot_manager=SnapshotManager(
            config.snapshots, config.canonical_index, snapshot_loader, window_size
        ),
        runtime_state=runtime_state if runtime_state is not None else RuntimeState(),
        snapshot_names=tuple(s.pkl_path.name for s in config.snapshots),
    )
    logger.info(
        "Engine built: %d snapshots, device %s, window size %d",
        len(config.snapshots),
        device,
        window_size,
    )
    return engine
