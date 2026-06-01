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


def _draw_debug_overlay(frame: np.ndarray, status: str) -> np.ndarray:
    """Bake the engine status into the frame's bottom-left corner, one metric
    per line, as white text over a transparent background. Returns the frame
    unchanged when there is no status to show yet.
    """
    if not status:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, frame.shape[0] // 40))
    text = "\n".join(status.split(" | "))
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
    ) -> None:
        self._interpolator = interpolator
        self._latent_navigator = latent_navigator
        self._weight_blender = weight_blender
        self._snapshot_manager = snapshot_manager
        self._runtime_state = runtime_state

        self._blender_cached: set[int] = set()
        self._last_frame_start: float | None = None
        self._last_log_time = time.monotonic()
        self._frames_since_log = 0
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
        kimg_a, kimg_b, _ = self._interpolator(position)
        self._snapshot_manager.prime(kimg_a, kimg_b)

    def start(self) -> None:
        """Start the snapshot manager's background loader thread."""
        self._snapshot_manager.start()

    def stop(self) -> None:
        """Stop the snapshot manager's background loader thread."""
        self._snapshot_manager.stop()

    def render_frame(self) -> np.ndarray:
        """Produce one frame as a uint8 HWC RGB array."""
        frame_start = time.monotonic()
        delta = (
            frame_start - self._last_frame_start
            if self._last_frame_start is not None
            else 0.0
        )
        self._last_frame_start = frame_start

        state = self._runtime_state.snapshot()
        latent_x = state.latent_x
        if state.anim_playing:
            latent_x = state.latent_x + state.anim_speed * delta
            self._runtime_state.update(latent_x=latent_x)

        kimg_a, kimg_b, alpha = self._interpolator(state.position)
        self._snapshot_manager.set_active_pair(kimg_a, kimg_b)

        # One consistent view of the loaded snapshots drives the whole frame:
        # the blender cache, the pair choice, and the blend. Re-reading the
        # snapshot manager would race the loader thread, which can evict a
        # snapshot between the pair choice and the blend.
        loaded = self._snapshot_manager.loaded_networks()
        self._sync_blender_cache(loaded)

        render_a, render_b = kimg_a, kimg_b
        if kimg_a not in loaded or kimg_b not in loaded:
            render_a, render_b = self._nearest_loaded_pair(loaded, kimg_a, kimg_b)
            logger.warning(
                "Snapshot pair (%d, %d) not ready; rendering nearest loaded (%d, %d)",
                kimg_a,
                kimg_b,
                render_a,
                render_b,
            )

        with torch.no_grad():
            ws = self._latent_navigator(latent_x, state.latent_y, state.truncation_psi)
            synthesis = self._weight_blender(render_a, render_b, alpha)
            image = synthesis(ws=ws.unsqueeze(0), noise_mode="const")[0]
            frame = _to_uint8_hwc(image)

        self._report(state.position, kimg_a, kimg_b, alpha)
        if state.debug:
            frame = _draw_debug_overlay(frame, self._last_status)
        self._limit_framerate(state.fps_cap, frame_start)
        return frame

    def _nearest_loaded_pair(
        self, loaded: dict[int, torch.nn.Module], kimg_a: int, kimg_b: int
    ) -> tuple[int, int]:
        kimgs = loaded.keys()
        nearest_a = min(kimgs, key=lambda kimg: abs(kimg - kimg_a))
        nearest_b = min(kimgs, key=lambda kimg: abs(kimg - kimg_b))
        return nearest_a, nearest_b

    def _sync_blender_cache(self, loaded: dict[int, torch.nn.Module]) -> None:
        """Mirror the weight blender's cache onto a single loaded-snapshot view.

        Caching from the same ``loaded`` dict the frame selects its pair from
        guarantees every snapshot the frame goes on to blend is resident in
        the blender.
        """
        for kimg, network in loaded.items():
            if kimg not in self._blender_cached:
                self._weight_blender.cache_snapshot(kimg, network)
                self._blender_cached.add(kimg)
        for kimg in self._blender_cached - loaded.keys():
            self._weight_blender.evict_snapshot(kimg)
            self._blender_cached.discard(kimg)

    def _limit_framerate(self, fps_cap: int, frame_start: float) -> None:
        if fps_cap <= 0:
            return
        remaining = (1.0 / fps_cap) - (time.monotonic() - frame_start)
        if remaining > 0.0:
            time.sleep(remaining)

    def _report(self, position: float, kimg_a: int, kimg_b: int, alpha: float) -> None:
        self._frames_since_log += 1
        elapsed = time.monotonic() - self._last_log_time
        if elapsed < 1.0:
            return
        self._last_status = (
            f"{self._frames_since_log / elapsed:.1f} fps | t={position:.3f} | "
            f"kimg {kimg_a}->{kimg_b} @ {alpha:.3f} | "
            f"loaded {len(self._snapshot_manager.loaded_kimgs())}, "
            f"pending {self._snapshot_manager.pending_count()}"
        )
        logger.info("%s", self._last_status)
        self._frames_since_log = 0
        self._last_log_time = time.monotonic()


def build_engine(
    config: EngineConfig, device: str | torch.device, window_size: int = 32
) -> Engine:
    """Construct the full engine component graph from a validated config.

    Loads the canonical mapping network and wires a snapshot loader that
    extracts each snapshot's synthesis network. Submodule imports happen lazily
    inside the loaders, so ``stylegan3/`` must be on ``sys.path`` before the
    returned engine is primed.
    """
    canonical_kimg = config.phase_config.canonical_mapping_kimg
    canonical_pkl = config.run_dir / f"network-snapshot-{canonical_kimg:06d}.pkl"
    canonical_mapping = load_canonical_mapping(canonical_pkl, device)

    def snapshot_loader(pkl_path) -> torch.nn.Module:
        return load_network_pkl(pkl_path)["G_ema"].synthesis.to(device)

    engine = Engine(
        interpolator=Interpolator(config.snapshots, config.phase_config),
        latent_navigator=LatentNavigator(canonical_mapping, z_dim=canonical_mapping.z_dim),
        weight_blender=WeightBlender(),
        snapshot_manager=SnapshotManager(
            config.snapshots, canonical_kimg, snapshot_loader, window_size
        ),
        runtime_state=RuntimeState(),
    )
    logger.info(
        "Engine built: %d snapshots, device %s, window size %d",
        len(config.snapshots),
        device,
        window_size,
    )
    return engine
