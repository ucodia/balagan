"""Per-frame render orchestration: the BalaGAN engine."""

import logging
import os
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
        use_cuda_graph: bool = True,
    ) -> None:
        self._interpolator = interpolator
        self._latent_navigator = latent_navigator
        self._weight_blender = weight_blender
        self._snapshot_manager = snapshot_manager
        self._runtime_state = runtime_state

        # StyleGAN's synthesis forward is launch-bound (~100 tiny kernels per
        # frame), so the GPU idles between launches. Capturing it as a CUDA graph
        # replays the whole sequence with one submission. Captured lazily on the
        # first CUDA frame; falls back to eager on non-CUDA or capture failure.
        self._use_cuda_graph = use_cuda_graph
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_static_ws: torch.Tensor | None = None
        self._graph_static_out: torch.Tensor | None = None
        self._graph_failed = False

        self._blender_cached: set[int] = set()
        self._last_frame_start: float | None = None
        self._next_deadline: float | None = None
        self._last_log_time = time.perf_counter()
        self._frames_since_log = 0
        self._last_status = ""

        # Diagnostic phase profiling, enabled with BALAGAN_PROFILE=1. Each phase
        # is CUDA-synchronized so GPU work is attributed to the phase that issued
        # it; _report prints per-phase mean ms. Remove once the bottleneck is
        # characterized. Off by default: the syncs serialize the pipeline.
        self._profile = os.environ.get("BALAGAN_PROFILE") == "1"
        self._cuda = torch.cuda.is_available()
        self._phase_sums: dict[str, float] = {}

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
        frame_start = time.perf_counter()
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
            mark = self._profile_mark("setup", frame_start) if self._profile else 0.0
            ws = self._latent_navigator(latent_x, latent_y, state.truncation_psi)
            ws_batched = ws.unsqueeze(0)
            mark = self._profile_mark("navigator", mark) if self._profile else 0.0

            graph_mode = (
                self._use_cuda_graph and not self._graph_failed and ws_batched.is_cuda
            )
            # Graph mode always blends into the persistent target so the captured
            # graph reads the same parameter tensors every frame.
            if graph_mode:
                synthesis = self._weight_blender.blend_into_target(
                    render_a, render_b, alpha
                )
            else:
                synthesis = self._weight_blender(render_a, render_b, alpha)
            mark = self._profile_mark("blend", mark) if self._profile else 0.0

            if graph_mode and self._graph is None:
                self._capture_graph(synthesis, ws_batched)
            if graph_mode and self._graph is not None:
                self._graph_static_ws.copy_(ws_batched)
                self._graph.replay()
                image = self._graph_static_out
            else:
                image = synthesis(ws=ws_batched, noise_mode="const")[0]
            mark = self._profile_mark("synthesis", mark) if self._profile else 0.0

            frame = _to_uint8_hwc(image)
            if self._profile:
                self._profile_mark("readback", mark)

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

    def _capture_graph(
        self, synthesis: torch.nn.Module, ws_batched: torch.Tensor
    ) -> None:
        """Capture the synthesis forward as a CUDA graph for launch-free replay.

        Warms up on a side stream first -- StyleGAN's custom ops cache plans and
        cuDNN selects algorithms on early calls -- then captures. Replay reads
        the synthesis parameters in place, so the weight blender can mutate them
        per frame and the graph reflects the new weights (verified bit-identical
        to eager). Any capture failure disables the path and falls back to eager.
        """
        try:
            static_ws = ws_batched.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    synthesis(ws=static_ws, noise_mode="const")
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = synthesis(ws=static_ws, noise_mode="const")[0]
        except Exception as exc:  # noqa: BLE001 -- any capture failure -> eager
            self._graph_failed = True
            logger.warning(
                "CUDA graph capture failed (%s: %s); using eager synthesis",
                type(exc).__name__,
                exc,
            )
            return
        self._graph = graph
        self._graph_static_ws = static_ws
        self._graph_static_out = static_out
        logger.info("CUDA graph captured for the synthesis forward")

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

    def _profile_mark(self, name: str, since: float) -> float:
        """Accumulate the time since ``since`` under ``name`` and return now.

        Synchronizes CUDA first so async GPU work is charged to the phase that
        launched it rather than bleeding into the next one.
        """
        if self._cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._phase_sums[name] = self._phase_sums.get(name, 0.0) + (now - since)
        return now

    def _report(self, position: float, kimg_a: int, kimg_b: int, alpha: float) -> None:
        self._frames_since_log += 1
        elapsed = time.perf_counter() - self._last_log_time
        if elapsed < 1.0:
            return
        frames = self._frames_since_log
        fps = f"{frames / elapsed:.1f} fps"
        blend = f"{kimg_a} → {kimg_b} ({round(alpha * 100)}%)"
        self._last_status = f"{fps} | {blend}"
        if self._profile and self._phase_sums:
            phases = " ".join(
                f"{name}={total / frames * 1e3:.1f}ms"
                for name, total in self._phase_sums.items()
            )
            logger.info("%s | t=%.3f | %s | %s", fps, position, blend, phases)
            self._phase_sums.clear()
        else:
            logger.info("%s | t=%.3f | %s", fps, position, blend)
        self._frames_since_log = 0
        self._last_log_time = time.perf_counter()


def build_engine(
    config: EngineConfig,
    device: str | torch.device,
    window_size: int = 32,
    runtime_state: RuntimeState | None = None,
    use_cuda_graph: bool = True,
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
    canonical_kimg = config.canonical_mapping_kimg
    canonical_pkl = config.snapshots_dir / f"network-snapshot-{canonical_kimg:06d}.pkl"
    canonical_mapping = load_canonical_mapping(canonical_pkl, device)

    def snapshot_loader(pkl_path) -> torch.nn.Module:
        return load_network_pkl(pkl_path)["G_ema"].synthesis.to(device)

    engine = Engine(
        interpolator=Interpolator(config.snapshots),
        latent_navigator=LatentNavigator(canonical_mapping, z_dim=canonical_mapping.z_dim),
        weight_blender=WeightBlender(),
        snapshot_manager=SnapshotManager(
            config.snapshots, canonical_kimg, snapshot_loader, window_size
        ),
        runtime_state=runtime_state if runtime_state is not None else RuntimeState(),
        use_cuda_graph=use_cuda_graph,
    )
    logger.info(
        "Engine built: %d snapshots, device %s, window size %d",
        len(config.snapshots),
        device,
        window_size,
    )
    return engine
