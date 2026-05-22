"""Tests for balagan.core.engine: per-frame render orchestration."""

import logging
from pathlib import Path

import numpy as np
import torch

from balagan.config import Phase, PhaseConfig, SnapshotInfo
from balagan.core.engine import Engine
from balagan.core.interpolator import Interpolator
from balagan.core.latent_navigator import LatentNavigator
from balagan.core.runtime_state import RuntimeState
from balagan.core.snapshot_manager import SnapshotManager
from balagan.core.weight_blender import WeightBlender


class StubMapping(torch.nn.Module):
    """Identity-broadcast stand-in for a StyleGAN2 MappingNetwork."""

    def __init__(self):
        super().__init__()
        self.z_dim = 4
        self.c_dim = 0
        self.w_dim = 4
        self.num_ws = 2
        self.register_buffer("w_avg", torch.zeros(4))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None):
        x = z.to(torch.float32).unsqueeze(1).repeat(1, self.num_ws, 1)
        if truncation_psi != 1:
            x = self.w_avg.lerp(x, truncation_psi)
        return x


class StubSynthesis(torch.nn.Module):
    """Stand-in for a StyleGAN2 SynthesisNetwork: emits a fixed-size image."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 4))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, ws, **block_kwargs):
        return torch.zeros(ws.shape[0], 3, 64, 64)


def _snapshot(kimg: int, fid: float) -> SnapshotInfo:
    return SnapshotInfo(kimg=kimg, fid_raw=fid, pkl_path=Path(f"snap-{kimg}.pkl"))


SNAPSHOTS = [
    _snapshot(0, 300.0),
    _snapshot(100, 250.0),
    _snapshot(200, 200.0),
    _snapshot(300, 150.0),
    _snapshot(400, 120.0),
    _snapshot(500, 100.0),
]
PHASE_CONFIG = PhaseConfig(
    kimg_range=(0, 500),
    smoothing_window=1,
    floor=1.0,
    canonical_mapping_kimg=200,
    phases=(Phase(0.0, 1.0, 0, 500),),
)


def make_engine(window_size: int = 6) -> tuple[Engine, RuntimeState, WeightBlender]:
    runtime_state = RuntimeState()
    runtime_state.update(fps_cap=0)  # uncapped: no frame-limiter sleep during tests
    weight_blender = WeightBlender()
    engine = Engine(
        interpolator=Interpolator(SNAPSHOTS, PHASE_CONFIG),
        latent_navigator=LatentNavigator(StubMapping(), z_dim=4),
        weight_blender=weight_blender,
        snapshot_manager=SnapshotManager(
            SNAPSHOTS, 200, lambda pkl_path: StubSynthesis(), window_size
        ),
        runtime_state=runtime_state,
    )
    return engine, runtime_state, weight_blender


def test_render_frame_returns_a_uint8_hwc_rgb_image():
    engine, _, _ = make_engine()
    engine.prime()
    frame = engine.render_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)


def test_renders_valid_frames_across_the_position_range():
    engine, state, _ = make_engine()
    engine.prime()
    for position in (0.0, 0.25, 0.5, 0.75, 1.0):
        state.update(position=position)
        frame = engine.render_frame()
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8


def test_anim_playing_advances_and_persists_latent_x():
    engine, state, _ = make_engine()
    engine.prime()
    state.update(anim_playing=True, anim_speed=1000.0)
    engine.render_frame()  # first frame: zero delta, no advance
    engine.render_frame()  # second frame: real delta advances latent_x
    assert state.snapshot().latent_x > 0.0


def test_anim_disabled_leaves_latent_x_unchanged():
    engine, state, _ = make_engine()
    engine.prime()
    engine.render_frame()
    engine.render_frame()
    assert state.snapshot().latent_x == 0.0


def test_falls_back_to_a_loaded_pair_when_target_not_ready(caplog):
    engine, state, _ = make_engine(window_size=3)
    engine.prime()  # primes the t=0.0 window only; loader thread not started
    state.update(position=1.0)
    with caplog.at_level(logging.WARNING):
        frame = engine.render_frame()
    assert frame.shape == (64, 64, 3)
    assert "not ready" in caplog.text


def test_blender_cache_evicts_snapshots_dropped_from_the_window():
    engine, state, blender = make_engine(window_size=3)
    engine.prime()
    engine.render_frame()
    assert blender.is_cached(0)
    state.update(position=1.0)
    engine.prime()  # re-prime: snapshot manager loads the far window, evicts the near
    engine.render_frame()
    assert not blender.is_cached(0)


def test_runtime_state_is_accessible():
    engine, state, _ = make_engine()
    assert engine.runtime_state is state


class _RacySnapshotManager:
    """Engine test double modeling a snapshot the loader is evicting mid-frame.

    Its granular reads disagree the way a concurrent eviction makes them:
    ``loaded_kimgs`` still lists the snapshot while ``get_synthesis`` already
    returns None for it. ``loaded_networks`` instead hands back a single
    atomic, reference-holding view -- the view render_frame must rely on so it
    never blends a snapshot it has not cached.
    """

    def __init__(self, kimg: int) -> None:
        self._kimg = kimg
        self._network = StubSynthesis()

    def set_active_pair(self, kimg_a: int, kimg_b: int) -> None:
        pass

    def is_pair_ready(self, kimg_a: int, kimg_b: int) -> bool:
        return False

    def loaded_kimgs(self) -> set[int]:
        return {self._kimg}

    def get_synthesis(self, kimg: int) -> torch.nn.Module | None:
        return None

    def loaded_networks(self) -> dict[int, torch.nn.Module]:
        return {self._kimg: self._network}

    def pending_count(self) -> int:
        return 0


def test_render_frame_blends_only_cached_snapshots_when_eviction_races_the_frame():
    """A fast position move can leave the engine selecting a snapshot the
    loader thread is concurrently evicting. render_frame must blend only
    snapshots it has cached, never raising and killing the render thread."""
    runtime_state = RuntimeState()
    runtime_state.update(fps_cap=0)
    engine = Engine(
        interpolator=Interpolator(SNAPSHOTS, PHASE_CONFIG),
        latent_navigator=LatentNavigator(StubMapping(), z_dim=4),
        weight_blender=WeightBlender(),
        snapshot_manager=_RacySnapshotManager(kimg=200),
        runtime_state=runtime_state,
    )
    frame = engine.render_frame()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


def test_debug_overlay_marks_the_frame_only_when_enabled():
    """With debug enabled, render_frame bakes the status overlay into the
    frame; with it disabled the frame is left untouched."""
    engine, state, _ = make_engine()
    engine.prime()
    engine._last_status = "30.0 fps | t=0.500 | kimg 0->100 @ 0.000 | loaded 6"
    plain = engine.render_frame()
    state.update(debug=True)
    overlaid = engine.render_frame()
    assert overlaid.shape == plain.shape == (64, 64, 3)
    assert np.array_equal(plain, np.full((64, 64, 3), 128, np.uint8))
    assert not np.array_equal(overlaid, plain)


def test_debug_overlay_skips_an_empty_status():
    """Before the first status report there is nothing to draw; the debug
    overlay must leave the frame untouched rather than fail."""
    engine, state, _ = make_engine()
    engine.prime()
    state.update(debug=True)
    frame = engine.render_frame()
    assert np.array_equal(frame, np.full((64, 64, 3), 128, np.uint8))
