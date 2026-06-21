"""Tests for balagan.core.engine: per-frame render orchestration."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from balagan.config import EngineConfig, SnapshotInfo
from balagan.core.engine import Engine, build_engine
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


def _snapshot(index: int) -> SnapshotInfo:
    return SnapshotInfo(index=index, pkl_path=Path(f"snap-{index}.pkl"))


SNAPSHOTS = [_snapshot(i) for i in range(6)]


def make_engine(window_size: int = 6) -> tuple[Engine, RuntimeState, WeightBlender]:
    runtime_state = RuntimeState()
    runtime_state.update(fps_cap=0)  # uncapped: no frame-limiter sleep during tests
    weight_blender = WeightBlender()
    engine = Engine(
        interpolator=Interpolator(SNAPSHOTS),
        latent_navigator=LatentNavigator(StubMapping(), z_dim=4),
        weight_blender=weight_blender,
        snapshot_manager=SnapshotManager(
            SNAPSHOTS, 2, lambda pkl_path: StubSynthesis(), window_size
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
    state.update(anim_playing=True, anim_speed_x=1000.0)
    engine.render_frame()  # first frame: zero delta, no advance
    engine.render_frame()  # second frame: real delta advances latent_x
    assert state.snapshot().latent_x > 0.0


def test_anim_playing_advances_latent_y_with_speed_y():
    engine, state, _ = make_engine()
    engine.prime()
    state.update(anim_playing=True, anim_speed_x=0.0, anim_speed_y=1000.0)
    engine.render_frame()  # first frame: zero delta, no advance
    engine.render_frame()  # second frame: real delta advances latent_y
    snap = state.snapshot()
    assert snap.latent_y > 0.0
    assert snap.latent_x == 0.0


def test_anim_default_leaves_latent_y_unchanged():
    engine, state, _ = make_engine()
    engine.prime()
    state.update(anim_playing=True)  # anim_speed_y defaults to 0.0
    engine.render_frame()
    engine.render_frame()
    snap = state.snapshot()
    assert snap.latent_y == 0.0
    assert snap.latent_x > 0.0


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


class _RacySnapshotManager:
    """Engine test double modeling a snapshot the loader is evicting mid-frame.

    Its granular reads disagree the way a concurrent eviction makes them:
    ``loaded_indices`` still lists the snapshot while ``get_synthesis`` already
    returns None for it. ``loaded_networks`` instead hands back a single
    atomic, reference-holding view -- the view render_frame must rely on so it
    never blends a snapshot it has not cached.
    """

    def __init__(self, index: int) -> None:
        self._index = index
        self._network = StubSynthesis()

    def set_active_pair(self, index_a: int, index_b: int) -> None:
        pass

    def is_pair_ready(self, index_a: int, index_b: int) -> bool:
        return False

    def loaded_indices(self) -> set[int]:
        return {self._index}

    def get_synthesis(self, index: int) -> torch.nn.Module | None:
        return None

    def loaded_networks(self) -> dict[int, torch.nn.Module]:
        return {self._index: self._network}

    def pending_count(self) -> int:
        return 0


def test_render_frame_blends_only_cached_snapshots_when_eviction_races_the_frame():
    """A fast position move can leave the engine selecting a snapshot the
    loader thread is concurrently evicting. render_frame must blend only
    snapshots it has cached, never raising and killing the render thread."""
    runtime_state = RuntimeState()
    runtime_state.update(fps_cap=0)
    engine = Engine(
        interpolator=Interpolator(SNAPSHOTS),
        latent_navigator=LatentNavigator(StubMapping(), z_dim=4),
        weight_blender=WeightBlender(),
        snapshot_manager=_RacySnapshotManager(index=2),
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
    engine._last_status = "30.0 fps | snap-0.pkl (100%) | snap-1.pkl (0%)"
    plain = engine.render_frame()
    state.update(debug=True)
    overlaid = engine.render_frame()
    assert overlaid.shape == plain.shape == (64, 64, 3)
    assert np.array_equal(plain, np.full((64, 64, 3), 128, np.uint8))
    assert not np.array_equal(overlaid, plain)


def test_build_engine_uses_the_injected_runtime_state(monkeypatch):
    """build_engine must wire in a caller-supplied RuntimeState so the GUI can
    keep one state object alive across engine rebuilds."""
    monkeypatch.setattr(
        "balagan.core.engine.load_canonical_mapping",
        lambda pkl, device: StubMapping(),
    )
    config = EngineConfig(
        snapshots_dir=Path("run"),
        canonical_index=2,
        snapshots=tuple(SNAPSHOTS),
    )
    sentinel = RuntimeState()
    engine = build_engine(config, "cpu", window_size=3, runtime_state=sentinel)
    assert engine.runtime_state is sentinel


def test_debug_overlay_draws_frame_counter_before_first_status():
    """The frame counter must be drawn every frame, even before the first
    per-second status report, so it can gauge end-to-end rendering delay."""
    engine, state, _ = make_engine()
    engine.prime()
    state.update(debug=True)
    frame = engine.render_frame()
    assert not np.array_equal(frame, np.full((64, 64, 3), 128, np.uint8))


class _FakeClock:
    """Deterministic stand-in for the ``time`` module the engine calls."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_limit_framerate_absorbs_post_render_overhead(monkeypatch):
    engine, _, _ = make_engine()
    clock = _FakeClock()
    monkeypatch.setattr("balagan.core.engine.time", clock)

    fps_cap = 30
    period = 1.0 / fps_cap
    render = 0.005
    post = 0.006
    starts = []
    for _ in range(12):
        starts.append(clock.now)
        clock.advance(render)
        engine._limit_framerate(fps_cap, starts[-1])
        clock.advance(post)

    periods = [b - a for a, b in zip(starts, starts[1:])]
    for measured in periods[1:]:
        assert measured == pytest.approx(period, abs=1e-9)


def test_limit_framerate_disabled_resets_deadline_and_never_sleeps(monkeypatch):
    engine, _, _ = make_engine()
    clock = _FakeClock()
    monkeypatch.setattr("balagan.core.engine.time", clock)

    engine._limit_framerate(30, clock.now)
    assert engine._next_deadline is not None

    clock.sleeps.clear()
    engine._limit_framerate(0, clock.now)
    assert engine._next_deadline is None
    assert clock.sleeps == []


def test_limit_framerate_resyncs_after_a_stall(monkeypatch):
    engine, _, _ = make_engine()
    clock = _FakeClock()
    monkeypatch.setattr("balagan.core.engine.time", clock)

    fps_cap = 30
    period = 1.0 / fps_cap
    engine._limit_framerate(fps_cap, clock.now)

    clock.advance(period * 5)
    clock.sleeps.clear()
    engine._limit_framerate(fps_cap, clock.now)
    assert clock.sleeps == []
    assert engine._next_deadline == pytest.approx(clock.now + period, abs=1e-9)

    frame_start = clock.now
    clock.advance(0.001)
    engine._limit_framerate(fps_cap, frame_start)
    assert clock.sleeps[-1] == pytest.approx(period - 0.001, abs=1e-9)
