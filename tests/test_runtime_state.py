"""Tests for balagan.core.runtime_state: thread-safe shared state."""

import dataclasses
import threading

import pytest

from balagan.core.runtime_state import RuntimeState, StateSnapshot


def test_defaults_match_the_spec():
    snap = RuntimeState().snapshot()
    assert isinstance(snap, StateSnapshot)
    assert snap.position == 0.0
    assert snap.latent_x == 0.0
    assert snap.latent_y == 0.0
    assert snap.anim_playing is False
    assert snap.anim_speed == 0.25
    assert snap.truncation_psi == 0.7
    assert snap.fps_cap == 30
    assert snap.spout_syphon_enabled is False
    assert snap.debug is False


def test_update_changes_a_single_field():
    state = RuntimeState()
    state.update(position=0.42)
    assert state.snapshot().position == 0.42


def test_update_changes_multiple_fields_atomically():
    state = RuntimeState()
    state.update(latent_x=1.5, latent_y=-2.5, anim_playing=True)
    snap = state.snapshot()
    assert snap.latent_x == 1.5
    assert snap.latent_y == -2.5
    assert snap.anim_playing is True


def test_snapshot_is_immutable():
    snap = RuntimeState().snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.position = 0.9


def test_snapshot_is_unaffected_by_later_updates():
    state = RuntimeState()
    before = state.snapshot()
    state.update(position=0.8)
    assert before.position == 0.0  # the earlier snapshot is a stable copy
    assert state.snapshot().position == 0.8


def test_update_rejects_an_unknown_field():
    state = RuntimeState()
    with pytest.raises(TypeError):
        state.update(nightmare_level=1.0)


def test_concurrent_updates_do_not_lose_writes():
    state = RuntimeState()
    iterations = 2000

    def hammer(field: str, final: float) -> None:
        for _ in range(iterations):
            state.update(**{field: 0.0})
        state.update(**{field: final})

    threads = [
        threading.Thread(target=hammer, args=("position", 1.0)),
        threading.Thread(target=hammer, args=("latent_x", 99.0)),
        threading.Thread(target=hammer, args=("truncation_psi", 0.5)),
        threading.Thread(target=hammer, args=("anim_speed", 3.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snap = state.snapshot()
    assert snap.position == 1.0
    assert snap.latent_x == 99.0
    assert snap.truncation_psi == 0.5
    assert snap.anim_speed == 3.0
