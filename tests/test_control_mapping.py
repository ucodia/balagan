"""Tests for the shared control-address mapping used by OSC and the web channel."""

import logging

from balagan.core.runtime_state import RuntimeState
from balagan.io.control_mapping import apply_control


def test_each_address_updates_its_field():
    state = RuntimeState()
    apply_control(state, "/position", 0.5)
    apply_control(state, "/seedX", 1.5)
    apply_control(state, "/seedY", -2.5)
    apply_control(state, "/seedAnim", 1)
    apply_control(state, "/seedSpeedX", 3.0)
    apply_control(state, "/seedSpeedY", -1.5)
    apply_control(state, "/truncation", 0.4)
    apply_control(state, "/fpsCap", 24)
    apply_control(state, "/debug", 1)

    snapshot = state.snapshot()
    assert snapshot.position == 0.5
    assert snapshot.latent_x == 1.5
    assert snapshot.latent_y == -2.5
    assert snapshot.anim_playing is True
    assert snapshot.anim_speed_x == 3.0
    assert snapshot.anim_speed_y == -1.5
    assert snapshot.truncation_psi == 0.4
    assert snapshot.fps_cap == 24
    assert snapshot.debug is True


def test_fps_cap_is_int_and_clamped():
    state = RuntimeState()
    apply_control(state, "/fpsCap", 200)
    assert state.snapshot().fps_cap == 120
    apply_control(state, "/fpsCap", -5)
    assert state.snapshot().fps_cap == 0
    apply_control(state, "/fpsCap", 30.9)
    assert state.snapshot().fps_cap == 30


def test_position_and_truncation_are_clamped():
    state = RuntimeState()
    apply_control(state, "/position", 1.5)
    apply_control(state, "/truncation", -0.2)
    snapshot = state.snapshot()
    assert snapshot.position == 1.0
    assert snapshot.truncation_psi == 0.0


def test_seed_anim_accepts_boolean_and_numeric():
    state = RuntimeState()
    apply_control(state, "/seedAnim", 0)
    assert state.snapshot().anim_playing is False
    apply_control(state, "/seedAnim", True)
    assert state.snapshot().anim_playing is True


def test_malformed_value_is_warned_and_ignored(caplog):
    state = RuntimeState()
    with caplog.at_level(logging.WARNING):
        applied = apply_control(state, "/position", "not-a-number")
    assert applied is False
    assert state.snapshot().position == 0.0
    assert "malformed" in caplog.text.lower()


def test_unknown_address_is_warned_and_ignored(caplog):
    state = RuntimeState()
    with caplog.at_level(logging.WARNING):
        applied = apply_control(state, "/nonexistent", 1.0)
    assert applied is False
