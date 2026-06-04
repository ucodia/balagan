"""Tests for balagan.io.osc_server: OSC control message handling."""

import logging
import time

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import SimpleUDPClient

from balagan.core.runtime_state import RuntimeState
from balagan.io.osc_server import OSCServer, _build_dispatcher


def _dispatch(dispatcher, address: str, *values) -> None:
    """Encode an OSC message and route it through the dispatcher (no socket)."""
    builder = OscMessageBuilder(address=address)
    for value in values:
        builder.add_arg(value)
    dispatcher.call_handlers_for_packet(builder.build().dgram, ("127.0.0.1", 0))


def test_position_message_updates_state():
    state = RuntimeState()
    _dispatch(_build_dispatcher(state), "/position", 0.5)
    assert state.snapshot().position == 0.5


def test_position_value_is_clamped_to_the_unit_range():
    state = RuntimeState()
    dispatcher = _build_dispatcher(state)
    _dispatch(dispatcher, "/position", 1.5)
    assert state.snapshot().position == 1.0
    _dispatch(dispatcher, "/position", -0.25)
    assert state.snapshot().position == 0.0


def test_seed_x_message_updates_latent_x():
    state = RuntimeState()
    _dispatch(_build_dispatcher(state), "/seedX", 1.5)
    assert state.snapshot().latent_x == 1.5


def test_seed_y_message_updates_latent_y():
    state = RuntimeState()
    _dispatch(_build_dispatcher(state), "/seedY", -2.5)
    assert state.snapshot().latent_y == -2.5


def test_seed_anim_message_toggles_playing():
    state = RuntimeState()
    dispatcher = _build_dispatcher(state)
    _dispatch(dispatcher, "/seedAnim", 1)
    assert state.snapshot().anim_playing is True
    _dispatch(dispatcher, "/seedAnim", 0)
    assert state.snapshot().anim_playing is False


def test_seed_speed_x_message_updates_state():
    state = RuntimeState()
    _dispatch(_build_dispatcher(state), "/seedSpeedX", 3.0)
    assert state.snapshot().anim_speed_x == 3.0


def test_seed_speed_y_message_updates_state():
    state = RuntimeState()
    _dispatch(_build_dispatcher(state), "/seedSpeedY", -1.5)
    assert state.snapshot().anim_speed_y == -1.5


def test_truncation_message_updates_and_clamps():
    state = RuntimeState()
    dispatcher = _build_dispatcher(state)
    _dispatch(dispatcher, "/truncation", 0.5)
    assert state.snapshot().truncation_psi == 0.5
    _dispatch(dispatcher, "/truncation", 2.0)
    assert state.snapshot().truncation_psi == 1.0


def test_malformed_message_is_warned_and_ignored(caplog):
    state = RuntimeState()
    dispatcher = _build_dispatcher(state)
    with caplog.at_level(logging.WARNING):
        _dispatch(dispatcher, "/position")  # no argument
    assert state.snapshot().position == 0.0  # unchanged from the default
    assert "malformed" in caplog.text.lower()


def test_server_receives_messages_over_udp():
    state = RuntimeState()
    server = OSCServer(state, host="127.0.0.1", port=0)
    server.start()
    try:
        client = SimpleUDPClient("127.0.0.1", server.port)
        client.send_message("/position", 0.5)
        deadline = time.monotonic() + 3.0
        while state.snapshot().position != 0.5 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.snapshot().position == 0.5
    finally:
        server.stop()
