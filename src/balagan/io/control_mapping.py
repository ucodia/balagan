"""Shared mapping from control addresses to runtime-state updates.

Both the OSC server and the WebTransport control channel speak the same
vocabulary, so the address list, type conversion, and clamp rules live here once.
Each control names a :class:`~balagan.core.runtime_state.StateSnapshot` field, a
converter for the incoming value, and an optional clamp range.
"""

import logging

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _to_bool(value) -> bool:
    return bool(int(value))


# address -> (state field, converter, clamp range or None)
_CONTROLS: dict[str, tuple[str, object, tuple[float, float] | None]] = {
    "/position": ("position", float, (0.0, 1.0)),
    "/seedX": ("latent_x", float, None),
    "/seedY": ("latent_y", float, None),
    "/seedAnim": ("anim_playing", _to_bool, None),
    "/seedSpeedX": ("anim_speed_x", float, None),
    "/seedSpeedY": ("anim_speed_y", float, None),
    "/truncation": ("truncation_psi", float, (0.0, 1.0)),
    "/fpsCap": ("fps_cap", int, (0, 120)),
    "/debug": ("debug", _to_bool, None),
}

CONTROL_ADDRESSES = tuple(_CONTROLS)


def apply_control(runtime_state, address: str, value) -> bool:
    """Convert, clamp, and apply one control message; return whether it applied.

    Unknown addresses and values that fail conversion are logged as warnings and
    ignored, so a single bad message never disrupts the control stream.
    """
    control = _CONTROLS.get(address)
    if control is None:
        logger.warning("Ignoring unknown control address: %s", address)
        return False

    field, converter, clamp_range = control
    try:
        converted = converter(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed control message: %s %r", address, value)
        return False

    if clamp_range is not None:
        converted = _clamp(converted, *clamp_range)
    runtime_state.update(**{field: converted})
    return True
