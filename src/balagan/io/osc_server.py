"""OSC control server: maps incoming OSC messages onto the runtime state."""

import logging
import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from balagan.core.runtime_state import RuntimeState

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _single_arg(address: str, args: tuple, converter):
    """Convert the first OSC argument, or log a warning and return None."""
    try:
        return converter(args[0])
    except (IndexError, TypeError, ValueError):
        logger.warning("Ignoring malformed OSC message: %s %r", address, args)
        return None


def _build_dispatcher(runtime_state: RuntimeState) -> Dispatcher:
    """Build the OSC dispatcher mapping the six endpoints to state updates.

    Out-of-range position and truncation values are clamped rather than
    rejected; messages with missing or wrongly-typed arguments are logged as
    warnings and ignored.
    """
    dispatcher = Dispatcher()

    def on_position(address: str, *args) -> None:
        value = _single_arg(address, args, float)
        if value is not None:
            runtime_state.update(position=_clamp(value, 0.0, 1.0))

    def on_seed_x(address: str, *args) -> None:
        value = _single_arg(address, args, float)
        if value is not None:
            runtime_state.update(latent_x=value)

    def on_seed_y(address: str, *args) -> None:
        value = _single_arg(address, args, float)
        if value is not None:
            runtime_state.update(latent_y=value)

    def on_seed_anim(address: str, *args) -> None:
        value = _single_arg(address, args, int)
        if value is not None:
            runtime_state.update(anim_playing=bool(value))

    def on_seed_speed(address: str, *args) -> None:
        value = _single_arg(address, args, float)
        if value is not None:
            runtime_state.update(anim_speed=value)

    def on_truncation(address: str, *args) -> None:
        value = _single_arg(address, args, float)
        if value is not None:
            runtime_state.update(truncation_psi=_clamp(value, 0.0, 1.0))

    dispatcher.map("/position", on_position)
    dispatcher.map("/seed/x", on_seed_x)
    dispatcher.map("/seed/y", on_seed_y)
    dispatcher.map("/seed/anim", on_seed_anim)
    dispatcher.map("/seed/speed", on_seed_speed)
    dispatcher.map("/truncation", on_truncation)
    return dispatcher


class OSCServer:
    """Receives OSC control messages on a background thread and applies them to
    the shared runtime state."""

    def __init__(
        self, runtime_state: RuntimeState, host: str = "0.0.0.0", port: int = 7700
    ) -> None:
        self._host = host
        self._port = port
        self._dispatcher = _build_dispatcher(runtime_state)
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the UDP socket and serve OSC messages on a background thread."""
        self._server = ThreadingOSCUDPServer(
            (self._host, self._port), self._dispatcher
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="osc-server", daemon=True
        )
        self._thread.start()
        logger.info("OSC server listening on %s:%d", self._host, self.port)

    def stop(self) -> None:
        """Stop serving and close the UDP socket."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        """The bound UDP port (resolved after start when constructed with port 0)."""
        if self._server is not None:
            return self._server.server_address[1]
        return self._port
