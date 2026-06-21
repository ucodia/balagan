"""OSC control server: maps incoming OSC messages onto the runtime state."""

import logging
import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from balagan.core.runtime_state import RuntimeState
from balagan.io.control_mapping import CONTROL_ADDRESSES, apply_control

logger = logging.getLogger(__name__)


def _build_dispatcher(runtime_state: RuntimeState) -> Dispatcher:
    """Build the OSC dispatcher mapping the control endpoints to state updates.

    Conversion and clamping are delegated to
    :func:`balagan.io.control_mapping.apply_control`, the shared vocabulary used
    by both OSC and the web control channel. Messages with missing arguments are
    logged as warnings and ignored.
    """
    dispatcher = Dispatcher()

    def handler(address: str, *args) -> None:
        if not args:
            logger.warning("Ignoring malformed OSC message: %s %r", address, args)
            return
        apply_control(runtime_state, address, args[0])

    for address in CONTROL_ADDRESSES:
        dispatcher.map(address, handler)
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

    def restart(self, port: int) -> None:
        """Rebind the server to a new port. Raises OSError if the port is
        unavailable, leaving the server stopped for the caller to recover."""
        self.stop()
        self._port = port
        self.start()

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
