"""WebTransport (HTTP/3 / QUIC) streaming output.

Encodes each rendered frame (see :mod:`balagan.io.video_encoder`) and pushes it
to subscribed browsers over WebTransport. Each frame is written on its own
**unidirectional** QUIC stream so a stalled or lost frame cannot head-of-line
block the next one. A small fixed header precedes each frame:

    flags (uint8, bit 0 = keyframe) | sequence (uint32) | timestamp_ms (uint64)

aioquic runs an asyncio event loop, but :meth:`WebStreamOutput.send` is called
from the synchronous render thread, so the loop runs on its own daemon thread and
``send`` marshals encoded chunks onto it via ``call_soon_threadsafe``. The render
thread is never blocked on the network: chunks land in a bounded queue that drops
the oldest frame on overflow, because stale frames are worthless for live
performance.

This module is the only ``io`` sink that needs network and encoder imports; they
are kept lazy so importing the module stays cheap on machines without aioquic.
"""

import asyncio
import logging
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# flags (uint8) | sequence (uint32) | timestamp in ms (uint64)
_HEADER = struct.Struct(">BIQ")
_KEYFRAME_FLAG = 0x01
_DEFAULT_QUEUE_SIZE = 8


class WebStreamOutput:
    """Streams encoded frames to browsers over WebTransport.

    Implements the same ``send``/``close`` contract as the Syphon/Spout sinks, so
    the render loop treats it identically. ``runtime_state`` is accepted now and
    used by the upstream control channel in a later phase.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        *,
        cert: Path,
        key: Path,
        runtime_state=None,
        host: str = "0.0.0.0",
        port: int = 4433,
        fps: int = 30,
        bitrate: int = 25_000_000,
        codec: str | None = None,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        from balagan.io.video_encoder import EncoderConfig, default_config

        self._name = name
        self._host = host
        self._port = port
        self._cert = Path(cert)
        self._key = Path(key)
        self._runtime_state = runtime_state
        self._queue_size = queue_size

        if codec is None:
            self._encoder_config = default_config(fps=fps, bitrate=bitrate)
        else:
            self._encoder_config = EncoderConfig(
                codec=codec,
                bitrate=bitrate,
                fps=fps,
                keyframe_interval=max(1, fps * 2),
                intra_refresh=False,
                options={"tune": "zerolatency"} if codec.startswith("libx") else {},
            )

        from balagan.io.video_encoder import VideoEncoder

        self._encoder = VideoEncoder(width, height, self._encoder_config)
        self._sequence = 0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._subscribers: set = set()
        self._pending: deque | None = None
        self._wakeup: asyncio.Event | None = None
        self._drain_task: asyncio.Task | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="web-stream", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("WebTransport server failed to start")
        logger.info(
            "WebTransport server listening on %s:%d (codec=%s)",
            self._host,
            self._port,
            self._encoder_config.codec,
        )

    # -- render-thread API ---------------------------------------------------

    def send(self, frame_uint8_rgb: np.ndarray) -> None:
        """Encode one ``[H, W, 3]`` uint8 RGB frame and queue it for delivery."""
        for chunk in self._encoder.encode(frame_uint8_rgb):
            seq = self._sequence
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            flags = _KEYFRAME_FLAG if chunk.is_keyframe else 0
            payload = _HEADER.pack(flags, seq, int(time.time() * 1000)) + chunk.data
            loop = self._loop
            if loop is None:
                return
            try:
                loop.call_soon_threadsafe(self._enqueue, payload)
            except RuntimeError:
                # Loop is shutting down; drop the frame rather than crash render.
                return

    def close(self) -> None:
        """Flush the encoder, stop the server, and join the loop thread."""
        try:
            self._encoder.close()
        except Exception:  # noqa: BLE001 — best-effort flush during teardown
            logger.exception("Encoder flush failed during close")
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._shutdown)
        self._thread.join(timeout=5)

    # -- loop-thread internals ----------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._pending = deque(maxlen=self._queue_size)
        self._wakeup = asyncio.Event()
        try:
            loop.run_until_complete(self._start_server())
            self._drain_task = loop.create_task(self._drain())
            self._ready.set()
            loop.run_forever()
        except Exception:  # noqa: BLE001 — surface startup failures to __init__
            logger.exception("WebTransport server thread crashed")
            self._ready.set()
        finally:
            if self._drain_task is not None:
                self._drain_task.cancel()
                try:
                    loop.run_until_complete(self._drain_task)
                except asyncio.CancelledError:
                    pass
            if self._server is not None:
                self._server.close()
            loop.close()

    async def _start_server(self) -> None:
        from aioquic.asyncio import serve
        from aioquic.h3.connection import H3_ALPN
        from aioquic.quic.configuration import QuicConfiguration

        config = QuicConfiguration(
            is_client=False,
            alpn_protocols=H3_ALPN,
            max_datagram_frame_size=65536,
        )
        config.load_cert_chain(str(self._cert), str(self._key))
        self._server = await serve(
            self._host,
            self._port,
            configuration=config,
            create_protocol=self._create_protocol,
        )

    def _create_protocol(self, *args, **kwargs):
        return _protocol_class()(*args, server=self, **kwargs)

    def _enqueue(self, payload: bytes) -> None:
        # deque(maxlen) drops the oldest payload when full: exactly the
        # drop-oldest backpressure we want for live frames.
        self._pending.append(payload)
        self._wakeup.set()

    async def _drain(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while self._pending:
                payload = self._pending.popleft()
                for protocol in list(self._subscribers):
                    protocol.send_frame(payload)

    def _shutdown(self) -> None:
        for protocol in list(self._subscribers):
            protocol.close()
        self._subscribers.clear()
        self._loop.stop()

    # -- subscriber registry (loop thread only) ------------------------------

    def register(self, protocol) -> None:
        self._subscribers.add(protocol)
        logger.info("WebTransport client connected (%d total)", len(self._subscribers))

    def unregister(self, protocol) -> None:
        self._subscribers.discard(protocol)
        logger.info(
            "WebTransport client disconnected (%d total)", len(self._subscribers)
        )

    def on_control(self, message: bytes) -> None:
        """Hook for the upstream control channel; wired up in a later phase."""


_PROTOCOL_CLASS = None


def _protocol_class():
    """Build and cache the WebTransport protocol class.

    aioquic's ``QuicConnectionProtocol`` is only importable at runtime, so the
    subclass is defined lazily on first use rather than at module import.
    """
    global _PROTOCOL_CLASS
    if _PROTOCOL_CLASS is None:
        _PROTOCOL_CLASS = _make_protocol_class()
    return _PROTOCOL_CLASS


def _make_protocol_class():
    from aioquic.asyncio import QuicConnectionProtocol
    from aioquic.h3.connection import H3Connection
    from aioquic.h3.events import HeadersReceived
    from aioquic.quic.events import ProtocolNegotiated

    class _Protocol(QuicConnectionProtocol):
        def __init__(self, *args, server: WebStreamOutput, **kwargs):
            super().__init__(*args, **kwargs)
            self._server = server
            self._http: H3Connection | None = None
            self._session_id: int | None = None

        def quic_event_received(self, event) -> None:
            if isinstance(event, ProtocolNegotiated):
                self._http = H3Connection(self._quic, enable_webtransport=True)
            if self._http is not None:
                for h3_event in self._http.handle_event(event):
                    self._h3_event_received(h3_event)

        def _h3_event_received(self, event) -> None:
            if isinstance(event, HeadersReceived):
                headers = dict(event.headers)
                if (
                    headers.get(b":method") == b"CONNECT"
                    and headers.get(b":protocol") == b"webtransport"
                ):
                    self._session_id = event.stream_id
                    self._http.send_headers(
                        stream_id=event.stream_id,
                        headers=[
                            (b":status", b"200"),
                            (b"sec-webtransport-http3-draft", b"draft02"),
                        ],
                    )
                    self._server.register(self)
                    self.transmit()

        def send_frame(self, payload: bytes) -> None:
            if self._session_id is None or self._http is None:
                return
            try:
                stream_id = self._http.create_webtransport_stream(
                    self._session_id, is_unidirectional=True
                )
                self._quic.send_stream_data(stream_id, payload, end_stream=True)
                self.transmit()
            except Exception:  # noqa: BLE001 — drop this frame, keep the session
                logger.debug("Dropping frame for a failed WebTransport stream")

        def connection_lost(self, exc) -> None:
            self._server.unregister(self)
            super().connection_lost(exc)

    return _Protocol
