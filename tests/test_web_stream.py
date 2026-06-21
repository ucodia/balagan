"""Integration tests for the WebTransport streaming output.

No browser is involved. The first tests confirm the QUIC server thread starts,
that ``send`` accepts frames without blocking the caller, and that ``close``
tears the thread down cleanly. The last test drives a real aioquic WebTransport
client over loopback and asserts encoded frames actually arrive on the wire. The
encoder is forced to libx264 so everything runs anywhere.
"""

import asyncio
import ssl
import struct

import numpy as np

from balagan.io.dev_cert import generate_self_signed_cert
from balagan.io.web_stream import WebStreamOutput


def _solid_frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _make_output(tmp_path, port: int = 0, **kwargs) -> WebStreamOutput:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    generate_self_signed_cert(cert, key)
    return WebStreamOutput(
        "test",
        64,
        64,
        cert=cert,
        key=key,
        port=port,
        fps=30,
        bitrate=1_000_000,
        codec="libx264",
        **kwargs,
    )


def test_server_starts_and_accepts_frames(tmp_path):
    output = _make_output(tmp_path)
    try:
        for i in range(10):
            output.send(_solid_frame(i * 20))
    finally:
        output.close()


def test_send_drops_rather_than_blocks_with_no_subscriber(tmp_path):
    # With no client connected, the bounded queue must drop frames instead of
    # blocking the render thread. Pushing far more than the queue size should
    # return promptly and never raise.
    output = _make_output(tmp_path, queue_size=4)
    try:
        for i in range(200):
            output.send(_solid_frame(i % 255))
    finally:
        output.close()


def _make_h3_client_class():
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.h3.connection import H3Connection
    from aioquic.h3.events import HeadersReceived, WebTransportStreamDataReceived
    from aioquic.quic.events import ProtocolNegotiated

    class _H3Client(QuicConnectionProtocol):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._http = None
            self._buffers = {}
            self._session_id = None
            self.frames = []
            self.session_ready = asyncio.get_event_loop().create_future()

        def quic_event_received(self, event):
            if isinstance(event, ProtocolNegotiated):
                self._http = H3Connection(self._quic, enable_webtransport=True)
            if self._http is not None:
                for h3_event in self._http.handle_event(event):
                    self._on_h3(h3_event)

        def _on_h3(self, event):
            if isinstance(event, HeadersReceived):
                if not self.session_ready.done():
                    self.session_ready.set_result(dict(event.headers))
            elif isinstance(event, WebTransportStreamDataReceived):
                buffer = self._buffers.get(event.stream_id, b"") + event.data
                if event.stream_ended:
                    self.frames.append(buffer)
                    self._buffers.pop(event.stream_id, None)
                else:
                    self._buffers[event.stream_id] = buffer

        async def connect_webtransport(self, authority: str):
            while self._http is None:
                await asyncio.sleep(0.01)
            stream_id = self._quic.get_next_available_stream_id()
            self._session_id = stream_id
            self._http.send_headers(
                stream_id=stream_id,
                headers=[
                    (b":method", b"CONNECT"),
                    (b":protocol", b"webtransport"),
                    (b":scheme", b"https"),
                    (b":authority", authority.encode()),
                    (b":path", b"/balagan"),
                ],
                end_stream=False,
            )
            self.transmit()

        def send_control(self, line: bytes):
            stream_id = self._http.create_webtransport_stream(
                self._session_id, is_unidirectional=False
            )
            self._quic.send_stream_data(stream_id, line, end_stream=False)
            self.transmit()

    return _H3Client


async def _stream_to_loopback_client(output: WebStreamOutput, port: int):
    from aioquic.asyncio import connect
    from aioquic.h3.connection import H3_ALPN
    from aioquic.quic.configuration import QuicConfiguration

    config = QuicConfiguration(
        is_client=True, alpn_protocols=H3_ALPN, max_datagram_frame_size=65536
    )
    config.verify_mode = ssl.CERT_NONE
    authority = f"127.0.0.1:{port}"

    async with connect(
        "127.0.0.1", port, configuration=config, create_protocol=_make_h3_client_class()
    ) as client:
        await client.connect_webtransport(authority)
        headers = await asyncio.wait_for(client.session_ready, 5)
        assert headers.get(b":status") == b"200"
        await asyncio.sleep(0.2)
        for i in range(15):
            output.send(_solid_frame(i * 15))
            await asyncio.sleep(0.03)
        await asyncio.sleep(0.5)
        return client.frames


def test_client_receives_encoded_frames_over_webtransport(tmp_path):
    port = 4456
    output = _make_output(tmp_path, port=port)
    try:
        frames = asyncio.run(_stream_to_loopback_client(output, port))
    finally:
        output.close()

    assert frames, "no frames arrived over WebTransport"
    flags, _seq, _ts = struct.unpack(">BIQ", frames[0][:13])
    assert flags & 0x01, "first delivered frame is not a keyframe"
    assert frames[0][13:17] == b"\x00\x00\x00\x01", "payload is not Annex B"


async def _send_control_from_loopback_client(port: int):
    from aioquic.asyncio import connect
    from aioquic.h3.connection import H3_ALPN
    from aioquic.quic.configuration import QuicConfiguration

    config = QuicConfiguration(
        is_client=True, alpn_protocols=H3_ALPN, max_datagram_frame_size=65536
    )
    config.verify_mode = ssl.CERT_NONE
    authority = f"127.0.0.1:{port}"

    async with connect(
        "127.0.0.1", port, configuration=config, create_protocol=_make_h3_client_class()
    ) as client:
        await client.connect_webtransport(authority)
        await asyncio.wait_for(client.session_ready, 5)
        client.send_control(b'{"addr": "/position", "value": 0.42}\n')
        await asyncio.sleep(0.5)


def test_control_message_updates_runtime_state_over_webtransport(tmp_path):
    from balagan.core.runtime_state import RuntimeState

    port = 4457
    state = RuntimeState()
    output = _make_output(tmp_path, port=port, runtime_state=state)
    try:
        asyncio.run(_send_control_from_loopback_client(port))
    finally:
        output.close()

    assert state.snapshot().position == 0.42
