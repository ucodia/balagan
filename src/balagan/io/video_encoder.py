"""Low-latency H.264/HEVC encoder for the WebTransport streaming output.

Takes ``[H, W, 3]`` uint8 RGB frames and yields encoded chunks. The encoder is
selected by platform, mirroring ``frame_output.py``'s Syphon/Spout dispatch:
VideoToolbox on macOS, NVENC on Windows, libx264 as a CI / no-hardware fallback.
Only the :class:`EncoderConfig` differs between platforms; the encode path is
identical.

**Bitstream framing is Annex B with in-band SPS/PPS** (repeated on every
keyframe). The browser's ``VideoDecoder.configure`` must therefore be called
**without** a ``description``. Both the hardware encoders and libx264 emit this
framing natively when used without a container, so no bitstream filter is needed.
"""

import logging
import sys
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderConfig:
    """Knobs that differ between platforms; the encode path stays identical.

    ``codec`` is the FFmpeg encoder name (e.g. ``h264_videotoolbox``,
    ``h264_nvenc``, ``libx264``). ``intra_refresh`` trades periodic IDR keyframes
    for a rolling intra refresh to avoid keyframe bitrate spikes; it is an NVENC
    capability and is ignored by encoders that do not support it.
    """

    codec: str
    bitrate: int
    fps: int
    keyframe_interval: int
    intra_refresh: bool
    options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedChunk:
    """One encoded access unit and whether it is an IDR/keyframe."""

    data: bytes
    is_keyframe: bool


def default_config(*, fps: int, bitrate: int, hevc: bool = False) -> EncoderConfig:
    """Build the platform-appropriate encoder config.

    macOS uses VideoToolbox with realtime CBR; Windows uses NVENC's
    ultra-low-latency preset with intra refresh; anything else falls back to
    libx264 ``tune=zerolatency`` (not a shipping path).
    """
    family = "hevc" if hevc else "h264"
    keyframe_interval = max(1, fps * 2)

    if sys.platform == "darwin":
        # Low-latency rate control does not combine with VBR, so use CBR only.
        return EncoderConfig(
            codec=f"{family}_videotoolbox",
            bitrate=bitrate,
            fps=fps,
            keyframe_interval=keyframe_interval,
            intra_refresh=False,
            options={"realtime": "true", "prio_speed": "true"},
        )
    if sys.platform == "win32":
        return EncoderConfig(
            codec=f"{family}_nvenc",
            bitrate=bitrate,
            fps=fps,
            keyframe_interval=keyframe_interval,
            intra_refresh=True,
            options={"preset": "p1", "tune": "ull", "rc": "cbr", "bf": "0"},
        )
    return EncoderConfig(
        codec="libx265" if hevc else "libx264",
        bitrate=bitrate,
        fps=fps,
        keyframe_interval=keyframe_interval,
        intra_refresh=False,
        options={"tune": "zerolatency"},
    )


class VideoEncoder:
    """Encodes successive RGB frames into Annex B H.264/HEVC chunks.

    A frame-size change transparently rebuilds the underlying codec context, so
    the next emitted chunk is a fresh keyframe.
    """

    def __init__(self, width: int, height: int, config: EncoderConfig) -> None:
        self._config = config
        self._width = width
        self._height = height
        self._pts = 0
        self._context = self._build_context(width, height)

    def _build_context(self, width: int, height: int):
        import av  # lazy: PyAV is a heavy, optional dependency

        context = av.CodecContext.create(self._config.codec, "w")
        context.width = width
        context.height = height
        context.pix_fmt = "yuv420p"
        context.framerate = self._config.fps
        context.bit_rate = self._config.bitrate
        context.gop_size = self._config.keyframe_interval
        options = dict(self._config.options)
        if self._config.intra_refresh:
            options["intra-refresh"] = "1"
            options["forced-idr"] = "0"
        context.options = options
        return context

    def encode(self, frame_uint8_rgb: np.ndarray) -> list[EncodedChunk]:
        """Encode one ``[H, W, 3]`` uint8 RGB frame, returning ready chunks.

        The list may be empty if the encoder is buffering; recreates the codec
        context when the frame size changes from the previous call.
        """
        import av

        height, width = frame_uint8_rgb.shape[:2]
        if (width, height) != (self._width, self._height):
            logger.info(
                "Frame size changed %dx%d -> %dx%d; rebuilding encoder",
                self._width,
                self._height,
                width,
                height,
            )
            self.close()
            self._width, self._height = width, height
            self._context = self._build_context(width, height)

        frame = av.VideoFrame.from_ndarray(frame_uint8_rgb, format="rgb24")
        frame.pts = self._pts
        self._pts += 1
        return [self._to_chunk(packet) for packet in self._context.encode(frame)]

    def close(self) -> list[EncodedChunk]:
        """Flush the encoder, returning any remaining chunks."""
        if self._context is None:
            return []
        chunks = [self._to_chunk(packet) for packet in self._context.encode(None)]
        self._context = None
        return chunks

    @staticmethod
    def _to_chunk(packet) -> EncodedChunk:
        return EncodedChunk(data=bytes(packet), is_keyframe=bool(packet.is_keyframe))
