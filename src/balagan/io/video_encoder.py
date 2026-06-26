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


# Default web encoder. libx264 with a low-latency preset is the default on every
# platform: the hardware H.264 encoders (VideoToolbox, NVENC) produce bitstreams
# that browser decoders buffer and stall on at each keyframe, whereas x264's
# `tune=zerolatency` emits the low-latency signaling browsers honour. The hardware
# encoders remain selectable by name for native consumers or experimentation.
DEFAULT_WEB_CODEC = "libx264"


def config_for(codec: str, *, fps: int, bitrate: int) -> EncoderConfig:
    """Build the encoder config for ``codec`` with its low-latency tuning.

    libx264/libx265 use ``preset=superfast tune=zerolatency``; VideoToolbox uses
    realtime CBR; NVENC uses the ultra-low-latency preset with intra refresh.
    Unknown codecs get no extra options.
    """
    # Long GOP on purpose: a keyframe is ~3x a delta frame, so a short interval
    # makes a periodic bandwidth spike that hitches the stream every couple of
    # seconds on a bandwidth-limited browser link. New viewers don't have to wait
    # for the next periodic keyframe — the web sink forces one on connect via
    # VideoEncoder.request_keyframe — so the interval only bounds loss/shed
    # recovery, where ~5 s is fine.
    keyframe_interval = max(1, fps * 5)
    intra_refresh = False
    if codec.startswith("libx"):
        options = {"preset": "superfast", "tune": "zerolatency"}
    elif codec.endswith("_videotoolbox"):
        # Low-latency rate control does not combine with VBR, so use CBR only.
        options = {"realtime": "true", "prio_speed": "true"}
    elif codec.endswith("_nvenc"):
        options = {"preset": "p1", "tune": "ull", "rc": "cbr", "bf": "0"}
        intra_refresh = True
    else:
        options = {}
    return EncoderConfig(
        codec=codec,
        bitrate=bitrate,
        fps=fps,
        keyframe_interval=keyframe_interval,
        intra_refresh=intra_refresh,
        options=options,
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
        self._force_keyframe = False
        self._context = self._build_context(width, height)

    def request_keyframe(self) -> None:
        """Force the next encoded frame to be a keyframe (IDR).

        Thread-safe (a single boolean flag): callers on other threads — e.g. the
        WebTransport loop when a new viewer connects — use this so a fresh client
        can start decoding immediately even with a long GOP, instead of waiting
        for the next periodic keyframe.
        """
        self._force_keyframe = True

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
        if self._force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
            self._force_keyframe = False
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
