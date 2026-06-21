"""Tests for the streaming video encoder.

The libx264 fallback path always runs so CI without hardware still exercises the
encoder. Hardware-encoder assertions are gated on platform and skipped when the
encoder (or its hardware) is unavailable.
"""

import sys

import numpy as np
import pytest

from balagan.io.video_encoder import (
    DEFAULT_WEB_CODEC,
    EncoderConfig,
    VideoEncoder,
    config_for,
)


def _libx264_config() -> EncoderConfig:
    return EncoderConfig(
        codec="libx264",
        bitrate=2_000_000,
        fps=30,
        keyframe_interval=30,
        intra_refresh=False,
    )


def _solid_frame(width: int, height: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_emits_chunks_with_keyframe_first():
    encoder = VideoEncoder(64, 64, _libx264_config())
    chunks = []
    for i in range(8):
        chunks.extend(encoder.encode(_solid_frame(64, 64, i * 30)))
    chunks.extend(encoder.close())

    assert chunks, "encoder produced no chunks"
    assert chunks[0].is_keyframe
    assert all(isinstance(c.data, bytes) and c.data for c in chunks)


def test_handles_frame_size_change():
    encoder = VideoEncoder(64, 64, _libx264_config())
    for i in range(4):
        encoder.encode(_solid_frame(64, 64, i * 30))

    resized = encoder.encode(_solid_frame(96, 96, 200))
    resized.extend(encoder.close())

    assert resized, "no chunks emitted after frame-size change"
    assert resized[0].is_keyframe


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"),
    reason="hardware encoders only exist on macOS (VideoToolbox) and Windows (NVENC)",
)
def test_default_web_codec_is_software():
    # The web path defaults to libx264 on every platform; hardware H.264 encoders
    # produce bitstreams that browser decoders stall on at keyframes.
    assert DEFAULT_WEB_CODEC == "libx264"
    config = config_for(DEFAULT_WEB_CODEC, fps=30, bitrate=4_000_000)
    assert config.options.get("tune") == "zerolatency"


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"),
    reason="hardware encoders only exist on macOS (VideoToolbox) and Windows (NVENC)",
)
def test_hardware_encoder_is_tuned_and_usable():
    codec = "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"
    config = config_for(codec, fps=30, bitrate=4_000_000)
    try:
        encoder = VideoEncoder(64, 64, config)
    except Exception:  # noqa: BLE001 — no hardware encoder on this machine
        pytest.skip(f"{codec} unavailable on this machine")

    chunks = []
    for i in range(4):
        chunks.extend(encoder.encode(_solid_frame(64, 64, i * 30)))
    chunks.extend(encoder.close())

    assert chunks
    assert chunks[0].is_keyframe
