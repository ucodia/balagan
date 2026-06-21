"""Platform-dispatched frame output for Syphon (macOS) and Spout (Windows).

The ``FrameOutput`` name resolves at import time to the platform's
implementation, or to a no-op stub on platforms with neither. ``build_output``
selects between that native sink and the WebTransport streaming sink based on the
``--output`` choice, keeping the two paths un-entangled.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_WEB_FPS = 30


@dataclass(frozen=True)
class OutputSettings:
    """Resolved output configuration, threaded from the CLI to the render loop.

    ``kind`` is ``auto`` / ``spout-syphon`` (both use the native sink) or ``web``
    (the WebTransport sink). The ``web_*`` fields are ignored by the native sink.
    """

    kind: str = "auto"
    name: str = "BalaGAN"
    web_port: int = 4433
    web_bitrate: int = 25_000_000
    web_codec: str | None = None
    web_cert: Path | None = None
    web_key: Path | None = None


def build_output(
    settings: OutputSettings, width: int, height: int, *, runtime_state=None
):
    """Construct the output sink for ``settings``.

    Returns the platform ``FrameOutput`` for ``auto``/``spout-syphon`` and a
    ``WebStreamOutput`` for ``web``. The web sink is imported lazily so its
    network/encoder dependencies are only loaded when actually selected.
    """
    if settings.kind == "web":
        from balagan.io.web_stream import WebStreamOutput

        fps = _DEFAULT_WEB_FPS
        if runtime_state is not None:
            fps = runtime_state.snapshot().fps_cap or _DEFAULT_WEB_FPS
        return WebStreamOutput(
            settings.name,
            width,
            height,
            cert=settings.web_cert,
            key=settings.web_key,
            runtime_state=runtime_state,
            port=settings.web_port,
            fps=fps,
            bitrate=settings.web_bitrate,
            codec=settings.web_codec,
        )
    return FrameOutput(settings.name, width, height)


class NoOpFrameOutput:
    """Frame output stub for platforms without Syphon or Spout."""

    def __init__(self, name: str, width: int, height: int) -> None:
        self._name = name
        self._width = width
        self._height = height

    def send(self, frame_uint8_rgb: np.ndarray) -> None:
        """Discard the frame: there is no output sink on this platform."""

    def close(self) -> None:
        """Nothing to release."""


if sys.platform == "darwin":
    from balagan.io.output_macos import SyphonOutput as FrameOutput
elif sys.platform == "win32":
    from balagan.io.output_windows import SpoutOutput as FrameOutput
else:
    FrameOutput = NoOpFrameOutput
