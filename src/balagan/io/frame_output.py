"""Platform-dispatched frame output for Syphon (macOS) and Spout (Windows).

The ``FrameOutput`` name resolves at import time to the platform's
implementation, or to a no-op stub on platforms with neither.
"""

import sys

import numpy as np


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
