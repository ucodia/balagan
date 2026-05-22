"""Windows frame output via Spout (SpoutGL).

UNVERIFIED: this module could not be exercised during development -- SpoutGL
ships Windows-only wheels, so it cannot be installed or run on the macOS
development/verification machine. The class implements the FrameOutput
interface; the SpoutGL calls below need verification on Windows before a
Windows deployment.
"""

import numpy as np
import SpoutGL
from OpenGL.GL import GL_RGBA


class SpoutOutput:
    """Publishes rendered frames to a Spout sender for Windows clients."""

    def __init__(self, name: str, width: int, height: int) -> None:
        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(name)
        self._width = width
        self._height = height
        self._rgba = np.empty((height, width, 4), dtype=np.uint8)
        self._rgba[:, :, 3] = 255  # opaque alpha, written once

    def send(self, frame_uint8_rgb: np.ndarray) -> None:
        """Publish a [H, W, 3] uint8 RGB frame to the Spout sender."""
        self._rgba[:, :, :3] = frame_uint8_rgb
        self._sender.sendImage(
            self._rgba.tobytes(), self._width, self._height, GL_RGBA, False, 0
        )

    def close(self) -> None:
        """Release the Spout sender."""
        self._sender.releaseSender()
