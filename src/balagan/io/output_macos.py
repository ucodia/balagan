"""macOS frame output via Syphon (syphon-python)."""

import numpy as np
import syphon
from syphon.utils.numpy import copy_image_to_mtl_texture
from syphon.utils.raw import create_mtl_texture


class SyphonOutput:
    """Publishes rendered frames to a Syphon Metal server for macOS clients.

    A single RGBA texture is allocated up front and overwritten each frame;
    the engine's RGB frame is copied into a pre-allocated opaque-alpha buffer.
    """

    def __init__(self, name: str, width: int, height: int) -> None:
        self._server = syphon.SyphonMetalServer(name)
        self._texture = create_mtl_texture(self._server.device, width, height)
        self._rgba = np.empty((height, width, 4), dtype=np.uint8)
        self._rgba[:, :, 3] = 255  # opaque alpha, written once

    def send(self, frame_uint8_rgb: np.ndarray) -> None:
        """Publish a [H, W, 3] uint8 RGB frame as a Syphon Metal texture."""
        self._rgba[:, :, :3] = frame_uint8_rgb
        copy_image_to_mtl_texture(self._rgba, self._texture)
        # Frames are top-row-first; Syphon clients expect bottom-row-first.
        self._server.publish_frame_texture(self._texture, is_flipped=True)

    def close(self) -> None:
        """Stop the Syphon server."""
        self._server.stop()
