"""Records rendered frames to an MP4 file via imageio's bundled ffmpeg.

Frames arrive as ``[H, W, 3]`` uint8 RGB arrays on the render thread; the
recorder encodes them to H.264 in real time. ``imageio-ffmpeg`` ships its own
ffmpeg binary, so recording works on macOS and Windows without a system
install. The render worker creates the recorder lazily when recording is
toggled on and closes it when toggled off.
"""

import logging
from pathlib import Path

import imageio
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Encodes successive RGB frames to an H.264 MP4 file."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._width = width
        self._height = height
        # macro_block_size=None disables imageio's silent resize-to-16-multiple;
        # the model canvas is a power of two, so it already satisfies H.264's
        # even-dimension requirement and must be preserved exactly.
        self._writer = imageio.get_writer(
            str(self._path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        logger.info(
            "Recording started: %s (%dx%d @ %d fps)", self._path, width, height, fps
        )

    @property
    def path(self) -> Path:
        return self._path

    def write(self, frame_uint8_rgb: np.ndarray) -> None:
        """Append one ``[H, W, 3]`` uint8 RGB frame to the video."""
        self._writer.append_data(frame_uint8_rgb)

    def close(self) -> None:
        """Finalize the file and release the encoder."""
        self._writer.close()
        logger.info("Recording saved: %s", self._path)
