"""Integration check for the video recorder: encode a few frames and read them
back to confirm a valid file is produced at the requested size."""

import imageio.v3 as iio
import numpy as np

from balagan.io.video_recorder import VideoRecorder


def test_records_frames_to_readable_mp4(tmp_path):
    width, height, count = 64, 64, 10
    path = tmp_path / "nested" / "clip.mp4"

    recorder = VideoRecorder(path, width, height, fps=30)
    for i in range(count):
        frame = np.full((height, width, 3), i * 20, dtype=np.uint8)
        recorder.write(frame)
    recorder.close()

    assert path.exists()
    assert path.stat().st_size > 0

    frames = iio.imread(path, index=None)  # all frames, shape [N, H, W, 3]
    assert frames.shape[0] == count
    assert frames.shape[1:3] == (height, width)
