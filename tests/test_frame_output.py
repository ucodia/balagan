"""Tests for balagan.io.frame_output: platform dispatch + no-op stub."""

import sys

import numpy as np

from balagan.io.frame_output import FrameOutput, NoOpFrameOutput


def test_frame_output_is_dispatched_for_the_platform():
    if sys.platform == "darwin":
        from balagan.io.output_macos import SyphonOutput

        assert FrameOutput is SyphonOutput
    elif sys.platform == "win32":
        from balagan.io.output_windows import SpoutOutput

        assert FrameOutput is SpoutOutput
    else:
        assert FrameOutput is NoOpFrameOutput
