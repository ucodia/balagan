"""Qt render worker: builds the engine and runs its per-frame loop on a thread."""

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

# Output frame rate used when recording while the FPS cap is unlimited (0).
_DEFAULT_RECORD_FPS = 30
_RECORDINGS_DIR = "recordings"


class RenderWorker(QThread):
    """Builds the engine for a config, primes it, and runs its loop on its own
    thread, emitting each rendered frame as a QImage plus the engine's status
    line. The heavy build and prime happen here so the Qt main thread never
    blocks. While the runtime state's Spout/Syphon checkbox is enabled, frames
    are also published to a lazily-created output.
    """

    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    loading_started = Signal(str)
    load_failed = Signal(str)
    recording_changed = Signal(str)
    recording_failed = Signal(str)

    def __init__(
        self, config, device, window_size: int, runtime_state, output_name: str
    ) -> None:
        super().__init__()
        self._config = config
        self._device = device
        self._window_size = window_size
        self._runtime_state = runtime_state
        self._output_name = output_name
        self._engine = None
        self._output = None
        self._recorder = None
        self._running = False

    def run(self) -> None:
        self._running = True
        total = len(self._config.snapshots)
        count = total if self._window_size <= 0 else min(self._window_size, total)
        self.loading_started.emit(f"Loading {count} snapshots…")
        try:
            from balagan.core.engine import build_engine

            self._engine = build_engine(
                self._config,
                self._device,
                self._window_size,
                self._runtime_state,
            )
            self._engine.prime()
        except Exception as exc:  # noqa: BLE001 — surface any build/prime failure
            logger.exception("Engine failed to load")
            self.load_failed.emit(str(exc))
            return

        # stop() may have arrived while we were building; bail before the loop.
        if not self._running:
            return

        self._engine.start()
        while self._running:
            frame = self._engine.render_frame()
            height, width = frame.shape[:2]
            image = QImage(
                frame.data, width, height, width * 3, QImage.Format.Format_RGB888
            ).copy()
            self.frame_ready.emit(image)
            self.status_changed.emit(self._engine.status())
            self._publish(frame, width, height)
            self._record(frame, width, height)
        if self._output is not None:
            self._output.close()
        self._stop_recording()
        self._engine.stop()

    def _publish(self, frame, width: int, height: int) -> None:
        if not self._runtime_state.snapshot().spout_syphon_enabled:
            return
        if self._output is None:
            from balagan.io.frame_output import FrameOutput

            self._output = FrameOutput(self._output_name, width, height)
        self._output.send(frame)

    def _record(self, frame, width: int, height: int) -> None:
        """Open, feed, or close the recorder to track the recording flag.

        The encoder is created on the first frame after recording is enabled and
        released on the first frame after it is disabled, so a single video file
        spans exactly one start/stop press.
        """
        enabled = self._runtime_state.snapshot().recording_enabled
        if enabled and self._recorder is None:
            if not self._start_recording(width, height):
                return
        elif not enabled and self._recorder is not None:
            self._stop_recording()
            return
        if self._recorder is not None:
            self._recorder.write(frame)

    def _start_recording(self, width: int, height: int) -> bool:
        """Create the recorder for the current frame size. On failure, clear the
        recording flag and report it, returning False."""
        fps = self._runtime_state.snapshot().fps_cap or _DEFAULT_RECORD_FPS
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(_RECORDINGS_DIR) / f"balagan_{stamp}.mp4"
        try:
            from balagan.io.video_recorder import VideoRecorder

            self._recorder = VideoRecorder(path, width, height, fps)
        except Exception as exc:  # noqa: BLE001 — surface any encoder failure
            logger.exception("Could not start recording")
            self._runtime_state.update(recording_enabled=False)
            self.recording_failed.emit(str(exc))
            return False
        self.recording_changed.emit(str(self._recorder.path.resolve()))
        return True

    def _stop_recording(self) -> None:
        if self._recorder is None:
            return
        recorder = self._recorder
        self._recorder = None
        recorder.close()
        self.recording_changed.emit("")

    def stop(self) -> None:
        """Stop the loop and wait for the thread to finish."""
        self._running = False
        self.wait()
