"""Qt render worker: runs the engine's per-frame loop on a background thread."""

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class RenderWorker(QThread):
    """Runs the engine loop on its own thread, emitting each rendered frame as a
    QImage plus the engine's status line. While the runtime state's Spout/Syphon
    checkbox is enabled, frames are also published to a lazily-created output.
    """

    frame_ready = Signal(QImage)
    status_changed = Signal(str)

    def __init__(self, engine, output_name: str) -> None:
        super().__init__()
        self._engine = engine
        self._output_name = output_name
        self._output = None
        self._running = False

    def run(self) -> None:
        self._running = True
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
        if self._output is not None:
            self._output.close()
        self._engine.stop()

    def _publish(self, frame, width: int, height: int) -> None:
        if not self._engine.runtime_state.snapshot().spout_syphon_enabled:
            return
        if self._output is None:
            from balagan.io.frame_output import FrameOutput

            self._output = FrameOutput(self._output_name, width, height)
        self._output.send(frame)

    def stop(self) -> None:
        """Stop the loop and wait for the thread to finish."""
        self._running = False
        self.wait()
