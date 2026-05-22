"""Qt main window: viewport on the left, control panel on the right."""

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from balagan.gui.control_panel import ControlPanel
from balagan.gui.render_worker import RenderWorker
from balagan.gui.viewport import Viewport


class MainWindow(QMainWindow):
    """The application window; owns the render worker that drives the engine loop."""

    def __init__(self, engine, run_dir, output_name: str) -> None:
        super().__init__()
        self.setWindowTitle(f"BalaGAN — {Path(run_dir).name}")

        runtime_state = engine.runtime_state
        self._viewport = Viewport(runtime_state)
        self._control_panel = ControlPanel(runtime_state)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._viewport, stretch=1)
        layout.addWidget(self._control_panel)
        self.setCentralWidget(central)

        self._render_worker = RenderWorker(engine, output_name)
        self._render_worker.frame_ready.connect(self._viewport.update_frame)
        self._render_worker.status_changed.connect(self._control_panel.update_status)
        self._render_worker.start()

    def closeEvent(self, event) -> None:
        self._render_worker.stop()
        super().closeEvent(event)
