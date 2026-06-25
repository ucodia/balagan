"""Qt main window: control panel on the left, viewport on the right.

Owns the engine lifecycle. The window opens with or without a run folder; the
render worker (re)builds the engine on its own thread whenever a folder is
selected or the canonical snapshot changes, so the swap never blocks the UI.
"""

import logging
from importlib.metadata import version
from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QWidget

from balagan.config import ConfigError, load_run
from balagan.gui.control_panel import ControlPanel
from balagan.gui.render_worker import RenderWorker
from balagan.gui.viewport import Viewport

logger = logging.getLogger(__name__)

_VERSION = version("balagan")


class MainWindow(QMainWindow):
    """The application window; owns the render worker that drives the engine loop."""

    def __init__(
        self,
        runtime_state,
        device,
        window_size: int,
        output_name: str,
        osc_server,
        initial_config=None,
    ) -> None:
        super().__init__()
        self._runtime_state = runtime_state
        self._device = device
        self._window_size = window_size
        self._output_name = output_name
        self._osc_server = osc_server
        self._config = None
        self._render_worker: RenderWorker | None = None

        self._viewport = Viewport(runtime_state)
        self._control_panel = ControlPanel(runtime_state, osc_server.port, window_size)
        self._control_panel.folder_selected.connect(self._on_folder_selected)
        self._control_panel.canonical_changed.connect(self._on_canonical_changed)
        self._control_panel.window_size_changed.connect(self._on_window_size_changed)
        self._control_panel.osc_port_changed.connect(self._on_osc_port_changed)

        # Square the canvas to the control column's height so it opens as a
        # filled square; expanding (minimum, not fixed) means it never floats.
        side = self._control_panel.minimumSizeHint().height()
        self._viewport.setMinimumSize(side, side)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._control_panel)
        layout.addWidget(self._viewport, stretch=1)
        self.setCentralWidget(central)

        self._update_title()
        if initial_config is not None:
            self._apply_config(initial_config)
        else:
            self._control_panel.set_engine_controls_enabled(False)

    def _apply_config(self, config) -> None:
        """Swap to a new engine config: tear down the old worker, repopulate the
        controls, and start a fresh worker that builds the new engine."""
        self._stop_worker()
        self._config = config
        self._control_panel.populate_canonical(config)
        self._control_panel.set_engine_controls_enabled(True)
        self._update_title()
        self._start_worker()

    def _on_folder_selected(self, path: str) -> None:
        try:
            config = load_run(Path(path))
        except ConfigError as exc:
            QMessageBox.warning(self, "Invalid folder", str(exc))
            return
        self._apply_config(config)

    def _on_canonical_changed(self, index: int) -> None:
        if self._config is None or index == self._config.canonical_index:
            return
        try:
            config = load_run(self._config.snapshots_dir, index)
        except ConfigError as exc:
            QMessageBox.warning(self, "Invalid canonical snapshot", str(exc))
            return
        self._apply_config(config)

    def _on_window_size_changed(self, window_size: int) -> None:
        if window_size == self._window_size:
            return
        self._window_size = window_size
        # The window size is baked into the snapshot manager at build time, so a
        # running engine must be rebuilt; with no folder yet it just takes effect
        # on the next load.
        if self._config is not None:
            self._stop_worker()
            self._start_worker()

    def _on_osc_port_changed(self, port: int) -> None:
        previous = self._osc_server.port
        if port == previous:
            return
        try:
            self._osc_server.restart(port)
        except OSError as exc:
            logger.warning("Could not bind OSC port %d: %s", port, exc)
            QMessageBox.warning(
                self, "OSC port unavailable", f"Could not listen on port {port}:\n{exc}"
            )
            self._osc_server.restart(previous)
        self._control_panel.set_osc_port(self._osc_server.port)

    def _on_recording_failed(self, message: str) -> None:
        QMessageBox.warning(
            self, "Recording failed", f"Could not start recording:\n{message}"
        )

    def _on_load_failed(self, message: str) -> None:
        self._config = None
        self._control_panel.set_engine_controls_enabled(False)
        self._update_title()
        QMessageBox.critical(self, "Engine failed to load", message)

    def _start_worker(self) -> None:
        worker = RenderWorker(
            self._config,
            self._device,
            self._window_size,
            self._runtime_state,
            self._output_name,
        )
        worker.frame_ready.connect(self._viewport.update_frame)
        worker.status_changed.connect(self._control_panel.update_status)
        worker.loading_started.connect(self._viewport.start_loading)
        worker.loading_started.connect(self._control_panel.update_status)
        worker.load_failed.connect(self._on_load_failed)
        worker.recording_changed.connect(self._control_panel.set_recording_path)
        worker.recording_failed.connect(self._on_recording_failed)
        self._render_worker = worker
        worker.start()

    def _stop_worker(self) -> None:
        if self._render_worker is None:
            return
        worker = self._render_worker
        self._render_worker = None
        # Drop the queued signals from the old engine before it goes away.
        worker.frame_ready.disconnect(self._viewport.update_frame)
        worker.status_changed.disconnect(self._control_panel.update_status)
        worker.loading_started.disconnect(self._viewport.start_loading)
        worker.loading_started.disconnect(self._control_panel.update_status)
        worker.load_failed.disconnect(self._on_load_failed)
        worker.recording_changed.disconnect(self._control_panel.set_recording_path)
        worker.recording_failed.disconnect(self._on_recording_failed)
        worker.stop()
        self._control_panel.set_recording_path("")

    def _update_title(self) -> None:
        title = f"BalaGAN v{_VERSION}"
        if self._config is not None:
            title += f" - {Path(self._config.snapshots_dir).resolve()}"
        self.setWindowTitle(title)

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)
