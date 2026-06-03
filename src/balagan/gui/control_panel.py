"""Qt control panel: sliders and inputs two-way bound to the runtime state."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_POSITION_STEPS = 1000
_SPEED_STEPS = 100
_SPEED_RANGE = 5.0
_REFRESH_INTERVAL_MS = 100


def _slider_to_speed(value: int) -> float:
    """Cubic map from a [-100, 100] slider to a [-5, 5] animation speed."""
    fraction = value / _SPEED_STEPS
    return _SPEED_RANGE * fraction**3


def _speed_to_slider(speed: float) -> int:
    """Inverse of _slider_to_speed, used to position the slider from a speed."""
    magnitude = (abs(speed) / _SPEED_RANGE) ** (1 / 3)
    return round(_SPEED_STEPS * magnitude * (1 if speed >= 0 else -1))


class ControlPanel(QWidget):
    """The right-hand panel. Every state-bound input writes through to the
    runtime state, and a timer pulls the state back into the widgets, so changes
    made elsewhere -- OSC, the viewport drag, the animation walk -- stay
    reflected here rather than leaving the widgets stale.

    The Snapshots group (folder picker and canonical-snapshot dropdown) is not
    state-bound: it drives the engine lifecycle through ``folder_selected`` and
    ``canonical_changed``, which the main window handles.
    """

    folder_selected = Signal(str)
    canonical_changed = Signal(int)
    window_size_changed = Signal(int)
    osc_port_changed = Signal(int)

    def __init__(self, runtime_state, osc_port: int, window_size: int) -> None:
        super().__init__()
        self._runtime_state = runtime_state
        self._osc_port = osc_port
        self._window_size = window_size
        self._osc_line = f"OSC listening on port {osc_port}"
        self._metrics = "starting…"
        state = runtime_state.snapshot()
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_snapshots_group())
        layout.addWidget(self._build_navigation_group(state))
        layout.addWidget(self._build_output_group(state))
        layout.addWidget(self._build_status_group())
        layout.addStretch(1)

        self.setMinimumWidth(round(self.sizeHint().width() * 1.3))

        self._bound_widgets = (
            self._position,
            self._seed_x,
            self._seed_y,
            self._play,
            self._speed,
            self._truncation,
            self._fps_cap,
            self._output,
            self._debug,
        )
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_from_state)
        self._refresh_timer.start(_REFRESH_INTERVAL_MS)

    def _build_snapshots_group(self) -> QGroupBox:
        group = QGroupBox("Snapshots")
        layout = QVBoxLayout(group)

        folder_row = QHBoxLayout()
        self._folder = QLineEdit()
        self._folder.setReadOnly(True)
        self._folder.setPlaceholderText("No folder selected")
        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._on_browse)
        folder_row.addWidget(self._folder, stretch=1)
        folder_row.addWidget(self._browse)
        layout.addLayout(folder_row)

        layout.addWidget(QLabel("Canonical snapshot"))
        self._canonical = QComboBox()
        self._canonical.activated.connect(self._on_canonical_activated)
        layout.addWidget(self._canonical)

        layout.addWidget(QLabel("Window size (0 = all)"))
        self._window_size_input = QSpinBox()
        self._window_size_input.setRange(0, 512)
        self._window_size_input.setValue(self._window_size)
        self._window_size_input.editingFinished.connect(
            lambda: self.window_size_changed.emit(self._window_size_input.value())
        )
        layout.addWidget(self._window_size_input)
        return group

    def _build_navigation_group(self, state) -> QGroupBox:
        group = QGroupBox("Navigation")
        layout = QVBoxLayout(group)

        self._position_label = QLabel(f"Position: {state.position:.3f}")
        layout.addWidget(self._position_label)
        self._position = QSlider(Qt.Orientation.Horizontal)
        self._position.setRange(0, _POSITION_STEPS)
        self._position.setValue(round(state.position * _POSITION_STEPS))
        self._position.valueChanged.connect(self._on_position)
        layout.addWidget(self._position)

        layout.addWidget(QLabel("Seed X / Y"))
        self._seed_x = QDoubleSpinBox()
        self._seed_x.setRange(-1e6, 1e6)
        self._seed_x.setSingleStep(1.0)
        self._seed_x.setValue(state.latent_x)
        self._seed_x.valueChanged.connect(
            lambda v: self._runtime_state.update(latent_x=v)
        )
        layout.addWidget(self._seed_x)
        self._seed_y = QDoubleSpinBox()
        self._seed_y.setRange(-1e6, 1e6)
        self._seed_y.setSingleStep(1.0)
        self._seed_y.setValue(state.latent_y)
        self._seed_y.valueChanged.connect(
            lambda v: self._runtime_state.update(latent_y=v)
        )
        layout.addWidget(self._seed_y)

        self._play = QPushButton("Pause" if state.anim_playing else "Play")
        self._play.setCheckable(True)
        self._play.setChecked(state.anim_playing)
        self._play.toggled.connect(self._on_play)
        layout.addWidget(self._play)

        layout.addWidget(QLabel("Speed"))
        self._speed = QSlider(Qt.Orientation.Horizontal)
        self._speed.setRange(-_SPEED_STEPS, _SPEED_STEPS)
        self._speed.setValue(_speed_to_slider(state.anim_speed))
        self._speed.valueChanged.connect(
            lambda v: self._runtime_state.update(anim_speed=_slider_to_speed(v))
        )
        layout.addWidget(self._speed)

        layout.addWidget(QLabel("Truncation"))
        self._truncation = QSlider(Qt.Orientation.Horizontal)
        self._truncation.setRange(0, _POSITION_STEPS)
        self._truncation.setValue(round(state.truncation_psi * _POSITION_STEPS))
        self._truncation.valueChanged.connect(
            lambda v: self._runtime_state.update(truncation_psi=v / _POSITION_STEPS)
        )
        layout.addWidget(self._truncation)
        return group

    def _build_output_group(self, state) -> QGroupBox:
        group = QGroupBox("Output")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("FPS limit (0 = unlimited)"))
        self._fps_cap = QSpinBox()
        self._fps_cap.setRange(0, 120)
        self._fps_cap.setValue(state.fps_cap)
        self._fps_cap.valueChanged.connect(
            lambda v: self._runtime_state.update(fps_cap=v)
        )
        layout.addWidget(self._fps_cap)

        self._output = QCheckBox("Enable Spout/Syphon output")
        self._output.setChecked(state.spout_syphon_enabled)
        self._output.toggled.connect(
            lambda on: self._runtime_state.update(spout_syphon_enabled=on)
        )
        layout.addWidget(self._output)

        self._debug = QCheckBox("Debug overlay")
        self._debug.setChecked(state.debug)
        self._debug.toggled.connect(lambda on: self._runtime_state.update(debug=on))
        layout.addWidget(self._debug)

        layout.addWidget(QLabel("OSC port"))
        self._osc_port_input = QSpinBox()
        self._osc_port_input.setRange(1024, 65535)
        self._osc_port_input.setValue(self._osc_port)
        self._osc_port_input.editingFinished.connect(
            lambda: self.osc_port_changed.emit(self._osc_port_input.value())
        )
        layout.addWidget(self._osc_port_input)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        self._status = QLabel(f"{self._metrics}\n{self._osc_line}")
        layout.addWidget(self._status)
        return group

    def populate_canonical(self, config) -> None:
        """Fill the folder field and canonical dropdown from a config, selecting
        its canonical kimg. Signals are blocked so this never re-triggers a
        rebuild."""
        self._folder.setText(str(config.snapshots_dir))
        self._canonical.blockSignals(True)
        self._canonical.clear()
        for snapshot in config.snapshots:
            self._canonical.addItem(f"{snapshot.kimg} kimg", snapshot.kimg)
        index = self._canonical.findData(config.canonical_mapping_kimg)
        if index >= 0:
            self._canonical.setCurrentIndex(index)
        self._canonical.blockSignals(False)

    def set_engine_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable the engine-dependent controls. The folder picker
        stays usable so a folder can always be (re)selected."""
        self._canonical.setEnabled(enabled)
        for widget in self._bound_widgets:
            widget.setEnabled(enabled)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select run folder")
        if path:
            self.folder_selected.emit(path)

    def _on_canonical_activated(self, _index: int) -> None:
        kimg = self._canonical.currentData()
        if kimg is not None:
            self.canonical_changed.emit(kimg)

    def _on_position(self, value: int) -> None:
        position = value / _POSITION_STEPS
        self._runtime_state.update(position=position)
        self._position_label.setText(f"Position: {position:.3f}")

    def _on_play(self, playing: bool) -> None:
        self._runtime_state.update(anim_playing=playing)
        self._play.setText("Pause" if playing else "Play")

    def _refresh_from_state(self) -> None:
        """Pull current values from the runtime state into the widgets, so
        changes made via OSC, the viewport drag, or the animation walk show up
        here. Signals are blocked across the writes, so refreshing never echoes
        a display-quantized value back into the state.
        """
        state = self._runtime_state.snapshot()
        for widget in self._bound_widgets:
            widget.blockSignals(True)
        self._position.setValue(round(state.position * _POSITION_STEPS))
        self._seed_x.setValue(state.latent_x)
        self._seed_y.setValue(state.latent_y)
        self._play.setChecked(state.anim_playing)
        self._speed.setValue(_speed_to_slider(state.anim_speed))
        self._truncation.setValue(round(state.truncation_psi * _POSITION_STEPS))
        self._fps_cap.setValue(state.fps_cap)
        self._output.setChecked(state.spout_syphon_enabled)
        self._debug.setChecked(state.debug)
        for widget in self._bound_widgets:
            widget.blockSignals(False)
        self._position_label.setText(f"Position: {state.position:.3f}")
        self._play.setText("Pause" if state.anim_playing else "Play")

    def update_status(self, status: str) -> None:
        """Slot for RenderWorker.status_changed; renders one metric per line so
        the label never wraps and the panel keeps a stable size."""
        if status:
            self._metrics = status.replace(" | ", "\n")
            self._refresh_status()

    def set_osc_port(self, port: int) -> None:
        """Reflect the port the server actually bound to: update the input and the
        Status line. Called by the main window after an OSC restart."""
        self._osc_port = port
        self._osc_line = f"OSC listening on port {port}"
        self._osc_port_input.blockSignals(True)
        self._osc_port_input.setValue(port)
        self._osc_port_input.blockSignals(False)
        self._refresh_status()

    def _refresh_status(self) -> None:
        self._status.setText(f"{self._metrics}\n{self._osc_line}")
