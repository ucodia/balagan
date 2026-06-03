"""Qt control panel: sliders and inputs two-way bound to the runtime state."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
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
    """The right-hand panel. Every input writes through to the runtime state,
    and a timer pulls the state back into the widgets, so changes made
    elsewhere -- OSC, the viewport drag, the animation walk -- stay reflected
    here rather than leaving the widgets stale."""

    def __init__(self, runtime_state) -> None:
        super().__init__()
        self._runtime_state = runtime_state
        state = runtime_state.snapshot()
        layout = QVBoxLayout(self)

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
        self._seed_x.valueChanged.connect(lambda v: runtime_state.update(latent_x=v))
        layout.addWidget(self._seed_x)
        self._seed_y = QDoubleSpinBox()
        self._seed_y.setRange(-1e6, 1e6)
        self._seed_y.setSingleStep(1.0)
        self._seed_y.setValue(state.latent_y)
        self._seed_y.valueChanged.connect(lambda v: runtime_state.update(latent_y=v))
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
            lambda v: runtime_state.update(anim_speed=_slider_to_speed(v))
        )
        layout.addWidget(self._speed)

        layout.addWidget(QLabel("Truncation"))
        self._truncation = QSlider(Qt.Orientation.Horizontal)
        self._truncation.setRange(0, _POSITION_STEPS)
        self._truncation.setValue(round(state.truncation_psi * _POSITION_STEPS))
        self._truncation.valueChanged.connect(
            lambda v: runtime_state.update(truncation_psi=v / _POSITION_STEPS)
        )
        layout.addWidget(self._truncation)

        layout.addWidget(QLabel("FPS cap (0 = uncapped)"))
        self._fps_cap = QSpinBox()
        self._fps_cap.setRange(0, 120)
        self._fps_cap.setValue(state.fps_cap)
        self._fps_cap.valueChanged.connect(lambda v: runtime_state.update(fps_cap=v))
        layout.addWidget(self._fps_cap)

        self._output = QCheckBox("Enable Spout/Syphon output")
        self._output.setChecked(state.spout_syphon_enabled)
        self._output.toggled.connect(
            lambda on: runtime_state.update(spout_syphon_enabled=on)
        )
        layout.addWidget(self._output)

        self._debug = QCheckBox("Debug overlay")
        self._debug.setChecked(state.debug)
        self._debug.toggled.connect(lambda on: runtime_state.update(debug=on))
        layout.addWidget(self._debug)

        self._status = QLabel("Status: starting…")
        layout.addWidget(self._status)
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
            self._status.setText("Status:\n" + status.replace(" | ", "\n"))
