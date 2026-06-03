"""Qt viewport: displays rendered frames and turns mouse drags into latent moves."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget


class Viewport(QWidget):
    """Shows the latest rendered frame scaled to fit, and drives latent_x/latent_y
    from left-button mouse drags (the Autolume drag pattern)."""

    def __init__(self, runtime_state) -> None:
        super().__init__()
        self._runtime_state = runtime_state
        self._image = QImage()
        self._dragging = False
        self._last_pos = None
        self._sized_to_frame = False
        self.setMinimumSize(512, 512)

    def update_frame(self, image: QImage) -> None:
        """Slot for RenderWorker.frame_ready: store the frame and request a repaint.

        On the first frame, grow the window so the viewport shows the snapshot at
        its native pixel resolution; afterwards the window stays freely resizable.
        """
        self._image = image
        if not self._sized_to_frame and not image.isNull():
            self._sized_to_frame = True
            self._resize_window_to_frame(image.width(), image.height())
        self.update()

    def _resize_window_to_frame(self, frame_width: int, frame_height: int) -> None:
        """Resize the top-level window by the viewport's gap to the frame size, so
        the viewport ends up at the frame's native resolution while the control
        panel and window chrome keep their own widths."""
        window = self.window()
        delta_w = frame_width - self.width()
        delta_h = frame_height - self.height()
        window.resize(window.width() + delta_w, window.height() + delta_h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image.isNull():
            return
        scaled = self._image.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(x, y, scaled)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging or self._last_pos is None:
            return
        position = event.position()
        dx = position.x() - self._last_pos.x()
        dy = position.y() - self._last_pos.y()
        self._last_pos = position
        # Autolume drag pattern: latent delta = pixel delta / font size * 4e-2.
        scale = 4e-2 / max(self.font().pointSizeF(), 1.0)
        state = self._runtime_state.snapshot()
        self._runtime_state.update(
            latent_x=state.latent_x + dx * scale,
            latent_y=state.latent_y + dy * scale,
        )

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
