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
        self._loading_text = None
        self._dragging = False
        self._last_pos = None
        self.setMinimumSize(256, 256)

    def start_loading(self, text: str) -> None:
        """Slot for RenderWorker.loading_started: show a message centered over the
        canvas until the first frame arrives."""
        self._loading_text = text
        self.update()

    def update_frame(self, image: QImage) -> None:
        """Slot for RenderWorker.frame_ready: store the frame and request a repaint."""
        self._image = image
        if not image.isNull():
            self._loading_text = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self._image.isNull():
            scaled = self._image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        if self._loading_text:
            painter.setPen(Qt.GlobalColor.white)
            font = painter.font()
            font.setPointSize(18)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._loading_text
            )

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
