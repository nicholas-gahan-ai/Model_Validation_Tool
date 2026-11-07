"""Live view: one or two image panes (side by side) sharing zoom and pan; clicks map to image px.
Also the seek bar with flag markers."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QWidget


def bgr_to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    q = QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888)
    q._keep = img  # keep the buffer alive with the image
    return q


class LiveView(QWidget):
    clicked = Signal(int, float, float)       # pane, x, y (image px)
    dragged_box = Signal(int, float, float, float, float)  # pane, x1, y1, x2, y2 (box-draw mode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LiveView")
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.images: list[QImage] = []
        self.titles: list[str] = []
        self.zoom = 1.0
        self.center = None                     # image-px point at the pane centre (None = fit)
        self._press = None
        self._drag_from = None
        self.box_mode = False                  # drag draws a box instead of panning
        self._box_rect = None
        self.placeholder: QPixmap | None = None
        self.placeholder_text = ""
        self._toast = ""
        self._flash = QElapsedTimer()
        self._anim = QTimer(self); self._anim.setInterval(16); self._anim.timeout.connect(self._tick)

    # ---- feedback ------------------------------------------------------------------------------------------------
    def flash(self, toast: str = ""):
        """Camera-style flash + a short toast, so an action like Snapshot visibly happened."""
        self._toast = toast
        self._flash.start()
        self._anim.start()
        self.update()

    def _tick(self):
        if not self._flash.isValid() or self._flash.elapsed() > 2200:
            self._anim.stop(); self._toast = ""; self._flash.invalidate()
        self.update()

    # ---- content -------------------------------------------------------------------------------------------------
    def set_images(self, imgs: list[np.ndarray], titles: list[str] | None = None):
        new = [bgr_to_qimage(i) for i in imgs]
        if self.images and new and (new[0].size() != self.images[0].size()):
            self.reset_view()
        self.images = new
        self.titles = titles or []
        self.update()

    def clear(self):
        self.images = []
        self.update()

    def reset_view(self):
        self.zoom, self.center = 1.0, None
        self.update()

    # ---- geometry ------------------------------------------------------------------------------------------------
    def pane_rects(self) -> list[QRectF]:
        n = max(1, len(self.images))
        gap = 4 if n > 1 else 0
        w = (self.width() - gap * (n - 1)) / n
        return [QRectF(i * (w + gap), 0, w, self.height()) for i in range(n)]

    def _transform(self, pane: QRectF, img: QImage):
        """(scale, offset_x, offset_y): widget = offset + image * scale."""
        iw, ih = img.width(), img.height()
        fit = min(pane.width() / iw, pane.height() / ih)
        s = fit * self.zoom
        cx, cy = self.center if self.center else (iw / 2, ih / 2)
        ox = pane.x() + pane.width() / 2 - cx * s
        oy = pane.y() + pane.height() / 2 - cy * s
        return s, ox, oy

    def to_image(self, pos: QPointF):
        for k, (pane, img) in enumerate(zip(self.pane_rects(), self.images)):
            if pane.contains(pos):
                s, ox, oy = self._transform(pane, img)
                return k, (pos.x() - ox) / s, (pos.y() - oy) / s
        return None

    # ---- paint ---------------------------------------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0C0D10"))
        if not self.images:
            if self.placeholder is not None:
                pm = self.placeholder
                p.setOpacity(0.25)
                p.drawPixmap(int((self.width() - pm.width()) / 2), int((self.height() - pm.height()) / 2 - 20), pm)
                p.setOpacity(1)
            if self.placeholder_text:
                p.setPen(QColor("#6B717C"))
                r = self.rect().adjusted(0, self.height() // 2 + 60, 0, 0)
                p.drawText(r, Qt.AlignHCenter | Qt.AlignTop, self.placeholder_text)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 2.5)
        for k, (pane, img) in enumerate(zip(self.pane_rects(), self.images)):
            s, ox, oy = self._transform(pane, img)
            p.save()
            p.setClipRect(pane)
            p.drawImage(QRectF(ox, oy, img.width() * s, img.height() * s), img)
            if k < len(self.titles) and self.titles[k]:
                f = QFont(); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(self.titles[k]) + 16
                p.fillRect(QRectF(pane.x() + 8, 8, tw, 22), QColor(0, 0, 0, 160))
                p.setPen(QColor("#E6E6E6"))
                p.drawText(QRectF(pane.x() + 8, 8, tw, 22), Qt.AlignCenter, self.titles[k])
            p.restore()
        if self._box_rect is not None:
            p.setPen(QPen(QColor("#FF5A5A"), 2, Qt.DashLine))
            p.drawRect(self._box_rect)
        if self._flash.isValid():
            t = self._flash.elapsed()
            if t < 250:  # white flash fading out
                p.fillRect(self.rect(), QColor(255, 255, 255, int(150 * (1 - t / 250))))
            if self._toast:
                f = QFont(); f.setPointSize(11); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(self._toast) + 28
                r = QRectF((self.width() - tw) / 2, self.height() - 64, tw, 34)
                p.setOpacity(1.0 if t < 1700 else max(0.0, 1 - (t - 1700) / 500))
                p.setPen(Qt.NoPen); p.setBrush(QColor(20, 22, 27, 230)); p.drawRoundedRect(r, 6, 6)
                p.setPen(QColor("#E6E6E6")); p.drawText(r, Qt.AlignCenter, self._toast)
                p.setOpacity(1.0)
        if self.zoom != 1.0:
            p.setPen(QColor("#9AA0AA"))
            p.drawText(self.rect().adjusted(0, 0, -10, -8), Qt.AlignRight | Qt.AlignBottom,
                       f"{self.zoom:.1f}×  (double-click: fit)")

    # ---- interaction ---------------------------------------------------------------------------------------------
    def wheelEvent(self, e):
        if not self.images:
            return
        hit = self.to_image(e.position())
        if not hit:
            return
        _, x, y = hit
        old = self.zoom
        self.zoom = float(np.clip(self.zoom * (1.25 if e.angleDelta().y() > 0 else 0.8), 1.0, 16.0))
        if self.zoom == 1.0:
            self.center = None
        else:
            img = self.images[0]
            cx, cy = self.center if self.center else (img.width() / 2, img.height() / 2)
            f = old / self.zoom  # keep the point under the cursor fixed
            self.center = (x + (cx - x) * f, y + (cy - y) * f)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position()
            self._drag_from = (e.position(), self.center)

    def mouseMoveEvent(self, e):
        if self._press is None or not self.images:
            return
        if self.box_mode:
            self._box_rect = QRectF(self._press, e.position()).normalized()
            self.update()
            return
        if self.zoom > 1.0:
            start, c0 = self._drag_from
            pane = self.pane_rects()[0]
            s, _, _ = self._transform(pane, self.images[0])
            img = self.images[0]
            cx, cy = c0 if c0 else (img.width() / 2, img.height() / 2)
            d = e.position() - start
            self.center = (cx - d.x() / s, cy - d.y() / s)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        moved = (e.position() - self._press).manhattanLength()
        if self.box_mode and self._box_rect is not None and moved > 6:
            a = self.to_image(self._box_rect.topLeft()); b = self.to_image(self._box_rect.bottomRight())
            if a and b and a[0] == b[0]:
                self.dragged_box.emit(a[0], a[1], a[2], b[1], b[2])
        elif moved < 5:
            hit = self.to_image(e.position())
            if hit:
                self.clicked.emit(*hit)
        self._press = None
        self._box_rect = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()


class SeekBar(QSlider):
    """Horizontal slider that paints flag markers. Clicking the groove or a marker jumps there;
    dragging the handle seeks on release. Emits seek_requested(position)."""
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.markers: list[int] = []
        self.setTracking(False)
        self.sliderReleased.connect(lambda: self.seek_requested.emit(self.value()))

    def set_markers(self, m):
        self.markers = sorted(set(m))
        self.update()

    def _groove(self):
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        return (self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self),
                self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self))

    def _x_of(self, v):
        gr, _ = self._groove()
        span = max(1, self.maximum() - self.minimum())
        return gr.x() + (v - self.minimum()) / span * gr.width()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.markers:
            return
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#E05252"))
        for v in self.markers:
            x = self._x_of(v)
            p.drawPolygon([QPointF(x, 0), QPointF(x - 4, 6), QPointF(x + 4, 6)])
            p.drawRect(QRectF(x - 1, 0, 2, self.height()))

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or not self.isEnabled():
            return super().mousePressEvent(e)
        x = e.position().x()
        for v in self.markers:
            if abs(self._x_of(v) - x) <= 4:
                self.setValue(v); self.seek_requested.emit(v)
                return
        gr, handle = self._groove()
        if handle.contains(e.position().toPoint()):
            return super().mousePressEvent(e)  # drag; seeks on release
        v = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(x - gr.x()), max(1, gr.width()))
        self.setValue(v)
        self.seek_requested.emit(v)
