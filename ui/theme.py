"""Dark Fusion theme + branding asset paths."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle

from core.settings import APP_ROOT

ASSETS = APP_ROOT / "assets"
UI_DIR = APP_ROOT / "ui"
BG, PANEL, TEXT, ACCENT, ACCENT_HOVER = "#16181D", "#1E2127", "#E6E6E6", "#7A1F1F", "#952828"
WORDMARK = ASSETS / "logo_wordmark_light.png"
ICON_PNG = ASSETS / "logo_icon.png"
ICON_ICO = ASSETS / "logo_icon.ico"


def wordmark_pixmap(height: int) -> QPixmap:
    """The Gahan AI wordmark with its empty margins cropped, scaled to `height` px (logical)."""
    from core.render import load_wordmark_bgra
    wm = load_wordmark_bgra()
    if wm is False:
        return QPixmap()
    h, w = wm.shape[:2]
    img = QImage(wm.data, w, h, wm.strides[0], QImage.Format_ARGB32).copy()   # BGRA bytes = ARGB32 on little-endian
    pm = QPixmap.fromImage(img)
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    pm = pm.scaledToHeight(int(height * dpr), Qt.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)
    return pm


class _Style(QProxyStyle):
    """Fusion without the platform icons on dialog buttons (they render as odd glyphs on some desktops)."""

    def styleHint(self, hint, option=None, widget=None, ret=None):
        if hint == QStyle.SH_DialogButtonBox_ButtonsHaveIcons:
            return 0
        return super().styleHint(hint, option, widget, ret)


def app_icon() -> QIcon:
    icon = QIcon()
    for p in (ICON_ICO, ICON_PNG, ASSETS / "logo_icon_256.png", ASSETS / "logo_icon_64.png",
              ASSETS / "logo_icon_32.png"):
        if p.is_file():
            icon.addFile(str(p))
    return icon


def apply_theme(app: QApplication) -> None:
    app.setStyle(_Style("Fusion"))
    pal = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.Base, "#111317"), (QPalette.AlternateBase, PANEL),
                    (QPalette.WindowText, TEXT), (QPalette.Text, TEXT), (QPalette.Button, "#2A2E36"),
                    (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#FFFFFF"),
                    (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, "#6B717C")):
        pal.setColor(role, QColor(c))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#5C626C"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5C626C"))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#5C626C"))
    app.setPalette(pal)
    qss = UI_DIR / "theme_dark.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text().replace("{UI}", UI_DIR.as_posix()))
    app.setWindowIcon(app_icon())
