"""The look of the desktop application, in one place.

Everything here is generated from :mod:`nncm.theme`; no widget module writes a
colour, a radius or a size of its own. Three things cannot be expressed in a Qt
style sheet and so are done in code, each with the reason attached:

* **tabular figures** (:func:`tabular`) — Qt style sheets have no
  ``font-variant-numeric``, so the font feature is set on the QFont. Left in
  the style sheet it would be silently dropped and every live value would
  jitter as its digits changed width;
* **the base palette** (:func:`_palette`) — native menus, tooltips and any
  region the style sheet does not reach fall back to the system theme unless
  the palette is set too;
* **the painted glyphs** (:func:`glyph_url`) — a combo box arrow and a check
  mark have to be images. They are painted once and cached, and if the cache
  cannot be written the rules that use them are left out, so Qt draws its own
  and the control keeps working.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from PySide6.QtCore import QPointF, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPen, QPixmap

from .. import theme

FONTS_DIR = theme.FONTS_DIR  # one place says where the type lives


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _families() -> list[str]:
    """The family stack, with any bundled faces registered first.

    Bundling keeps the interface identical on every machine — including the
    real italic, because a synthesised oblique reads as cheapness. Nothing
    breaks without it: an absent font directory costs polish, not function,
    since the stack falls through to the system UI face.

    Registration needs a running QApplication, which is why it happens here
    rather than at import time.
    """
    if FONTS_DIR.is_dir():
        for path in sorted(FONTS_DIR.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                QFontDatabase.addApplicationFont(str(path))
    return list(theme.FONT_STACK)


def font(
    size: int = theme.FONT_BODY,
    weight: int = theme.WEIGHT_REGULAR,
    *,
    italic: bool = False,
    figures: bool = False,
    mono: bool = False,
) -> QFont:
    """A font from the scale. ``figures=True`` asks for tabular digits."""
    value = QFont()
    value.setFamilies(list(theme.FONT_MONO_STACK) if mono else _families())
    value.setPixelSize(size)
    value.setWeight(QFont.Weight(weight))
    value.setItalic(italic)
    if figures:
        tabular(value)
    return value


def tabular(value: QFont) -> QFont:
    """Fixed-advance digits, so columns line up and live values do not jitter.

    This is *not* a job for a monospace face — that brings a technical accent
    nobody asked for. It cannot be moved into the style sheet: Qt has no
    ``font-variant-numeric`` and would drop it without a word.
    """
    if hasattr(QFont, "Tag"):  # Qt 6.7+; older Qt simply keeps proportional digits
        value.setFeature(QFont.Tag("tnum"), 1)
    return value


# ---------------------------------------------------------------------------
# Painted glyphs
# ---------------------------------------------------------------------------
def _cache_dir() -> Path | None:
    root = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    if not root:
        return None
    path = Path(root) / "glyphs"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _paint(name: str, colour: str, size: int) -> QPixmap:
    """Draw one glyph on a 16-unit grid: no fill, 1.5 stroke, round caps."""
    scale = size / 16.0
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(colour))
    pen.setWidthF(1.6 * scale if name.startswith("chevron") else 1.5 * scale)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    if name == "chevron-down":
        painter.drawPolyline([_pt(4.5, 6.5, scale), _pt(8, 10, scale), _pt(11.5, 6.5, scale)])
    elif name == "check":
        painter.drawPolyline([_pt(3.8, 8.4, scale), _pt(6.6, 11.2, scale), _pt(12.2, 5.2, scale)])
    elif name == "dash":
        painter.drawLine(_pt(4, 8, scale), _pt(12, 8, scale))
    painter.end()
    return pixmap


def _pt(x: float, y: float, scale: float) -> QPointF:
    return QPointF(x * scale, y * scale)


@lru_cache(maxsize=32)
def glyph_url(name: str, colour: str, size: int = 16) -> str | None:
    """Path to a cached glyph PNG, or ``None`` if it could not be written.

    Cached because a repaint must never redraw an immutable asset, and on disk
    because a Qt style sheet can only reference an image by URL. Forward
    slashes on every platform — Qt does not read a backslash in a URL.
    """
    directory = _cache_dir()
    if directory is None:
        return None
    target = directory / f"{name}-{colour.lstrip('#')}-{size}.png"
    if not target.exists():
        try:
            if not _paint(name, colour, size).save(str(target), "PNG"):
                return None
        except OSError:
            return None
    return target.as_posix()


# ---------------------------------------------------------------------------
# Palette and style sheet
# ---------------------------------------------------------------------------
def _palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(theme.WINDOW))
    palette.setColor(QPalette.WindowText, QColor(theme.TEXT))
    palette.setColor(QPalette.Base, QColor(theme.SURFACE))
    palette.setColor(QPalette.AlternateBase, QColor(theme.SURFACE_ALT))
    palette.setColor(QPalette.Text, QColor(theme.TEXT))
    palette.setColor(QPalette.PlaceholderText, QColor(theme.TERTIARY))
    palette.setColor(QPalette.Button, QColor(theme.SURFACE))
    palette.setColor(QPalette.ButtonText, QColor(theme.TEXT))
    palette.setColor(QPalette.Highlight, QColor(theme.ACCENT_SOFT))
    palette.setColor(QPalette.HighlightedText, QColor(theme.TEXT))
    palette.setColor(QPalette.Link, QColor(theme.ACCENT))
    palette.setColor(QPalette.ToolTipBase, QColor(theme.SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(theme.TEXT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.TERTIARY))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme.TERTIARY))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(theme.TERTIARY))
    return palette


_SHEET = Template(
    """
QWidget { color: $TEXT; }
QMainWindow, QDialog, QWidget#Body { background: $WINDOW; }

/* -- containers ------------------------------------------------------- */
QFrame#Card, QFrame#Surface {
    background: $SURFACE; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTAINER}px;
}
QFrame#ActionBar { background: transparent; border: none; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QSplitter::handle { background: transparent; }

/* -- type ------------------------------------------------------------- */
QLabel#PaneHeader { font-size: ${FONT_TITLE}px; font-weight: $WEIGHT_SEMIBOLD; color: $TEXT;
                    padding: 2px 2px 10px 2px; }
QLabel#CardTitle { font-size: ${FONT_TITLE}px; font-weight: $WEIGHT_SEMIBOLD; color: $TEXT; }
QLabel#SectionTitle { font-size: ${FONT_SECTION}px; font-weight: $WEIGHT_SEMIBOLD; color: $TEXT; }
QLabel#Caption, QLabel#MetricCaption {
    font-size: ${FONT_CAPTION}px; font-weight: $WEIGHT_MEDIUM; color: $SUBTLE; }
QLabel#MetricUnit, QLabel#Unit {
    font-size: ${FONT_UNIT}px; font-weight: $WEIGHT_MEDIUM; color: $TERTIARY; }
QLabel#Explanation {
    font-size: ${FONT_CAPTION}px; font-style: italic; color: $SUBTLE; }
QLabel#EmptyState { font-size: ${FONT_BODY}px; color: $SUBTLE; }
QLabel#FieldLabel { font-size: ${FONT_BODY}px; color: $TEXT; }
QLabel#Danger { font-size: ${FONT_BODY}px; color: $DANGER; }

/* -- buttons ---------------------------------------------------------- */
QPushButton {
    background: $SURFACE; color: $TEXT; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px; padding: 6px 14px;
    font-size: ${FONT_CONTROL}px; font-weight: $WEIGHT_MEDIUM;
}
QPushButton:hover { background: $SURFACE_ALT; }
QPushButton:pressed { background: $BORDER; }
QPushButton:disabled { color: $TERTIARY; background: $SURFACE; }
QPushButton[primary="true"] {
    background: $ACCENT; color: $ACCENT_TEXT; border-color: $ACCENT;
}
QPushButton[primary="true"]:hover { background: $ACCENT_HOVER; border-color: $ACCENT_HOVER; }
QPushButton[primary="true"]:pressed { background: $ACCENT_HOVER; }
QPushButton[primary="true"]:disabled { background: $BORDER; border-color: $BORDER; color: $TERTIARY; }
QPushButton#Quiet {
    background: transparent; border: ${BORDER_WIDTH}px solid transparent; color: $TERTIARY;
    padding: 2px 8px; border-radius: ${RADIUS_SMALL}px;
}
QPushButton#Quiet:hover { background: $SURFACE_ALT; border-color: $BORDER; color: $TEXT; }

/* -- fields ----------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QAbstractSpinBox {
    background: $SURFACE; color: $TEXT; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px; padding: 4px 8px;
    /* Tall enough for the digits it holds: a value showing only its top half
       is worse than no value. */
    min-height: 20px;
    font-size: ${FONT_BODY}px; selection-background-color: $ACCENT_SOFT;
    selection-color: $TEXT;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QPlainTextEdit:hover {
    border-color: $TERTIARY;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border-color: $ACCENT;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: $TERTIARY; background: $SURFACE_ALT;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: $SURFACE; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px; padding: 4px;
    selection-background-color: $ACCENT_SOFT; selection-color: $TEXT;
    outline: none;
}
QCheckBox { font-size: ${FONT_BODY}px; color: $TEXT; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_INDICATOR}px; background: $SURFACE;
}
QCheckBox::indicator:hover { border-color: $TERTIARY; }
QCheckBox::indicator:checked { background: $ACCENT; border-color: $ACCENT; }

/* -- tables ----------------------------------------------------------- */
QTableView, QTableWidget {
    background: $SURFACE; alternate-background-color: $SURFACE;
    border: none; gridline-color: transparent;
    font-size: ${FONT_BODY}px; outline: none;
    selection-background-color: $ACCENT_SOFT; selection-color: $TEXT;
}
QTableView::item, QTableWidget::item {
    border: none; border-bottom: ${BORDER_WIDTH}px solid $BORDER; padding: 0px 8px;
}
QTableView::item:selected, QTableWidget::item:selected {
    background: $ACCENT_SOFT; color: $TEXT;
}
QHeaderView { background: $SURFACE; border: none; }
QHeaderView::section {
    background: $SURFACE; color: $SUBTLE; border: none;
    border-bottom: ${BORDER_WIDTH}px solid $BORDER;
    padding: 6px 8px; font-size: ${FONT_CAPTION}px; font-weight: $WEIGHT_MEDIUM;
}
QHeaderView::section:hover { color: $TEXT; }
QTableCornerButton::section { background: $SURFACE; border: none; }

/* -- tabs ------------------------------------------------------------- */
QTabWidget::pane { border: none; background: transparent; }
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab {
    background: transparent; color: $SUBTLE;
    padding: 7px 14px; margin-right: 4px;
    border: none; border-bottom: 2px solid transparent;
    font-size: ${FONT_CONTROL}px; font-weight: $WEIGHT_MEDIUM;
}
QTabBar::tab:hover { color: $TEXT; }
QTabBar::tab:selected { color: $TEXT; border-bottom: 2px solid $ACCENT; }

/* -- menus, status, tooltips ------------------------------------------ */
QMenuBar { background: $WINDOW; color: $TEXT; border: none; padding: 2px 8px; }
QMenuBar::item {
    background: transparent; padding: 4px 10px; border-radius: ${RADIUS_INDICATOR}px;
    font-size: ${FONT_BODY}px;
}
QMenuBar::item:selected { background: $SURFACE_ALT; }
QMenu {
    background: $SURFACE; color: $TEXT; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px; padding: 6px;
}
QMenu::item {
    padding: 6px 24px 6px 24px; border-radius: ${RADIUS_SMALL}px; font-size: ${FONT_BODY}px;
}
QMenu::item:selected { background: $SURFACE_ALT; }
QMenu::item:disabled { color: $TERTIARY; }
QMenu::separator { height: ${BORDER_WIDTH}px; background: $BORDER; margin: 6px 8px; }
QStatusBar {
    background: transparent; border: none; color: $TERTIARY; font-size: ${FONT_UNIT}px;
}
QStatusBar::item { border: none; }
QToolTip {
    background: $SURFACE; color: $TEXT; border: ${BORDER_WIDTH}px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px; padding: 6px 8px; font-size: ${FONT_UNIT}px;
}
QDockWidget {
    color: $SUBTLE; font-size: ${FONT_CAPTION}px; font-weight: $WEIGHT_MEDIUM;
    titlebar-close-icon: none; titlebar-normal-icon: none;
}
QDockWidget::title { background: transparent; padding: 6px 2px; }

/* -- progress and scrollbars ------------------------------------------ */
QProgressBar {
    background: $SURFACE_ALT; border: none; border-radius: 3px;
    height: 6px; text-align: center; color: $SUBTLE; font-size: ${FONT_CAPTION}px;
}
QProgressBar::chunk { background: $ACCENT; border-radius: 3px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: $BORDER; border-radius: ${RADIUS_CONTROL}px; min-height: 28px;
}
QScrollBar::handle:horizontal {
    background: $BORDER; border-radius: ${RADIUS_CONTROL}px; min-width: 28px;
}
QScrollBar::handle:hover { background: $TERTIARY; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""
)

_GLYPH_SHEET = Template(
    """
QComboBox::down-arrow { image: url("$CHEVRON"); width: 14px; height: 14px; }
QCheckBox::indicator:checked { image: url("$CHECK"); }
"""
)


def stylesheet() -> str:
    """The whole style sheet, tokens substituted in.

    The glyph rules are appended only when the images exist. Without them Qt
    draws its own arrow and tick: less tidy, still usable — ornament is never
    allowed to take a control with it.
    """
    sheet = _SHEET.substitute(theme.tokens())
    chevron = glyph_url("chevron-down", theme.TERTIARY)
    check = glyph_url("check", theme.ACCENT_TEXT)
    if chevron and check:
        sheet += _GLYPH_SHEET.substitute(CHEVRON=chevron, CHECK=check)
    return sheet


def apply(app) -> None:
    """Dress a QApplication: one style, one palette, one font, one style sheet."""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setFont(font())
    app.setStyleSheet(stylesheet())
