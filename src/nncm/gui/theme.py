"""The look of the desktop application, in one place.

This is the only module in the application that calls ``setStyleSheet``. Every
colour and metric it uses comes from :mod:`nncm.theme`, which is Qt-free so the
charts can read the same tokens without ``core`` importing anything from
``gui``. No widget carries a style sheet of its own, ever.

Three things a Qt style sheet cannot express, done in code with the reason
attached:

* **tabular figures** (:func:`tabular`) — Qt has no ``font-variant-numeric``,
  so the font feature is set on the QFont. Left in the sheet it would be
  silently dropped and every live value would jitter as its digits changed
  width;
* **the base palette** — native menus, tooltips, file dialogs and message
  boxes partly ignore the sheet and read the palette instead;
* **role properties** — a role is a dynamic property (``role``, ``variant``)
  that the sheet selects on, so :func:`restyle` re-polishes a widget whose
  role changed after it was shown.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from .. import theme as T

logger = logging.getLogger(__name__)

FONTS_DIR = T.FONTS_DIR  # one place says where the type lives

FONT_HINTING = QFont.PreferVerticalHinting
"""Hint the stems, not the spacing.

Under Qt's default preference the shaper rounds every glyph advance to a whole
pixel, which quantises the kerning away: at 22 px the string "Project" measures
the same width with kerning switched on as with it switched off. The result is
the uneven word colour this setting exists to remove — pairs that should tuck
together do not, and the slack lands in whichever gap the rounding favoured.

Vertical hinting keeps the advances sub-pixel, so kerning survives, while still
snapping horizontal stems to the pixel grid — crisper on screen than
``PreferNoHinting`` at the same spacing.
"""


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
def load_fonts() -> bool:
    """Register the bundled Inter faces with Qt.

    Returns True if either the variable or the static family is available
    afterwards. A missing face is logged rather than raised — the app still
    runs on the platform sans-serif, it just does not look like itself.
    """
    if FONTS_DIR.is_dir():
        for path in sorted(FONTS_DIR.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                if QFontDatabase.addApplicationFont(str(path)) == -1:
                    logger.warning("Qt refused the bundled font file: %s", path.name)
    else:
        logger.warning("Bundled font directory missing: %s", FONTS_DIR)

    families = QFontDatabase.families()
    if T.FONT_VARIABLE_FAMILY not in families:
        # Not fatal — the stack falls through to the static faces, which look
        # the same and merely spend four files doing it.
        logger.info("%s not registered; using the static faces.",
                    T.FONT_VARIABLE_FAMILY)
    return any(name in families
               for name in (T.FONT_VARIABLE_FAMILY, T.FONT_FAMILY))


def font(
    size: int = T.FONT_BODY,
    weight: int = T.WEIGHT_REGULAR,
    *,
    italic: bool = False,
    figures: bool = False,
    mono: bool = False,
) -> QFont:
    """A font from the scale. ``figures=True`` asks for tabular digits."""
    value = QFont()
    value.setFamilies(list(T.FONT_MONO_STACK) if mono else list(T.FONT_STACK))
    value.setPixelSize(size)
    value.setWeight(QFont.Weight(weight))  # PySide6 rejects a bare int here
    value.setItalic(italic)
    value.setHintingPreference(FONT_HINTING)
    if figures:
        tabular(value)
    return value


def tabular(value: QFont) -> QFont:
    """Fixed-advance digits, so columns line up and live values do not jitter.

    This is *not* a job for a monospace face — that brings a technical accent
    nobody asked for.
    """
    if hasattr(QFont, "Tag"):  # Qt 6.7+; older Qt keeps proportional digits
        value.setFeature(QFont.Tag("tnum"), 1)
    return value


# ---------------------------------------------------------------------------
# The style sheet
# ---------------------------------------------------------------------------
def stylesheet() -> str:
    """The whole style sheet, built from the tokens.

    Written as one f-string rather than a template so a token that does not
    exist is an error here rather than a literal ``$NAME`` on screen.
    """
    ink = T.ink
    family = ", ".join(f'"{name}"' for name in T.FONT_STACK[:-1]) + ", sans-serif"
    grab = (T.SPLITTER_GRAB - T.SPLITTER_VISUAL) // 2

    return f"""
/* -- Global ------------------------------------------------------------ */
* {{
    font-family: {family};
    font-size: {T.FONT_BODY}px;
    font-weight: {T.WEIGHT_REGULAR};
    color: {ink(T.INK_PRIMARY)};
    outline: none;
}}
QMainWindow, QDialog, QWidget#Content, QWidget#Body {{ background: {T.CANVAS}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* -- Type -------------------------------------------------------------- */
QLabel {{ background: transparent; border: none; padding: 0px; }}
QLabel[role="title"] {{
    font-size: {T.FONT_TITLE}px; font-weight: {T.WEIGHT_SEMIBOLD};
}}
QLabel[role="heading"], QLabel[role="brand"] {{
    font-size: {T.FONT_HEADING}px; font-weight: {T.WEIGHT_SEMIBOLD};
}}
QLabel[role="caption"] {{
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {ink(T.INK_TERTIARY)};
}}
QLabel[role="unit"] {{
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {ink(T.INK_TERTIARY)};
}}
QLabel[role="explanation"] {{
    font-size: {T.FONT_CAPTION}px; font-style: italic;
    color: {ink(T.INK_TERTIARY)};
}}
QLabel[role="empty"] {{
    font-size: {T.FONT_CAPTION}px; color: {ink(T.INK_TERTIARY)};
}}
QLabel[role="field"] {{ font-size: {T.FONT_BODY}px; color: {ink(T.INK_PRIMARY)}; }}
QLabel[role="metric"] {{
    font-size: {T.FONT_METRIC}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {ink(T.INK_PRIMARY)};
}}

/* Status banners — colour and words together, never colour alone. */
QLabel[role="error"] {{
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {T.ERROR}; background: {T.ERROR_BG};
    border-radius: {T.RADIUS}px; padding: 6px 10px;
}}
QLabel[role="warning"] {{
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {T.WARNING}; background: {T.WARNING_BG};
    border-radius: {T.RADIUS}px; padding: 6px 10px;
}}
QLabel[role="success"] {{
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    color: {T.SUCCESS}; background: {T.SUCCESS_BG};
    border-radius: {T.RADIUS}px; padding: 6px 10px;
}}

/* -- Splash ------------------------------------------------------------ */
/* Frameless, so it draws the edge the window manager would otherwise give
   it. The panel radius rather than the control one: this is a surface. */
QWidget#Splash {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER_STRONG)};
    border-radius: {T.RADIUS_PANEL}px;
}}

/* -- Panels ------------------------------------------------------------ */
QFrame[role="panel"] {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER)};
    border-radius: {T.RADIUS_PANEL}px;
}}
QFrame[role="bare"] {{ background: transparent; border: none; }}
QFrame[role="rule"] {{
    background: {ink(T.SURFACE_BORDER)}; border: none; max-height: 1px;
}}

/* -- Fields ------------------------------------------------------------ */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER)};
    border-radius: {T.RADIUS}px;
    padding: 5px 9px;
    min-height: {T.CONTROL_HEIGHT - 14}px;
    font-size: {T.FONT_BODY}px;
    color: {ink(T.INK_PRIMARY)};
    selection-background-color: {ink(T.SURFACE_SELECTED)};
    selection-color: {ink(T.INK_PRIMARY)};
}}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {ink(T.SURFACE_BORDER_STRONG)};
}}
/* 2 px ring, padding down 1 px each side so the control's box never moves. */
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {ink(T.INK_GLYPH)};
    padding: 4px 8px;
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {ink(T.SURFACE_WASH)};
    color: {ink(T.INK_DISABLED)};
    border-color: {ink(T.SURFACE_BORDER)};
}}
QLineEdit[readOnlyValue="true"], QPlainTextEdit[readOnlyValue="true"] {{
    background: {ink(T.SURFACE_WASH)};
}}
QComboBox::drop-down {{ border: none; width: {T.ICON_SIZE + 8}px; }}
QComboBox::down-arrow {{
    width: {T.ICON_SIZE}px; height: {T.ICON_SIZE}px;
}}
QComboBox QAbstractItemView {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER_STRONG)};
    border-radius: {T.RADIUS}px;
    padding: 4px;
    selection-background-color: {ink(T.SURFACE_SELECTED)};
    selection-color: {ink(T.INK_PRIMARY)};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent; border: none; width: 0px;
}}

/* -- Checkboxes -------------------------------------------------------- */
QCheckBox, QRadioButton {{
    spacing: 8px; font-size: {T.FONT_BODY}px; color: {ink(T.INK_PRIMARY)};
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: {T.ICON_SIZE}px; height: {T.ICON_SIZE}px;
    border: 1px solid {ink(T.SURFACE_BORDER_STRONG)};
    background: {T.CANVAS};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: {T.ICON_SIZE // 2}px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ink(T.INK_GLYPH)};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {T.ACTION}; border-color: {T.ACTION};
}}
/* The state goes AFTER the sub-control. Written the other way round Qt draws
   a border around the whole checkbox row instead of its indicator. */
QCheckBox::indicator:focus, QRadioButton::indicator:focus {{
    border: 2px solid {ink(T.INK_GLYPH)};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {ink(T.INK_DISABLED)}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {ink(T.SURFACE_WASH)}; border-color: {ink(T.SURFACE_BORDER)};
}}

/* -- Buttons ----------------------------------------------------------- */
QPushButton {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER)};
    border-radius: {T.RADIUS}px;
    padding: 6px 16px;
    min-height: {T.CONTROL_HEIGHT - 14}px;
    font-size: {T.FONT_LABEL}px;
    font-weight: {T.WEIGHT_MEDIUM};
    color: {ink(T.INK_PRIMARY)};
}}
QPushButton:hover {{
    background: {ink(T.SURFACE_HOVER)};
    border-color: {ink(T.SURFACE_BORDER_STRONG)};
}}
QPushButton:pressed {{ background: {ink(T.SURFACE_PRESSED)}; }}
QPushButton:focus {{ border: 2px solid {ink(T.INK_GLYPH)}; padding: 5px 15px; }}
QPushButton:disabled {{
    background: {ink(T.SURFACE_WASH)};
    border-color: {ink(T.SURFACE_BORDER)};
    color: {ink(T.INK_DISABLED)};
}}
QPushButton[variant="primary"] {{
    background: {T.ACTION}; border-color: {T.ACTION}; color: {T.ACTION_TEXT};
}}
QPushButton[variant="primary"]:hover {{
    background: {T.ACTION_HOVER}; border-color: {T.ACTION_HOVER};
}}
QPushButton[variant="primary"]:pressed {{
    background: {T.ACTION_PRESSED}; border-color: {T.ACTION_PRESSED};
}}
/* Its own pressed colour, so the ring stays visible against the fill. */
QPushButton[variant="primary"]:focus {{
    border: 2px solid {T.ACTION_PRESSED}; padding: 5px 15px;
}}
QPushButton[variant="primary"]:disabled {{
    background: {ink(T.SURFACE_BORDER)}; border-color: {ink(T.SURFACE_BORDER)};
    color: {ink(T.INK_DISABLED)};
}}
QPushButton[variant="danger"] {{
    background: {T.CANVAS}; border-color: {T.ERROR}; color: {T.ERROR};
}}
QPushButton[variant="danger"]:hover {{
    background: {T.ERROR}; border-color: {T.ERROR}; color: {T.ACTION_TEXT};
}}
QPushButton[variant="danger"]:pressed {{
    background: {T.ERROR_PRESSED}; border-color: {T.ERROR_PRESSED};
    color: {T.ACTION_TEXT};
}}
QPushButton[variant="danger"]:focus {{
    border: 2px solid {T.ERROR_PRESSED}; padding: 5px 15px;
}}
QPushButton[variant="quiet"] {{
    background: transparent; border: 1px solid transparent;
    color: {ink(T.INK_SECONDARY)}; padding: 6px 10px;
}}
QPushButton[variant="quiet"]:hover {{
    background: {ink(T.SURFACE_HOVER)}; color: {ink(T.INK_PRIMARY)};
}}
QPushButton[variant="quiet"]:pressed {{ background: {ink(T.SURFACE_PRESSED)}; }}
QPushButton[variant="quiet"]:focus {{
    border: 2px solid {ink(T.INK_GLYPH)}; padding: 5px 9px;
}}
QPushButton[variant="quiet"]:disabled {{
    background: transparent; border-color: transparent;
    color: {ink(T.INK_DISABLED)};
}}

/* -- Navigation rail --------------------------------------------------- */
QWidget#Sidebar {{ background: {ink(T.SURFACE_WASH)}; border: none; }}
/* A destination list, not a row of buttons. Left-aligned so the labels form a
   readable column; the current item carries fill AND weight AND ink, because
   a fill on its own is a colour-only signal. */
QPushButton[variant="nav"] {{
    background: transparent; border: 1px solid transparent;
    border-radius: {T.RADIUS}px;
    padding: 6px 10px;
    min-height: {T.CONTROL_HEIGHT - 12}px;
    text-align: left;
    font-size: {T.FONT_LABEL}px;
    font-weight: {T.WEIGHT_MEDIUM};
    color: {ink(T.INK_SECONDARY)};
}}
QPushButton[variant="nav"]:hover {{
    background: {ink(T.SURFACE_HOVER)}; color: {ink(T.INK_PRIMARY)};
}}
QPushButton[variant="nav"]:pressed {{ background: {ink(T.SURFACE_PRESSED)}; }}
QPushButton[variant="nav"]:checked {{
    background: {ink(T.SURFACE_SELECTED)};
    color: {ink(T.INK_PRIMARY)};
    font-weight: {T.WEIGHT_SEMIBOLD};
}}
QPushButton[variant="nav"]:focus {{
    border: 2px solid {ink(T.INK_GLYPH)}; padding: 5px 9px;
}}

/* -- Tables ------------------------------------------------------------ */
/* Alternating wash, no gridlines, no cell borders — never all three. */
QTableView, QTreeView, QListView {{
    background: {T.CANVAS};
    alternate-background-color: {ink(T.SURFACE_WASH)};
    border: none;
    gridline-color: transparent;
    font-size: {T.FONT_BODY}px;
    outline: none;
    selection-background-color: {ink(T.SURFACE_SELECTED)};
    selection-color: {ink(T.INK_PRIMARY)};
}}
QTableView::item, QTreeView::item, QListView::item {{
    border: none; padding: 0px 8px;
}}
QTableView:focus, QTreeView:focus, QListView:focus {{
    border: 2px solid {ink(T.INK_GLYPH)}; border-radius: {T.RADIUS}px;
}}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{
    background: {ink(T.SURFACE_WASH)};
    color: {ink(T.INK_SECONDARY)};
    border: none;
    border-bottom: 1px solid {ink(T.SURFACE_BORDER)};
    padding: 6px 8px;
    font-size: {T.FONT_CAPTION}px;
    font-weight: {T.WEIGHT_SEMIBOLD};
}}
QHeaderView::section:hover {{ color: {ink(T.INK_PRIMARY)}; }}
QTableCornerButton::section {{
    background: {ink(T.SURFACE_WASH)}; border: none;
    border-bottom: 1px solid {ink(T.SURFACE_BORDER)};
}}

/* -- Tabs -------------------------------------------------------------- */
QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {ink(T.INK_SECONDARY)};
    border: none; border-bottom: 2px solid transparent;
    padding: 8px 16px;
    font-size: {T.FONT_LABEL}px; font-weight: {T.WEIGHT_MEDIUM};
}}
QTabBar::tab:hover:!selected {{ color: {ink(T.INK_PRIMARY)}; }}
QTabBar::tab:selected {{
    color: {ink(T.INK_PRIMARY)};
    border-bottom-color: {ink(T.INK_PRIMARY)};
}}
QTabBar::tab:focus {{ border-bottom: 2px solid {ink(T.INK_GLYPH)}; }}

/* -- Menus, status, tooltips ------------------------------------------- */
QMenuBar {{
    background: {T.CANVAS};
    border-bottom: 1px solid {ink(T.SURFACE_BORDER)};
    padding: 2px 8px;
    font-size: {T.FONT_LABEL}px;
}}
QMenuBar::item {{
    background: transparent; padding: 5px 10px;
    border-radius: {T.RADIUS - 2}px; color: {ink(T.INK_PRIMARY)};
}}
QMenuBar::item:selected {{ background: {ink(T.SURFACE_HOVER)}; }}
QMenu {{
    background: {T.CANVAS};
    border: 1px solid {ink(T.SURFACE_BORDER_STRONG)};
    border-radius: {T.RADIUS_PANEL}px;
    padding: 4px;
    font-size: {T.FONT_LABEL}px;
}}
QMenu::item {{
    padding: 6px 28px 6px 12px; border-radius: {T.RADIUS - 2}px;
    color: {ink(T.INK_PRIMARY)};
}}
QMenu::item:selected {{ background: {ink(T.SURFACE_HOVER)}; }}
QMenu::item:disabled {{ color: {ink(T.INK_DISABLED)}; }}
QMenu::separator {{
    height: 1px; background: {ink(T.SURFACE_BORDER)}; margin: 4px 8px;
}}
QStatusBar {{
    background: {T.CANVAS};
    border-top: 1px solid {ink(T.SURFACE_BORDER)};
    font-size: {T.FONT_CAPTION}px;
    color: {ink(T.INK_TERTIARY)};
    padding: 2px {T.MARGIN_WINDOW}px;
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{
    font-size: {T.FONT_CAPTION}px; color: {ink(T.INK_TERTIARY)};
}}
QToolTip {{
    background: {T.ink_hex(0.92)}; color: {T.CANVAS};
    border: none; border-radius: {T.RADIUS}px;
    padding: 6px 10px; font-size: {T.FONT_CAPTION}px;
}}
QDockWidget {{
    color: {ink(T.INK_SECONDARY)};
    font-size: {T.FONT_CAPTION}px; font-weight: {T.WEIGHT_MEDIUM};
    titlebar-close-icon: none; titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: transparent; padding: 6px 2px; text-align: left;
}}

/* -- Progress, splitters, scrollbars ----------------------------------- */
QProgressBar {{
    background: {ink(T.SURFACE_WASH)}; border: none; border-radius: 3px;
    height: 6px; text-align: center;
    font-size: {T.FONT_CAPTION}px; color: {ink(T.INK_SECONDARY)};
}}
QProgressBar::chunk {{ background: {T.ACTION}; border-radius: 3px; }}
QSplitter::handle {{ background: {ink(T.SURFACE_BORDER)}; }}
QSplitter::handle:hover {{ background: {ink(T.SURFACE_BORDER_STRONG)}; }}
QSplitter::handle:horizontal {{
    width: {T.SPLITTER_VISUAL}px; margin: 0 {grab}px;
}}
QSplitter::handle:vertical {{
    height: {T.SPLITTER_VISUAL}px; margin: {grab}px 0;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {ink(T.INK_GLYPH)}; border-radius: 5px; min-height: 32px;
}}
QScrollBar::handle:horizontal {{
    background: {ink(T.INK_GLYPH)}; border-radius: 5px; min-width: 32px;
}}
QScrollBar::handle:hover {{ background: {ink(T.INK_TERTIARY)}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
""" + _glyph_qss()


def _glyph_qss() -> str:
    """The two glyphs a style sheet can only take as images.

    Each rule is left out entirely if its image could not be written: Qt then
    draws its own and the control keeps working. A control is never allowed to
    fail over an ornament.
    """
    from .icons import glyph_url

    rules = []
    chevron = glyph_url("chevron-down")
    if chevron is not None:
        rules.append(f'QComboBox::down-arrow {{ image: url("{chevron}"); }}')
    # The tick sits on the action fill, so it takes that fill's text colour
    # rather than a rung of the ink ladder.
    check = glyph_url("check", T.ACTION_TEXT)
    if check is not None:
        rules.append(f'QCheckBox::indicator:checked {{ image: url("{check}"); }}')
    return "\n" + "\n".join(rules) + "\n" if rules else ""


# ---------------------------------------------------------------------------
# Application-level setup
# ---------------------------------------------------------------------------
def _palette() -> QPalette:
    """The base palette, for what a style sheet does not reach.

    QMessageBox and QFileDialog are the ones that matter here: they partly
    ignore the sheet, and without this they would arrive in the platform
    theme instead of the application's.
    """
    palette = QPalette()
    canvas = QColor(T.CANVAS)
    primary = QColor(T.ink_hex(T.INK_PRIMARY))
    palette.setColor(QPalette.Window, canvas)
    palette.setColor(QPalette.WindowText, primary)
    palette.setColor(QPalette.Base, canvas)
    palette.setColor(QPalette.AlternateBase, QColor(T.ink_hex(T.SURFACE_WASH)))
    palette.setColor(QPalette.Text, primary)
    palette.setColor(QPalette.PlaceholderText, QColor(T.ink_hex(T.INK_TERTIARY)))
    palette.setColor(QPalette.Button, canvas)
    palette.setColor(QPalette.ButtonText, primary)
    palette.setColor(QPalette.Highlight, QColor(T.ACTION))
    palette.setColor(QPalette.HighlightedText, QColor(T.ACTION_TEXT))
    palette.setColor(QPalette.Link, QColor(T.ACTION))
    palette.setColor(QPalette.ToolTipBase, QColor(T.ink_hex(0.92)))
    palette.setColor(QPalette.ToolTipText, canvas)
    disabled = QColor(T.ink_hex(T.INK_DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled)
    return palette


def apply_theme(app: QApplication) -> None:
    """Dress a QApplication: one style, one palette, one font, one sheet."""
    if not load_fonts():
        logger.warning(
            "%s is not available; the window falls back to the platform "
            "sans-serif. Type sizes and spacing are unaffected.", T.FONT_FAMILY
        )
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setFont(font())
    app.setStyleSheet(stylesheet())


def apply_mpl_theme() -> None:
    """Theme matplotlib to match. A figure does not read a style sheet."""
    T.apply_matplotlib_style()


def restyle(widget: QWidget) -> None:
    """Re-read the sheet after a ``role`` or ``variant`` property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


# Kept as the old name so call sites reading "apply the theme" still say so.
apply = apply_theme
