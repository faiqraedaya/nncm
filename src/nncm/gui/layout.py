"""Layout and widget factories.

Every layout in the application comes from here. Qt's own defaults are off the
grid — 9 px margins, 6 px spacing — so a layout built with a bare constructor
is a layout that has quietly left the design system. Going through a factory
makes that impossible rather than merely discouraged.

The same argument applies to the role labels: a role is spelled once, here,
rather than at every call site that needs a caption.
"""

from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import theme as T
from .theme import restyle


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def vbox(parent: QWidget | None = None, *, margin: int = 0,
         spacing: int | None = None) -> QVBoxLayout:
    layout = QVBoxLayout(parent) if parent is not None else QVBoxLayout()
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(T.SPACING_ROW if spacing is None else spacing)
    return layout


def hbox(parent: QWidget | None = None, *, margin: int = 0,
         spacing: int | None = None) -> QHBoxLayout:
    layout = QHBoxLayout(parent) if parent is not None else QHBoxLayout()
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(T.SPACING_ROW if spacing is None else spacing)
    return layout


def grid(parent: QWidget | None = None, *, margin: int = 0,
         spacing: int | None = None) -> QGridLayout:
    layout = QGridLayout(parent) if parent is not None else QGridLayout()
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setHorizontalSpacing(T.SPACING_ROW if spacing is None else spacing)
    layout.setVerticalSpacing(T.SPACING_ROW if spacing is None else spacing)
    return layout


def form(parent: QWidget | None = None, *, margin: int = 0) -> QFormLayout:
    layout = QFormLayout(parent) if parent is not None else QFormLayout()
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setHorizontalSpacing(T.SPACING_GROUP)
    layout.setVerticalSpacing(T.SPACING_ROW)
    # One alignment for every form in the application. Inconsistent label
    # alignment between windows is the most visible flaw in a multi-page tool.
    layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    return layout


def window_layout(parent: QWidget) -> QVBoxLayout:
    """The outermost layout of a window or page: window margins, group spacing."""
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(
        T.MARGIN_WINDOW, T.MARGIN_WINDOW, T.MARGIN_WINDOW, T.MARGIN_WINDOW
    )
    layout.setSpacing(T.SPACING_GROUP)
    return layout


def panel(title: str = "", *, margin: int | None = None,
          spacing: int | None = None) -> tuple[QFrame, QVBoxLayout]:
    """A bordered surface for one subject, with its heading already placed.

    Returns the frame and the layout to fill. Content inside stays borderless:
    a bordered input inside a bordered group inside a bordered pane is three
    outlines on one control.
    """
    frame = QFrame()
    frame.setProperty("role", "panel")
    inner = vbox(
        frame,
        margin=T.MARGIN_GROUP if margin is None else margin,
        spacing=T.SPACING_ROW if spacing is None else spacing,
    )
    if title:
        inner.addWidget(heading(title))
    return frame, inner


def bare_panel(*, margin: int = 0, spacing: int | None = None) -> tuple[QFrame, QVBoxLayout]:
    """A grouping frame that draws no edge of its own."""
    frame = QFrame()
    frame.setProperty("role", "bare")
    return frame, vbox(frame, margin=margin, spacing=spacing)


def rule() -> QFrame:
    """A 1 px divider. Never place one across a relationship."""
    line = QFrame()
    line.setProperty("role", "rule")
    line.setFixedHeight(1)  # a rule has no text to clip
    line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return line


def splitter(*widgets: QWidget, orientation=Qt.Horizontal,
             sizes: Iterable[int] | None = None,
             collapsible: bool = False) -> QSplitter:
    """A splitter with its panes already added.

    Taking the panes as arguments is the point: ``setSizes`` before
    ``addWidget`` is silently ignored, and passing them in makes that ordering
    impossible to get wrong.
    """
    split = QSplitter(orientation)
    split.setHandleWidth(T.SPLITTER_GRAB)
    split.setChildrenCollapsible(collapsible)
    for index, widget in enumerate(widgets):
        split.addWidget(widget)
        split.setCollapsible(index, collapsible)
    if sizes is not None:
        split.setSizes(list(sizes))
    return split


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
def _label(text: str, role: str, *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", role)
    label.setWordWrap(wrap)
    return label


def title(text: str) -> QLabel:
    """The name of the current page. One per screen."""
    return _label(text, "title")


def brand(text: str) -> QLabel:
    """The application name in the navigation rail — heading size, not title.

    Two 22 px titles on one screen compete, and the page title has to win.
    """
    return _label(text, "brand")


def heading(text: str) -> QLabel:
    return _label(text, "heading")


def caption(text: str = "") -> QLabel:
    return _label(text, "caption", wrap=True)


def field_label(text: str) -> QLabel:
    return _label(text, "field", wrap=True)


def unit_label(text: str) -> QLabel:
    return _label(text, "unit")


def metric_value(text: str = "") -> QLabel:
    return _label(text, "metric")


def empty_state(text: str) -> QLabel:
    """What is absent, and which action fills it. Never a blank rectangle."""
    label = _label(text, "empty", wrap=True)
    label.setAlignment(Qt.AlignCenter)
    return label


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status_label(text: str = "", role: str = "") -> QLabel:
    """An inline banner for a validation or calculation outcome.

    Hides itself when empty, so it takes no height at rest. Reporting here
    rather than in a modal keeps the results on screen while the message is
    read, and lets the message be as long as it needs to be.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    set_status(label, text, role)
    return label


def set_status(label: QLabel, text: str, role: str = "") -> None:
    """Set an inline status. ``role`` is error, warning, success, or empty."""
    label.setText(text)
    label.setProperty("role", role if text else "")
    label.setVisible(bool(text))
    restyle(label)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def button(text: str, *, variant: str = "", on_click: Callable[[], None] | None = None,
           tip: str = "") -> QPushButton:
    widget = QPushButton(text)
    if variant:
        widget.setProperty("variant", variant)
    if on_click is not None:
        widget.clicked.connect(on_click)
    if tip:
        widget.setToolTip(tip)
    return widget


def set_variant(widget: QWidget, variant: str | None) -> None:
    """Move a variant between widgets — e.g. the single primary of a page."""
    widget.setProperty("variant", variant)
    restyle(widget)


def action_row(primary: QPushButton | None, *others: QPushButton) -> QHBoxLayout:
    """The one placement of a page's actions, used by every page.

    Secondary actions first in reading order, the single primary last on the
    trailing edge. A primary sized to its label on one page and stretched
    across another is the most visible inconsistency in a multi-page tool, so
    there is exactly one helper and every page calls it.
    """
    row = hbox(spacing=T.SPACING_ROW)
    row.addStretch(1)
    for widget in others:
        row.addWidget(widget)
    if primary is not None:
        row.addWidget(primary)
    return row


def value_field(widget: QWidget) -> QWidget:
    """A field holding a number: never narrower than the values it must show."""
    widget.setMinimumWidth(T.FIELD_MIN_W)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return widget


def read_only(widget: QWidget) -> QWidget:
    """Mark a field as a readout, so it reads as a value not as somewhere to type."""
    widget.setProperty("readOnlyValue", "true")
    restyle(widget)
    return widget
