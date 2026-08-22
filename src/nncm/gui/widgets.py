"""The pieces every page is built from.

A component here reaches only for role tokens, never a raw value, and hides
itself when it has nothing to say. The rules it exists to keep:

* a container owns its edge — a table inside a card draws no border of its own;
* every figure carries a name and a unit, taken from :mod:`nncm.quantities`, so
  a label and its explanation cannot drift apart;
* a table sorts on double-click, by model values, blanks last, and its totals
  live inside it as its last row;
* an empty view says what is absent, in a sentence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSize,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..quantities import Quantity, describe
from . import style

# Padding, translated from the guide's CSS order (top/right/bottom/left) into
# Qt's (left, top, right, bottom).
WINDOW_MARGINS = (12, 16, 16, 16)
CARD_MARGINS = (14, 16, 12, 16)
METRIC_MARGINS = (14, 12, 12, 12)

NAME_COLUMN_FLOOR = 140   # below this a name stops identifying its row
NAME_COLUMN_CEILING = 320  # above it, the name is stealing width from figures


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def elide_middle(text: str, keep: int = 46) -> str:
    """Shorten a path from the middle: both ends identify it, the middle does not."""
    text = str(text)
    if len(text) <= keep:
        return text
    head = (keep - 1) // 2
    tail = keep - 1 - head
    return f"{text[:head]}…{text[-tail:]}"


def format_number(value: Any) -> str:
    """A figure for a table cell. No value renders blank, never a zero."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,d}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,d}"
        return f"{value:,.6g}"
    text = str(value)
    return "" if text in {"nan", "None", "NaT"} else text


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Type and containers
# ---------------------------------------------------------------------------
class PaneHeader(QLabel):
    """A label and nothing else. Space does the separating, not a rule."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("PaneHeader")


class Caption(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("Caption")
        self.setWordWrap(True)


class Explanation(QLabel):
    """The one job italic has: a sentence saying what a setting changes.

    A wrapped label in a grid comes back one line short unless the height is
    measured explicitly — Qt asks for a height before it knows the width. So
    the height is recomputed whenever the width changes, and only then.

    The measuring is done against the font rather than by asking QLabel, whose
    own answer is floored by the minimum height already set on it. Asked in a
    loop like this one, that answer can only ever grow: a label measured once
    while it was briefly narrow would keep the four lines it needed then, for
    the rest of the session, however wide it later became.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("Explanation")
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(policy)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt naming)
        margins = self.contentsMargins()
        usable = max(width - margins.left() - margins.right(), 1)
        wrapped = self.fontMetrics().boundingRect(
            0, 0, usable, 1 << 20, Qt.TextWordWrap | self.alignment(), self.text()
        )
        return wrapped.height() + margins.top() + margins.bottom()

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        super().setText(text)
        self.setMinimumHeight(self.heightForWidth(self.width()))

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.setMinimumHeight(self.heightForWidth(self.width()))


class EmptyState(QLabel):
    """What is absent, and why — never a blank rectangle, never "No data"."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("EmptyState")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)


class Card(QFrame):
    """One surface for one subject. Content inside it stays borderless."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*CARD_MARGINS)
        self._layout.setSpacing(12)
        if title:
            heading = QLabel(title)
            heading.setObjectName("CardTitle")
            self._layout.addWidget(heading)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class MetricCell(QWidget):
    """Caption above a figure, with its unit beside it. A readout, not a control."""

    def __init__(self, caption: str, help_text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*METRIC_MARGINS)
        layout.setSpacing(2)
        self._caption = QLabel(caption)
        self._caption.setObjectName("MetricCaption")
        self._value = QLabel("")
        self._value.setObjectName("MetricValue")
        self._value.setFont(style.font(theme.FONT_METRIC, theme.WEIGHT_MEDIUM, figures=True))
        self._unit = QLabel("")
        self._unit.setObjectName("MetricUnit")
        figure = QHBoxLayout()
        figure.setContentsMargins(0, 0, 0, 0)
        figure.setSpacing(4)
        figure.addWidget(self._value)
        figure.addWidget(self._unit, 0, Qt.AlignBottom)
        figure.addStretch(1)
        layout.addWidget(self._caption)
        layout.addLayout(figure)
        if help_text:
            self.setToolTip(help_text)

    def set_value(self, value: str | None, unit: str = "") -> None:
        """A quantity with no meaningful value shows a dash, never a zero.

        A zero is a figure and would be read as one. The dash says there is no
        figure yet, and the unit goes with it — a unit with nothing to measure
        is furniture.
        """
        self._value.setText(value if value else "–")
        self._unit.setText(unit if value else "")


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
class _NoWheel(QObject):
    """Stops a wheel over an unfocused field from silently editing it.

    Scrolling a form should scroll the form. A spin box that eats the gesture
    changes a value the reader was only passing over.
    """

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        if event.type() == event.Type.Wheel and not watched.hasFocus():
            event.ignore()
            return True
        return False


_NO_WHEEL = _NoWheel()


def _prepare_numeric(box) -> None:
    # Right-aligned, wide enough for the longest realistic value: a figure
    # showing its own tail is worse than no figure. The steppers are off
    # because styling a Qt spin box sub-control replaces the native ones, and
    # a painted arrow is one more decorative asset to keep working for a
    # gesture the keyboard already offers.
    box.setButtonSymbols(box.ButtonSymbols.NoButtons)
    box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    box.setMinimumWidth(104)
    box.setMaximumWidth(168)
    box.setFont(style.tabular(box.font()))
    box.setFocusPolicy(Qt.StrongFocus)
    box.installEventFilter(_NO_WHEEL)


def integer_field(low: int, high: int, value: int = 0, special: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    if special:
        box.setSpecialValueText(special)
    box.setValue(value)
    _prepare_numeric(box)
    return box


def decimal_field(low: float, high: float, value: float = 0.0, decimals: int = 3) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setValue(value)
    _prepare_numeric(box)
    return box


def choice_field(options: Iterable[str], current: str = "") -> QComboBox:
    """A fixed set of valid answers is a list to pick from, not a box to type in."""
    box = QComboBox()
    box.addItems(list(options))
    if current:
        box.setCurrentText(current)
    box.installEventFilter(_NO_WHEEL)
    return box


class PathField(QWidget):
    """A path and the one control that changes it."""

    def __init__(
        self,
        caption: str,
        mode: str = "open",
        file_filter: str = "Excel workbooks (*.xlsx)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._caption = caption
        self._mode = mode
        self._filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Not set")
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.cwd())
        if self._mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, self._caption, start)
        elif self._mode == "save":
            chosen, _ = QFileDialog.getSaveFileName(self, self._caption, start, self._filter)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, self._caption, start, self._filter)
        if chosen:
            self.edit.setText(chosen)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None

    def set_path(self, path: Path | str | None) -> None:
        self.edit.setText(str(path) if path else "")
        self.edit.setToolTip(str(path) if path else "")


class RangeField(QWidget):
    """Two editors reading as one quantity: a floor, a ceiling, and how it is spaced."""

    def __init__(self, low: float, high: float, decimals: int = 3, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.minimum = decimal_field(low, high, low, decimals)
        self.maximum = decimal_field(low, high, high, decimals)
        joiner = QLabel("to")
        joiner.setObjectName("Unit")
        self.log = QCheckBox("Log-spaced")
        self.log.setToolTip(describe("log_spacing").help)
        layout.addWidget(self.minimum)
        layout.addWidget(joiner)
        layout.addWidget(self.maximum)
        layout.addWidget(self.log)
        layout.addStretch(1)

    def values(self) -> tuple[float, float, bool]:
        return self.minimum.value(), self.maximum.value(), self.log.isChecked()

    def set_values(self, minimum: float, maximum: float, log: bool) -> None:
        self.minimum.setValue(minimum)
        self.maximum.setValue(maximum)
        self.log.setChecked(log)


class Form(QWidget):
    """One row is label, editor, unit — with the explanation under both columns.

    Every label, unit and sentence comes from :mod:`nncm.quantities`, so there
    is only ever one copy of each to keep true.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(9)
        # The editor column takes the width it needs and the unit follows it;
        # a number stranded at the far side of the pane has lost its label.
        self._grid.setColumnStretch(2, 1)
        # A form is as tall as its rows and no shorter. Left free to shrink it
        # gives back the height of the last explanation first, which reads as a
        # sentence sliced in half by the edge of a card.
        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)
        self._row = 0

    def add(self, key: str, editor: QWidget, *, span: bool = False, label: str | None = None) -> QWidget:
        quantity: Quantity = describe(key)
        name = QLabel(label if label is not None else quantity.label)
        name.setObjectName("FieldLabel")
        name.setWordWrap(True)  # wrap the label rather than clip the field
        name.setBuddy(editor)
        self._grid.addWidget(name, self._row, 0, Qt.AlignLeft | Qt.AlignVCenter)
        if span:
            self._grid.addWidget(editor, self._row, 1, 1, 2)
        else:
            self._grid.addWidget(editor, self._row, 1)
            unit = QLabel(quantity.unit)
            unit.setObjectName("Unit")
            unit.setMinimumWidth(56)  # units line up down the pane
            self._grid.addWidget(unit, self._row, 2, Qt.AlignLeft | Qt.AlignVCenter)
        self._row += 1
        if quantity.help:
            note = Explanation(quantity.help)
            self._grid.addWidget(note, self._row, 1, 1, 2)
            self._row += 1
        return editor

    def add_switch(self, key: str, box: QCheckBox) -> QCheckBox:
        """A checkbox names itself, so it takes the whole row."""
        quantity = describe(key)
        box.setText(quantity.label)
        self._grid.addWidget(box, self._row, 0, 1, 3)
        self._row += 1
        if quantity.help:
            note = Explanation(quantity.help)
            self._grid.addWidget(note, self._row, 0, 1, 3)
            self._row += 1
        return box

    def add_widget(self, widget: QWidget) -> QWidget:
        self._grid.addWidget(widget, self._row, 0, 1, 3)
        self._row += 1
        return widget


# ---------------------------------------------------------------------------
# Advisory notes
# ---------------------------------------------------------------------------
class Advisories(QWidget):
    """A dot beside a sentence. With nothing to report, the block hides itself."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self.hide()

    def show_notes(self, notes: Sequence[tuple[str, str]]) -> None:
        """``notes`` is a sequence of (severity, sentence); severity is a token name."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for severity, sentence in notes:
            self._layout.addWidget(_Advisory(severity, sentence))
        self.setVisible(bool(notes))


class _Advisory(QWidget):
    _COLOURS = {"subtle": theme.SUBTLE, "warning": theme.WARNING, "danger": theme.DANGER}

    def __init__(self, severity: str, sentence: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(QSize(8, 8))
        colour = self._COLOURS.get(severity, theme.SUBTLE)
        dot.setStyleSheet(f"background: {colour}; border-radius: 4px;")
        # Held in a top-aligned spacer so the dot sits on the first line of a
        # message that wraps.
        holder = QVBoxLayout()
        holder.setContentsMargins(0, 4, 0, 0)
        holder.addWidget(dot)
        holder.addStretch(1)
        layout.addLayout(holder)
        text = QLabel(sentence)
        text.setWordWrap(True)
        text.setObjectName("FieldLabel")
        layout.addWidget(text, 1)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
class ActionBar(QFrame):
    """Where a stage's actions live: reversible on the left, the one Primary on the right.

    Reading surfaces carry no buttons, so the figures get the whole width.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ActionBar")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setMaximumWidth(260)
        self._progress.hide()
        self._layout.addWidget(self._progress)
        self._layout.addStretch(1)

    def add_secondary(self, text: str, on_click: Callable[[], None], tip: str = "") -> QPushButton:
        """A reversible action. Added in reading order, left of the Primary."""
        button = QPushButton(text)
        button.clicked.connect(on_click)
        if tip:
            button.setToolTip(tip)
        self._layout.addWidget(button)
        return button

    def add_primary(self, text: str, on_click: Callable[[], None], tip: str = "") -> QPushButton:
        button = QPushButton(text)
        button.setProperty("primary", "true")
        button.clicked.connect(on_click)
        if tip:
            button.setToolTip(tip)
        self._layout.addWidget(button)
        return button

    def show_progress(self, done: int, total: int, unit: str = "") -> None:
        """Stepped progress: every step is a thing that actually finished."""
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._progress.setFormat(f"%v of at most %m {unit}".strip())
        self._progress.show()

    def clear_progress(self) -> None:
        self._progress.hide()
        self._progress.reset()


def set_primary(button: QPushButton, primary: bool) -> None:
    """Move the single Primary between two buttons as the project's state moves."""
    button.setProperty("primary", "true" if primary else None)
    button.style().unpolish(button)
    button.style().polish(button)


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------
class LogView(QPlainTextEdit):
    """Append-only record of what each stage did.

    Real monospace, deliberately: this is pre-formatted, space-aligned output,
    which is the one thing a monospace face is for.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFrameShape(QFrame.NoFrame)
        self.setFont(style.font(theme.FONT_UNIT, mono=True))
        self.setPlaceholderText("Nothing has run yet in this session.")

    def append_line(self, text: str) -> None:
        for line in str(text).rstrip("\n").split("\n"):
            self.appendPlainText(line)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
class PlotArea(QFrame):
    """A drawing surface that says what is absent while it is empty.

    Borderless on purpose: it lives inside a card, and two nested edges is the
    most common way a clean interface stops being clean.
    """

    def __init__(self, empty_text: str = "Nothing to plot yet.", parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._empty = EmptyState(empty_text)
        self._layout.addWidget(self._empty)

    def show_message(self, text: str) -> None:
        self._drop_canvas()
        self._empty.setText(text)
        self._empty.show()

    def show_figure(self, figure) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._drop_canvas()
        self._empty.hide()
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setStyleSheet(f"background: {theme.SURFACE};")
        self._layout.addWidget(self._canvas)
        self._canvas.draw_idle()

    def _drop_canvas(self) -> None:
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
class FrameModel(QAbstractTableModel):
    """A DataFrame as a table, plus an optional totals row that lives inside it.

    Sorting reads :data:`SortRole` — the model's own value — so ``"1,000"``
    can never sort before ``"9"``, and a cell with no value stays blank rather
    than being rendered as a zero.
    """

    SortRole = Qt.UserRole + 1

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._totals: dict[str, Any] = {}
        self._numeric: list[bool] = []
        self._figures = style.font(theme.FONT_BODY, figures=True)
        self._bold = style.font(theme.FONT_BODY, theme.WEIGHT_SEMIBOLD, figures=True)

    # -- data ------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, totals: dict[str, Any] | None = None) -> None:
        self.beginResetModel()
        self._frame = frame.reset_index(drop=True)
        self._totals = dict(totals or {})
        self._numeric = [
            pd.api.types.is_numeric_dtype(self._frame[c]) and not pd.api.types.is_bool_dtype(self._frame[c])
            for c in self._frame.columns
        ]
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def is_totals(self, row: int) -> bool:
        return bool(self._totals) and row == len(self._frame)

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame) + (1 if self._totals else 0)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame.columns)

    def _value(self, row: int, column: int) -> Any:
        name = self._frame.columns[column]
        if self.is_totals(row):
            return self._totals.get(name)
        value = self._frame.iat[row, column]
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        value = self._value(row, column)
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            text = format_number(value)
            if self.is_totals(row) and column == 0 and not text:
                return ""
            return text
        if role == self.SortRole:
            return value
        if role == Qt.TextAlignmentRole:
            align = Qt.AlignRight if self._numeric[column] else Qt.AlignLeft
            return int(align | Qt.AlignVCenter)
        if role == Qt.FontRole:
            if self.is_totals(row):
                return self._bold
            return self._figures if self._numeric[column] else None
        if role == Qt.BackgroundRole and self.is_totals(row):
            return QColor(theme.SURFACE_ALT)
        return None

    def flags(self, index: QModelIndex):
        if self.is_totals(index.row()):
            return Qt.ItemIsEnabled  # present and readable, but not a row to select
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section: int, orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if orientation != Qt.Horizontal or section >= len(self._frame.columns):
            return None
        quantity = describe(str(self._frame.columns[section]))
        if role == Qt.DisplayRole:
            return quantity.with_unit()
        if role == Qt.ToolTipRole:
            return quantity.help or quantity.with_unit()
        return None


class SortProxy(QSortFilterProxyModel):
    """Sorting that keeps its promises: model values, blanks last, totals pinned.

    Blanks sort last in *both* directions — no figure is not a small figure —
    and the totals row stays where it belongs however the table is ordered.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.setSortRole(FrameModel.SortRole)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        descending = self.sortOrder() == Qt.DescendingOrder
        if model.is_totals(left.row()):
            return descending
        if model.is_totals(right.row()):
            return not descending
        a = model.data(left, FrameModel.SortRole)
        b = model.data(right, FrameModel.SortRole)
        if a is None and b is None:
            return False
        if a is None:
            return descending
        if b is None:
            return not descending
        if is_number(a) and is_number(b):
            return float(a) < float(b)
        return str(a).casefold() < str(b).casefold()


class DataTable(QWidget):
    """A reading surface: no gridlines, no buttons, one hairline under each row.

    Sorts on **double**-click, cycling ascending, descending, then back to the
    model's own order — a single click on the way to dragging a column edge
    must not reorder the table under the reader. Sorting permutes the view
    only; the frame handed in is never rewritten.
    """

    def __init__(self, empty_text: str = "Nothing here yet.", parent: QWidget | None = None):
        super().__init__(parent)
        self._model = FrameModel(self)
        self._proxy = SortProxy(self)
        self._proxy.setSourceModel(self._model)
        self._sorted: tuple[int, int] | None = None
        self._key_columns: list[str] = []

        self.view = QTableView()
        self.view.setModel(self._proxy)
        self.view.setShowGrid(False)
        self.view.setAlternatingRowColors(False)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setWordWrap(False)
        self.view.setSortingEnabled(False)  # our own gesture, see below
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(theme.ROW_HEIGHT)
        header = self.view.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setHighlightSections(False)
        header.setMinimumSectionSize(NAME_COLUMN_FLOOR // 2)
        header.sectionDoubleClicked.connect(self._cycle_sort)

        self._empty = EmptyState(empty_text)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.view)
        self._layout.addWidget(self._empty)
        self.view.hide()

    # -- content ----------------------------------------------------------
    def show_frame(
        self,
        frame: pd.DataFrame,
        totals: dict[str, Any] | None = None,
        key_columns: Sequence[str] | None = None,
    ) -> None:
        self._key_columns = [c for c in (key_columns or []) if c in frame.columns]
        self._model.set_frame(frame, totals)
        self._sorted = None
        self._proxy.sort(-1)
        empty = frame.empty
        self.view.setVisible(not empty)
        self._empty.setVisible(empty)
        if not empty:
            self._size_columns()

    def show_message(self, text: str) -> None:
        self._empty.setText(text)
        self._model.set_frame(pd.DataFrame())
        self.view.hide()
        self._empty.show()

    def set_detail_visible(self, visible: bool) -> None:
        """Leave on screen only the columns a decision is made on."""
        if not self._key_columns:
            return
        for index, name in enumerate(self._model.frame.columns):
            self.view.setColumnHidden(index, not visible and str(name) not in self._key_columns)

    # -- behaviour --------------------------------------------------------
    def _cycle_sort(self, column: int) -> None:
        order = Qt.AscendingOrder
        if self._sorted and self._sorted[0] == column:
            if self._sorted[1] == Qt.AscendingOrder:
                order = Qt.DescendingOrder
            else:
                # Third click returns the model's own order, so there is
                # always a way back.
                self._sorted = None
                self._proxy.sort(-1)
                self.view.horizontalHeader().setSortIndicatorShown(False)
                return
        self._sorted = (column, order)
        self.view.horizontalHeader().setSortIndicatorShown(True)
        self.view.horizontalHeader().setSortIndicator(column, order)
        self._proxy.sort(column, order)

    def _size_columns(self) -> None:
        header = self.view.horizontalHeader()
        self.view.resizeColumnsToContents()
        for index in range(self._model.columnCount()):
            if index == 0:
                width = min(max(self.view.columnWidth(0), NAME_COLUMN_FLOOR), NAME_COLUMN_CEILING)
                self.view.setColumnWidth(0, width)
            else:
                self.view.setColumnWidth(index, max(self.view.columnWidth(index), 88))
        header.setStretchLastSection(True)
