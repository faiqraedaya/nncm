"""The application's navigation rail.

Five destinations in the order the workflow runs, each with its label and its
icon. A rail rather than a tab bar because five destinations with multi-word
labels do not fit a single row without abbreviating them, and because adding a
sixth would cost horizontal space the content needs.

The rail is a list of places, not a row of buttons: the labels align in a
column, and the current page is marked by fill *and* weight *and* ink, so the
selection never rests on a background tint alone.

The numbers stay on the labels. This is a pipeline — the order in which the
stages are run is part of what the user needs to know, and a rail is where
that ordering is legible.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QSizePolicy, QWidget

from .. import theme as T
from . import layout as ly
from .icons import icon

# key, label, icon name. The order is the navigation order.
PAGES = (
    ("project", "1 · Project", "folder"),
    ("sample", "2 · Sample", "scatter"),
    ("phast", "3 · Phast", "exchange"),
    ("train", "4 · Train", "network"),
    ("predict", "5 · Predict", "trend"),
)


class Sidebar(QWidget):
    """Vertical navigation. Emits the index of the page the user picked."""

    selected = Signal(int)

    # The longest label ("3 · Phast" is not it — "1 · Project" is) plus its
    # icon, its padding and the rail's own margins, with nothing eliding.
    MIN_W = 164
    DEFAULT_W = 192

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setMinimumWidth(self.MIN_W)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        panel = ly.vbox(self, margin=T.SPACING_ROW, spacing=T.SPACING_GROUP)

        brand = ly.brand("NNCM")
        brand.setContentsMargins(T.SPACING_ROW, T.SPACING_ROW, T.SPACING_ROW, 0)
        brand.setToolTip("Neural network consequence modelling")
        panel.addWidget(brand)

        nav = ly.vbox(spacing=2)  # one list, so the items sit tight together
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list = []

        for index, (_key, label, glyph) in enumerate(PAGES):
            button = ly.button(label, variant="nav")
            button.setCheckable(True)
            button.setIcon(icon(glyph))
            button.setIconSize(QSize(T.ICON_SIZE, T.ICON_SIZE))
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked, i=index: self.selected.emit(i))
            self._group.addButton(button, index)
            self._buttons.append(button)
            nav.addWidget(button)

        panel.addLayout(nav)
        panel.addStretch(1)

        self._buttons[0].setChecked(True)

    def set_current(self, index: int) -> None:
        """Mark a page as current without re-emitting the selection."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def current(self) -> int:
        return self._group.checkedId()

    @staticmethod
    def page_title(index: int) -> str:
        """The page's own name, without the stage number the rail carries."""
        label = PAGES[index][1]
        return label.split("·", 1)[-1].strip()
