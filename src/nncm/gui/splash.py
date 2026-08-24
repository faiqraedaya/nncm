"""The window shown while the application is still starting.

Opening NNCM takes several seconds, almost all of it spent importing pandas,
matplotlib and openpyxl before the first widget can be built. Without
something on screen the user gets a taskbar entry and nothing else, which is
indistinguishable from a launch that failed.

**This module must stay cheap to import.** It reaches for PySide6 and the
Qt-free tokens in :mod:`nncm.theme`, and nothing else — a splash that has to
wait for the slow imports before it can be shown is not a splash. The startup
sequence in :func:`nncm.gui.startup.run` imports it first, puts it on screen,
and only then pulls in the modules that cost the time.

The stages it reports are the real ones, named in the user's terms. It shows
what is happening rather than a bar that fills at a rate nobody measured.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressBar, QWidget

from .. import theme as T
from . import layout as ly


class Splash(QWidget):
    """A frameless card: the name, what is loading, and an activity bar."""

    # Wide enough for the longest stage message without wrapping, and no
    # taller than the three things it holds. A splash that reflows as its
    # message changes draws attention to itself instead of to the wait.
    WIDTH = 380

    def __init__(self, steps: int = 1):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setObjectName("Splash")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._steps = max(steps, 1)
        self._done = 0

        panel = ly.vbox(self, margin=T.SPACING_SECTION, spacing=T.SPACING_ROW)
        panel.addWidget(ly.brand("NNCM"))
        panel.addWidget(ly.heading("Neural Network Consequence Modelling"))
        self._caption = ly.caption("Starting…")
        panel.addWidget(self._caption)

        # Determinate, because the stages are countable and each one reported
        # is a thing that actually finished. Shown at its resting position
        # from the first paint, never animated into place.
        self._bar = QProgressBar()
        self._bar.setRange(0, self._steps)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        panel.addWidget(self._bar)

        self.setFixedWidth(self.WIDTH)  # no text of its own resizes with it
        self.adjustSize()

    def step(self, message: str) -> None:
        """Say what is happening now, and repaint before it starts.

        The startup work runs on the UI thread — it is a chain of imports, and
        imports cannot be moved off it — so the event loop has to be pumped by
        hand or the splash would show its first message and then freeze for the
        whole wait.
        """
        self._caption.setText(message)
        self._bar.setValue(min(self._done, self._steps))
        self._done += 1
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish(self, window: QWidget) -> None:
        """Close once the real window is up, so nothing flashes empty."""
        self._bar.setValue(self._steps)
        self.close()
        window.activateWindow()
        window.raise_()

    def center_on_screen(self) -> None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return
        area = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(area.center())
        self.move(frame.topLeft())
