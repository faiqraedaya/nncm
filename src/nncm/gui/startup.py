"""Bringing the application up, in an order the user can watch.

Opening NNCM costs several seconds, and almost none of it is work this package
does: it is the import of pandas, matplotlib and openpyxl that :mod:`nncm.gui.app`
pulls in transitively. That cost is unavoidable here — they are what the pages
are built out of — but being told nothing while it happens is not.

So the order matters, and it is the whole point of this module:

1. the QApplication and the theme, which are cheap;
2. the splash, which imports nothing heavier than the tokens;
3. *then* the expensive imports, each one announced before it starts.

Putting this in :mod:`nncm.gui.app` would not work. That module is the
expensive import, so anything it contains can only run once the wait is
already over.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Cheap: PySide6, plus the Qt-free tokens. Nothing here reaches for pandas.
from . import theme as gui_theme
from .splash import Splash

# Named in the user's terms, and each one is a stage that actually happens.
# Kept beside the calls below so a stage cannot be added without a message.
_STAGES = (
    "Loading the charting library…",
    "Loading the workflow pages…",
    "Opening the project…",
)


def run(project_root: Path | None = None) -> int:
    """Start the application, showing progress while it loads."""
    app = QApplication.instance() or QApplication(sys.argv)

    # Before the splash, so the splash is themed like everything else. Both
    # are cheap; between them they cost a tenth of a second.
    gui_theme.apply_theme(app)

    splash = Splash(steps=len(_STAGES))
    splash.center_on_screen()
    splash.show()
    app.processEvents()  # paint it before the first slow import begins

    try:
        splash.step(_STAGES[0])
        gui_theme.apply_mpl_theme()

        splash.step(_STAGES[1])
        # The expensive one — pandas, matplotlib and openpyxl arrive here.
        from .app import MainWindow

        splash.step(_STAGES[2])
        window = MainWindow(project_root)
        window.show()
    except BaseException:
        # A failed start must not leave a frameless window with no way to
        # close it sitting on top of everything.
        splash.close()
        raise

    splash.finish(window)
    return app.exec()
