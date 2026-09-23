"""The window: a navigation rail, five stages, one log, one status line.

The rail is the workflow, top to bottom, and it names the stages in the order
they are run. The content area names the current page once, in one place,
driven by the rail — five separately-placed titles drift, one placement
cannot.

The menu bar offers every action the window has, so the whole application is
reachable from the keyboard. Each stage repeats its own action as the single
primary in its action bar; nothing else carries a button.

Two things the window owns on behalf of every page: whether detail columns are
shown (a table should hold the columns a decision is
made on, and nothing else), and the busy state — the status bar's progress
indicator exists only while something is running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QWidget,
)

from .. import theme as T
from ..config import Project, default_project_root
from . import layout as ly
from . import theme as gui_theme
from .icons import icon
from .pages import Page, PhastPage, PredictPage, ProjectPage, SamplePage, TrainPage
from .sidebar import Sidebar
from .widgets import LogView
from .workers import TaskRunner


class MainWindow(QMainWindow):
    # Both derived from the content, not from a general floor.
    #
    # Width: the Predict page puts a form beside a results table and the Train
    # page puts two figure panels side by side; below this the rail plus either
    # pair starts clipping rather than merely tightening.
    MIN_W = Sidebar.MIN_W + 780
    # Height: everything that does not scroll, plus the floor the page body
    # keeps for itself. Guessing this number is how a window ends up with a
    # minimum size at which a field is still sliced in half.
    LOG_DOCK_H = 128
    MIN_H = (
        Page.BODY_MIN_H     # the page body's own floor
        + 44                # the page title header
        + T.SPACING_GROUP   # header to body
        + 40                # the action bar
        + T.MARGIN_WINDOW * 2
        + 30                # menu bar
        + 26                # status bar
        + LOG_DOCK_H + 28   # the log and its title
    )

    def __init__(self, project_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("NNCM — Neural network consequence modelling")
        self.setMinimumSize(QSize(self.MIN_W, self.MIN_H))
        self.resize(1360, 900)
        self.project = Project.create(Path(project_root or default_project_root()))
        self.runner = TaskRunner(self)
        self.settings = QSettings("nncm", "nncm")
        self.detail_visible = False
        self._task_name = ""
        self._sidebar_width = Sidebar.DEFAULT_W

        # -- shell ---------------------------------------------------------
        self.sidebar = Sidebar()
        self.sidebar.selected.connect(self.show_page)

        content = QWidget()
        content.setObjectName("Content")
        content_layout = ly.window_layout(content)

        self.rail_toggle = ly.button("", variant="quiet", on_click=self.toggle_sidebar,
                                     tip="Show or hide the navigation rail (Ctrl+B)")
        self.rail_toggle.setIcon(icon("panel-left"))
        self.rail_toggle.setIconSize(QSize(T.ICON_SIZE, T.ICON_SIZE))
        self.rail_toggle.setCheckable(True)
        self.rail_toggle.setChecked(True)
        self.page_title = ly.title("")
        header = ly.hbox(spacing=T.SPACING_ROW)
        header.addWidget(self.rail_toggle)
        header.addWidget(self.page_title)
        header.addStretch(1)
        content_layout.addLayout(header)

        self.stack = QStackedWidget()
        self.project_page = ProjectPage(self)
        self.sample_page = SamplePage(self)
        self.phast_page = PhastPage(self)
        self.train_page = TrainPage(self)
        self.predict_page = PredictPage(self)
        for page in self._pages():
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)

        body = ly.splitter(self.sidebar, content,
                           sizes=[Sidebar.DEFAULT_W, self.width() - Sidebar.DEFAULT_W])
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        self.splitter = body
        self.setCentralWidget(body)

        # -- log -----------------------------------------------------------
        self.log_view = LogView()
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("LogDock")  # QSettings needs it to restore
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [self.LOG_DOCK_H], Qt.Vertical)

        # -- status bar ----------------------------------------------------
        self._status = QLabel("")
        self.statusBar().addWidget(self._status)
        # The progress indicator exists only while something is running.
        self._busy = QProgressBar()
        self._busy.setRange(0, 0)          # indeterminate
        self._busy.setTextVisible(False)
        self._busy.setMaximumWidth(160)
        self._busy.hide()
        self.statusBar().addPermanentWidget(self._busy)
        self.statusBar().setSizeGripEnabled(False)

        self._build_menus()
        self._restore_settings()
        self._load_project(self.project.root)
        self.show_page(self.sidebar.current())
        self.report("Ready")

    # -- navigation --------------------------------------------------------
    def show_page(self, index: int) -> None:
        """Move to a page, and let the shared header name it."""
        self.stack.setCurrentIndex(index)
        self.sidebar.set_current(index)
        self.page_title.setText(Sidebar.page_title(index))

    def toggle_sidebar(self) -> None:
        """Hide or show the rail, restoring the width it had.

        The toggle stays visible either way — a rail with no way back is a
        destination list the user has lost.
        """
        visible = not self.sidebar.isVisible()
        if not visible:
            self._sidebar_width = max(self.splitter.sizes()[0], Sidebar.MIN_W)
        self.sidebar.setVisible(visible)
        self.rail_toggle.setChecked(visible)
        if visible:
            self.splitter.setSizes(
                [self._sidebar_width, max(self.width() - self._sidebar_width, 1)]
            )
        if hasattr(self, "sidebar_action"):
            self.sidebar_action.setChecked(visible)

    # -- menus -------------------------------------------------------------
    def _build_menus(self) -> None:
        def action(text: str, handler: Callable[[], None], shortcut=None, tip: str = "") -> QAction:
            item = QAction(text, self)
            item.triggered.connect(handler)
            if shortcut is not None:
                item.setShortcut(shortcut)
            if tip:
                item.setStatusTip(tip)  # every non-obvious item says what it does
            return item

        files = self.menuBar().addMenu("&File")
        files.addAction(action("New project…", self._new_project, QKeySequence.New,
                               "Create a project directory and start a fresh configuration."))
        files.addAction(action("Open project…", self._open_project, QKeySequence.Open,
                               "Open an existing project directory."))
        files.addSeparator()
        files.addAction(action("Save configuration", self.save_configuration, QKeySequence.Save,
                               "Write the settings on the Project and Train pages to nncm.json."))
        files.addSeparator()
        files.addAction(action("Quit", self.close, QKeySequence.Quit, "Close the window."))

        run = self.menuBar().addMenu("&Run")
        run.addAction(action("Generate cases", self.sample_page._generate, None,
                             "Draw the sampling design and write cases.csv."))
        run.addAction(action("Write Phast input workbook", self.phast_page._export, None,
                             "Write the sampled cases into a copy of the Safeti template."))
        run.addAction(action("Extract training data", self.phast_page._import, None,
                             "Read a Phast result workbook into the training dataset."))
        run.addAction(action("Train model", self.train_page._train, None,
                             "Train on the project dataset and save a new run."))
        run.addAction(action("Predict", self.predict_page._predict, None,
                             "Predict the consequences of the inputs on the Predict page."))
        run.addAction(action("Predict from CSV…", self.predict_page._predict_csv, None,
                             "Predict every row of a CSV and write the results beside it."))

        view = self.menuBar().addMenu("&View")
        self.sidebar_action = QAction("Show navigation rail", self, checkable=True)
        self.sidebar_action.setChecked(True)
        self.sidebar_action.setShortcut(QKeySequence("Ctrl+B"))
        self.sidebar_action.setStatusTip("Hide the rail to give the page its width.")
        self.sidebar_action.triggered.connect(self.toggle_sidebar)
        view.addAction(self.sidebar_action)
        view.addSeparator()

        self.detail_action = QAction("Show detail columns", self, checkable=True)
        self.detail_action.setStatusTip(
            "Show every column, not only the ones a decision is made on."
        )
        self.detail_action.toggled.connect(self.set_detail_visible)
        view.addAction(self.detail_action)

        self.log_action = QAction("Show log", self, checkable=True)
        self.log_action.setChecked(True)
        self.log_action.setShortcut(QKeySequence("Ctrl+L"))
        self.log_action.setStatusTip("Show what each stage reported while it ran.")
        self.log_action.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(self.log_action.setChecked)
        view.addAction(self.log_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(action("About NNCM", self._about, None, "What this application does."))

    # -- shared services ---------------------------------------------------
    def log(self, message: str) -> None:
        self.log_view.append_line(message)

    def report(self, message: str) -> None:
        """The status line: what just happened, quiet enough to ignore."""
        self._status.setText(message)

    def set_busy(self, busy: bool, message: str = "") -> None:
        """Show or clear the running indicator, and say what is running.

        Released on failure as well as on success — a bar left spinning after
        an error is a window that looks hung.
        """
        self._busy.setVisible(busy)
        if message:
            self.report(message)

    def begin_task(self, name: str) -> None:
        self._task_name = name
        self.set_busy(True, f"{name}…")

    def finish_task(self, message: str) -> None:
        self.set_busy(False, f"{message} in {self.runner.elapsed_ms:,} ms")
        self._task_name = ""

    def show_failure(self, title: str, message: str) -> None:
        """A genuine failure: the log keeps it, the status line summarises it.

        Validation and calculation problems report inline on the page that
        raised them; this is for the unrecoverable ones.
        """
        first = message.splitlines()[0] if message else ""
        self.log(f"! {title}: {first}")
        self.set_busy(False, f"{title}: {first}")
        QMessageBox.critical(self, title, message)

    def run_task(
        self,
        task: Callable[..., Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        on_progress: Callable[[int, int], None] | None = None,
        pass_log: bool = True,
        quiet: bool = False,
        cancellable: bool = False,
    ) -> bool:
        """Run one stage in the background; one stage at a time."""
        started = self.runner.start(
            task, self.log, on_done, on_error, on_progress=on_progress,
            pass_log=pass_log, cancellable=cancellable,
        )
        if not started and not quiet:
            self.report("Another stage is still running.")
        return started

    # -- project -----------------------------------------------------------
    def open_project(self, root: Path) -> None:
        self._load_project(root)

    def reload_model(self) -> None:
        self.predict_page.reload_model()

    def save_configuration(self) -> None:
        self.project_page.collect()
        self.train_page.collect()
        try:
            self.project.config.validate()
        except ValueError as exc:
            # Bad input is reported where the mistake is, naming the value.
            self.project_page.report_problem(str(exc))
            self.show_page(0)
            self.report("The configuration is not valid.")
            return
        self.project_page.report_problem("")
        path = self.project.save_config()
        self.log(f"configuration saved to {path}")
        self.refresh_project_state()
        self.report(f"Saved {path.name}")

    def refresh_project_state(self) -> None:
        """Reflect what is on disk.

        The project names the window, which is where a desktop application is
        expected to say which document it is showing; each stage carries the
        figures that belong to it.
        """
        project = self.project
        name = project.root.name or str(project.root)
        self.setWindowTitle(f"{name} — NNCM")
        self.sample_page.refresh_counts()
        self.phast_page.refresh_state()

    # -- view toggles ------------------------------------------------------
    def set_detail_visible(self, visible: bool) -> None:
        self.detail_visible = visible
        for page in self._pages():
            page.set_detail_visible(visible)

    def _pages(self):
        return (
            self.project_page,
            self.sample_page,
            self.phast_page,
            self.train_page,
            self.predict_page,
        )

    # -- settings ----------------------------------------------------------
    def _restore_settings(self) -> None:
        """Bring back the rail, its width, and the page last worked on."""
        settings = self.settings
        width = int(settings.value("sidebar/width", Sidebar.DEFAULT_W))
        self._sidebar_width = max(width, Sidebar.MIN_W)
        self.splitter.setSizes(
            [self._sidebar_width, max(self.width() - self._sidebar_width, 1)]
        )
        if settings.value("sidebar/visible", "true") == "false":
            self.toggle_sidebar()
        if settings.value("log/visible", "true") == "false":
            self.log_dock.setVisible(False)
        page = int(settings.value("window/page", 0))
        self.sidebar.set_current(max(0, min(page, self.stack.count() - 1)))

    def _save_settings(self) -> None:
        settings = self.settings
        if self.sidebar.isVisible():
            self._sidebar_width = max(self.splitter.sizes()[0], Sidebar.MIN_W)
        settings.setValue("sidebar/width", self._sidebar_width)
        settings.setValue("sidebar/visible", "true" if self.sidebar.isVisible() else "false")
        settings.setValue("log/visible", "true" if self.log_dock.isVisible() else "false")
        settings.setValue("window/page", self.stack.currentIndex())

    # -- internals ---------------------------------------------------------
    def _load_project(self, root: Path) -> None:
        self.project = Project.create(Path(root))
        self.log(f"project: {self.project.root}")
        for page in self._pages():
            page.on_project_changed()
        self.set_detail_visible(self.detail_visible)
        self.refresh_project_state()

    def _new_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a directory for the new project", str(self.project.root.parent)
        )
        if chosen:
            self._load_project(Path(chosen))
            self.report(f"Opened {Path(chosen).name}")

    def _open_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Open a project directory", str(self.project.root)
        )
        if chosen:
            self._load_project(Path(chosen))
            self.report(f"Opened {Path(chosen).name}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About NNCM",
            f"<p style='font-size:{T.FONT_HEADING}px; font-weight:600;'>NNCM</p>"
            "<p>Neural network consequence modelling. Samples release scenarios, "
            "drives them through Phast or Safeti, and trains a surrogate model "
            "that predicts consequence results in milliseconds rather than "
            "minutes.</p>"
            f"<p style='color:{T.ink_hex(T.INK_SECONDARY)};'>The five destinations "
            "in the rail are the workflow, top to bottom. Everything the window "
            "can do is in the menu bar.</p>",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.runner.busy:
            answer = QMessageBox.question(
                self,
                "A stage is still running",
                f"{self._task_name or 'A stage'} has not finished. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            # Stop what can be stopped, then let the thread return: a QThread
            # destroyed while still running aborts the whole process.
            self.runner.request_stop()
            self.report("Stopping…")
            self.runner.wait()
        self._save_settings()
        event.accept()
