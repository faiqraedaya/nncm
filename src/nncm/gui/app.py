"""The window: five stages, one log, one status line.

The project names the window; every figure belongs to the stage that produced
it, so there is no summary bar competing with the stage on screen.

The menu bar offers every action the window has, so the whole application is
reachable from the keyboard. Each stage repeats its own action as the single
Primary in its action bar; nothing else carries a button.

Two things the window owns on behalf of every page: whether explanations are
shown (they roughly double the height of a form, so they are a toggle), and
whether detail columns are shown (a table should hold the columns a decision is
made on, and nothing else).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..config import Project, default_project_root
from . import style
from .pages import PhastPage, PredictPage, ProjectPage, SamplePage, TrainPage
from .widgets import WINDOW_MARGINS, LogView
from .workers import TaskRunner


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("NNCM — Neural network consequence modelling")
        self.resize(1280, 860)
        self.project = Project.create(Path(project_root or default_project_root()))
        self.runner = TaskRunner(self)
        self.descriptions_visible = True
        self.detail_visible = False
        self._task_name = ""

        body = QWidget()
        body.setObjectName("Body")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(*WINDOW_MARGINS)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.project_page = ProjectPage(self)
        self.sample_page = SamplePage(self)
        self.phast_page = PhastPage(self)
        self.train_page = TrainPage(self)
        self.predict_page = PredictPage(self)
        for label, page in (
            ("1 · Project", self.project_page),
            ("2 · Sample", self.sample_page),
            ("3 · Phast", self.phast_page),
            ("4 · Train", self.train_page),
            ("5 · Predict", self.predict_page),
        ):
            self.tabs.addTab(page, label)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(body)

        self.log_view = LogView()
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [190], Qt.Vertical)

        self._status = QLabel("")
        self.statusBar().addWidget(self._status)
        self.statusBar().setSizeGripEnabled(False)

        self._build_menus()
        self._load_project(self.project.root)
        self.report("Ready")

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
        self.descriptions_action = QAction("Show descriptions", self, checkable=True)
        self.descriptions_action.setChecked(True)
        self.descriptions_action.setStatusTip(
            "Show the sentence under each setting. They roughly double the height of a form."
        )
        self.descriptions_action.toggled.connect(self.set_descriptions_visible)
        view.addAction(self.descriptions_action)

        self.detail_action = QAction("Show detail columns", self, checkable=True)
        self.detail_action.setStatusTip(
            "Show every column, not only the ones a decision is made on."
        )
        self.detail_action.toggled.connect(self.set_detail_visible)
        view.addAction(self.detail_action)

        self.log_action = QAction("Show log", self, checkable=True)
        self.log_action.setChecked(True)
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

    def begin_task(self, name: str) -> None:
        self._task_name = name
        self.report(f"{name}…")

    def finish_task(self, message: str) -> None:
        self.report(f"{message} in {self.runner.elapsed_ms:,} ms")
        self._task_name = ""

    def show_failure(self, title: str, message: str) -> None:
        first = message.splitlines()[0] if message else ""
        self.log(f"! {title}: {first}")
        self.report(f"{title}: {first}")
        QMessageBox.critical(self, title, message)

    def run_task(
        self,
        task: Callable[..., Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        on_progress: Callable[[int, int], None] | None = None,
        pass_log: bool = True,
        quiet: bool = False,
    ) -> bool:
        """Run one stage in the background; one stage at a time."""
        started = self.runner.start(
            task, self.log, on_done, on_error, on_progress=on_progress, pass_log=pass_log
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
            self.show_failure("The configuration is not valid", str(exc))
            return
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
    def set_descriptions_visible(self, visible: bool) -> None:
        self.descriptions_visible = visible
        for label in self.tabs.findChildren(QLabel, "Explanation"):
            label.setVisible(visible)

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

    # -- internals ---------------------------------------------------------
    def _load_project(self, root: Path) -> None:
        self.project = Project.create(Path(root))
        self.log(f"project: {self.project.root}")
        for page in self._pages():
            page.on_project_changed()
        self.set_descriptions_visible(self.descriptions_visible)
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
            "<p style='font-size:15px; font-weight:600;'>NNCM</p>"
            "<p>Neural network consequence modelling. Samples release scenarios, "
            "drives them through Phast or Safeti, and trains a surrogate model "
            "that predicts consequence results in milliseconds rather than "
            "minutes.</p>"
            f"<p style='color:{theme.SUBTLE};'>The five tabs are the workflow, "
            "left to right. Everything the window can do is in the menu bar.</p>",
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
        event.accept()


def run(project_root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    style.apply(app)
    theme.apply_matplotlib_style()
    window = MainWindow(project_root)
    window.show()
    return app.exec()
