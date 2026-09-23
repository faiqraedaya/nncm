"""Tests for the desktop application's threading and failure handling.

How anything looks is out of scope. What is checked is what would hang or
crash the window: work off the UI thread, results back on it, a stoppable
task, and a failure that releases the busy state.
"""

from __future__ import annotations

import os
import time

import pytest

# A desktop test needs a display; offscreen gives Qt one that needs no screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from nncm.gui import theme as gui_theme  # noqa: E402
from nncm.gui.workers import TaskRunner  # noqa: E402


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    gui_theme.apply_theme(application)
    return application


def _wait_for(app, condition, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def test_window_builds_and_every_stage_paints(app, tmp_path):
    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    window.show()
    assert window.stack.count() == 5
    for index in range(window.stack.count()):
        window.show_page(index)
        app.processEvents()
        assert window.page_title.text()

    window.save_configuration()
    assert (tmp_path / "project" / "nncm.json").exists()
    window.close()


def test_a_failure_reports_inline_and_releases_the_busy_state(app, tmp_path):
    """A calculation that failed must not leave the window looking hung."""
    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    page = window.sample_page
    window.begin_task("Generating cases")
    page.generate_button.setEnabled(False)
    assert window._busy.isVisibleTo(window)

    page._failed("ValueError: the design has no materials")

    assert page.status.property("role") == "error"
    assert not window._busy.isVisibleTo(window)
    assert page.generate_button.isEnabled()
    window.close()


def test_task_callbacks_come_back_on_the_ui_thread(app):
    """Handlers connected to a worker signal run on the worker thread unless
    the receiver lives in the main thread; every Qt call in them then fails."""
    holder = QWidget()
    runner = TaskRunner(holder)
    ui_thread = app.thread()
    seen: dict[str, object] = {}

    def task(log):
        log("halfway")
        seen["ran_on"] = QThread.currentThread()
        return 42

    def done(result):
        seen["result"] = result
        seen["finished_on"] = QThread.currentThread()

    def logged(message):
        seen.setdefault("logged_on", QThread.currentThread())

    assert runner.start(task, on_log=logged, on_done=done, on_error=seen.setdefault)
    _wait_for(app, lambda: "result" in seen)

    assert seen.get("result") == 42
    assert seen["ran_on"] is not ui_thread
    assert seen["finished_on"] is ui_thread
    assert seen["logged_on"] is ui_thread
    assert not runner.busy


def test_a_cancellable_task_stops_when_asked(app):
    """Closing the window mid-run stops the task and waits for its thread; a
    QThread destroyed while running aborts the process."""
    holder = QWidget()
    runner = TaskRunner(holder)
    errors: list[str] = []

    def task(log, should_stop):
        while not should_stop():
            time.sleep(0.005)
        raise RuntimeError("stopped")

    assert runner.start(task, on_log=lambda _: None, on_done=lambda _: None,
                        on_error=errors.append, cancellable=True)
    assert runner.request_stop()
    runner.wait()
    _wait_for(app, lambda: bool(errors))
    assert errors and "stopped" in errors[0]
    assert not runner.busy
    # A task started without the flag cannot be asked to stop.
    assert runner.start(lambda log: None, on_log=lambda _: None,
                        on_done=lambda _: None, on_error=errors.append)
    assert not runner.request_stop()
    runner.wait()


def test_plots_never_touch_pyplot():
    """Charts are drawn on worker threads; a pyplot figure would create Qt
    objects there and leak through pyplot's global registry."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from nncm.config import SamplingConfig
    from nncm.sampling import generate_cases, plot_case_distributions

    plt.close("all")
    cases, _ = generate_cases(SamplingConfig(n_vessels=8, n_leaks_per_vessel=2))
    figure = plot_case_distributions(cases)
    assert plt.get_fignums() == []
    assert figure.axes
