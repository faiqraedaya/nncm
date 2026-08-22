"""Background execution for the desktop application.

Sampling, workbook I/O, model loading and training all take seconds to
minutes. Running them on the Qt thread freezes the window (and Windows paints
it "not responding"), so every stage runs in a worker thread that streams its
log lines — and, where the work has countable steps, its progress — back to the
interface.

**Every callback comes back on the UI thread.** The runner is a QObject living
in the main thread and the worker's signals reach it by queued connection, so
the page code that handles a result may touch widgets freely. Handing those
signals straight to a plain function or a lambda would run them on the worker
thread instead, where a Qt call fails with "cannot set parent, new parent is in
a different thread" and the thread ends up waiting on itself.

Progress is only ever reported for steps that have actually finished. Nothing
here invents a rate or animates over unknown work.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot


class Worker(QObject):
    """Runs one task off the UI thread, reporting what it did as it goes."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[..., Any], pass_log: bool = True, pass_progress: bool = False):
        super().__init__()
        self._task = task
        self._pass_log = pass_log
        self._pass_progress = pass_progress

    @Slot()
    def run(self) -> None:
        kwargs: dict[str, Any] = {}
        if self._pass_log:
            kwargs["log"] = self.log.emit
        if self._pass_progress:
            kwargs["progress"] = self.progress.emit
        try:
            result = self._task(**kwargs)
        except Exception as exc:  # surfaced in the interface, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
            return
        self.finished.emit(result)


class TaskRunner(QObject):
    """Owns a worker and its thread, and delivers every result on the UI thread."""

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._started: float = 0.0
        self._on_log: Callable[[str], None] | None = None
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._on_progress: Callable[[int, int], None] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def elapsed_ms(self) -> int:
        """How long the last task took — reported in the status bar, so a
        regression is visible to everyone rather than only to whoever profiles."""
        return int((time.perf_counter() - self._started) * 1000)

    def start(
        self,
        task: Callable[..., Any],
        on_log: Callable[[str], None],
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        on_progress: Callable[[int, int], None] | None = None,
        pass_log: bool = True,
    ) -> bool:
        if self.busy:
            return False
        self._on_log, self._on_done = on_log, on_done
        self._on_error, self._on_progress = on_error, on_progress

        thread = QThread(self)
        worker = Worker(task, pass_log=pass_log, pass_progress=on_progress is not None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # Queued, and to slots of *this* object: this is what puts the handlers
        # back on the UI thread. Connecting to a bare function or a lambda here
        # would run them wherever the signal was emitted.
        worker.log.connect(self._handle_log, Qt.QueuedConnection)
        worker.progress.connect(self._handle_progress, Qt.QueuedConnection)
        worker.finished.connect(self._handle_finished, Qt.QueuedConnection)
        worker.failed.connect(self._handle_failed, Qt.QueuedConnection)

        self._thread, self._worker = thread, worker
        self._started = time.perf_counter()
        thread.start()
        return True

    # -- handlers, all on the UI thread -----------------------------------
    @Slot(str)
    def _handle_log(self, message: str) -> None:
        if self._on_log is not None:
            self._on_log(message)

    @Slot(int, int)
    def _handle_progress(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)

    @Slot(object)
    def _handle_finished(self, result: Any) -> None:
        callback = self._on_done
        self._retire()
        if callback is not None:
            callback(result)

    @Slot(str)
    def _handle_failed(self, message: str) -> None:
        callback = self._on_error
        self._retire()
        if callback is not None:
            callback(message)

    def _retire(self) -> None:
        """Shut the thread down. Safe here, and only here: the task has already
        returned, and this runs on the UI thread rather than on the thread it
        is closing — a thread cannot wait on itself."""
        thread, self._thread = self._thread, None
        worker, self._worker = self._worker, None
        self._on_log = self._on_done = self._on_error = self._on_progress = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
