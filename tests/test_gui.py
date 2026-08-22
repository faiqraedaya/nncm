"""Tests for the interface rules that are easy to break and hard to spot.

Deliberately excluded: how anything looks. What is checked here is the
behaviour the design guide asks for — sorting by model values, blanks last,
totals pinned inside the table, a fixed categorical palette that refuses to
cycle, and one registry entry behind every label the interface shows.
"""

from __future__ import annotations

import os
import pathlib

import pandas as pd
import pytest

from nncm import theme
from nncm.config import Material, SamplingConfig, TrainingConfig
from nncm.quantities import QUANTITIES, describe

# A desktop test needs a display; offscreen gives Qt one that needs no screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from nncm.gui import style, widgets  # noqa: E402


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    style.apply(application)
    return application


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def test_series_palette_is_finite_and_refuses_to_cycle():
    assert theme.series_colors(3) == list(theme.SERIES)
    with pytest.raises(ValueError):
        theme.series_colors(len(theme.SERIES) + 1)


def test_stylesheet_substitutes_every_token(app):
    sheet = app.styleSheet()
    assert "$" not in sheet, "a token was left unsubstituted in the style sheet"
    assert theme.ACCENT in sheet and theme.SURFACE in sheet


# ---------------------------------------------------------------------------
# Labels, units and explanations
# ---------------------------------------------------------------------------
def test_every_configuration_field_shown_has_a_registry_entry():
    """The interface must never invent a label for a setting it edits."""
    shown = {
        *(f.name for f in SamplingConfig.__dataclass_fields__.values()),
        *(f.name for f in TrainingConfig.__dataclass_fields__.values()),
    } - {
        # Not edited in the window: kept in nncm.json only.
        "materials", "group_column", "head_units", "weight_decay",
        "patience_early_stop", "patience_reduce_lr", "test_size", "valid_size",
        "log_targets",
    }
    missing = sorted(name for name in shown if name not in QUANTITIES)
    assert not missing, f"no label, unit or explanation for: {missing}"


def test_every_phast_target_has_a_name_and_a_unit():
    from nncm.phast.output_reader import DEFAULT_TARGETS

    for target in DEFAULT_TARGETS:
        quantity = describe(target.name)
        assert quantity.label != target.name, f"{target.name} has no readable name"
        assert quantity.unit or "fraction" in quantity.help, f"{target.name} has no unit"


def test_an_unknown_quantity_degrades_to_a_readable_label():
    quantity = describe("Some_new_target")
    assert quantity.label == "Some new target"
    assert quantity.unit == ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _table(app, totals=None):
    table = widgets.DataTable()
    frame = pd.DataFrame(
        {"material": ["METHANE", "PROPANE", "AMMONIA"], "Release_rate": [1000.0, 9.0, None]}
    )
    table.show_frame(frame, totals=totals)
    return table


def _column(table, column: int) -> list[str]:
    proxy = table._proxy
    return [proxy.data(proxy.index(row, column)) for row in range(proxy.rowCount())]


def test_sorting_uses_model_values_not_rendered_text(app):
    table = _table(app)
    table._cycle_sort(1)
    assert _column(table, 1)[:2] == ["9", "1,000"], "1,000 must not sort before 9"


def test_blanks_sort_last_in_both_directions(app):
    table = _table(app)
    table._cycle_sort(1)
    assert _column(table, 1)[-1] == ""
    table._cycle_sort(1)
    assert _column(table, 1)[-1] == "", "no figure is not a small figure"


def test_third_sort_returns_the_model_order(app):
    table = _table(app)
    original = _column(table, 0)
    for _ in range(3):
        table._cycle_sort(1)
    assert _column(table, 0) == original


def test_totals_row_stays_last_however_the_table_is_sorted(app):
    table = _table(app, totals={"material": "All materials", "Release_rate": 1009.0})
    for _ in range(2):
        table._cycle_sort(1)
        assert _column(table, 0)[-1] == "All materials"


def test_totals_row_is_readable_but_not_selectable(app):
    table = _table(app, totals={"material": "All materials"})
    model = table._model
    last = model.rowCount() - 1
    flags = model.flags(model.index(last, 0))
    assert flags & Qt.ItemIsEnabled
    assert not flags & Qt.ItemIsSelectable


def test_a_column_with_no_meaningful_total_is_blank_not_zero(app):
    table = _table(app, totals={"material": "All materials"})
    model = table._model
    last = model.rowCount() - 1
    assert model.data(model.index(last, 1), Qt.DisplayRole) == ""


def test_headers_carry_the_unit_and_the_explanation(app):
    table = _table(app)
    model = table._model
    assert model.headerData(1, Qt.Horizontal, Qt.DisplayRole) == "Release rate (kg/s)"
    assert model.headerData(1, Qt.Horizontal, Qt.ToolTipRole)


def test_empty_table_says_what_is_absent(app):
    table = widgets.DataTable()
    table.show_frame(pd.DataFrame({"material": []}))
    assert table._empty.isVisibleTo(table)
    assert table._empty.text().strip().endswith(".")


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def test_advisories_hide_themselves_with_nothing_to_report(app):
    notes = widgets.Advisories()
    notes.show_notes([("warning", "Outside the training envelope.")])
    assert notes.isVisibleTo(notes.parentWidget()) or not notes.isHidden()
    notes.show_notes([])
    assert notes.isHidden()


def test_a_form_row_carries_its_label_unit_and_sentence(app):
    from PySide6.QtWidgets import QLabel

    form = widgets.Form()
    form.add("orifice_mm", widgets.decimal_field(0.0, 10.0, 1.0))
    texts = [label.text() for label in form.findChildren(QLabel)]
    assert "Orifice diameter" in texts
    assert "mm" in texts
    assert any(label.objectName() == "Explanation" for label in form.findChildren(QLabel))


def test_numeric_fields_are_right_aligned_and_keep_their_own_wheel(app):
    field = widgets.decimal_field(0.0, 10.0, 1.0)
    assert field.alignment() & Qt.AlignRight
    assert field.buttonSymbols() == field.ButtonSymbols.NoButtons


def test_paths_elide_from_the_middle():
    elided = widgets.elide_middle("C:/a/very/long/project/path/that/keeps/going/on.csv", 24)
    assert elided.startswith("C:/a/very") and elided.endswith("on.csv") and len(elided) == 24


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
def test_window_builds_and_every_stage_paints(app, tmp_path):
    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    window.show()
    assert window.tabs.count() == 5
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        app.processEvents()

    window.set_descriptions_visible(False)
    window.set_detail_visible(True)
    window.set_detail_visible(False)
    window.set_descriptions_visible(True)

    window.save_configuration()
    assert (tmp_path / "project" / "nncm.json").exists()
    window.close()


def test_each_stage_offers_exactly_one_primary_action(app, tmp_path):
    """One Primary per surface — the rule most easily broken by accident."""
    from PySide6.QtWidgets import QPushButton

    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    for page in window._pages():
        primaries = [
            button
            for button in page.actions.findChildren(QPushButton)
            if button.property("primary") == "true"
        ]
        assert len(primaries) == 1, f"{type(page).__name__} has {len(primaries)} primaries"
    window.close()


def test_reading_surfaces_carry_no_buttons(app, tmp_path):
    """Tables and plots get the whole width; the actions live elsewhere."""
    from PySide6.QtWidgets import QPushButton

    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    for page in window._pages():
        for table in page.findChildren(widgets.DataTable):
            assert not table.findChildren(QPushButton)
        for plot in page.findChildren(widgets.PlotArea):
            assert not plot.findChildren(QPushButton)
    window.close()


def test_materials_table_totals_the_weights(app, tmp_path):
    from nncm.gui.app import MainWindow

    window = MainWindow(tmp_path / "project")
    page = window.project_page
    page._fill_materials([Material("METHANE", weight=1.0), Material("PROPANE", weight=3.0)])
    table = page.materials_table
    last = table.rowCount() - 1
    assert table.item(last, 0).text() == "2 materials"
    assert table.item(last, 3).text() == "4"
    assert not table.item(last, 4).text(), "a column with no meaningful total stays blank"
    assert page._material_rows() == [
        Material("METHANE", weight=1.0),
        Material("PROPANE", weight=3.0),
    ]
    window.close()


def test_bundled_faces_register_and_are_the_ones_used(app):
    """Bundled type is only worth bundling if the window actually gets it."""
    from PySide6.QtGui import QFontDatabase, QFontInfo

    from nncm.gui.style import FONTS_DIR

    if not any(FONTS_DIR.glob("*.ttf")) and not any(FONTS_DIR.glob("*.otf")):
        pytest.skip("no faces bundled; the stack falls back to the system face")

    first = theme.FONT_STACK[0]
    assert first in QFontDatabase.families(), f"{first} was not registered"
    for size, weight, italic in (
        (theme.FONT_BODY, theme.WEIGHT_REGULAR, False),
        (theme.FONT_CONTROL, theme.WEIGHT_MEDIUM, False),
        (theme.FONT_TITLE, theme.WEIGHT_SEMIBOLD, False),
        (theme.FONT_CAPTION, theme.WEIGHT_REGULAR, True),  # the real italic
    ):
        info = QFontInfo(style.font(size, weight, italic=italic))
        assert info.family() == first, f"{size}px/{weight} fell back to {info.family()}"
        assert info.italic() == italic


def test_charts_are_set_in_the_same_face_as_the_window():
    """One typeface, one voice — chrome and figures alike.

    Qt and matplotlib keep separate font lists, so a face registered for the
    window is not automatically available to a chart. Without this the figures
    would quietly fall back to a second typeface.
    """
    import matplotlib
    from matplotlib import font_manager

    from nncm.gui.style import FONTS_DIR

    theme.apply_matplotlib_style()
    if not any(FONTS_DIR.glob("*.ttf")) and not any(FONTS_DIR.glob("*.otf")):
        pytest.skip("no faces bundled; both toolkits fall back to the system face")

    resolved = font_manager.findfont(
        font_manager.FontProperties(family=matplotlib.rcParams["font.sans-serif"])
    )
    assert FONTS_DIR.samefile(pathlib.Path(resolved).parent), (
        f"charts would be drawn in {resolved}, not the bundled face"
    )


# ---------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------
def test_task_callbacks_come_back_on_the_ui_thread(app):
    """The bug this guards: handlers connected to a worker signal run on the
    worker thread unless the receiver is a QObject living in the main thread.
    Every Qt call in a page's result handler then fails, and the runner ends up
    waiting on the very thread it is closing."""
    import time

    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QWidget

    from nncm.gui.workers import TaskRunner

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
        seen["message"] = message

    assert runner.start(task, on_log=logged, on_done=done, on_error=seen.setdefault)

    deadline = time.time() + 20
    while "result" not in seen and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert seen.get("result") == 42, "the task never reported back"
    assert seen["ran_on"] is not ui_thread, "the task ran on the UI thread; the window would freeze"
    assert seen["finished_on"] is ui_thread, "the result handler ran off the UI thread"
    assert seen["logged_on"] is ui_thread, "log lines arrived off the UI thread"
    assert not runner.busy


def test_plots_never_touch_pyplot():
    """A pyplot figure is built on whichever backend the window loaded, so a
    chart drawn on a worker thread would create Qt objects there. It also keeps
    every figure in a global registry, which is a leak across runs."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from nncm.config import SamplingConfig
    from nncm.sampling import generate_cases, plot_case_distributions

    plt.close("all")
    cases, _ = generate_cases(SamplingConfig(n_vessels=8, n_leaks_per_vessel=2))
    figure = plot_case_distributions(cases)
    assert plt.get_fignums() == [], "the chart registered a figure with pyplot"
    assert figure.axes, "the figure came back empty"


def test_a_chart_given_one_caveat_writes_a_sentence(app):
    """The bug this guards: a lone caveat passed as a string was joined
    character by character, printing the subtitle vertically down the figure
    and collapsing the space left for the panels."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.0, 5.0))
    top = theme.figure_header(figure, "Case design coverage", "260 scenarios across 130 vessels")
    subtitle = [t.get_text() for t in figure.texts if t.get_text() != "Case design coverage"]
    assert subtitle == ["260 scenarios across 130 vessels"]
    assert top > 0.7, f"the header ate the figure: panels start at {top:.2f}"


def test_the_material_panel_folds_its_tail_instead_of_dropping_it(app):
    """More materials than the panel has rows is a fold, not a cut: the bars
    still account for every vessel in the design."""
    from nncm.config import Material, SamplingConfig, default_materials
    from nncm.sampling import MAX_MATERIAL_BARS, generate_cases, plot_case_distributions

    materials = default_materials()
    assert len(materials) > MAX_MATERIAL_BARS, "the catalogue no longer exercises the fold"
    cases, _ = generate_cases(SamplingConfig(n_vessels=len(materials) * 3, n_leaks_per_vessel=1))
    figure = plot_case_distributions(cases)

    bars = figure.axes[-1]
    labels = [t.get_text() for t in bars.get_yticklabels()]
    assert len(labels) <= MAX_MATERIAL_BARS
    assert sum(label.startswith("Other (") for label in labels) == 1
    counted = sum(int(t.get_text().replace(",", "")) for t in bars.texts)
    assert counted == cases["vessel_name"].nunique(), "the bars lost vessels"


def test_a_sentence_is_measured_at_the_width_it_is_given(app):
    """The bug this guards: QLabel floors heightForWidth with the minimum
    height already set on it, so a label measured once while it was narrow kept
    those lines for good — every form row inherited the extra space."""
    note = widgets.Explanation(
        "Mass given to every generated vessel. Large enough that steady-state "
        "releases are not limited by the inventory."
    )
    narrow = note.heightForWidth(180)
    note.setMinimumHeight(narrow)          # what a first, cramped layout does
    assert note.heightForWidth(900) < narrow, "the measured height only ever grew"
