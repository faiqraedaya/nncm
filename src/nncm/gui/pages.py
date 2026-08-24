"""The five stages of the workflow, one page each.

**Actions live in three places, and nowhere else**: the menu bar, which offers
every action the window has; one action bar at the foot of each stage, holding
that stage's single Primary; and one toolbar row above each editable table.
The reading surfaces — tables, plots, metric cells — carry no buttons at all,
so the figures get the whole width.

Each page reads its labels, units and explanations from
:mod:`nncm.quantities`, and its state from the project on disk, so what is on
screen is what would happen if the stage were run right now.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from .. import theme as T
from ..config import Material, Range
from ..pipeline import dataset_summary, run_phast_export, run_phast_import, run_sampling
from ..predict import ModelBundle, format_quantity
from ..quantities import describe
from ..sampling import load_cases, plot_case_distributions
from .dialogs import MaterialDialog
from .widgets import (
    ActionBar,
    Advisories,
    Card,
    DataTable,
    Explanation,
    Form,
    MetricCell,
    PathField,
    PlotArea,
    RangeField,
    choice_field,
    decimal_field,
    elide_middle,
    format_number,
    integer_field,
    set_primary,
)
from . import layout as ly
from . import theme as gui_theme

CASE_KEY_COLUMNS = ["vessel_name", "material", "temperature_degC", "pressure_barg", "orifice_mm"]


def scroll_pane(content: QWidget) -> QScrollArea:
    """A scrolling pane around one widget, with no edge of its own.

    Horizontal scrolling is off: text wraps or elides, it never asks the
    reader to scroll sideways to finish a sentence.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)   # fill the window when the content fits
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(content)
    return area


class Page(QWidget):
    """A stage: a body, an inline status line, and one action bar at the foot.

    The page does not draw its own title — the shell names the current page
    once, in the shared content header, driven by the navigation. Five
    separately-placed titles drift; one placement cannot.

    The body scrolls when the window is shorter than the stage needs. Without
    it Qt buys the missing height by compressing whatever will compress, and
    what compresses first is the explanation at the foot of a card — so a short
    window would silently cut a sentence in half rather than admit it ran out
    of room. The status line and the action bar stay put: the stage's one
    primary is never something you have to scroll to find, and neither is the
    message saying why it did not work.
    """

    title = ""

    # Two cards stacked, or one card holding a table with its metric row: the
    # shortest arrangement any page here is legible at.
    BODY_MIN_H = 340

    #: Whether the whole body scrolls as one flow.
    #:
    #: True for a page that is a single column of cards taller than the window.
    #: False for a page whose body is a splitter — a splitter can only hand out
    #: height it has been given, and inside a scroll area it is given its own
    #: minimum instead, so every pane collapses to its floor and the page
    #: scrolls past panes that should have shared the screen. Those pages let
    #: the panes that need it scroll individually.
    scrolls = True

    def __init__(self, window):
        super().__init__()
        self.window = window
        self._outer = ly.vbox(self, spacing=T.SPACING_GROUP)
        self.body = ly.vbox(spacing=T.SPACING_GROUP)
        content = QWidget()
        content.setLayout(self.body)
        if self.scrolls:
            self._scroll = scroll_pane(content)
            # A floor for the body, so the window's minimum is a size the
            # content is actually usable at. Without it Qt shrinks the page to
            # nothing and the shortfall shows as a field sliced in half.
            self._scroll.setMinimumHeight(self.BODY_MIN_H)
            self._outer.addWidget(self._scroll, 1)
        else:
            self._scroll = None
            content.setMinimumHeight(self.BODY_MIN_H)
            self._outer.addWidget(content, 1)
        # Validation and calculation failures report here, below the results
        # and above the action that produced them — not in a modal, which
        # would cover the figures the message is about.
        self.status = ly.status_label()
        self._outer.addWidget(self.status)
        self.actions = ActionBar()
        self._outer.addWidget(self.actions)

    @property
    def project(self):
        return self.window.project

    def log(self, message: str) -> None:
        self.window.log(message)

    def report_problem(self, message: str, role: str = "error") -> None:
        """Say inline what went wrong, or clear it with an empty message."""
        ly.set_status(self.status, message, role if message else "")

    def fail(self, title: str, message: str, *controls) -> None:
        """The one failure path: inline, busy released, controls handed back.

        A calculation that did not work is not an interruption. The results
        already on screen stay readable while the message is read, and the
        message can be as long as it needs to be. The whole text goes to the
        log, which is where the rest of a traceback belongs.
        """
        first = message.splitlines()[0] if message else "no detail given"
        self.log(f"! {title}: {message}")
        self.report_problem(f"{title}. {first}")
        self.actions.clear_progress()
        self.window.set_busy(False, title)
        for control in controls:
            control.setEnabled(True)

    def on_project_changed(self) -> None:
        """Refresh from the project on disk after it changes."""

    def set_detail_visible(self, visible: bool) -> None:
        """React to View > Show detail columns."""


# ---------------------------------------------------------------------------
# 1. Project and design
# ---------------------------------------------------------------------------
MATERIAL_COLUMNS = [
    ("name", "Material"),
    ("kind", "Kind"),
    ("composition", "Composition"),
    ("weight", "Weight"),
    ("temperature", "Temperature"),
    ("pressure", "Pressure"),
    ("molecular_weight", "MW"),
    ("normal_boiling_point_K", "Tb"),
    ("critical_temperature_K", "Tc"),
    ("critical_pressure_bara", "Pc"),
]
MATERIAL_DETAIL = {"molecular_weight", "normal_boiling_point_K", "critical_temperature_K",
                   "critical_pressure_bara"}
MATERIAL_KEY_COLUMN = 0
MATERIAL_EDIT_COLUMN = len(MATERIAL_COLUMNS)


class ProjectPage(Page):
    title = "Project"
    scrolls = False   # the body is a splitter; each pane scrolls for itself

    def __init__(self, window):
        super().__init__(window)
        top = ly.hbox(spacing=T.SPACING_GROUP)

        design = Card("Sampling design")
        self.design_form = Form()
        self.vessels = integer_field(1, 1_000_000, 500)
        self.leaks = integer_field(1, 100, 6)
        self.sampler = choice_field(["lhs", "sobol", "random"], "lhs")
        self.seed = integer_field(0, 10**6, 42)
        self.inventory = decimal_field(1.0, 1e9, 50_000.0, 1)
        self.design_form.add("n_vessels", self.vessels)
        self.design_form.add("n_leaks_per_vessel", self.leaks)
        self.design_form.add("sampler", self.sampler, span=True)
        self.design_form.add("seed", self.seed)
        self.design_form.add("mass_inventory_kg", self.inventory)
        design.add(self.design_form)
        design.body().addStretch(1)
        top.addWidget(design, 1)

        ranges = Card("Input ranges")
        self.ranges_form = Form()
        self.ranges: dict[str, RangeField] = {}
        for key, low, high in (
            ("temperature", -273.0, 2000.0),
            ("pressure", 0.0, 5000.0),
            ("orifice", 0.001, 5000.0),
            ("elevation", 0.0, 500.0),
        ):
            field = RangeField(low, high)
            self.ranges[key] = field
            self.ranges_form.add(key, field, span=True)
        ranges.add(self.ranges_form)
        ranges.body().addStretch(1)
        top.addWidget(ranges, 1)
        settings_content = QWidget()
        settings_content.setLayout(top)
        # The settings scroll inside their own pane. With the explanations
        # shown they are taller than half the window, and the alternative is
        # for them to push the materials table off the foot of the page.
        settings = scroll_pane(settings_content)
        settings.setMinimumHeight(220)

        materials = Card("Materials")
        materials.add(
            Explanation(
                "Every vessel draws one material. A material with no components "
                "is a pure component; per-material ranges override the design "
                "ranges above. Double-click a row, or use Edit, to change one."
            )
        )
        toolbar = ly.hbox(spacing=T.SPACING_ROW)
        add_button = ly.button("Add material", on_click=self._add_material,
                               tip="Add a material to the sampling design.")
        remove_button = ly.button("Remove", variant="danger", on_click=self._remove_material,
                                  tip="Remove the selected material from the design.")
        toolbar.addWidget(add_button)
        toolbar.addStretch(1)
        # Remove sits apart from Add: both change the table, only one of them
        # can be undone by pressing the other.
        toolbar.addWidget(remove_button)
        materials.add_layout(toolbar)

        self.materials_table = QTableWidget(0, len(MATERIAL_COLUMNS) + 1)
        self.materials_table.setHorizontalHeaderLabels(
            [label for _, label in MATERIAL_COLUMNS] + [""]
        )
        for index, (key, _) in enumerate(MATERIAL_COLUMNS):
            item = self.materials_table.horizontalHeaderItem(index)
            quantity = describe(key)
            item.setToolTip(quantity.help or quantity.label)
        self.materials_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.materials_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.materials_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Wash, and only wash: gridlines as well would be two devices doing
        # one job, and the wash costs no height and draws no line.
        self.materials_table.setShowGrid(False)
        self.materials_table.setAlternatingRowColors(True)
        self.materials_table.setTextElideMode(Qt.ElideRight)
        self.materials_table.setFrameShape(QTableWidget.NoFrame)
        self.materials_table.verticalHeader().setVisible(False)
        self.materials_table.verticalHeader().setDefaultSectionSize(T.ROW_HEIGHT)
        header = self.materials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        header.setTextElideMode(Qt.ElideRight)
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # the elastic column
        # The Edit column holds a cell widget, which ResizeToContents does not
        # measure — left to itself the column collapses and the button clips.
        header.setSectionResizeMode(MATERIAL_EDIT_COLUMN, QHeaderView.Fixed)
        self.materials_table.setColumnWidth(MATERIAL_EDIT_COLUMN, 72)
        _align_headers(self.materials_table, MATERIAL_NUMERIC_COLUMNS)
        # Tall enough to hold its own totals row without scrolling to it.
        self.materials_table.setMinimumHeight(T.ROW_HEIGHT * 5 + 44)
        self.materials_table.cellDoubleClicked.connect(lambda row, _column: self._edit_material(row))
        materials.add(self.materials_table, 1)
        # A splitter rather than a plain stack: the settings above are tall
        # enough with their explanations shown to push the table off the foot
        # of the page, and a materials list showing none of its materials is
        # the landing page failing at the one thing it is for. The user can
        # rebalance the two, and the table keeps a share by default.
        self.body.addWidget(
            ly.splitter(settings, materials, orientation=Qt.Vertical,
                        sizes=[440, 360]),
            1,
        )

        self.actions.add_secondary(
            "Reload from disk", self._reload, "Discard edits and read nncm.json again."
        )
        self.save_button = self.actions.add_primary(
            "Save configuration", self._save, "Write these settings to nncm.json."
        )

    def _add_material(self) -> None:
        dialog = MaterialDialog(Material("NEW MATERIAL"), self)
        if dialog.exec():
            self._fill_materials(self._material_rows() + [dialog.material()])

    # -- materials --------------------------------------------------------
    def _material_rows(self) -> list[Material]:
        return [self.materials_table.item(row, MATERIAL_KEY_COLUMN).data(Qt.UserRole)
                for row in range(self.materials_table.rowCount())
                if self.materials_table.item(row, MATERIAL_KEY_COLUMN) is not None
                and self.materials_table.item(row, MATERIAL_KEY_COLUMN).data(Qt.UserRole) is not None]

    def _fill_materials(self, materials: list[Material]) -> None:
        table = self.materials_table
        table.setRowCount(0)
        for material in materials:
            self._append_material(material)
        self._append_totals(materials)
        self.set_detail_visible(self.window.detail_visible)

    def _append_material(self, material: Material) -> None:
        table = self.materials_table
        row = table.rowCount()
        table.insertRow(row)
        values = {
            "name": material.name,
            "kind": "Mixture" if material.is_mixture else "Pure",
            "composition": ", ".join(
                f"{c.component} {c.fraction:g}" for c in material.components
            ) or "pure component",
            "weight": format_number(material.weight),
            "temperature": _range_text(material.temperature, "design range"),
            "pressure": _range_text(material.pressure, "design range"),
            **{
                key: format_number(material.properties.get(key))
                for key in MATERIAL_DETAIL
            },
        }
        for column, (key, _) in enumerate(MATERIAL_COLUMNS):
            text = values.get(key, "")
            item = QTableWidgetItem(elide_middle(text, 44) if key == "composition" else text)
            item.setToolTip(text)
            if key in {"weight"} or key in MATERIAL_DETAIL:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFont(gui_theme.tabular(item.font()))
            if column == MATERIAL_KEY_COLUMN:
                item.setData(Qt.UserRole, material)
            table.setItem(row, column, item)
        edit = ly.button("Edit", variant="quiet", tip="Open this material in full.")
        edit.clicked.connect(lambda _checked=False, r=row: self._edit_material(r))
        table.setCellWidget(row, MATERIAL_EDIT_COLUMN, edit)

    def _append_totals(self, materials: list[Material]) -> None:
        """Totals belong inside the table, as its last row."""
        table = self.materials_table
        row = table.rowCount()
        table.insertRow(row)
        total = sum(m.weight for m in materials)
        cells = {0: f"{len(materials)} materials", 3: format_number(total)}
        for column in range(len(MATERIAL_COLUMNS)):
            # Columns with no meaningful total are left blank: an averaged
            # boiling point is a figure that means nothing.
            item = QTableWidgetItem(cells.get(column, ""))
            item.setFlags(Qt.ItemIsEnabled)
            item.setBackground(_totals_brush())
            font = gui_theme.font(T.FONT_BODY, T.WEIGHT_SEMIBOLD, figures=True)
            item.setFont(font)
            if column == 3:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, column, item)

    def _edit_material(self, row: int) -> None:
        materials = self._material_rows()
        if not 0 <= row < len(materials):
            return  # the totals row is not a material
        dialog = MaterialDialog(materials[row], self)
        if dialog.exec():
            materials[row] = dialog.material()
            self._fill_materials(materials)

    def _remove_material(self) -> None:
        rows = {index.row() for index in self.materials_table.selectedIndexes()}
        materials = self._material_rows()
        keep = [m for index, m in enumerate(materials) if index not in rows]
        if len(keep) == len(materials):
            self.report_problem("Select a material row first.", "warning")
            return
        self.report_problem("")
        self._fill_materials(keep)

    def set_detail_visible(self, visible: bool) -> None:
        for column, (key, _) in enumerate(MATERIAL_COLUMNS):
            self.materials_table.setColumnHidden(column, key in MATERIAL_DETAIL and not visible)

    # -- state ------------------------------------------------------------
    def on_project_changed(self) -> None:
        config = self.project.config
        sampling = config.sampling
        self.vessels.setValue(sampling.n_vessels)
        self.leaks.setValue(sampling.n_leaks_per_vessel)
        self.sampler.setCurrentText(sampling.sampler)
        self.seed.setValue(sampling.seed)
        self.inventory.setValue(sampling.mass_inventory_kg)
        for key, field in self.ranges.items():
            value: Range = getattr(sampling, key)
            field.set_values(value.min, value.max, value.log)
        self._fill_materials(list(sampling.materials))

    def _reload(self) -> None:
        self.window.open_project(self.project.root)

    def collect(self) -> None:
        """Copy what is on screen into the project's configuration."""
        sampling = self.project.config.sampling
        sampling.n_vessels = self.vessels.value()
        sampling.n_leaks_per_vessel = self.leaks.value()
        sampling.sampler = self.sampler.currentText()
        sampling.seed = self.seed.value()
        sampling.mass_inventory_kg = self.inventory.value()
        for key, field in self.ranges.items():
            minimum, maximum, log = field.values()
            setattr(sampling, key, Range(minimum, maximum, log and minimum > 0))
        sampling.materials = self._material_rows()

    def _save(self) -> None:
        self.window.save_configuration()


def _range_text(value: Range | None, absent: str) -> str:
    if value is None:
        return absent
    spacing = ", log" if value.log else ""
    return f"{value.min:g} to {value.max:g}{spacing}"


MATERIAL_NUMERIC_COLUMNS = frozenset(
    index for index, (key, _) in enumerate(MATERIAL_COLUMNS)
    if key == "weight" or key in MATERIAL_DETAIL
)


def _align_headers(table: QTableWidget, numeric_columns) -> None:
    """A header aligns with the cells beneath it.

    Centred headers over left-aligned data is the most common misalignment in
    a Qt table, and the one a reader notices first.
    """
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        if item is None:
            continue
        align = Qt.AlignRight if column in numeric_columns else Qt.AlignLeft
        item.setTextAlignment(align | Qt.AlignVCenter)


def _totals_brush():
    """The totals row sits on the selected-surface rung, not on a grey."""
    return QBrush(QColor(T.ink_hex(T.SURFACE_PRESSED)))


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------
class SamplePage(Page):
    title = "Sampled cases"
    scrolls = False   # the body is a splitter, and both panes fill the height

    def __init__(self, window):
        super().__init__(window)
        self._cases: pd.DataFrame | None = None

        table_card = Card("Scenarios")
        figures = ly.hbox(spacing=T.SPACING_GROUP)
        self.counts: dict[str, MetricCell] = {}
        for key, caption, tip in (
            ("scenarios", "Scenarios", "Leak scenarios in the design."),
            ("vessels", "Vessels", describe("vessel_id").help),
            ("materials", "Materials", "Materials the design draws from."),
        ):
            cell = MetricCell(caption, tip)
            self.counts[key] = cell
            figures.addWidget(cell)
        figures.addStretch(1)
        table_card.add_layout(figures)
        self.table = DataTable(
            "No cases yet. Generate a sampling design to fill this table."
        )
        table_card.add(self.table, 1)
        plot_card = Card("Coverage")
        self.plot = PlotArea(
            "No design to draw yet. Generating cases also draws what they cover."
        )
        plot_card.add(self.plot, 1)
        self.body.addWidget(ly.splitter(table_card, plot_card, sizes=[520, 680]), 1)

        self.generate_button = self.actions.add_primary(
            "Generate cases",
            self._generate,
            "Draw the design from the ranges on the Project page and write cases.csv.",
        )

    def on_project_changed(self) -> None:
        if not self.project.cases_path.exists():
            self._cases = None
            self.refresh_counts()
            self.table.show_message(
                "No cases in this project yet. Generate a sampling design to fill this table."
            )
            self.plot.show_message("No design to draw yet.")
            return
        try:
            self._show(load_cases(self.project.cases_path))
        except Exception as exc:
            self.log(f"! could not load existing cases: {exc}")
            self.table.show_message(f"The case table could not be read: {exc}")

    def set_detail_visible(self, visible: bool) -> None:
        self.table.set_detail_visible(visible)

    def _generate(self) -> None:
        project = self.project
        self.report_problem("")
        self.generate_button.setEnabled(False)
        self.actions.show_running()
        self.window.begin_task("Generating cases")
        self.window.run_task(
            lambda log: run_sampling(project, log=log),
            on_done=self._done,
            on_error=self._failed,
        )

    def _done(self, result: Any) -> None:
        self.generate_button.setEnabled(True)
        self.actions.clear_progress()
        cases, _report = result
        self._show(cases)
        self.window.refresh_project_state()
        self.window.finish_task(f"Generated {len(cases):,} scenarios")

    def _failed(self, message: str) -> None:
        self.fail("Sampling failed", message, self.generate_button)

    def refresh_counts(self) -> None:
        cases = self._cases
        if cases is None:
            for cell in self.counts.values():
                cell.set_value(None)
            return
        self.counts["scenarios"].set_value(f"{len(cases):,}")
        self.counts["vessels"].set_value(
            f"{cases['vessel_name'].nunique():,}" if "vessel_name" in cases else None
        )
        self.counts["materials"].set_value(
            f"{cases['material'].nunique():,}" if "material" in cases else None
        )

    def _show(self, cases: pd.DataFrame) -> None:
        self._cases = cases
        self.refresh_counts()
        self.table.show_frame(cases, key_columns=CASE_KEY_COLUMNS)
        self.table.set_detail_visible(self.window.detail_visible)
        try:
            self.plot.show_figure(plot_case_distributions(cases))
        except Exception as exc:
            # A figure that will not draw must never take the table with it.
            self.log(f"! could not draw the design: {exc}")
            self.plot.show_message(f"The coverage plot could not be drawn: {exc}")


# ---------------------------------------------------------------------------
# 3. Phast exchange
# ---------------------------------------------------------------------------
class PhastPage(Page):
    """The round trip out to Phast and back.

    Two steps, one action bar: the Primary is whichever step the project's
    state says comes next, so there is never more than one on the surface.
    """

    title = "Phast exchange"

    def __init__(self, window):
        super().__init__(window)
        top = ly.hbox(spacing=T.SPACING_GROUP)

        export = Card("1 · Write the input workbook")
        export.add(
            Explanation(
                "Writes the sampled cases into a copy of the Safeti template. "
                "Import the file into Phast, run the study, then export its "
                "results and come back to step 2."
            )
        )
        self.export_form = Form()
        self.export_path = PathField("Save Phast input workbook", mode="save")
        self.split_rows = integer_field(0, 1_000_000, 0, special="one file")
        self.split_rows.setSingleStep(1000)
        self.export_form.add("export_path", self.export_path, span=True)
        self.export_form.add("max_rows_per_workbook", self.split_rows)
        export.add(self.export_form)
        self.export_notes = Advisories()
        export.add(self.export_notes)
        export.body().addStretch(1)
        top.addWidget(export, 1)

        extract = Card("2 · Read the result workbook")
        extract.add(
            Explanation(
                "Joins the Discharge, Dispersion and Fire sheets back onto the "
                "sampled cases and writes the training dataset."
            )
        )
        self.import_form = Form()
        self.import_path = PathField("Select Phast result workbook")
        self.merge_box = QCheckBox()
        self.import_form.add("result_path", self.import_path, span=True)
        self.import_form.add_switch("merge", self.merge_box)
        extract.add(self.import_form)
        extract.body().addStretch(1)
        top.addWidget(extract, 1)
        self.body.addLayout(top)

        dataset = Card("Training dataset")
        figures = ly.hbox(spacing=T.SPACING_GROUP)
        self.dataset_metrics: dict[str, MetricCell] = {}
        for key, caption, tip in (
            ("rows", "Rows", "Result rows available for training."),
            ("vessels", "Vessels", "Distinct vessels the rows come from."),
            ("materials", "Materials", "Materials present in the dataset."),
            ("targets", "Targets present", "Configured targets found in the dataset."),
        ):
            cell = MetricCell(caption, tip)
            self.dataset_metrics[key] = cell
            figures.addWidget(cell)
        figures.addStretch(1)
        dataset.add_layout(figures)
        self.table = DataTable("No dataset yet. Read a Phast result workbook to build one.")
        dataset.add(self.table, 1)
        self.body.addWidget(dataset, 1)

        self.export_button = self.actions.add_secondary(
            "Write input workbook", self._export, "Write the sampled cases into a Phast workbook."
        )
        self.import_button = self.actions.add_primary(
            "Extract training data", self._import, "Read a Phast result workbook into the dataset."
        )

    # -- state ------------------------------------------------------------
    def on_project_changed(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_path.set_path(self.project.phast_input_dir / f"phast_input_{stamp}.xlsx")
        self.split_rows.setValue(self.project.config.phast.max_rows_per_workbook)
        self.export_notes.show_notes([])
        self.refresh_dataset()

    def set_detail_visible(self, visible: bool) -> None:
        self.table.set_detail_visible(visible)

    def refresh_state(self) -> None:
        """Which step comes next. Cheap: it does not re-read the dataset."""
        self._move_primary()

    def _move_primary(self) -> None:
        """One primary on the surface, on the step the project is actually at.

        The buttons are reordered as well as restyled. The primary sits on the
        trailing edge of the action bar on every other page, and a primary that
        moved to the middle of a row here would be the one visible
        inconsistency in the set.
        """
        ready_to_extract = self.project.cases_path.exists() and any(
            self.project.phast_input_dir.glob("*.xlsx")
        )
        set_primary(self.import_button, ready_to_extract)
        set_primary(self.export_button, not ready_to_extract)
        self.actions.order(
            [self.export_button, self.import_button] if ready_to_extract
            else [self.import_button, self.export_button]
        )

    def refresh_dataset(self) -> None:
        summary = dataset_summary(self.project)
        self.dataset_metrics["rows"].set_value(f"{summary.rows:,}" if summary.rows else None, "rows")
        self.dataset_metrics["vessels"].set_value(f"{summary.vessels:,}" if summary.vessels else None)
        self.dataset_metrics["materials"].set_value(f"{summary.materials:,}" if summary.materials else None)
        self.dataset_metrics["targets"].set_value(f"{len(summary.targets)}" if summary.targets else None)
        if not summary.rows:
            self.table.show_message(
                "No dataset in this project yet. Read a Phast result workbook to build one."
            )
        else:
            try:
                frame = pd.read_csv(self.project.training_data_path)
                self.table.show_frame(frame, key_columns=CASE_KEY_COLUMNS + list(summary.targets))
                self.table.set_detail_visible(self.window.detail_visible)
            except Exception as exc:
                self.log(f"! could not preview the dataset: {exc}")
                self.table.show_message(f"The dataset could not be read: {exc}")
        self._move_primary()

    # -- actions ----------------------------------------------------------
    def _export(self) -> None:
        project = self.project
        project.config.phast.max_rows_per_workbook = self.split_rows.value()
        target = self.export_path.path()
        self.report_problem("")
        self.export_button.setEnabled(False)
        self.actions.show_running()
        self.window.begin_task("Writing the Phast workbook")
        self.window.run_task(
            lambda log: run_phast_export(project, output_path=target, log=log),
            on_done=self._export_done,
            on_error=lambda message: self._failed(self.export_button, "Export failed", message),
        )

    def _export_done(self, report) -> None:
        self.export_button.setEnabled(True)
        self.actions.clear_progress()
        notes = [("subtle", f"Wrote {elide_middle(str(path), 60)}") for path in report.files]
        notes.append(
            ("subtle", f"{report.n_vessels:,} vessels and {report.n_leaks:,} leaks. Import this "
                       "into Phast, run it, then export the results.")
        )
        self.export_notes.show_notes(notes)
        self.window.refresh_project_state()
        self.window.finish_task(f"Wrote {report.n_leaks:,} leak rows")

    def _import(self) -> None:
        workbook = self.import_path.path()
        if workbook is None or not workbook.exists():
            self.report_problem(
                "Choose a Phast result workbook first — the workbook Phast "
                "exported after the run, holding the Discharge, Dispersion "
                "and Fire sheets.", "warning"
            )
            return
        project = self.project
        append = self.merge_box.isChecked()
        self.report_problem("")
        self.import_button.setEnabled(False)
        self.actions.show_running()
        self.window.begin_task("Reading the Phast results")
        self.window.run_task(
            lambda log: run_phast_import(project, workbook, log=log, append=append),
            on_done=self._import_done,
            on_error=lambda message: self._failed(self.import_button, "Extraction failed", message),
        )

    def _import_done(self, result: Any) -> None:
        self.import_button.setEnabled(True)
        self.actions.clear_progress()
        frame, _report = result
        self.refresh_dataset()
        self.window.refresh_project_state()
        self.window.finish_task(f"Extracted {len(frame):,} rows")

    def _failed(self, button: QPushButton, title: str, message: str) -> None:
        self.fail(title, message, button)


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
METRIC_COLUMNS = ["target", "r2", "median_ape", "mae", "rmse", "n_test"]


class TrainPage(Page):
    title = "Train"

    def __init__(self, window):
        super().__init__(window)
        settings = Card("Hyperparameters")
        settings.add(
            Explanation(
                "Training splits by vessel, so rows that share a vessel never "
                "land on both sides of the split and the scores stay honest."
            )
        )
        columns = ly.hbox(spacing=T.SPACING_SECTION)
        left, right = Form(), Form()
        self.epochs = integer_field(1, 10_000, 400)
        self.batch_size = integer_field(8, 8192, 256)
        self.learning_rate = decimal_field(1e-5, 1.0, 1e-3, 5)
        self.learning_rate.setSingleStep(1e-4)
        self.dropout = decimal_field(0.0, 0.9, 0.1, 2)
        self.dropout.setSingleStep(0.05)
        self.hidden = QLineEdit()
        self.hidden.setPlaceholderText("128, 128, 96")
        self.targets = QLineEdit()
        self.targets.setPlaceholderText("Release_rate, Velocity, Distance_to_LFL, Flame_length")
        self.use_weather = QCheckBox()
        self.use_properties = QCheckBox()
        left.add("epochs", self.epochs)
        left.add("batch_size", self.batch_size)
        left.add("learning_rate", self.learning_rate)
        left.add("dropout", self.dropout)
        right.add("hidden_units", self.hidden, span=True)
        right.add("targets", self.targets, span=True)
        right.add_switch("use_weather", self.use_weather)
        right.add_switch("use_material_properties", self.use_properties)
        columns.addWidget(left, 1)
        columns.addWidget(right, 1)
        settings.add_layout(columns)
        self.body.addWidget(settings)

        scores = Card("Scores on held-out vessels")
        self.metrics = DataTable("No run yet. Train a model to score it here.")
        scores.add(self.metrics, 1)
        parity = Card("Predicted against Phast")
        self.plot = PlotArea("No run yet. Training draws a parity plot for every target.")
        parity.add(self.plot, 1)
        self.body.addWidget(ly.splitter(scores, parity, sizes=[520, 680]), 1)

        self.notes = Advisories()
        self.body.addWidget(self.notes)

        self.train_button = self.actions.add_primary(
            "Train model", self._train, "Train on the project dataset and save a new run."
        )

    # -- state ------------------------------------------------------------
    def on_project_changed(self) -> None:
        config = self.project.config.training
        self.epochs.setValue(config.epochs)
        self.batch_size.setValue(config.batch_size)
        self.learning_rate.setValue(config.learning_rate)
        self.dropout.setValue(config.dropout)
        self.hidden.setText(", ".join(str(units) for units in config.hidden_units))
        self.targets.setText(", ".join(config.targets))
        self.use_weather.setChecked(config.use_weather)
        self.use_properties.setChecked(config.use_material_properties)
        self._show_latest_run()

    def _show_latest_run(self) -> None:
        run_dir = self.project.latest_run_dir()
        if run_dir is None:
            self.metrics.show_message(
                "No trained run in this project yet. Train a model to score it here."
            )
            self.plot.show_message("No run yet. Training draws a parity plot for every target.")
            self.notes.show_notes([])
            return
        registry = self.project.registry()
        entry = next(
            (r for r in registry.get("runs", []) if r.get("run_id") == run_dir.name), None
        )
        if entry:
            self._show_metrics(entry.get("metrics", {}))
        self._show_parity(run_dir)

    def collect(self):
        config = self.project.config.training
        config.epochs = self.epochs.value()
        config.batch_size = self.batch_size.value()
        config.learning_rate = self.learning_rate.value()
        config.dropout = self.dropout.value()
        hidden = [int(u) for u in self.hidden.text().replace(" ", "").split(",") if u]
        if hidden:
            config.hidden_units = hidden
        targets = [t.strip() for t in self.targets.text().split(",") if t.strip()]
        if targets:
            config.targets = targets
            config.log_targets = [t for t in config.log_targets if t in targets] or targets
        config.use_weather = self.use_weather.isChecked()
        config.use_material_properties = self.use_properties.isChecked()
        return config

    # -- actions ----------------------------------------------------------
    def _train(self) -> None:
        from ..training import train_model

        if not self.project.training_data_path.exists():
            self.report_problem(
                "There is no dataset to train on yet. Read a Phast result "
                "workbook on the Phast page to build one.", "warning"
            )
            return
        project = self.project
        config = self.collect()
        project.save_config()
        self.report_problem("")
        self.train_button.setEnabled(False)
        self.notes.show_notes([])
        self.actions.show_progress(0, config.epochs, "epochs")
        self.window.begin_task("Training")
        self.window.run_task(
            lambda log, progress: train_model(project, config=config, log=log, progress=progress),
            on_done=self._done,
            on_error=self._failed,
            on_progress=self._progress,
        )

    def _progress(self, done: int, total: int) -> None:
        # Every step is an epoch that actually finished; early stopping can end
        # the run before the total, which is why the bar says "at most".
        self.actions.show_progress(done, total, "epochs")

    def _done(self, result) -> None:
        self.train_button.setEnabled(True)
        self.actions.clear_progress()
        self._show_metrics(result.metrics)
        self._show_parity(result.run_dir)
        mean_r2 = result.metrics.get("mean_r2")
        self.window.refresh_project_state()
        self.window.reload_model()
        self.window.finish_task(
            f"Trained {result.run_id}, mean R2 {mean_r2:.4f}" if mean_r2 == mean_r2
            else f"Trained {result.run_id}"
        )

    def _failed(self, message: str) -> None:
        self.fail("Training failed", message, self.train_button)

    # -- results ----------------------------------------------------------
    def _show_metrics(self, metrics: dict) -> None:
        per_target = metrics.get("per_target", {})
        if not per_target:
            self.metrics.show_message("The run produced no scored target.")
            return
        rows = []
        for target, scores in per_target.items():
            rows.append(
                {
                    "target": describe(target).with_unit(),
                    "r2": scores.get("r2"),
                    "median_ape": scores.get("median_ape"),
                    "mae": scores.get("mae"),
                    "rmse": scores.get("rmse"),
                    "n_test": scores.get("n_test"),
                }
            )
        frame = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        # Only R2 and the row count mean anything across targets: an MAE
        # averaged over kilograms per second and metres is a figure that means
        # nothing, so those columns are left blank.
        totals = {
            "target": "All targets",
            "r2": metrics.get("mean_r2"),
            "n_test": metrics.get("n_test"),
        }
        self.metrics.show_frame(frame, totals=totals)

        thin = [
            (
                "warning",
                f"{describe(target).label} was scored on {scores.get('n_test', 0)} rows — "
                "too few for the score to mean much.",
            )
            for target, scores in per_target.items()
            if "r2" not in scores or scores.get("n_test", 0) < 30
        ]
        self.notes.show_notes(thin)

    def _show_parity(self, run_dir) -> None:
        from ..training import plot_parity

        try:
            self.plot.show_figure(plot_parity(run_dir))
        except Exception as exc:
            self.log(f"! could not draw the parity plot: {exc}")
            self.plot.show_message(f"The parity plot could not be drawn: {exc}")


# ---------------------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------------------
STABILITY_CLASSES = ["A", "B", "C", "D", "E", "F"]


class PredictPage(Page):
    title = "Predict"
    scrolls = False   # the body is a splitter; the input column scrolls itself

    def __init__(self, window):
        super().__init__(window)
        self.bundle: ModelBundle | None = None

        left = QWidget()
        left_column = ly.vbox(left, spacing=T.SPACING_GROUP)

        model_card = Card("Model")
        figures = ly.hbox(spacing=T.SPACING_GROUP)
        self.model_metrics: dict[str, MetricCell] = {}
        for key, caption, tip in (
            ("run", "Run", "The trained run currently loaded."),
            ("mean_r2", "Mean R2", describe("mean_r2").help),
            ("n_test", "Test rows", describe("n_test").help),
        ):
            cell = MetricCell(caption, tip)
            self.model_metrics[key] = cell
            figures.addWidget(cell)
        figures.addStretch(1)
        model_card.add_layout(figures)
        self.model_notes = Advisories()
        model_card.add(self.model_notes)
        left_column.addWidget(model_card)

        inputs = Card("Inputs")
        form = Form()
        self.temperature = decimal_field(-273.0, 2000.0, 25.0, 2)
        self.pressure = decimal_field(0.0, 5000.0, 20.0, 2)
        self.orifice = decimal_field(0.01, 5000.0, 25.0, 2)
        self.material = choice_field([], "")
        self.wind = decimal_field(0.1, 60.0, 5.0, 2)
        self.stability = choice_field(STABILITY_CLASSES, "D")
        self.mc_samples = integer_field(0, 500, 50, special="none")
        form.add("temperature_degC", self.temperature)
        form.add("pressure_barg", self.pressure)
        form.add("orifice_mm", self.orifice)
        form.add("material", self.material, span=True)
        form.add("wind_speed_ms", self.wind)
        form.add("stability_index", self.stability, span=True)
        form.add("mc_samples", self.mc_samples)
        inputs.add(form)
        inputs.body().addStretch(1)
        left_column.addWidget(inputs, 1)
        left = scroll_pane(left)
        left.setMinimumWidth(420)

        right = QWidget()
        right_column = ly.vbox(right, spacing=T.SPACING_GROUP)
        results = Card("Predicted consequences")
        self.results = DataTable("Nothing predicted yet. Set the inputs, then Predict.")
        results.add(self.results, 1)
        right_column.addWidget(results, 1)
        self.domain_notes = Advisories()
        right_column.addWidget(self.domain_notes)
        self.body.addWidget(ly.splitter(left, right, sizes=[460, 740]), 1)

        self.batch_button = self.actions.add_secondary(
            "Predict from CSV", self._predict_csv, "Predict every row of a CSV and write the results beside it."
        )
        self.predict_button = self.actions.add_primary(
            "Predict", self._predict, "Predict the consequences of the inputs on the left."
        )

    # -- model ------------------------------------------------------------
    def on_project_changed(self) -> None:
        # The design's materials are the honest default list: the model may
        # narrow it, but an empty box names nothing at all.
        self._set_materials([m.name for m in self.project.config.sampling.materials])
        self.reload_model()

    def _set_materials(self, names: list[str]) -> None:
        current = self.material.currentText()
        self.material.clear()
        self.material.addItems(names)
        if current in names:
            self.material.setCurrentText(current)

    def reload_model(self) -> None:
        """Load the latest run off the UI thread — it pulls in TensorFlow."""
        run_dir = self.project.latest_run_dir()
        if run_dir is None:
            self.bundle = None
            self._set_model_metrics(None)
            self.model_notes.show_notes(
                [("subtle", "No trained model in this project yet. Train one on the Train page.")]
            )
            self.predict_button.setEnabled(False)
            return
        self.bundle = None
        self.predict_button.setEnabled(False)
        self.model_notes.show_notes([("subtle", f"Loading {run_dir.name}...")])
        self.window.begin_task(f"Loading {run_dir.name}")
        if not self.window.run_task(
            lambda: ModelBundle(run_dir),
            on_done=self._model_loaded,
            on_error=self._model_failed,
            pass_log=False,
            quiet=True,
        ):
            # Another stage is mid-run; the model is loaded again when it ends.
            message = "The model will load when the running stage finishes."
            self.model_notes.show_notes([("subtle", message)])
            self.window.report(message)

    def _model_loaded(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._set_model_metrics(bundle)
        self.model_notes.show_notes([])
        self.predict_button.setEnabled(True)
        self._set_materials(
            bundle.spec.material_categories
            or [m.name for m in self.project.config.sampling.materials]
        )
        self.window.finish_task(f"Loaded {bundle.run_id}")

    def _model_failed(self, message: str) -> None:
        self.bundle = None
        self._set_model_metrics(None)
        self.model_notes.show_notes(
            [("danger", f"The model could not be loaded: {message.splitlines()[0]}")]
        )
        self.log(f"! {message}")

    def _set_model_metrics(self, bundle: ModelBundle | None) -> None:
        if bundle is None:
            for cell in self.model_metrics.values():
                cell.set_value(None)
            return
        metrics = bundle.metrics
        mean_r2 = metrics.get("mean_r2")
        self.model_metrics["run"].set_value(bundle.run_id.replace("run_", ""))
        self.model_metrics["mean_r2"].set_value(
            f"{mean_r2:.4f}" if isinstance(mean_r2, float) and mean_r2 == mean_r2 else None
        )
        self.model_metrics["n_test"].set_value(
            f"{metrics.get('n_test'):,}" if metrics.get("n_test") else None, "rows"
        )

    # -- prediction -------------------------------------------------------
    def _inputs(self) -> dict[str, Any]:
        from ..phast.output_reader import STABILITY_INDEX

        row: dict[str, Any] = {
            "temperature_degC": self.temperature.value(),
            "pressure_barg": self.pressure.value(),
            "orifice_mm": self.orifice.value(),
            "wind_speed_ms": self.wind.value(),
            "stability_index": STABILITY_INDEX.get(self.stability.currentText(), 4),
        }
        material = self.material.currentText().strip()
        if material:
            row["material"] = material
            properties = {
                m.name: m.properties for m in self.project.config.sampling.materials
            }.get(material, {})
            for key, value in properties.items():
                row[f"mat_{key}"] = value
        return row

    def _predict(self) -> None:
        if self.bundle is None:
            self.report_problem(
                "No model is loaded yet. Train one on the Train page, or open "
                "a project that already has a run.", "warning"
            )
            return
        self.report_problem("")
        try:
            prediction = self.bundle.predict_one(
                self._inputs(), mc_samples=self.mc_samples.value()
            )
        except Exception as exc:
            self.fail("Prediction failed", str(exc))
            return

        rows = []
        for target, value in prediction.values.items():
            quantity = describe(target)
            spread = prediction.uncertainty.get(target)
            rows.append(
                {
                    "quantity": quantity.label,
                    "prediction": format_quantity(value),
                    "value_unit": quantity.unit,
                    "spread": format_quantity(spread) if spread is not None else "",
                }
            )
        self.results.show_frame(
            pd.DataFrame(rows, columns=["quantity", "prediction", "value_unit", "spread"])
        )
        self.domain_notes.show_notes(
            [
                ("warning", f"{describe(column).label}: {message}")
                for column, message in prediction.domain_warnings.items()
            ]
        )
        self.window.report("Predicted 1 scenario")

    def _predict_csv(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        if self.bundle is None:
            self.report_problem(
                "No model is loaded yet. Train one on the Train page, or open "
                "a project that already has a run.", "warning"
            )
            return
        self.report_problem("")
        chosen, _filter = QFileDialog.getOpenFileName(
            self, "Select a CSV of input rows", str(self.project.root), "CSV files (*.csv)"
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            frame = pd.read_csv(path)
            predictions = self.bundle.predict_frame(frame, mc_samples=self.mc_samples.value())
            target = path.with_name(path.stem + "_predictions.csv")
            pd.concat([frame, predictions], axis=1).to_csv(target, index=False)
        except Exception as exc:
            self.fail("Batch prediction failed", str(exc))
            return
        self.log(f"predicted {len(frame)} rows -> {target}")
        self.domain_notes.show_notes(
            [("subtle", f"{len(frame):,} rows written to {elide_middle(str(target), 60)}")]
        )
        self.window.report(f"Predicted {len(frame):,} rows")
