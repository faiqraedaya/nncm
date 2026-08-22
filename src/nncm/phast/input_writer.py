"""Turn a sampled case table into a Phast/Safeti-importable workbook.

This replaces the manual copy-paste step: the generated cases are written into
a copy of the shipped template (which already carries a configured study,
weather set and parameter sets), addressed by Safeti attribute code and
converted into whatever units the template's header row declares.

Two rules keep Phast's importer happy, and both matter more than they look:

* **The template is patched, never re-saved.** Rows are injected into the
  worksheet XML by :mod:`nncm.phast.patcher`; every other part of the file
  stays byte-identical. Re-saving through a spreadsheet library rewrites
  strings, drops the shared-string table and renames parts, and Phast rejects
  the result even though Excel opens it.
* **Only the cells a scenario needs are written.** Every extra default is
  another chance to trip a validation rule inside Phast, so the writer touches
  two sheets and about a dozen columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Material, PhastConfig
from . import units
from .patcher import TemplatePatcher
from .workbook import SafetiWorkbook

VESSEL_SHEET = "Pressure vessel"
LEAK_SHEET = "Leak"
COMPONENT_SHEET = "COMPONENT"
MIXTURE_SHEET = "MIXTURE"
STUDY_SHEET = "Study"

# Property method Safeti assigns to materials created from a spreadsheet; the
# real project sheets carry this value on every declared component and mixture.
PROPERTY_METHOD_TEMPLATE = "PhastMC"

# Attribute codes are Safeti's stable identifiers; labels shift between builds.
VESSEL_NAME_COL = "label:Name"
VESSEL_MATERIAL_COL = "Material"
LEAK_NAME_COL = "label:Name"
LEAK_VESSEL_COL = "label:Pressure vessel"


@dataclass
class WriteReport:
    files: list[Path] = field(default_factory=list)
    n_vessels: int = 0
    n_leaks: int = 0
    materials: list[str] = field(default_factory=list)
    unit_map: dict[str, str] = field(default_factory=dict)
    sheets_touched: list[str] = field(default_factory=list)
    columns_written: dict[str, list[str]] = field(default_factory=dict)
    materials_declared: dict[str, list[str]] = field(default_factory=dict)
    verified: bool = False
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"wrote {self.n_vessels} vessels and {self.n_leaks} leaks",
            "units used: " + ", ".join(f"{k}={v}" for k, v in sorted(self.unit_map.items())),
            "sheets modified: " + (", ".join(self.sheets_touched) or "none"),
            "read-back check: " + ("passed" if self.verified else "see warnings below"),
        ]
        for sheet, names in self.materials_declared.items():
            lines.append(f"  materials on {sheet}: {', '.join(names)}")
        for sheet, columns in self.columns_written.items():
            lines.append(f"  {sheet}: {', '.join(columns)}")
        lines += [f"  -> {p}" for p in self.files]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def write_input_workbook(
    cases: pd.DataFrame,
    output_path: Path,
    config: PhastConfig,
    template: Path | None = None,
    materials: list[Material] | None = None,
) -> WriteReport:
    """Write ``cases`` into copies of the Safeti template.

    Splits into several workbooks when ``config.max_rows_per_workbook`` is set,
    because Phast's importer slows to a crawl on very large sheets.
    """
    template_path = Path(template or config.template)
    output_path = Path(output_path)
    report = WriteReport()

    chunk_size = config.max_rows_per_workbook or len(cases)
    if chunk_size <= 0:
        chunk_size = len(cases)

    # Never split a vessel across workbooks: a leak must land in the same file
    # as the vessel it references or the Phast import breaks the link.
    vessel_order = list(dict.fromkeys(cases["vessel_name"]))
    per_vessel = cases.groupby("vessel_name", sort=False)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_rows = 0
    for vessel in vessel_order:
        size = len(per_vessel.get_group(vessel))
        if current and current_rows + size > chunk_size:
            chunks.append(current)
            current, current_rows = [], 0
        current.append(vessel)
        current_rows += size
    if current:
        chunks.append(current)

    for chunk_idx, vessels in enumerate(chunks, start=1):
        subset = cases[cases["vessel_name"].isin(vessels)]
        path = output_path
        if len(chunks) > 1:
            path = output_path.with_name(f"{output_path.stem}_part{chunk_idx:02d}{output_path.suffix}")
        _write_single(subset, path, config, template_path, report, materials or [])

    report.materials = sorted(cases["material"].astype(str).unique().tolist())
    return report


def _write_single(
    cases: pd.DataFrame,
    path: Path,
    config: PhastConfig,
    template_path: Path,
    report: WriteReport,
    materials: list[Material],
) -> None:
    with SafetiWorkbook(template_path, read_only=True) as workbook:
        vessel_schema = workbook.schema(VESSEL_SHEET)
        leak_schema = workbook.schema(LEAK_SHEET)

        # Units come from the template itself, so a template configured in psi
        # or inches still receives correct numbers.
        temperature_unit = vessel_schema.unit("Temperature")
        pressure_unit = vessel_schema.unit("Pressure")
        mass_unit = vessel_schema.unit("MassInventory")
        orifice_unit = leak_schema.unit("HoleDiameter")
        elevation_unit = leak_schema.unit("Elevation", entity="DischargeParameters")
        report.unit_map = {
            "temperature": temperature_unit,
            "pressure": pressure_unit,
            "mass_inventory": mass_unit,
            "orifice": orifice_unit,
            "elevation": elevation_unit,
        }

        study_name = _resolve_study_name(workbook, config)

        vessels = cases.drop_duplicates("vessel_name")
        vessel_records: list[dict[str | int, Any]] = []
        for row in vessels.itertuples(index=False):
            record: dict[str | int, Any] = {
                "label:Use": "Yes",
                "label:Study": study_name,
                VESSEL_NAME_COL: row.vessel_name,
                VESSEL_MATERIAL_COL: row.material,
                "MaterialToTrack": row.material,
                "Temperature": units.convert(row.temperature_degC, "temperature", temperature_unit),
                "Pressure": units.convert(row.pressure_barg, "pressure", pressure_unit),
                "MassInventory": units.convert(
                    getattr(row, "mass_inventory_kg", 50_000.0), "mass", mass_unit
                ),
            }
            if config.folder_name:
                record["idx:3"] = config.folder_name  # first Folder column
            record.update(_defaults(vessel_schema, config.vessel_defaults, report))
            vessel_records.append(record)

        leak_records: list[dict[str | int, Any]] = []
        for row in cases.itertuples(index=False):
            record = {
                "label:Use": "Yes",
                "label:Study": study_name,
                LEAK_VESSEL_COL: row.vessel_name,
                LEAK_NAME_COL: row.leak_name,
                "HoleDiameter": units.convert(row.orifice_mm, "diameter", orifice_unit),
            }
            elevation = getattr(row, "elevation_m", None)
            if elevation is not None and pd.notna(elevation):
                record["Elevation"] = units.convert(float(elevation), "length", elevation_unit)
            if config.folder_name:
                record["idx:3"] = config.folder_name
            record.update(_defaults(leak_schema, config.leak_defaults, report))
            leak_records.append(record)

        vessel_rows = _to_indexed(vessel_records, vessel_schema)
        leak_rows = _to_indexed(leak_records, leak_schema)
        report.columns_written = {
            VESSEL_SHEET: _column_labels(vessel_records, vessel_schema),
            LEAK_SHEET: _column_labels(leak_records, leak_schema),
        }

        material_rows: dict[str, list[dict[int, Any]]] = {}
        used = sorted(set(cases["material"].astype(str)))
        catalogue = {m.name: m for m in materials}
        undefined = [name for name in used if name not in catalogue]
        if undefined and config.write_material_rows:
            report.warnings.append(
                f"no definition for material(s) {undefined} — those vessels would "
                "reference a material the study never defines"
            )
        if config.write_material_rows:
            material_rows = _material_rows(
                workbook, [catalogue[name] for name in used if name in catalogue], report
            )

    # Patch the template rather than re-saving it: everything outside the rows
    # we add stays byte-for-byte identical, which is what Phast's importer needs.
    with TemplatePatcher(template_path) as patcher:
        n_vessels = patcher.add_rows(
            VESSEL_SHEET, vessel_rows, start_row=vessel_schema.data_start_row
        )
        n_leaks = patcher.add_rows(LEAK_SHEET, leak_rows, start_row=leak_schema.data_start_row)
        for sheet_name, sheet_rows in material_rows.items():
            patcher.add_rows(sheet_name, sheet_rows)
        saved = patcher.save(path)
        report.sheets_touched = sorted(patcher.report.rows_added)

    if config.verify_after_write:
        _verify_written(
            saved,
            {
                VESSEL_SHEET: (vessel_rows, vessel_schema.data_start_row),
                LEAK_SHEET: (leak_rows, leak_schema.data_start_row),
            },
            report,
        )
        _verify_materials(saved, cases, report)

    report.files.append(saved)
    report.n_vessels += n_vessels
    report.n_leaks += n_leaks


def _verify_written(
    path: Path,
    written: dict[str, tuple[list[dict[int, Any]], int]],
    report: WriteReport,
    sample: int = 50,
) -> None:
    """Read the saved workbook back and confirm the cells really landed.

    Cheap insurance: a malformed row would otherwise only surface when Phast
    refuses the import, long after the export looked successful.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet_name, (rows, start_row) in written.items():
            if not rows:
                continue
            worksheet = workbook[sheet_name]
            indices = sorted({0, len(rows) - 1, *range(0, len(rows), max(1, len(rows) // sample))})
            wanted = {start_row + i: rows[i] for i in indices}
            seen = 0
            last_populated = 0
            for row_number, row in enumerate(
                worksheet.iter_rows(min_row=start_row, values_only=True), start=start_row
            ):
                if any(v is not None and str(v).strip() != "" for v in row):
                    last_populated = row_number
                expected = wanted.get(row_number)
                if expected is None:
                    continue
                seen += 1
                for column, value in expected.items():
                    actual = row[column - 1] if column - 1 < len(row) else None
                    if isinstance(value, float):
                        if actual is None or abs(float(actual) - value) > max(1e-6, abs(value) * 1e-9):
                            report.warnings.append(
                                f"{sheet_name} row {row_number} col {column}: wrote {value}, read back {actual}"
                            )
                    elif str(actual) != str(value):
                        report.warnings.append(
                            f"{sheet_name} row {row_number} col {column}: wrote '{value}', read back '{actual}'"
                        )
            expected_last = start_row + len(rows) - 1
            if last_populated != expected_last:
                report.warnings.append(
                    f"{sheet_name}: expected data to end at row {expected_last}, found {last_populated}"
                )
            if seen != len(wanted):
                report.warnings.append(
                    f"{sheet_name}: only {seen} of {len(wanted)} sampled rows were readable"
                )
    finally:
        workbook.close()


def _verify_materials(path: Path, cases: pd.DataFrame, report: WriteReport) -> None:
    """Check every vessel names the material its case says, and that it exists.

    A vessel pointing at a material the study never defines imports as an
    incomplete model, and a mis-assigned material silently trains the surrogate
    on the wrong fluid — neither shows up as an error anywhere else.
    """
    with SafetiWorkbook(path, read_only=True) as workbook:
        vessels = {
            str(row.get("Name")).strip(): str(row.get("Material", "")).strip()
            for row in workbook.data_rows(VESSEL_SHEET)
            if row.get("Name")
        }
        defined = _existing_names(workbook, COMPONENT_SHEET) | _existing_names(
            workbook, MIXTURE_SHEET
        )

    expected = (
        cases.drop_duplicates("vessel_name")
        .set_index("vessel_name")["material"]
        .astype(str)
        .to_dict()
    )
    mismatched = [
        f"{name}: expected '{material}', sheet says '{vessels.get(name, '')}'"
        for name, material in expected.items()
        if vessels.get(name, "") != material
    ]
    if mismatched:
        report.warnings.append(
            f"{len(mismatched)} vessel(s) carry the wrong material — e.g. {mismatched[0]}"
        )

    undeclared = sorted(
        {m for m in expected.values() if m.strip().casefold() not in defined}
    )
    if undeclared:
        report.warnings.append(
            f"material(s) not defined in the workbook: {undeclared} — "
            "add them to the material catalogue or declare them in the template"
        )
    report.verified = not report.warnings


def _to_indexed(records: list[dict[str | int, Any]], schema) -> list[dict[int, Any]]:
    """Resolve column keys to 1-based column indices once for the whole batch."""
    resolved: dict[str | int, int] = {}
    rows: list[dict[int, Any]] = []
    for record in records:
        row: dict[int, Any] = {}
        for key, value in record.items():
            if value is None:
                continue
            if key not in resolved:
                resolved[key] = _resolve_column(key, schema)
            row[resolved[key]] = value
        rows.append(row)
    return rows


def _resolve_column(key: str | int, schema) -> int:
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.startswith("idx:"):
        return int(key[4:])
    return schema.column(key).index


def _column_labels(records: list[dict[str | int, Any]], schema) -> list[str]:
    """Human-readable list of the columns this export touches."""
    labels: list[str] = []
    for key in dict.fromkeys(k for record in records for k in record):
        index = _resolve_column(key, schema)
        column = next((c for c in schema.columns if c.index == index), None)
        labels.append(column.label or column.attribute or f"col{index}" if column else f"col{index}")
    return labels


def _defaults(schema, defaults: dict[str, Any], report: WriteReport) -> dict[str | int, Any]:
    """Filter configured defaults down to columns this template actually has."""
    resolved: dict[str | int, Any] = {}
    for key, value in defaults.items():
        if schema.has(key):
            resolved[key] = value
            _check_option(schema, key, value, report)
        else:
            message = f"{schema.name}: template has no column '{key}' — default ignored"
            if message not in report.warnings:
                report.warnings.append(message)
    return resolved


def _check_option(schema, key: str | int, value: Any, report: WriteReport) -> None:
    """Reject enumeration values the template's own pick-list does not offer.

    Phast rejects the row (or silently defaults it) when an enumeration string
    does not match exactly; the template lists the legal values, so check here.
    """
    if not isinstance(value, str):
        return
    column = schema.column(key)
    choices = column.enumerations
    if choices and value not in choices:
        message = (
            f"{schema.name}.{column.label or column.attribute}: '{value}' is not one of "
            f"{list(choices[:6])}{' …' if len(choices) > 6 else ''}"
        )
        if message not in report.warnings:
            report.warnings.append(message)


def _resolve_study_name(workbook: SafetiWorkbook, config: PhastConfig) -> str:
    """Use the study already defined in the template so the link resolves."""
    try:
        rows = workbook.data_rows(STUDY_SHEET)
    except Exception:
        return config.study_name
    for row in rows:
        for key in ("Name", "label:Name"):
            if key in row and str(row[key]).strip():
                return str(row[key]).strip()
        name = row.get("Name") or row.get("col2")
        if name:
            return str(name).strip()
    return config.study_name


def _material_rows(
    workbook: SafetiWorkbook,
    materials: list[Material],
    report: WriteReport,
) -> dict[str, list[dict[int, Any]]]:
    """Declare every material the vessels reference, the way the real sheets do.

    * A pure component gets one COMPONENT row: ``Yes | Physical Properties
      System | Materials | (folder) | NAME | PhastMC``.
    * A mixture gets one MIXTURE block: the first row carries the mixture name,
      property template and its first component; each further component adds a
      row holding only the component and its fraction. Fractions go in the Mole
      (or Mass) column as percentages summing to 100, which is how Safeti
      stores a composition.
    """
    rows: dict[str, list[dict[int, Any]]] = {}
    pure = [m for m in materials if not m.is_mixture]
    mixtures = [m for m in materials if m.is_mixture]

    if pure:
        records, declared = _component_records(workbook, pure, report)
        if records:
            rows[COMPONENT_SHEET] = records
            report.materials_declared[COMPONENT_SHEET] = declared

    if mixtures:
        records, declared = _mixture_records(workbook, mixtures, report)
        if records:
            rows[MIXTURE_SHEET] = records
            report.materials_declared[MIXTURE_SHEET] = declared

    return rows


def _component_records(
    workbook: SafetiWorkbook, materials: list[Material], report: WriteReport
) -> tuple[list[dict[int, Any]], list[str]]:
    if COMPONENT_SHEET not in workbook.sheet_names:
        report.warnings.append(
            f"template has no {COMPONENT_SHEET} sheet — pure components not declared"
        )
        return [], []

    schema = workbook.schema(COMPONENT_SHEET)
    existing = _existing_names(workbook, COMPONENT_SHEET)
    records: list[dict[str | int, Any]] = []
    declared: list[str] = []
    for material in materials:
        if material.name.strip().casefold() in existing:
            continue  # already in the template; referencing it is enough
        record: dict[str | int, Any] = {
            "label:Use": "Yes",
            "label:Physical Properties System": "Physical Properties System",
            "label:Materials": "Materials",
            "label:Name": material.name,
        }
        if schema.has("CreationTemplate"):
            record["CreationTemplate"] = PROPERTY_METHOD_TEMPLATE
        records.append(record)
        declared.append(material.name)
    return _to_indexed(records, schema), declared


def _mixture_records(
    workbook: SafetiWorkbook, materials: list[Material], report: WriteReport
) -> tuple[list[dict[int, Any]], list[str]]:
    if MIXTURE_SHEET not in workbook.sheet_names:
        report.warnings.append(f"template has no {MIXTURE_SHEET} sheet — mixtures not declared")
        return [], []

    schema = workbook.schema(MIXTURE_SHEET)
    existing = _existing_names(workbook, MIXTURE_SHEET)
    records: list[dict[str | int, Any]] = []
    declared: list[str] = []
    for material in materials:
        if material.name.strip().casefold() in existing:
            continue
        fraction_column = "XLSMass" if material.composition_basis == "mass" else "XLSMole"
        for index, component in enumerate(material.normalised_components()):
            if index == 0:
                record: dict[str | int, Any] = {
                    "label:Use": "Yes",
                    "label:Physical Properties System": "Physical Properties System",
                    "label:Materials": "Materials",
                    "label:Name": material.name,
                    "XLSComponent": component.component,
                    fraction_column: round(component.fraction, 6),
                }
                if schema.has("CreationTemplate"):
                    record["CreationTemplate"] = PROPERTY_METHOD_TEMPLATE
            else:
                # Continuation row: component and fraction only, exactly as
                # Safeti writes the remaining components of a stream.
                record = {
                    "XLSComponent": component.component,
                    fraction_column: round(component.fraction, 6),
                }
            records.append(record)
        declared.append(material.name)
    return _to_indexed(records, schema), declared


def _existing_names(workbook: SafetiWorkbook, sheet: str) -> set[str]:
    """Material names already present in a sheet, so template rows are kept."""
    names: set[str] = set()
    for row in workbook.data_rows(sheet):
        value = row.get("Name")
        if value:
            names.add(str(value).strip().casefold())
    return names
