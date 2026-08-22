"""Schema reader for Safeti Excel interchange workbooks.

Read-only by design. Writing goes through :mod:`nncm.phast.patcher`, which
edits the workbook's XML in place — re-saving a template through openpyxl
produces a file Phast will not import.


Every sheet in a Safeti input workbook carries its own machine-readable header:

===  ==========================================================================
r1   group header (e.g. " Material ")
r2   sub-group header
r3   entity class the column belongs to (``OnshoreOperatingConditions``)
r4   attribute code (``Temperature``) — stable across Safeti versions
r5   display label (``Temperature``/``Pressure (gauge)``) — what users read
r6   unit for the column (``degC``, ``bar``, ``mm``)
r7+  per-column pick-lists (enumeration values and alternative units)
===  ==========================================================================

``A7`` holds the first data row (63 in the shipped template) and ``A8`` the
index of the first attribute column, so the data region is discovered rather
than hard-coded. Addressing columns by *attribute code* — falling back to the
display label — is what keeps the writer working when the template shifts
columns between Safeti releases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl

HEADER_ENTITY_ROW = 3
HEADER_ATTRIBUTE_ROW = 4
HEADER_LABEL_ROW = 5
HEADER_UNIT_ROW = 6
FIRST_OPTION_ROW = 7
DEFAULT_DATA_START_ROW = 63

# Safeti enumerations are "<code> <description>" ("0 No", "1 Pressure/temperature"),
# plus the bare Yes/No used by the "Use" column.
_ENUM_RE = re.compile(r"^(\d+\s+\S|Yes$|No$)")


class SchemaError(RuntimeError):
    """Raised when a workbook does not look like a Safeti interchange sheet."""


@dataclass(frozen=True)
class ColumnSpec:
    index: int          # 1-based Excel column index
    entity: str         # row 3
    attribute: str      # row 4
    label: str          # row 5
    unit: str           # row 6
    options: tuple[str, ...] = ()  # rows 7+: pick-list of accepted values

    @property
    def key(self) -> str:
        return self.attribute or self.label

    @property
    def enumerations(self) -> tuple[str, ...]:
        """Pick-list entries that are Safeti enumerations (``0 No``, ``Yes``).

        The same block also lists alternative units for numeric columns, which
        are not values a data cell may take.
        """
        return tuple(o for o in self.options if _ENUM_RE.match(o))


@dataclass
class SheetSchema:
    name: str
    data_start_row: int
    first_attribute_col: int
    columns: list[ColumnSpec]
    _by_attribute: dict[str, list[ColumnSpec]] = field(default_factory=dict, repr=False)
    _by_label: dict[str, list[ColumnSpec]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for column in self.columns:
            if column.attribute:
                self._by_attribute.setdefault(column.attribute.casefold(), []).append(column)
            if column.label:
                self._by_label.setdefault(column.label.casefold(), []).append(column)

    # -- lookup -----------------------------------------------------------
    def column(self, key: str | int, entity: str | None = None) -> ColumnSpec:
        """Resolve a column by index, attribute code, or display label.

        Keys may be prefixed to force a lookup mode: ``attr:Temperature`` or
        ``label:Pressure (gauge)``. Ambiguous labels (the sheets repeat
        ``Folder`` many times) must be disambiguated with ``entity`` or an
        explicit index.
        """
        if isinstance(key, int):
            match = [c for c in self.columns if c.index == key]
            if not match:
                raise KeyError(f"{self.name}: no column at index {key}")
            return match[0]

        mode, _, name = key.partition(":")
        if mode not in {"attr", "label"}:
            mode, name = "auto", key

        candidates: list[ColumnSpec] = []
        if mode in {"auto", "attr"}:
            candidates = list(self._by_attribute.get(name.casefold(), []))
        if not candidates and mode in {"auto", "label"}:
            candidates = list(self._by_label.get(name.casefold(), []))
        if entity:
            candidates = [c for c in candidates if c.entity.casefold() == entity.casefold()]
        if not candidates:
            raise KeyError(f"{self.name}: no column matching '{key}'" + (f" (entity={entity})" if entity else ""))
        if len(candidates) > 1:
            detail = ", ".join(f"{c.index}:{c.entity}" for c in candidates)
            raise KeyError(f"{self.name}: '{key}' is ambiguous ({detail}); pass entity= or a column index")
        return candidates[0]

    def has(self, key: str | int, entity: str | None = None) -> bool:
        try:
            self.column(key, entity)
        except KeyError:
            return False
        return True

    def unit(self, key: str | int, entity: str | None = None) -> str:
        return self.column(key, entity).unit

    def describe(self) -> str:
        lines = [f"{self.name}: data starts at row {self.data_start_row}, {len(self.columns)} columns"]
        lines += [
            f"  {c.index:>4} {c.attribute or '-':<32} {c.label:<45} {c.unit}"
            for c in self.columns
            if c.attribute or c.label
        ]
        return "\n".join(lines)


class SafetiWorkbook:
    """Read/modify a Safeti interchange workbook while preserving its layout."""

    def __init__(self, path: Path, read_only: bool = False):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"workbook not found: {self.path}")
        self._read_only = read_only
        self.workbook = openpyxl.load_workbook(
            self.path, data_only=read_only, read_only=read_only
        )
        self._schemas: dict[str, SheetSchema] = {}

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "SafetiWorkbook":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.workbook.close()

    @property
    def sheet_names(self) -> list[str]:
        return list(self.workbook.sheetnames)

    # -- schema ------------------------------------------------------------
    def schema(self, sheet_name: str) -> SheetSchema:
        if sheet_name in self._schemas:
            return self._schemas[sheet_name]
        if sheet_name not in self.workbook.sheetnames:
            raise SchemaError(f"sheet '{sheet_name}' not in {self.path.name}")
        worksheet = self.workbook[sheet_name]

        # Read the whole header block: rows 1-6 describe the columns, rows 7 up
        # to the first data row hold each column's pick-list.
        header: dict[int, list[Any]] = {}
        for row_idx, row in enumerate(
            worksheet.iter_rows(min_row=1, max_row=DEFAULT_DATA_START_ROW - 1, values_only=True),
            start=1,
        ):
            header[row_idx] = list(row)

        def cell(row: int, col_zero_based: int) -> str:
            values = header.get(row, [])
            if col_zero_based >= len(values):
                return ""
            value = values[col_zero_based]
            return "" if value is None else str(value).strip()

        if cell(HEADER_ENTITY_ROW, 0) != "Safeti":
            raise SchemaError(
                f"sheet '{sheet_name}' does not carry a Safeti header (A3 = '{cell(HEADER_ENTITY_ROW, 0)}')"
            )

        data_start_row = _as_int(header.get(7, [None])[0], DEFAULT_DATA_START_ROW)
        first_attribute_col = _as_int(header.get(8, [None])[0], 2)

        width = max((len(v) for v in header.values()), default=0)

        def options(col: int) -> tuple[str, ...]:
            values = []
            for row_idx in range(FIRST_OPTION_ROW, data_start_row):
                value = cell(row_idx, col)
                if value:
                    values.append(value)
            return tuple(values)

        columns = [
            ColumnSpec(
                index=col + 1,
                entity=cell(HEADER_ENTITY_ROW, col),
                attribute=cell(HEADER_ATTRIBUTE_ROW, col),
                label=cell(HEADER_LABEL_ROW, col),
                unit=cell(HEADER_UNIT_ROW, col),
                options=options(col),
            )
            for col in range(width)
        ]
        # Row 4 column A holds the Safeti file version, not an attribute code.
        columns = [
            c if c.index != 1 else ColumnSpec(1, c.entity, "", c.label, c.unit, c.options)
            for c in columns
        ]

        schema = SheetSchema(sheet_name, data_start_row, first_attribute_col, columns)
        self._schemas[sheet_name] = schema
        return schema

    # -- data --------------------------------------------------------------
    def data_rows(self, sheet_name: str) -> list[dict[str, Any]]:
        """Existing populated rows of a sheet, keyed by attribute/label."""
        schema = self.schema(sheet_name)
        worksheet = self.workbook[sheet_name]
        rows: list[dict[str, Any]] = []
        for row in worksheet.iter_rows(min_row=schema.data_start_row, values_only=True):
            if not any(v is not None and str(v).strip() != "" for v in row):
                continue
            record: dict[str, Any] = {}
            for column in schema.columns:
                if column.index - 1 >= len(row):
                    continue
                value = row[column.index - 1]
                if value is None or str(value).strip() == "":
                    continue
                record[column.key or f"col{column.index}"] = value
            rows.append(record)
        return rows

    def first_free_row(self, sheet_name: str) -> int:
        schema = self.schema(sheet_name)
        worksheet = self.workbook[sheet_name]
        last = schema.data_start_row - 1
        for row_idx, row in enumerate(
            worksheet.iter_rows(min_row=schema.data_start_row, values_only=True),
            start=schema.data_start_row,
        ):
            if any(v is not None and str(v).strip() != "" for v in row):
                last = row_idx
        return last + 1


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
