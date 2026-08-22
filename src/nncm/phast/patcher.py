"""Add rows to a Safeti template workbook without rewriting the rest of it.

Loading the template with openpyxl and saving it produces a *different* file:
every string becomes an inline string, ``xl/sharedStrings.xml`` disappears,
``fileVersion`` is dropped, and all 61 comment/VML parts are renamed. Excel
opens the result happily; Phast's importer does not.

So the writer never round-trips the workbook. It copies the template's zip
entries byte-for-byte and rewrites only:

* the worksheet parts that receive rows (new ``<row>`` elements appended to
  ``<sheetData>``, plus an updated ``<dimension>``), and
* ``xl/sharedStrings.xml``, extended with any string we add.

Everything else — styles, data validations, merged cells, comments, VML
drawings, content types, document properties — is passed through untouched.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKBOOK_PART = "xl/workbook.xml"
WORKBOOK_RELS_PART = "xl/_rels/workbook.xml.rels"
CONTENT_TYPES_PART = "[Content_Types].xml"
SHARED_STRINGS_PART = "xl/sharedStrings.xml"
SHARED_STRINGS_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
)
SHARED_STRINGS_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
)

_SHEET_RE = re.compile(r"<sheet\b[^>]*/>")
_RELATIONSHIP_RE = re.compile(r"<Relationship\b[^>]*/>")
_ATTR_RE = re.compile(r'([\w:]+)="([^"]*)"')
_SI_RE = re.compile(r"<si\b.*?</si>|<si\b[^>]*/>", re.S)
_TEXT_RE = re.compile(r"<t\b[^>]*>(.*?)</t>", re.S)
_ROW_RE = re.compile(r'<row\b[^>]*\br="(\d+)"')
_DIMENSION_RE = re.compile(r"<dimension\b[^>]*/>")
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class PatchError(RuntimeError):
    """Raised when the template does not have the structure we need to patch."""


@dataclass
class PatchReport:
    parts_rewritten: list[str] = field(default_factory=list)
    parts_copied: int = 0
    rows_added: dict[str, int] = field(default_factory=dict)
    strings_added: int = 0

    def summary(self) -> str:
        rows = ", ".join(f"{sheet}: {count}" for sheet, count in self.rows_added.items())
        return (
            f"rows added — {rows or 'none'}; "
            f"{len(self.parts_rewritten)} parts rewritten, {self.parts_copied} copied verbatim"
        )


def column_letter(index: int) -> str:
    """1 -> A, 27 -> AA."""
    if index < 1:
        raise ValueError(f"column index must be >= 1 (got {index})")
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def column_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - 64)
    return index


def _escape(text: str) -> str:
    text = _ILLEGAL_XML.sub("", str(text))
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_number(value: float) -> str:
    """Compact but lossless-enough numeric text (Excel keeps 15 digits)."""
    if isinstance(value, bool):  # bool is an int subclass — handled by the caller
        raise TypeError("bools are written as t=\"b\"")
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.12g}"
    return "0" if text in {"-0", "-0.0"} else text


class TemplatePatcher:
    """Append data rows to a Safeti workbook, preserving every other byte."""

    def __init__(self, template_path: Path):
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"template not found: {self.template_path}")
        self._zip = zipfile.ZipFile(self.template_path)
        self._names = self._zip.namelist()
        self._patched: dict[str, bytes] = {}
        self._sheet_parts = self._map_sheets()
        self._load_shared_strings()
        self.report = PatchReport()

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "TemplatePatcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._zip.close()

    # -- structure ---------------------------------------------------------
    def _map_sheets(self) -> dict[str, str]:
        workbook = self._read_text(WORKBOOK_PART)
        rels = self._read_text(WORKBOOK_RELS_PART)
        targets: dict[str, str] = {}
        for relationship in _RELATIONSHIP_RE.findall(rels):
            attrs = dict(_ATTR_RE.findall(relationship))
            if "Id" in attrs and "Target" in attrs:
                targets[attrs["Id"]] = attrs["Target"]

        sheets: dict[str, str] = {}
        for sheet in _SHEET_RE.findall(workbook):
            attrs = dict(_ATTR_RE.findall(sheet))
            name = attrs.get("name")
            rid = attrs.get("r:id") or attrs.get("relationships:id")
            if not name or rid not in targets:
                continue
            target = targets[rid].lstrip("/")
            sheets[name] = target if target.startswith("xl/") else f"xl/{target}"
        if not sheets:
            raise PatchError(f"{self.template_path.name}: no worksheets found")
        return sheets

    def _read_text(self, part: str) -> str:
        source = self._patched.get(part)
        data = source if source is not None else self._zip.read(part)
        return data.decode("utf-8")

    def sheet_part(self, sheet_name: str) -> str:
        if sheet_name not in self._sheet_parts:
            raise PatchError(f"sheet '{sheet_name}' not in {self.template_path.name}")
        return self._sheet_parts[sheet_name]

    def existing_max_row(self, sheet_name: str) -> int:
        xml = self._read_text(self.sheet_part(sheet_name))
        rows = [int(r) for r in _ROW_RE.findall(xml)]
        return max(rows) if rows else 0

    # -- shared strings ----------------------------------------------------
    def _load_shared_strings(self) -> None:
        self._string_index: dict[str, int] = {}
        self._new_strings: list[str] = []
        self._extra_string_cells = 0

        if SHARED_STRINGS_PART not in self._names:
            self._has_shared_strings = False
            self._existing_string_count = 0
            self._unique_count = 0
            return

        self._has_shared_strings = True
        xml = self._read_text(SHARED_STRINGS_PART)
        # Match the <sst> element itself — the file opens with an XML
        # declaration, so the first '>' belongs to that, not to <sst>.
        header_match = re.search(r"<sst\b[^>]*>", xml)
        if header_match is None:
            raise PatchError("sharedStrings.xml has no <sst> element")
        attrs = dict(_ATTR_RE.findall(header_match.group(0)))
        self._existing_string_count = int(attrs.get("count", 0) or 0)

        items = _SI_RE.findall(xml)
        self._unique_count = len(items)
        for index, item in enumerate(items):
            # Only reuse plain-text entries: rich-text entries carry formatting
            # runs we do not want to inherit into a data cell.
            if "<r>" in item or "<r " in item:
                continue
            texts = _TEXT_RE.findall(item)
            if len(texts) == 1:
                self._string_index.setdefault(_unescape(texts[0]), index)

    def _string_id(self, text: str) -> int:
        if text in self._string_index:
            return self._string_index[text]
        index = self._unique_count + len(self._new_strings)
        self._new_strings.append(text)
        self._string_index[text] = index
        return index

    def _build_shared_strings(self) -> bytes:
        xml = self._read_text(SHARED_STRINGS_PART) if self._has_shared_strings else (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'count="0" uniqueCount="0"></sst>'
        )
        additions = "".join(
            f"<si><t xml:space=\"preserve\">{_escape(text)}</t></si>" for text in self._new_strings
        )
        if "<sst" not in xml:
            raise PatchError("sharedStrings.xml has no <sst> element")

        if re.search(r"<sst\b[^>]*/>", xml):  # empty table written self-closed
            xml = re.sub(
                r"<sst\b([^>]*)/>",
                lambda m: f"<sst{m.group(1)}>{additions}</sst>",
                xml,
                count=1,
            )
        else:
            xml = xml.replace("</sst>", f"{additions}</sst>")

        count = self._existing_string_count + self._extra_string_cells
        unique = self._unique_count + len(self._new_strings)
        header_match = re.search(r"<sst\b[^>]*>", xml)
        header = header_match.group(0)
        updated = _set_attribute(_set_attribute(header, "count", str(count)), "uniqueCount", str(unique))
        return (xml[: header_match.start()] + updated + xml[header_match.end() :]).encode("utf-8")

    # -- row writing -------------------------------------------------------
    def add_rows(
        self,
        sheet_name: str,
        rows: list[dict[int, Any]],
        start_row: int | None = None,
    ) -> int:
        """Append ``rows`` (keyed by 1-based column index) to a sheet."""
        if not rows:
            return 0
        part = self.sheet_part(sheet_name)
        xml = self._read_text(part)

        first_row = start_row if start_row is not None else self.existing_max_row(sheet_name) + 1
        existing_max = self.existing_max_row(sheet_name)
        if first_row <= existing_max:
            raise PatchError(
                f"{sheet_name}: refusing to write over existing data "
                f"(row {first_row} <= last populated row {existing_max})"
            )

        chunks: list[str] = []
        max_column = 0
        for offset, row in enumerate(rows):
            row_number = first_row + offset
            cells = []
            columns = [
                c
                for c in sorted(row)
                if row[c] is not None and not (isinstance(row[c], str) and row[c] == "")
            ]
            for column in columns:
                cells.append(self._cell_xml(column_letter(column) + str(row_number), row[column]))
            if columns:
                max_column = max(max_column, columns[-1])
            # `spans` is the hint Excel itself writes; harmless but keeps the
            # generated rows structurally identical to hand-made ones.
            spans = f' spans="{columns[0]}:{columns[-1]}"' if columns else ""
            chunks.append(f'<row r="{row_number}"{spans}>{"".join(cells)}</row>')

        block = "".join(chunks)
        if "</sheetData>" in xml:
            xml = xml.replace("</sheetData>", f"{block}</sheetData>", 1)
        elif re.search(r"<sheetData\b[^>]*/>", xml):
            xml = re.sub(r"<sheetData\b([^>]*)/>", lambda m: f"<sheetData{m.group(1)}>{block}</sheetData>", xml, count=1)
        else:
            raise PatchError(f"{sheet_name}: no <sheetData> element to extend")

        xml = _extend_dimension(xml, max_column, first_row + len(rows) - 1)
        self._patched[part] = xml.encode("utf-8")
        self.report.rows_added[sheet_name] = self.report.rows_added.get(sheet_name, 0) + len(rows)
        return len(rows)

    def _cell_xml(self, reference: str, value: Any) -> str:
        if isinstance(value, bool):
            return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
        if isinstance(value, (int, float)):
            return f'<c r="{reference}"><v>{_format_number(value)}</v></c>'
        text = str(value)
        if self._has_shared_strings:
            self._extra_string_cells += 1
            return f'<c r="{reference}" t="s"><v>{self._string_id(text)}</v></c>'
        return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{_escape(text)}</t></is></c>'

    # -- output ------------------------------------------------------------
    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._patched:
            # Nothing to change: hand back an exact copy of the template.
            shutil.copyfile(self.template_path, output_path)
            self.report.parts_copied = len(self._names)
            return output_path

        if self._new_strings or self._extra_string_cells:
            self._patched[SHARED_STRINGS_PART] = self._build_shared_strings()
            self.report.strings_added = len(self._new_strings)

        parts = dict(self._patched)
        names = list(self._names)
        if SHARED_STRINGS_PART in parts and SHARED_STRINGS_PART not in names:
            names.append(SHARED_STRINGS_PART)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as target:
            for name in names:
                if name in parts:
                    info = _info_for(self._zip, name)
                    target.writestr(info, parts[name])
                    self.report.parts_rewritten.append(name)
                else:
                    info = self._zip.getinfo(name)
                    target.writestr(info, self._zip.read(name))
                    self.report.parts_copied += 1

            if SHARED_STRINGS_PART in parts and SHARED_STRINGS_PART not in self._names:
                self._register_shared_strings_part(target)
        return output_path

    def _register_shared_strings_part(self, target: zipfile.ZipFile) -> None:
        """Declare a newly created shared-string table (rare: template had none)."""
        content_types = self._read_text(CONTENT_TYPES_PART)
        if SHARED_STRINGS_PART not in content_types:
            override = (
                f'<Override PartName="/{SHARED_STRINGS_PART}" ContentType="{SHARED_STRINGS_TYPE}"/>'
            )
            content_types = content_types.replace("</Types>", f"{override}</Types>")
            target.writestr(CONTENT_TYPES_PART, content_types.encode("utf-8"))

        rels = self._read_text(WORKBOOK_RELS_PART)
        if "sharedStrings.xml" not in rels:
            used = {int(m) for m in re.findall(r'Id="rId(\d+)"', rels)}
            new_id = f"rId{max(used) + 1 if used else 1}"
            relationship = (
                f'<Relationship Id="{new_id}" Type="{SHARED_STRINGS_REL}" Target="sharedStrings.xml"/>'
            )
            rels = rels.replace("</Relationships>", f"{relationship}</Relationships>")
            target.writestr(WORKBOOK_RELS_PART, rels.encode("utf-8"))


def _info_for(source: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    """Reuse the template entry's metadata so only the payload differs."""
    original = source.getinfo(name)
    info = zipfile.ZipInfo(name, date_time=original.date_time)
    info.compress_type = original.compress_type
    info.external_attr = original.external_attr
    info.internal_attr = original.internal_attr
    info.create_system = original.create_system
    return info


def _set_attribute(tag: str, name: str, value: str) -> str:
    if re.search(rf'\b{name}="[^"]*"', tag):
        return re.sub(rf'\b{name}="[^"]*"', f'{name}="{value}"', tag, count=1)
    return tag[:-1].rstrip() + f' {name}="{value}"' + tag[-1]


def _extend_dimension(xml: str, max_column: int, max_row: int) -> str:
    match = _DIMENSION_RE.search(xml)
    if not match:
        return xml
    attrs = dict(_ATTR_RE.findall(match.group(0)))
    ref = attrs.get("ref", "")
    if ":" not in ref:
        return xml
    start, end = ref.split(":", 1)
    end_letters = re.match(r"([A-Z]+)(\d+)", end)
    if not end_letters:
        return xml
    current_column = column_index(end_letters.group(1))
    current_row = int(end_letters.group(2))
    new_ref = f"{start}:{column_letter(max(current_column, max_column))}{max(current_row, max_row)}"
    return xml[: match.start()] + f'<dimension ref="{new_ref}"/>' + xml[match.end() :]


def _unescape(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
