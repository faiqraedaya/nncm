"""Extract a training table directly from a Phast/Safeti result workbook.

Phast exports one sheet per consequence type (``Discharge``, ``Flammable
Dispersion``, ``Jet fire``, ...). Each row is one *scenario x weather*
combination, identified by a backslash-delimited ``Path`` whose last segment is
the scenario (leak) name. This module joins those sheets on
``(scenario_key, weather, hole_size)``, decodes the weather category into
numeric features, and merges back onto the sampled case table so the true
inputs — including material — travel with every target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExtractionConfig

# Path + scenario + weather uniquely identifies a result row. The scenario name
# alone collides in real projects (the same equipment name recurs under several
# studies), and path+weather collides for time-varying releases, which export
# one row per release segment under a single path.
KEY_COLUMNS = ["path_key", "scenario_label", "weather"]

# Pasquill stability class -> ordinal index used as a model feature.
STABILITY_INDEX = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

_WEATHER_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*/\s*([A-F])", re.IGNORECASE)


@dataclass
class TargetSource:
    """One extracted quantity: which sheet and column it comes from.

    ``column`` may be the exact header or a prefix of it. Headers that carry a
    threshold — ``Distance downwind to intensity level 1 (6.3 kW/m2) (m)`` —
    differ between studies because the level is a study parameter, so matching
    ignores the parenthetical part. The threshold actually used is recorded in
    the extraction report.
    """

    name: str
    sheet: str
    column: str
    required: bool = False


DEFAULT_TARGETS: list[TargetSource] = [
    TargetSource("Release_rate", "Discharge", "Peak Flowrate (kg/s)", required=True),
    TargetSource("Velocity", "Discharge", "Velocity (m/s)", required=True),
    TargetSource("Release_temperature", "Discharge", "Temperature (degC)"),
    TargetSource("Liquid_mass_fraction", "Discharge", "Liquid mass fraction in material (fraction)"),
    TargetSource("Droplet_diameter", "Discharge", "Droplet diameter (um)"),
    TargetSource("Expanded_diameter", "Discharge", "Expanded diameter (m)"),
    TargetSource("Release_duration", "Discharge", "End time of release (s)"),
    TargetSource("Distance_to_LFL", "Flammable Dispersion", "Distance to LFL (m)"),
    TargetSource("Distance_to_LFL_fraction", "Flammable Dispersion", "Distance to LFL fraction (m)"),
    TargetSource("Distance_to_UFL", "Flammable Dispersion", "Distance to UFL (m)"),
    TargetSource("Flash_fire_mass", "Flammable Dispersion", "Max flash fire flammable mass (kg)"),
    TargetSource("Flame_length", "Jet fire", "Flame length (m)"),
    TargetSource("Flame_emissive_power", "Jet fire", "Flame emissive power (kW/m2)"),
    TargetSource("Jet_fire_distance_level1", "Jet fire", "Distance downwind to intensity level 1"),
    TargetSource("Jet_fire_distance_level2", "Jet fire", "Distance downwind to intensity level 2"),
    TargetSource("Jet_fire_distance_level3", "Jet fire", "Distance downwind to intensity level 3"),
    TargetSource("Pool_diameter", "Early Pool Fire", "Pool diameter (m)"),
    TargetSource("Explosion_distance_level1", "Explosions", "Distance downwind to overpressure 1"),
    TargetSource("Explosion_distance_level2", "Explosions", "Distance downwind to overpressure 2"),
    TargetSource("Explosion_distance_level3", "Explosions", "Distance downwind to overpressure 3"),
]

INPUT_ECHO = {
    "Temperature (input) (degC)": "temperature_degC",
    "Pressure (input) (bar)": "pressure_barg",
    "Hole size (mm)": "orifice_mm",
    "Material": "material_id",
}


@dataclass
class ExtractionReport:
    sheets_found: list[str] = field(default_factory=list)
    rows_per_sheet: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0
    n_matched_cases: int = 0
    n_unmatched: int = 0
    matched_by: dict[str, int] = field(default_factory=dict)
    column_map: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, str] = field(default_factory=dict)
    dropped_missing: int = 0
    dropped_nonpositive: int = 0
    target_coverage: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"sheets read: {', '.join(self.sheets_found) or 'none'}",
            f"rows after join: {self.n_rows}",
            f"matched to sampled cases: {self.n_matched_cases} (unmatched {self.n_unmatched})"
            + (
                f" — by leak name {self.matched_by.get('leak', 0)}, "
                f"by vessel name {self.matched_by.get('vessel', 0)}"
                if self.matched_by
                else ""
            ),
            f"dropped — missing required targets: {self.dropped_missing}, non-positive: {self.dropped_nonpositive}",
            "target coverage:",
        ]
        lines += [
            f"  {name:<28} {cov * 100:5.1f} %"
            for name, cov in sorted(self.target_coverage.items(), key=lambda kv: -kv[1])
        ]
        if self.thresholds:
            lines.append(
                "thresholds used: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.thresholds.items()))
            )
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def parse_weather(value: Any) -> tuple[float, str]:
    """``'Category 5/D'`` -> ``(5.0, 'D')``. Returns ``(nan, '')`` if unparsable."""
    if value is None:
        return float("nan"), ""
    match = _WEATHER_RE.search(str(value))
    if not match:
        return float("nan"), ""
    return float(match.group(1)), match.group(2).upper()


def normalise_path(path: Any) -> str:
    """Canonical form of a Phast result ``Path`` (separator- and space-insensitive)."""
    if path is None:
        return ""
    return "\\".join(part.strip() for part in str(path).replace("/", "\\").strip("\\ ").split("\\"))


def scenario_key(path: Any) -> str:
    """Last segment of a Phast result ``Path`` — the deepest named object."""
    normalised = normalise_path(path)
    return normalised.split("\\")[-1] if normalised else ""


def identifiers(path: Any, scenario_label: Any) -> list[str]:
    """Every name a result row could be identified by, most specific first.

    Where the equipment sits in the study decides how Phast reports it. In a
    study with route/scenario groups the path runs all the way down to the leak
    and ``Scenario`` holds the hole size; in a flat study the path stops at the
    vessel and ``Scenario`` holds the leak name. Matching against both — plus
    every intermediate path segment — makes the join independent of that.
    """
    names: list[str] = []
    label = "" if scenario_label is None else str(scenario_label).strip()
    if label:
        names.append(label)
    normalised = normalise_path(path)
    if normalised:
        names.extend(reversed(normalised.split("\\")))
    seen: set[str] = set()
    return [n for n in names if n and not (n in seen or seen.add(n))]


def _normalise_header(name: str) -> str:
    """Header without its parenthetical units/thresholds, for tolerant matching."""
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", str(name))).strip().casefold()


def resolve_column(frame: pd.DataFrame, wanted: str) -> str | None:
    """Find ``wanted`` in ``frame``: exact header first, then threshold-agnostic."""
    if wanted in frame.columns:
        return wanted
    target = _normalise_header(wanted)
    matches = [c for c in frame.columns if _normalise_header(c) == target]
    if matches:
        return matches[0]
    prefixed = [c for c in frame.columns if _normalise_header(c).startswith(target)]
    return prefixed[0] if len(prefixed) == 1 else None


def threshold_of(header: str) -> str:
    """The threshold a result column was computed at, e.g. ``4 kW/m2``."""
    for group in re.findall(r"\(([^)]*)\)", str(header)):
        text = group.strip()
        if any(ch.isdigit() for ch in text) and text.lower() not in {"m", "m2", "kg", "s"}:
            return text
    return ""


def read_output_sheets(path: Path, sheets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read result sheets into tidy frames keyed by scenario/weather/hole size."""
    path = Path(path)
    frames: dict[str, pd.DataFrame] = {}
    excel = pd.ExcelFile(path, engine="openpyxl")
    wanted = sheets or excel.sheet_names
    for sheet in wanted:
        if sheet not in excel.sheet_names:
            continue
        frame = excel.parse(sheet)
        if frame.empty or "Path" not in frame.columns:
            continue
        frame = frame.copy()
        frame["path_key"] = frame["Path"].map(normalise_path)
        frame["scenario_key"] = frame["Path"].map(scenario_key)
        frame["scenario_label"] = (
            frame["Scenario"].astype(str).str.strip() if "Scenario" in frame else ""
        )
        frame["weather"] = frame["Weather"].astype(str).str.strip() if "Weather" in frame else ""
        if "Hole size (mm)" in frame.columns:
            frame["hole_size_mm"] = pd.to_numeric(frame["Hole size (mm)"], errors="coerce")
        else:
            frame["hole_size_mm"] = np.nan
        frames[sheet] = frame
    excel.close()
    return frames


def build_training_table(
    output_path: Path,
    cases: pd.DataFrame | None = None,
    config: ExtractionConfig | None = None,
    targets: list[TargetSource] | None = None,
) -> tuple[pd.DataFrame, ExtractionReport]:
    """Join Phast result sheets into one training row per scenario x weather."""
    config = config or ExtractionConfig()
    targets = targets or DEFAULT_TARGETS
    report = ExtractionReport()

    needed_sheets = list(dict.fromkeys(t.sheet for t in targets))
    frames = read_output_sheets(output_path, needed_sheets)
    report.sheets_found = list(frames)
    report.rows_per_sheet = {name: len(f) for name, f in frames.items()}
    missing_sheets = [s for s in needed_sheets if s not in frames]
    if missing_sheets:
        report.warnings.append(f"result sheets absent from workbook: {', '.join(missing_sheets)}")

    if not frames:
        raise ValueError(f"no usable result sheets in {output_path}")

    # Scenario universe = every (path, weather) seen anywhere, so scenarios that
    # only appear on one sheet (BLEVE, pool fire) are not silently dropped.
    identity_columns = KEY_COLUMNS + ["scenario_key", "hole_size_mm"] + list(INPUT_ECHO)
    identity_frames = [
        frame[[c for c in identity_columns if c in frame.columns]] for frame in frames.values()
    ]
    merged = (
        pd.concat(identity_frames, ignore_index=True)
        .sort_values(KEY_COLUMNS)
        .groupby(KEY_COLUMNS, as_index=False)
        .first()
    )
    # Input echo columns give a fallback when the sampled case table is absent.
    merged = merged.rename(columns={k: v for k, v in INPUT_ECHO.items() if k in merged.columns})

    for sheet, group in _group_by_sheet(targets):
        if sheet not in frames:
            continue
        frame = frames[sheet]
        resolved = {t.name: resolve_column(frame, t.column) for t in group}
        available = [t for t in group if resolved[t.name]]
        for missing in [t for t in group if not resolved[t.name]]:
            report.warnings.append(f"'{missing.column}' not found in sheet '{sheet}'")
        if not available:
            continue
        for target in available:
            source = resolved[target.name]
            report.column_map[target.name] = source
            threshold = threshold_of(source)
            if threshold:
                report.thresholds[target.name] = threshold
        columns = KEY_COLUMNS + [resolved[t.name] for t in available]
        subset = frame[columns].copy()
        duplicates = subset.duplicated(KEY_COLUMNS).sum()
        if duplicates:
            report.warnings.append(
                f"sheet '{sheet}': {duplicates} duplicate scenario/weather rows — keeping the worst case"
            )
            for t in available:
                subset[resolved[t.name]] = pd.to_numeric(subset[resolved[t.name]], errors="coerce")
            subset = subset.groupby(KEY_COLUMNS, as_index=False).max()
        subset = subset.rename(columns={resolved[t.name]: t.name for t in available})
        merged = merged.merge(subset, on=KEY_COLUMNS, how="left")

    for target in targets:
        if target.name in merged.columns:
            merged[target.name] = pd.to_numeric(merged[target.name], errors="coerce")

    weather_parsed = merged["weather"].map(parse_weather)
    merged["wind_speed_ms"] = [w[0] for w in weather_parsed]
    merged["stability_class"] = [w[1] for w in weather_parsed]
    merged["stability_index"] = merged["stability_class"].map(STABILITY_INDEX).astype("float64")

    if config.weather_filter:
        keep = merged["weather"].isin(config.weather_filter)
        merged = merged[keep].reset_index(drop=True)

    # Attach the sampled inputs (material, exact T/P/d, vessel grouping).
    if cases is not None and not cases.empty:
        merged, matched_by = _join_cases(merged, cases)
        report.n_matched_cases = int(merged["leak_name"].notna().sum())
        report.n_unmatched = int(len(merged) - report.n_matched_cases)
        report.matched_by = matched_by
        if report.n_unmatched:
            report.warnings.append(
                f"{report.n_unmatched} result rows did not match any sampled case "
                "(renamed in Phast, or results from a different study)"
            )
            if config.drop_unmatched:
                merged = merged[merged["leak_name"].notna()].reset_index(drop=True)
    else:
        report.warnings.append(
            "no case table — inputs, material and vessel grouping come from the result sheets"
        )
    # Rows no case covered still need a grouping key and a material, or the
    # split falls back to random rows and the material is lost.
    _fill_from_results(merged)

    # Quality gates: Phast leaves cells blank for scenarios that did not run.
    before = len(merged)
    required = [c for c in config.require_targets if c in merged.columns]
    if required:
        merged = merged.dropna(subset=required)
    report.dropped_missing = before - len(merged)

    before = len(merged)
    for column in config.drop_nonpositive:
        if column in merged.columns:
            # A missing value is not a non-positive one: a sparse target is
            # masked in training, so its blank rows must survive this gate.
            merged = merged[~(merged[column] <= 0)]
    report.dropped_nonpositive = before - len(merged)

    merged = merged.reset_index(drop=True)
    report.n_rows = len(merged)
    if len(merged):
        report.target_coverage = {
            t.name: float(merged[t.name].notna().mean())
            for t in targets
            if t.name in merged.columns
        }
    return merged, report


def _join_cases(merged: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach each result row to its sampled case.

    Leak names are tried first; a row that only identifies its vessel still
    picks up that vessel's material, conditions and grouping key, with the hole
    size taken from the result sheet.
    """
    case_columns = [c for c in cases.columns if c != "phase_hint"]
    by_leak = {
        str(row["leak_name"]): row for _, row in cases[case_columns].iterrows()
    }
    vessel_columns = [c for c in case_columns if c not in {"leak_name", "case_id", "orifice_mm"}]
    by_vessel = {
        str(name): group.iloc[0][vessel_columns]
        for name, group in cases.groupby("vessel_name", sort=False)
    }

    matched_by = {"leak": 0, "vessel": 0, "none": 0}
    attached: list[dict[str, Any]] = []
    for path, label in zip(merged["path_key"], merged["scenario_label"]):
        record: dict[str, Any] = {}
        for name in identifiers(path, label):
            if name in by_leak:
                record = by_leak[name].to_dict()
                matched_by["leak"] += 1
                break
            if name in by_vessel:
                record = by_vessel[name].to_dict()
                record["leak_name"] = name  # vessel-level match: no leak identity
                matched_by["vessel"] += 1
                break
        else:
            matched_by["none"] += 1
        attached.append(record)

    # Columns fixed up front, so a study with no match at all still yields
    # the case columns (all empty) rather than none.
    case_frame = pd.DataFrame(
        attached, index=merged.index, columns=list(dict.fromkeys([*case_columns, "leak_name"]))
    )
    for column in case_frame.columns:
        if column in {"temperature_degC", "pressure_barg", "orifice_mm"} and column in merged.columns:
            # Sampled values win; the echoed value stays as the fallback.
            merged[column] = case_frame[column].where(case_frame[column].notna(), merged[column])
        else:
            merged[column] = case_frame[column]
    return merged, matched_by


def _fill_from_results(frame: pd.DataFrame) -> None:
    """Backfill grouping key and material for rows the case join did not cover.

    The grouping key exists to keep leaks sharing one vessel's state out of
    both train and test, so the fallback groups on exactly that state —
    material, temperature and pressure as Phast echoes them. The path cannot
    be used: depending on the study layout it ends at the vessel or at the
    leak, and its parent is the vessel or the whole study.
    """
    if frame.empty:
        return
    state_columns = [c for c in ("material_id", "temperature_degC", "pressure_barg") if c in frame.columns]
    if state_columns:
        fallback = frame[state_columns].astype(str).agg("|".join, axis=1)
    else:
        fallback = frame["path_key"]
    fallback = "phast:" + fallback
    if "vessel_id" in frame.columns:
        frame["vessel_id"] = frame["vessel_id"].astype("object").where(
            frame["vessel_id"].notna(), fallback
        )
    else:
        frame["vessel_id"] = fallback

    if "material_id" in frame.columns:
        fallback = frame["material_id"].where(
            frame["material_id"].isna(), frame["material_id"].astype(str).str.strip()
        )
        if "material" in frame.columns:
            frame["material"] = frame["material"].astype("object").where(
                frame["material"].notna(), fallback
            )
        else:
            frame["material"] = fallback


def _group_by_sheet(targets: list[TargetSource]):
    ordered: dict[str, list[TargetSource]] = {}
    for target in targets:
        ordered.setdefault(target.sheet, []).append(target)
    return ordered.items()


def save_training_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
