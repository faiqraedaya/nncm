"""Stage-level operations shared by the CLI and the GUI.

The GUI and the CLI must do *the same thing*; both call these functions rather
than re-implementing each stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Project
from .phast.input_writer import WriteReport, write_input_workbook
from .phast.output_reader import ExtractionReport, build_training_table, save_training_table
from .sampling import SamplingReport, generate_cases, load_cases, plot_case_distributions, save_cases

LogFn = Callable[[str], None]


@dataclass
class StageResult:
    ok: bool
    message: str
    detail: str = ""


def run_sampling(project: Project, log: LogFn | None = None) -> tuple[pd.DataFrame, SamplingReport]:
    emit = log or (lambda m: None)
    project.ensure_dirs()
    cases, report = generate_cases(project.config.sampling)
    save_cases(cases, project.cases_path)
    emit(report.summary())
    emit(f"cases written to {project.cases_path}")
    try:
        plot_case_distributions(cases, project.plots_dir / "case_distributions.png")
        emit(f"design plot written to {project.plots_dir / 'case_distributions.png'}")
    except Exception as exc:  # plotting must never block the pipeline
        emit(f"! could not plot case distributions: {exc}")
    return cases, report


def run_phast_export(
    project: Project,
    cases: pd.DataFrame | None = None,
    output_path: Path | None = None,
    log: LogFn | None = None,
    passthrough: bool = False,
) -> WriteReport:
    emit = log or (lambda m: None)
    project.ensure_dirs()

    if passthrough:
        # Diagnostic: an exact copy of the template with no rows added. If Phast
        # rejects this too, the problem is the template or the Phast version,
        # not anything NNCM writes.
        from .phast.patcher import TemplatePatcher

        target = Path(output_path or project.phast_input_dir / "template_passthrough.xlsx")
        with TemplatePatcher(project.config.phast.template_path()) as patcher:
            saved = patcher.save(target)
        emit(f"wrote an unmodified copy of the template to {saved}")
        return WriteReport(files=[saved], verified=True)

    if cases is None:
        if not project.cases_path.exists():
            raise FileNotFoundError(f"no cases at {project.cases_path}; generate a sample first")
        cases = load_cases(project.cases_path)
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project.phast_input_dir / f"phast_input_{stamp}.xlsx"
    report = write_input_workbook(
        cases,
        Path(output_path),
        project.config.phast,
        materials=project.config.sampling.materials,
    )
    emit(report.summary())
    return report


def run_phast_import(
    project: Project,
    output_workbook: Path,
    log: LogFn | None = None,
    append: bool = False,
) -> tuple[pd.DataFrame, ExtractionReport]:
    """Turn a Phast result workbook into training data joined to the case table."""
    emit = log or (lambda m: None)
    project.ensure_dirs()
    cases = load_cases(project.cases_path) if project.cases_path.exists() else None
    if cases is None:
        emit("! no case table found — falling back to the inputs echoed in the result sheets")
    frame, report = build_training_table(Path(output_workbook), cases, project.config.extraction)
    filled = _attach_material_properties(frame, project.config.sampling.materials)
    if filled:
        emit(f"material descriptors taken from the catalogue for {filled} rows the case table did not cover")

    if append and project.training_data_path.exists():
        existing = pd.read_csv(project.training_data_path)
        before = len(existing)
        frame = pd.concat([existing, frame], ignore_index=True)
        key = [c for c in ("path_key", "scenario_label", "weather") if c in frame.columns]
        if key:
            frame = frame.drop_duplicates(key, keep="last").reset_index(drop=True)
        emit(f"appended to existing dataset ({before} -> {len(frame)} rows)")

    save_training_table(frame, project.training_data_path)
    emit(report.summary())
    emit(f"training data written to {project.training_data_path}")
    return frame, report


def _attach_material_properties(frame: pd.DataFrame, materials) -> int:
    """Fill ``mat_*`` descriptors from the catalogue where the case join left none.

    Without this, rows from results the case table does not cover carry no
    descriptors and training imputes the median fluid for them. Returns the
    number of rows filled.
    """
    if "material" not in frame.columns or frame.empty:
        return 0
    catalogue = {m.name.strip().casefold(): m.properties for m in materials if m.properties}
    names = sorted({k for props in catalogue.values() for k in props})
    columns = [f"mat_{k}" for k in names]
    for column in columns:
        if column not in frame.columns:
            frame[column] = float("nan")
    if not columns:
        return 0
    lacking = frame[columns].isna().all(axis=1)
    keys = frame["material"].astype(str).str.strip().str.casefold()
    known = lacking & keys.isin(catalogue)
    for name, column in zip(names, columns):
        frame.loc[known, column] = keys[known].map(lambda k, n=name: catalogue[k].get(n, float("nan")))
    return int(known.sum())


@dataclass
class DatasetSummary:
    """What the training dataset holds, as figures rather than as a sentence.

    The GUI needs the numbers to fill metric cells and the CLI needs a
    paragraph; both read this, so the two can never disagree about how many
    rows there are.
    """

    rows: int = 0
    columns: int = 0
    vessels: int = 0
    materials: int = 0
    targets: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_text(self) -> str:
        if not self.rows:
            return "no training dataset yet"
        lines = [f"{self.rows} rows, {self.columns} columns"]
        if self.vessels:
            lines.append(f"{self.vessels} vessels (grouping key)")
        if self.materials:
            lines.append(f"{self.materials} materials")
        for target, stats in self.targets.items():
            if not stats:
                lines.append(f"{target}: no values")
                continue
            lines.append(
                f"{target}: n={int(stats['n'])}  min={stats['min']:.4g}  "
                f"median={stats['median']:.4g}  max={stats['max']:.4g}"
            )
        return "\n".join(lines)


def dataset_summary(project: Project) -> DatasetSummary:
    """Read the project dataset and describe it. An absent file is not an error."""
    if not project.training_data_path.exists():
        return DatasetSummary()
    frame = pd.read_csv(project.training_data_path)
    summary = DatasetSummary(rows=len(frame), columns=len(frame.columns))
    if "vessel_id" in frame.columns:
        summary.vessels = int(frame["vessel_id"].nunique())
    if "material" in frame.columns:
        summary.materials = int(frame["material"].nunique())
    for target in project.config.training.targets:
        if target not in frame.columns:
            continue
        series = pd.to_numeric(frame[target], errors="coerce").dropna()
        summary.targets[target] = (
            {}
            if series.empty
            else {
                "n": float(len(series)),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }
        )
    return summary


def dataset_overview(project: Project) -> str:
    return dataset_summary(project).as_text()
