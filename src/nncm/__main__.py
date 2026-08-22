"""CLI entry point — ``nncm <command>`` or ``python -m nncm``.

Every command operates on a *project* directory (``--project``, default
``./workspace``) holding config, cases, workbooks, datasets and model runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import NncmConfig, Project, default_project_root


def _project(args: argparse.Namespace) -> Project:
    root = Path(args.project) if args.project else default_project_root()
    return Project.create(root)


def _log(message: str) -> None:
    print(message)


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.project) if args.project else default_project_root()
    project = Project.create(root, NncmConfig(), overwrite=args.force)
    print(f"project ready at {project.root}")
    print(f"config: {project.config_path}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .pipeline import run_sampling

    project = _project(args)
    if args.vessels:
        project.config.sampling.n_vessels = args.vessels
    if args.leaks:
        project.config.sampling.n_leaks_per_vessel = args.leaks
    if args.seed is not None:
        project.config.sampling.seed = args.seed
    if args.vessels or args.leaks or args.seed is not None:
        project.save_config()
    run_sampling(project, log=_log)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .pipeline import run_phast_export

    project = _project(args)
    if args.template:
        project.config.phast.template = str(Path(args.template).resolve())
    if args.split:
        project.config.phast.max_rows_per_workbook = args.split
    run_phast_export(
        project,
        output_path=Path(args.out) if args.out else None,
        log=_log,
        passthrough=args.passthrough,
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from .pipeline import run_phast_import

    project = _project(args)
    run_phast_import(project, Path(args.workbook), log=_log, append=args.append)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .training import train_model

    project = _project(args)
    config = project.config.training
    if args.epochs:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    train_model(project, config=config, log=_log)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import pandas as pd

    from .predict import ModelBundle, format_quantity, load_latest

    project = _project(args)
    bundle = ModelBundle(Path(args.run)) if args.run else load_latest(project)

    if args.csv:
        frame = pd.read_csv(args.csv)
        predictions = bundle.predict_frame(frame, mc_samples=args.mc)
        out = Path(args.out) if args.out else Path(args.csv).with_name(
            Path(args.csv).stem + "_predictions.csv"
        )
        pd.concat([frame, predictions], axis=1).to_csv(out, index=False)
        print(f"{len(frame)} rows predicted -> {out}")
        return 0

    if args.temperature is None or args.pressure is None or args.orifice is None:
        print("provide --temperature, --pressure and --orifice (or --csv)", file=sys.stderr)
        return 2
    inputs = {
        "temperature_degC": args.temperature,
        "pressure_barg": args.pressure,
        "orifice_mm": args.orifice,
        "wind_speed_ms": args.wind,
        "stability_index": args.stability,
    }
    if args.material:
        inputs["material"] = args.material
    prediction = bundle.predict_one(inputs, mc_samples=args.mc)
    for target, value in prediction.values.items():
        spread = prediction.uncertainty.get(target)
        suffix = f"  ± {format_quantity(spread)}" if spread is not None else ""
        print(f"{target:<28} {format_quantity(value)}{suffix}")
    for column, warning in prediction.domain_warnings.items():
        print(f"! {column}: {warning}", file=sys.stderr)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .pipeline import dataset_overview

    project = _project(args)
    print(f"project: {project.root}")
    print(f"cases:   {project.cases_path if project.cases_path.exists() else '(none)'}")
    print(f"dataset: {project.training_data_path if project.training_data_path.exists() else '(none)'}")
    print(dataset_overview(project))
    registry = project.registry()
    print(f"runs: {len(registry.get('runs', []))}, latest: {registry.get('latest') or '(none)'}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch

    root = Path(args.project) if args.project else default_project_root()
    return launch(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nncm", description="Neural Network Consequence Modelling")
    parser.add_argument("--project", "-p", help="project directory (default: ./workspace)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a project directory with a default config")
    p.add_argument("--force", action="store_true", help="overwrite an existing config")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("sample", help="generate the case table (LHS/Sobol design)")
    p.add_argument("--vessels", type=int, help="number of vessels")
    p.add_argument("--leaks", type=int, help="leaks per vessel")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("export", help="write cases into a Phast-importable workbook")
    p.add_argument("--template", help="Safeti template workbook (overrides config)")
    p.add_argument("--out", help="output workbook path")
    p.add_argument("--split", type=int, help="max leak rows per workbook")
    p.add_argument(
        "--passthrough",
        action="store_true",
        help="write an unmodified copy of the template (to test whether Phast accepts it at all)",
    )
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="extract training data from a Phast result workbook")
    p.add_argument("workbook", help="Phast/Safeti output workbook")
    p.add_argument("--append", action="store_true", help="merge into the existing dataset")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("train", help="train a model on the project dataset")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict from the latest (or a chosen) run")
    p.add_argument("--run", help="run directory (default: latest)")
    p.add_argument("--temperature", type=float, help="degC")
    p.add_argument("--pressure", type=float, help="barg")
    p.add_argument("--orifice", type=float, help="mm")
    p.add_argument("--material")
    p.add_argument("--wind", type=float, default=5.0, help="wind speed m/s")
    p.add_argument("--stability", type=float, default=4.0, help="Pasquill index (A=1 .. F=6)")
    p.add_argument("--csv", help="predict for every row of a CSV instead")
    p.add_argument("--out", help="output CSV for --csv mode")
    p.add_argument("--mc", type=int, default=0, help="MC-dropout samples for uncertainty")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("info", help="show project status")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("gui", help="launch the unified desktop application")
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
