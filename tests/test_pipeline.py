"""Tests for the functions whose failure gives a wrong answer without an error.

Model quality is out of scope (that is what the run metrics are for), as is
anything needing Phast itself. Workbook tests need the client template, which
is not in the repository, and skip without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nncm.config import Material, MixtureComponent, NncmConfig, Project, Range, SamplingConfig
from nncm.features import (
    FeatureSpec,
    build_features,
    domain_report,
    fit_feature_spec,
    resolve_materials,
)
from nncm.phast import units
from nncm.phast.output_reader import build_training_table, identifiers, parse_weather
from nncm.sampling import generate_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "Safeti Template Input Sheet.xlsx"
EXAMPLE_OUTPUT = REPO_ROOT / "templates" / "Safeti Example Real Output Sheet.xlsx"

requires_template = pytest.mark.skipif(not TEMPLATE.exists(), reason="Safeti template not present")
requires_output = pytest.mark.skipif(
    not EXAMPLE_OUTPUT.exists(), reason="example Safeti output not present"
)


def _materials() -> list[Material]:
    return [
        Material("METHANE", properties={"molecular_weight": 16.04}),
        Material("PROPANE", properties={"molecular_weight": 44.1}),
    ]


def _sampling_config(**overrides) -> SamplingConfig:
    config = SamplingConfig(
        n_vessels=20,
        n_leaks_per_vessel=5,
        temperature=Range(-50, 150),
        pressure=Range(1, 100, log=True),
        orifice=Range(1, 500, log=True),
        materials=_materials(),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_stale_template_path_is_cleared_but_a_custom_one_is_kept(tmp_path):
    # Written on Windows, opened anywhere: the shipped template's old absolute
    # path must be forgotten, whichever separator it uses.
    stale = NncmConfig.from_dict(
        {"version": 4, "phast": {"template": r"C:\Dev\nncm\templates\Safeti Template Input Sheet.xlsx"}}
    )
    assert stale.phast.template == ""
    assert stale.migrated

    custom = tmp_path / "My Own Study.xlsx"
    kept = NncmConfig.from_dict({"version": 4, "phast": {"template": str(custom)}})
    assert kept.phast.template == str(custom)


def test_project_config_round_trip(tmp_path: Path):
    project = Project.create(tmp_path / "proj")
    project.config.sampling.n_vessels = 123
    project.config.sampling.materials = [Material("ETHANE", properties={"molecular_weight": 30.07})]
    project.save_config()

    reopened = Project.open(tmp_path / "proj")
    assert reopened.config.sampling.n_vessels == 123
    assert reopened.config.sampling.materials[0].properties["molecular_weight"] == 30.07
    assert isinstance(reopened.config.sampling.pressure, Range)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_sampling_is_reproducible_and_stays_in_each_envelope():
    config = _sampling_config(
        materials=[
            Material("METHANE", temperature=Range(-160, -80)),
            Material("PROPANE", temperature=Range(0, 60)),
        ]
    )
    first, _ = generate_cases(config)
    second, _ = generate_cases(config)
    pd.testing.assert_frame_equal(first, second)

    assert len(first) == 100 and first["leak_name"].is_unique
    assert first.loc[first["material"] == "METHANE", "temperature_degC"].between(-160, -80).all()
    assert first.loc[first["material"] == "PROPANE", "temperature_degC"].between(0, 60).all()
    assert first["pressure_barg"].between(1, 100).all()
    # Stratified hole sizes: every vessel spans more than a decade.
    spans = first.groupby("vessel_name")["orifice_mm"].agg(lambda s: np.log10(s.max() / s.min()))
    assert (spans > 1.0).all()


def test_each_material_gets_its_own_stratified_design():
    """LHS per material: one vessel in each of its n temperature strata."""
    cases, _ = generate_cases(_sampling_config(n_vessels=40))
    for _, group in cases.drop_duplicates("vessel_name").groupby("material"):
        unit = (group["temperature_degC"].to_numpy() + 50) / 200
        strata = np.floor(unit * len(group)).astype(int)
        assert sorted(strata) == list(range(len(group)))


def test_different_designs_never_share_vessel_names():
    first, _ = generate_cases(_sampling_config(seed=1))
    second, _ = generate_cases(_sampling_config(seed=2))
    assert not set(first["vessel_name"]) & set(second["vessel_name"])


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "quantity", "unit", "expected"),
    [
        (25.0, "temperature", "degK", 298.15),
        (25.0, "temperature", "degF", 77.0),
        (10.0, "pressure", "psi", 145.03774),
        (100.0, "diameter", "in", 3.937008),
        (1000.0, "mass", "tonne", 1.0),
    ],
)
def test_unit_conversion_and_inverse(value, quantity, unit, expected):
    converted = units.convert(value, quantity, unit)
    assert converted == pytest.approx(expected, rel=1e-5)
    assert units.to_canonical(converted, quantity, unit) == pytest.approx(value, rel=1e-9)


def test_unknown_unit_raises_rather_than_writing_wrong_numbers():
    with pytest.raises(units.UnitError):
        units.convert(1.0, "pressure", "furlongs")


# ---------------------------------------------------------------------------
# Workbook writing (needs the client template)
# ---------------------------------------------------------------------------
def _phast_config():
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    return config


def _sheet_part(workbook_path: Path, sheet_name: str) -> str:
    from nncm.phast.patcher import TemplatePatcher

    with TemplatePatcher(workbook_path) as patcher:
        return patcher.sheet_part(sheet_name)


@requires_template
def test_written_workbook_round_trips(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    cases, _ = generate_cases(_sampling_config(n_vessels=5, n_leaks_per_vessel=3))
    out = tmp_path / "input.xlsx"
    report = write_input_workbook(cases, out, _phast_config(), materials=_materials())
    assert (report.n_vessels, report.n_leaks) == (5, 15)
    assert report.verified, report.warnings

    with SafetiWorkbook(out, read_only=True) as workbook:
        vessels = {v["Name"]: v for v in workbook.data_rows("Pressure vessel")}
        leaks = {l["Name"]: l for l in workbook.data_rows("Leak")}
    for case in cases.itertuples():
        assert leaks[case.leak_name]["HoleDiameter"] == pytest.approx(case.orifice_mm)
        assert leaks[case.leak_name]["Pressure vessel"] == case.vessel_name
        vessel = vessels[case.vessel_name]
        assert vessel["Temperature"] == pytest.approx(case.temperature_degC)
        assert vessel["Pressure"] == pytest.approx(case.pressure_barg)
        assert vessel["Material"] == case.material


@requires_template
def test_export_changes_only_the_sheets_it_writes(tmp_path: Path):
    """Phast rejects a re-saved template, so every other part must survive intact."""
    import zipfile

    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=4, n_leaks_per_vessel=3))
    out = tmp_path / "input.xlsx"
    write_input_workbook(cases, out, _phast_config(), materials=_materials())

    with zipfile.ZipFile(TEMPLATE) as template, zipfile.ZipFile(out) as written:
        assert template.namelist() == written.namelist()
        changed = [n for n in template.namelist() if template.read(n) != written.read(n)]
    assert sorted(changed) == sorted(
        [
            _sheet_part(TEMPLATE, "Pressure vessel"),
            _sheet_part(TEMPLATE, "Leak"),
            _sheet_part(TEMPLATE, "COMPONENT"),
            "xl/sharedStrings.xml",
        ]
    )


@requires_template
def test_patcher_refuses_to_overwrite_existing_rows():
    from nncm.phast.patcher import PatchError, TemplatePatcher

    with TemplatePatcher(TEMPLATE) as patcher:
        assert patcher.existing_max_row("Weather") >= 63
        with pytest.raises(PatchError):
            patcher.add_rows("Weather", [{1: "Yes"}], start_row=63)


@requires_template
def test_pure_components_and_mixtures_are_declared(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    materials = [
        Material("METHANE"),
        Material(
            "NATURAL GAS",
            components=[
                MixtureComponent("METHANE", 0.90),
                MixtureComponent("ETHANE", 0.07),
                MixtureComponent("PROPANE", 0.03),
            ],
        ),
    ]
    cases, _ = generate_cases(_sampling_config(materials=materials, n_vessels=8))
    out = tmp_path / "materials.xlsx"
    report = write_input_workbook(cases, out, _phast_config(), materials=materials)
    assert report.warnings == []

    with SafetiWorkbook(out, read_only=True) as workbook:
        components = workbook.data_rows("COMPONENT")
        mixture_rows = workbook.data_rows("MIXTURE")
        vessels = workbook.data_rows("Pressure vessel")

    # Mixture: header row carries the name, continuation rows only components,
    # fractions normalised to percent.
    assert len(mixture_rows) == 3
    assert mixture_rows[0]["Name"] == "NATURAL GAS"
    assert all("Name" not in row for row in mixture_rows[1:])
    assert sum(r["XLSMole"] for r in mixture_rows) == pytest.approx(100.0)
    declared = {r["Name"] for r in components} | {r["Name"] for r in mixture_rows if "Name" in r}
    assert {v["Material"] for v in vessels} <= declared


@requires_template
def test_workbook_splitting_keeps_vessels_intact(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    cases, _ = generate_cases(_sampling_config(n_vessels=6, n_leaks_per_vessel=4))
    config = _phast_config()
    config.max_rows_per_workbook = 10
    report = write_input_workbook(cases, tmp_path / "split.xlsx", config, materials=_materials())

    assert len(report.files) > 1
    for path in report.files:
        with SafetiWorkbook(path, read_only=True) as workbook:
            vessel_names = {v["Name"] for v in workbook.data_rows("Pressure vessel")}
            leak_parents = {l["Pressure vessel"] for l in workbook.data_rows("Leak")}
        assert leak_parents <= vessel_names


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [("Category 5/D", (5.0, "D")), ("Category 1.5/F", (1.5, "F")), ("nonsense", (None, ""))],
)
def test_weather_parsing(text, expected):
    speed, stability = parse_weather(text)
    assert np.isnan(speed) if expected[0] is None else speed == expected[0]
    assert stability == expected[1]


def test_identifiers_cover_both_study_layouts():
    # Flat study: the path stops at the vessel, Scenario holds the leak.
    flat = identifiers("Study\\PV00001", "PV00001_L01")
    assert flat[0] == "PV00001_L01" and "PV00001" in flat
    # Routed study: the path runs down to the leak, Scenario holds the hole size.
    routed = identifiers("Study\\ Unit 1 \\AL-101_V\\Scenario group\\AL-101_V_005mm", 5)
    assert routed[0] == "5"
    assert {"AL-101_V_005mm", "AL-101_V"} <= set(routed)
    assert len(set(routed)) == len(routed) and all(routed)


def _write_result_workbook(path: Path, rows: list[dict]) -> Path:
    """Minimal stand-in for a Phast result workbook."""
    common = [
        {"Path": r["path"], "Scenario": r["scenario"], "Weather": "Category 5/D", "Hole size (mm)": r["hole"]}
        for r in rows
    ]
    discharge = pd.DataFrame(
        [
            {
                **c,
                "Material": r.get("material", "N-BUTANE"),
                "Temperature (input) (degC)": r.get("temperature", 60.0),
                "Pressure (input) (bar)": 5.0,
                "Peak Flowrate (kg/s)": r["rate"],
                "Velocity (m/s)": 120.0,
            }
            for c, r in zip(common, rows)
        ]
    )
    jet = pd.DataFrame(
        [
            {
                **c,
                "Flame length (m)": r.get("flame", 20.0),
                # Threshold differs from the default 6.3 kW/m2 on purpose.
                "Distance downwind to intensity level 1 (4 kW/m2) (m)": 33.0,
            }
            for c, r in zip(common, rows)
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        discharge.to_excel(writer, sheet_name="Discharge", index=False)
        jet.to_excel(writer, sheet_name="Jet fire", index=False)
    return path


def test_join_matches_leaks_and_falls_back_to_vessels(tmp_path: Path):
    cases, _ = generate_cases(_sampling_config(n_vessels=2, n_leaks_per_vessel=2))
    by_leak = [
        {"path": f"Study\\{c.vessel_name}", "scenario": c.leak_name, "hole": 1.0, "rate": 10.0}
        for c in cases.itertuples()
    ]
    by_vessel = [
        {"path": f"Study\\{name}", "scenario": "RENAMED_IN_PHAST", "hole": 50.0, "rate": 5.0}
        for name in cases["vessel_name"].unique()
    ]
    workbook = _write_result_workbook(tmp_path / "results.xlsx", by_leak + by_vessel)

    frame, report = build_training_table(workbook, cases=cases)

    assert report.matched_by == {"leak": 4, "vessel": 2, "none": 0}
    leaks = frame[frame["leak_name"].isin(cases["leak_name"])]
    for row in leaks.itertuples():
        case = cases[cases["leak_name"] == row.leak_name].iloc[0]
        # Sampled inputs win over the values Phast echoes.
        assert row.orifice_mm == pytest.approx(case["orifice_mm"])
        assert row.material == case["material"]
        assert row.vessel_id == case["vessel_id"]
    renamed = frame[~frame["leak_name"].isin(cases["leak_name"])]
    assert (renamed["orifice_mm"] == 50.0).all()
    assert renamed["material"].notna().all()
    assert frame["vessel_id"].nunique() == 2


def test_unmatched_results_still_group_by_vessel_state(tmp_path: Path):
    """No match at all must not crash, and must not fall back to a random split."""
    rows = [
        {"path": "Other\\V1", "scenario": "A", "hole": 10.0, "rate": 1.0, "temperature": 20.0},
        {"path": "Other\\V1", "scenario": "B", "hole": 50.0, "rate": 9.0, "temperature": 20.0},
        {"path": "Other\\V2", "scenario": "C", "hole": 10.0, "rate": 2.0, "temperature": 80.0},
    ]
    workbook = _write_result_workbook(tmp_path / "foreign.xlsx", rows)
    cases, _ = generate_cases(_sampling_config(n_vessels=2, n_leaks_per_vessel=1))

    for case_table in (cases, None):
        frame, _ = build_training_table(workbook, cases=case_table)
        assert len(frame) == 3
        assert frame["material"].eq("N-BUTANE").all()
        # Two vessel states (20 degC and 80 degC), so two groups.
        assert frame["vessel_id"].nunique() == 2


def test_a_sparse_target_is_not_dropped_by_the_nonpositive_gate(tmp_path: Path):
    from nncm.config import ExtractionConfig

    rows = [
        {"path": "S\\V1", "scenario": "A", "hole": 10.0, "rate": 1.0, "flame": None},
        {"path": "S\\V1", "scenario": "B", "hole": 20.0, "rate": 2.0, "flame": 0.0},
        {"path": "S\\V1", "scenario": "C", "hole": 30.0, "rate": 3.0, "flame": 12.0},
    ]
    workbook = _write_result_workbook(tmp_path / "sparse.xlsx", rows)
    config = ExtractionConfig(drop_nonpositive=["Release_rate", "Flame_length"])
    frame, report = build_training_table(workbook, config=config)

    assert sorted(frame["scenario_label"]) == ["A", "C"]  # blank kept, zero dropped
    assert report.dropped_nonpositive == 1


def test_result_columns_resolve_despite_project_specific_thresholds():
    from nncm.phast.output_reader import resolve_column, threshold_of

    frame = pd.DataFrame(
        columns=[
            "Distance downwind to intensity level 1 (4 kW/m2) (m)",
            "Distance to LFL (m)",
            "Distance to LFL fraction (m)",
        ]
    )
    assert resolve_column(frame, "Distance downwind to intensity level 1 (6.3 kW/m2) (m)") == (
        "Distance downwind to intensity level 1 (4 kW/m2) (m)"
    )
    # Similar names must not collide.
    assert resolve_column(frame, "Distance to LFL (m)") == "Distance to LFL (m)"
    assert resolve_column(frame, "Distance to LFL fraction (m)") == "Distance to LFL fraction (m)"
    assert resolve_column(frame, "Flame length (m)") is None
    assert threshold_of("Distance downwind to intensity level 1 (4 kW/m2) (m)") == "4 kW/m2"


@requires_output
def test_extracts_training_data_from_real_output():
    frame, report = build_training_table(EXAMPLE_OUTPUT)

    assert len(frame) > 1000
    assert (frame["Release_rate"] > 0).all()
    assert frame["stability_class"].isin(list("ABCDEF")).all()
    assert frame["vessel_id"].notna().all()
    assert not frame.duplicated(["path_key", "scenario_label", "weather"]).any()


# ---------------------------------------------------------------------------
# Features and domain checks
# ---------------------------------------------------------------------------
def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature_degC": [20.0, -40.0, 150.0, 60.0],
            "pressure_barg": [10.0, 1.0, 100.0, 5.0],
            "orifice_mm": [5.0, 100.0, 250.0, 20.0],
            "elevation_m": [1.0, 1.0, 1.0, 1.0],
            "wind_speed_ms": [5.0, 2.0, 10.0, 5.0],
            "stability_index": [4.0, 6.0, 4.0, 4.0],
            "material": ["METHANE", "PROPANE", "METHANE", "PROPANE"],
            "mat_molecular_weight": [16.04, 44.1, 16.04, 44.1],
        }
    )


def test_features_agree_between_training_and_inference():
    """Same spec after a JSON round trip; one row equals its batch row; a row
    naming only its material gets that material's descriptors."""
    frame = _feature_frame()
    spec = FeatureSpec.from_dict(fit_feature_spec(frame).to_dict())
    batch = build_features(frame, spec)
    pd.testing.assert_frame_equal(build_features(frame.iloc[[1]], spec), batch.iloc[[1]])

    by_name = frame.drop(columns=["mat_molecular_weight"]).iloc[[1]]
    pd.testing.assert_frame_equal(build_features(by_name, spec), batch.iloc[[1]])


def test_unknown_material_without_descriptors_is_refused():
    spec = fit_feature_spec(_feature_frame())
    row = pd.DataFrame([{"temperature_degC": 20.0, "pressure_barg": 10.0, "orifice_mm": 5.0}])
    with pytest.raises(ValueError, match="XENON"):
        resolve_materials(row.assign(material="XENON"), spec, strict=True)
    with pytest.raises(ValueError):
        resolve_materials(row, spec, strict=True)
    # A new material with its descriptors given is allowed, and flagged.
    given = row.assign(material="XENON", mat_molecular_weight=131.3)
    resolve_materials(given, spec, strict=True)
    assert "material" in domain_report(given.iloc[0].to_dict(), spec)


def test_domain_report_flags_extrapolation():
    spec = fit_feature_spec(_feature_frame())
    base = {"temperature_degC": 20.0, "pressure_barg": 10.0, "orifice_mm": 50.0}

    assert domain_report({**base, "material": "METHANE"}, spec) == {}
    # NumPy scalars, as a pandas row carries them, are checked too.
    assert "temperature_degC" in domain_report({**base, "temperature_degC": np.int64(900)}, spec)
    # Inside the global range but outside what PROPANE was trained on.
    assert "pressure_barg" in domain_report({**base, "material": "PROPANE"}, spec)
    # A setting every training case shared cannot be varied.
    assert "elevation_m" in domain_report({**base, "elevation_m": 5.0}, spec)


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------
def test_masked_scaler_ignores_missing_values():
    from nncm.training import _fit_masked_scaler

    scaler = _fit_masked_scaler(np.array([[1.0, 2.0], [3.0, 2.0], [5.0, np.nan]]))
    assert scaler.mean_[0] == pytest.approx(3.0)
    assert scaler.scale_[0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert scaler.scale_[1] == 1.0  # constant column: no division by zero


def test_grouped_split_never_shares_a_vessel_between_partitions():
    from nncm.training import _grouped_split

    groups = np.repeat(np.arange(50), 8).astype(str)
    train, val, test = _grouped_split(groups, test_size=0.2, valid_size=0.2, seed=7)

    assert len(train) + len(val) + len(test) == len(groups)
    parts = [set(groups[i]) for i in (train, val, test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def _partial_dataset(n_vessels: int = 40) -> pd.DataFrame:
    """Two materials; one target present for only the large holes."""
    rng = np.random.default_rng(0)
    rows = []
    for vessel in range(n_vessels):
        material, weight = ("METHANE", 16.04) if vessel % 2 else ("HYDROGEN", 2.02)
        temperature = rng.uniform(-20, 120)
        pressure = 10 ** rng.uniform(0, 2)
        for _ in range(6):
            orifice = 10 ** rng.uniform(0, 2.5)
            rate = pressure * orifice**2 * 1e-4 * np.sqrt(weight)
            rows.append(
                {
                    "vessel_id": vessel,
                    "material": material,
                    "mat_molecular_weight": weight,
                    "temperature_degC": temperature,
                    "pressure_barg": pressure,
                    "orifice_mm": orifice,
                    "Release_rate": rate,
                    "Distance_to_LFL": 3.0 * rate**0.5 if orifice > 50 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_training_and_prediction_end_to_end(tmp_path: Path):
    from nncm.predict import ModelBundle
    from nncm.training import train_model

    project = Project.create(tmp_path / "masked")
    frame = _partial_dataset()
    config = project.config.training
    config.targets = ["Release_rate", "Distance_to_LFL"]
    config.log_targets = list(config.targets)
    config.epochs = 6
    config.hidden_units = [16, 16]
    config.head_units = [8]

    result = train_model(project, frame=frame, config=config, log=lambda _: None)

    # Every row with at least one target is used; the sparse one is scored on
    # its own rows only.
    assert result.n_train + result.n_val + result.n_test == len(frame)
    sparse = result.metrics["per_target"]["Distance_to_LFL"]
    assert sparse["n_test"] < result.metrics["per_target"]["Release_rate"]["n_test"]

    bundle = ModelBundle(result.run_dir)
    base = {"temperature_degC": 20.0, "pressure_barg": 10.0, "orifice_mm": 60.0}
    methane = bundle.predict_one({**base, "material": "METHANE"}, mc_samples=5)
    hydrogen = bundle.predict_one({**base, "material": "HYDROGEN"})
    # The material name alone must reach the model as its descriptors.
    assert methane.values != hydrogen.values
    assert all(v >= 0 for v in methane.values.values()), "log targets never invert below zero"
    assert set(methane.uncertainty) == set(config.targets)

    batch = bundle.predict_batch(pd.DataFrame([{**base, "material": "METHANE"}]))
    assert batch["Release_rate"].iloc[0] == pytest.approx(methane.values["Release_rate"], rel=1e-5)
    assert "domain_warnings" in batch
