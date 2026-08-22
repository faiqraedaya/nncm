"""Tests for the stages that are expensive to debug by hand.

Deliberately excluded: model quality (that is what the metrics artifacts are
for) and anything requiring Phast itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nncm.config import Material, NncmConfig, Project, Range, SamplingConfig
from nncm.features import FeatureSpec, build_features, domain_report, fit_feature_spec
from nncm.phast import units
from nncm.phast.output_reader import normalise_path, parse_weather, scenario_key
from nncm.sampling import generate_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "Safeti Template Input Sheet.xlsx"
EXAMPLE_OUTPUT = REPO_ROOT / "templates" / "Safeti Example Real Output Sheet.xlsx"

requires_template = pytest.mark.skipif(not TEMPLATE.exists(), reason="Safeti template not present")
requires_output = pytest.mark.skipif(
    not EXAMPLE_OUTPUT.exists(), reason="example Safeti output not present"
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _materials() -> list[Material]:
    """Catalogue matching the materials used by :func:`_sampling_config`."""
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
        materials=[
            Material("METHANE", properties={"molecular_weight": 16.04}),
            Material("PROPANE", properties={"molecular_weight": 44.1}),
        ],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_sampling_is_reproducible_and_respects_bounds():
    config = _sampling_config()
    first, report = generate_cases(config)
    second, _ = generate_cases(config)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100
    assert report.n_vessels == 20
    assert first["temperature_degC"].between(-50, 150).all()
    assert first["pressure_barg"].between(1, 100).all()
    assert first["orifice_mm"].between(1, 500).all()
    assert first["leak_name"].is_unique


def test_log_sampling_spreads_across_decades():
    """Linear sampling of a 1-500 mm range starves the small holes."""
    cases, _ = generate_cases(_sampling_config(n_vessels=200))
    decades = np.log10(cases["orifice_mm"])
    below_10mm = (cases["orifice_mm"] < 10).mean()
    assert below_10mm > 0.25, "log sampling should place a large share below 10 mm"
    assert decades.max() - decades.min() > 2.0


def test_per_material_ranges_override_global_ranges():
    config = _sampling_config(
        materials=[
            Material("METHANE", temperature=Range(-160, -80)),
            Material("PROPANE", temperature=Range(0, 60)),
        ]
    )
    cases, _ = generate_cases(config)
    methane = cases[cases["material"] == "METHANE"]["temperature_degC"]
    propane = cases[cases["material"] == "PROPANE"]["temperature_degC"]
    assert methane.between(-160, -80).all()
    assert propane.between(0, 60).all()


def test_every_vessel_spans_the_hole_size_range():
    cases, _ = generate_cases(_sampling_config(n_leaks_per_vessel=6))
    spans = cases.groupby("vessel_name")["orifice_mm"].agg(lambda s: np.log10(s.max() / s.min()))
    assert (spans > 1.0).all(), "stratification should give each vessel a wide size range"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "quantity", "unit", "expected"),
    [
        (25.0, "temperature", "degC", 25.0),
        (25.0, "temperature", "degK", 298.15),
        (25.0, "temperature", "degF", 77.0),
        (10.0, "pressure", "bar", 10.0),
        (10.0, "pressure", "psi", 145.03774),
        (100.0, "diameter", "mm", 100.0),
        (100.0, "diameter", "in", 3.937008),
        (1.0, "length", "ft", 3.28084),
        (1000.0, "mass", "tonne", 1.0),
    ],
)
def test_unit_conversion(value, quantity, unit, expected):
    assert units.convert(value, quantity, unit) == pytest.approx(expected, rel=1e-5)


def test_unit_roundtrip():
    for quantity, unit in [("temperature", "degF"), ("pressure", "psi"), ("mass", "lb")]:
        converted = units.convert(42.0, quantity, unit)
        assert units.to_canonical(converted, quantity, unit) == pytest.approx(42.0, rel=1e-9)


def test_unknown_unit_raises_rather_than_writing_wrong_numbers():
    with pytest.raises(units.UnitError):
        units.convert(1.0, "pressure", "furlongs")


# ---------------------------------------------------------------------------
# Workbook schema and writing
# ---------------------------------------------------------------------------
@requires_template
def test_template_schema_is_discovered():
    from nncm.phast.workbook import SafetiWorkbook

    with SafetiWorkbook(TEMPLATE) as workbook:
        schema = workbook.schema("Pressure vessel")
        assert schema.data_start_row == 63
        assert schema.column("Temperature").unit == "degC"
        assert schema.column("Pressure").label == "Pressure (gauge)"
        assert schema.column("label:Name").index == 15
        leak = workbook.schema("Leak")
        assert leak.column("HoleDiameter").unit == "mm"
        assert leak.column("label:Pressure vessel").index == 15


@requires_template
def test_ambiguous_column_names_are_rejected():
    from nncm.phast.workbook import SafetiWorkbook

    with SafetiWorkbook(TEMPLATE) as workbook:
        schema = workbook.schema("Pressure vessel")
        with pytest.raises(KeyError):
            schema.column("label:Folder")  # appears many times


@requires_template
def test_written_workbook_round_trips(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    cases, _ = generate_cases(_sampling_config(n_vessels=5, n_leaks_per_vessel=3))
    out = tmp_path / "input.xlsx"
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    report = write_input_workbook(cases, out, config, materials=_materials())

    assert report.n_vessels == 5
    assert report.n_leaks == 15
    assert out.exists()

    with SafetiWorkbook(out, read_only=True) as workbook:
        vessels = workbook.data_rows("Pressure vessel")
        leaks = workbook.data_rows("Leak")
    assert len(vessels) == 5
    assert len(leaks) == 15

    first_case = cases.iloc[0]
    written_leak = next(l for l in leaks if l["Name"] == first_case["leak_name"])
    assert written_leak["HoleDiameter"] == pytest.approx(first_case["orifice_mm"])
    written_vessel = next(v for v in vessels if v["Name"] == first_case["vessel_name"])
    assert written_vessel["Temperature"] == pytest.approx(first_case["temperature_degC"])
    assert written_vessel["Pressure"] == pytest.approx(first_case["pressure_barg"])
    assert written_vessel["Material"] == first_case["material"]


@requires_template
def test_export_changes_only_the_sheets_it_writes(tmp_path: Path):
    """Phast rejects a re-saved template, so every other part must survive intact."""
    import zipfile

    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=4, n_leaks_per_vessel=3))
    out = tmp_path / "input.xlsx"
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    write_input_workbook(cases, out, config, materials=_materials())

    with zipfile.ZipFile(TEMPLATE) as template, zipfile.ZipFile(out) as written:
        assert template.namelist() == written.namelist(), "no part may be added, dropped or renamed"
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
def test_export_produces_well_formed_parts_and_shared_strings(tmp_path: Path):
    import re
    import zipfile
    from xml.etree import ElementTree

    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=4, n_leaks_per_vessel=3))
    out = tmp_path / "input.xlsx"
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    write_input_workbook(cases, out, config, materials=_materials())

    with zipfile.ZipFile(out) as written:
        for name in written.namelist():
            if name.endswith((".xml", ".rels")):
                ElementTree.fromstring(written.read(name))  # raises if malformed
        sheet = written.read(_sheet_part(TEMPLATE, "Pressure vessel")).decode("utf-8")
        shared = written.read("xl/sharedStrings.xml").decode("utf-8")

    # Strings must go through the shared-string table, as Excel writes them.
    assert "inlineStr" not in sheet
    assert 't="s"' in sheet
    header = re.search(r"<sst\b[^>]*>", shared).group(0)
    unique = int(re.search(r'uniqueCount="(\d+)"', header).group(1))
    assert unique == len(re.findall(r"<si\b", shared))
    with zipfile.ZipFile(TEMPLATE) as template:
        original = template.read("xl/sharedStrings.xml").decode("utf-8")
    original_count = int(re.search(r'count="(\d+)"', original).group(1))
    assert int(re.search(r'\bcount="(\d+)"', header).group(1)) > original_count


@requires_template
def test_patcher_refuses_to_overwrite_existing_rows(tmp_path: Path):
    from nncm.phast.patcher import PatchError, TemplatePatcher

    with TemplatePatcher(TEMPLATE) as patcher:
        # The template ships 16 populated weather rows from row 63 down.
        assert patcher.existing_max_row("Weather") >= 63
        with pytest.raises(PatchError):
            patcher.add_rows("Weather", [{1: "Yes"}], start_row=63)


@requires_template
def test_unknown_enumeration_is_reported(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=2, n_leaks_per_vessel=2))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    config.vessel_defaults["FlashFlag"] = "1 Pressure/Temperature"  # wrong capitalisation

    report = write_input_workbook(cases, tmp_path / "bad.xlsx", config, materials=_materials())
    assert any("Specified condition" in w for w in report.warnings)


@requires_template
def test_export_read_back_check_passes(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=6, n_leaks_per_vessel=4))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    report = write_input_workbook(cases, tmp_path / "input.xlsx", config, materials=_materials())

    assert report.warnings == []
    assert report.verified
    assert report.sheets_touched == ["COMPONENT", "Leak", "Pressure vessel"]


def _sheet_part(workbook_path: Path, sheet_name: str) -> str:
    from nncm.phast.patcher import TemplatePatcher

    with TemplatePatcher(workbook_path) as patcher:
        return patcher.sheet_part(sheet_name)


@requires_template
def test_pure_components_and_mixtures_are_declared(tmp_path: Path):
    """Vessels may only reference materials the workbook itself defines."""
    from nncm.config import MixtureComponent
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    materials = [
        Material("METHANE", properties={"molecular_weight": 16.04}),
        Material(
            "NATURAL GAS",
            components=[
                MixtureComponent("METHANE", 90.0),
                MixtureComponent("ETHANE", 7.0),
                MixtureComponent("PROPANE", 3.0),
            ],
        ),
    ]
    cases, _ = generate_cases(_sampling_config(materials=materials, n_vessels=8))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    out = tmp_path / "materials.xlsx"
    report = write_input_workbook(cases, out, config, materials=materials)

    assert report.warnings == []
    assert report.materials_declared["COMPONENT"] == ["METHANE"]
    assert report.materials_declared["MIXTURE"] == ["NATURAL GAS"]

    with SafetiWorkbook(out, read_only=True) as workbook:
        components = workbook.data_rows("COMPONENT")
        mixture_rows = workbook.data_rows("MIXTURE")
        vessels = workbook.data_rows("Pressure vessel")

    # Pure component: a single row naming it in the study's material list.
    methane = next(r for r in components if r["Name"] == "METHANE")
    assert methane["Use"] == "Yes"
    assert methane["Physical Properties System"] == "Physical Properties System"
    assert methane["Materials"] == "Materials"
    assert methane["CreationTemplate"] == "PhastMC"

    # Mixture: header row carries the name, continuation rows only components.
    assert len(mixture_rows) == 3
    header = mixture_rows[0]
    assert header["Name"] == "NATURAL GAS"
    assert header["XLSComponent"] == "METHANE"
    assert header["XLSMole"] == pytest.approx(90.0)
    assert header["CreationTemplate"] == "PhastMC"
    for continuation in mixture_rows[1:]:
        assert "Name" not in continuation
        assert "Use" not in continuation
        assert continuation["XLSComponent"] in {"ETHANE", "PROPANE"}
    assert sum(r["XLSMole"] for r in mixture_rows) == pytest.approx(100.0)

    # Every material a vessel references is defined in the workbook.
    declared = {r["Name"] for r in components} | {
        r["Name"] for r in mixture_rows if "Name" in r
    }
    assert {v["Material"] for v in vessels} <= declared


@requires_template
def test_each_vessel_keeps_the_material_its_case_assigns(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    cases, _ = generate_cases(_sampling_config(n_vessels=12, n_leaks_per_vessel=3))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    out = tmp_path / "assignment.xlsx"
    write_input_workbook(cases, out, config, materials=_materials())

    with SafetiWorkbook(out, read_only=True) as workbook:
        written = {r["Name"]: r["Material"] for r in workbook.data_rows("Pressure vessel")}
        leaks = workbook.data_rows("Leak")

    expected = cases.drop_duplicates("vessel_name").set_index("vessel_name")["material"].to_dict()
    assert written == expected

    # A leak inherits its material through the vessel it points at.
    leak_to_vessel = {r["Name"]: r["Pressure vessel"] for r in leaks}
    for leak_name, vessel_name in leak_to_vessel.items():
        case = cases[cases["leak_name"] == leak_name].iloc[0]
        assert vessel_name == case["vessel_name"]
        assert written[vessel_name] == case["material"]


@requires_template
def test_material_without_a_definition_is_reported(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook

    cases, _ = generate_cases(_sampling_config(n_vessels=6))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)

    # Catalogue is missing PROPANE, which some vessels use.
    report = write_input_workbook(
        cases,
        tmp_path / "missing.xlsx",
        config,
        materials=[Material("METHANE")],
    )
    assert any("PROPANE" in w for w in report.warnings)
    assert not report.verified


def test_mixture_fractions_are_normalised_to_percent():
    from nncm.config import MixtureComponent

    material = Material(
        "TEST GAS",
        components=[MixtureComponent("METHANE", 0.9), MixtureComponent("ETHANE", 0.1)],
    )
    normalised = material.normalised_components()
    assert sum(c.fraction for c in normalised) == pytest.approx(100.0)
    assert normalised[0].fraction == pytest.approx(90.0)


def test_material_validation_rejects_bad_compositions():
    from nncm.config import MixtureComponent

    with pytest.raises(ValueError):
        Material("BAD", components=[MixtureComponent("METHANE", -1.0)]).validate()
    with pytest.raises(ValueError):
        Material("BAD", components=[MixtureComponent("", 100.0)]).validate()
    with pytest.raises(ValueError):
        Material("BAD", composition_basis="volume").validate()


def test_duplicate_material_names_are_rejected():
    config = _sampling_config(materials=[Material("METHANE"), Material("methane")])
    with pytest.raises(ValueError, match="duplicate material"):
        config.validate()


@requires_template
def test_workbook_splitting_keeps_vessels_intact(tmp_path: Path):
    from nncm.phast.input_writer import write_input_workbook
    from nncm.phast.workbook import SafetiWorkbook

    cases, _ = generate_cases(_sampling_config(n_vessels=6, n_leaks_per_vessel=4))
    config = NncmConfig().phast
    config.template = str(TEMPLATE)
    config.max_rows_per_workbook = 10
    report = write_input_workbook(cases, tmp_path / "split.xlsx", config, materials=_materials())

    assert len(report.files) > 1
    for path in report.files:
        with SafetiWorkbook(path, read_only=True) as workbook:
            vessel_names = {v["Name"] for v in workbook.data_rows("Pressure vessel")}
            leak_parents = {l["Pressure vessel"] for l in workbook.data_rows("Leak")}
        assert leak_parents <= vessel_names, "a leak must ship with its vessel"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Category 5/D", (5.0, "D")),
        ("Category 1.5/F", (1.5, "F")),
        ("Category 10/D", (10.0, "D")),
        ("nonsense", (None, "")),
    ],
)
def test_weather_parsing(text, expected):
    speed, stability = parse_weather(text)
    if expected[0] is None:
        assert np.isnan(speed)
    else:
        assert speed == expected[0]
    assert stability == expected[1]


def test_path_helpers():
    path = "Study\\Unit 1\\AL-101_V\\Scenario group\\AL-101_V_005mm"
    assert scenario_key(path) == "AL-101_V_005mm"
    assert normalise_path("Study\\ Unit 1 \\AL-101_V ") == "Study\\Unit 1\\AL-101_V"


@requires_output
def test_extracts_training_data_from_real_output():
    from nncm.phast.output_reader import build_training_table

    frame, report = build_training_table(EXAMPLE_OUTPUT)

    assert len(frame) > 1000
    assert report.n_rows == len(frame)
    for column in ("Release_rate", "Velocity", "temperature_degC", "pressure_barg", "orifice_mm"):
        assert column in frame.columns
    assert (frame["Release_rate"] > 0).all()
    assert frame["wind_speed_ms"].notna().all()
    assert frame["stability_class"].isin(list("ABCDEF")).all()
    # Each result row must be unique on its key — duplicates mean a bad join.
    assert not frame.duplicated(["path_key", "scenario_label", "weather"]).any()


@requires_output
def test_case_join_recovers_sampled_inputs():
    from nncm.phast.output_reader import build_training_table

    raw, _ = build_training_table(EXAMPLE_OUTPUT)
    sample = raw.head(20)
    cases = pd.DataFrame(
        {
            "case_id": range(len(sample)),
            "vessel_id": 1,
            "vessel_name": "PV1",
            "leak_name": sample["scenario_key"].to_numpy(),
            "material": "METHANE",
            "temperature_degC": 11.0,
            "pressure_barg": 99.0,
            "orifice_mm": 7.0,
        }
    )
    joined, report = build_training_table(EXAMPLE_OUTPUT, cases=cases)
    matched = joined[joined["leak_name"].notna()]
    assert report.n_matched_cases > 0
    # Sampled inputs win over the values echoed by Phast.
    assert (matched["pressure_barg"] == 99.0).all()
    assert (matched["material"] == "METHANE").all()


# ---------------------------------------------------------------------------
# Result identification across study layouts
# ---------------------------------------------------------------------------
def test_identifiers_cover_both_study_layouts():
    from nncm.phast.output_reader import identifiers

    # Flat study: the path stops at the vessel, Scenario holds the leak.
    assert identifiers("Study\\PV00001", "PV00001_L01")[0] == "PV00001_L01"
    assert "PV00001" in identifiers("Study\\PV00001", "PV00001_L01")

    # Routed study: the path runs down to the leak, Scenario holds the hole size.
    routed = identifiers("Study\\Unit 1\\AL-101_V\\Scenario group\\AL-101_V_005mm", 5)
    assert routed[0] == "5"
    assert "AL-101_V_005mm" in routed
    assert "AL-101_V" in routed

    # No duplicates, no empties.
    assert len(set(routed)) == len(routed)
    assert all(routed)


def _write_result_workbook(path: Path, rows: list[dict]) -> Path:
    """Minimal stand-in for a Phast result workbook."""
    discharge = pd.DataFrame(
        [
            {
                "Path": r["path"],
                "Scenario": r["scenario"],
                "Weather": r["weather"],
                "Hole size (mm)": r["hole"],
                "Material": "N-BUTANE",
                "Temperature (input) (degC)": 60.0,
                "Pressure (input) (bar)": 5.0,
                "Peak Flowrate (kg/s)": r["rate"],
                "Velocity (m/s)": 120.0,
            }
            for r in rows
        ]
    )
    jet = pd.DataFrame(
        [
            {
                "Path": r["path"],
                "Scenario": r["scenario"],
                "Weather": r["weather"],
                "Hole size (mm)": r["hole"],
                "Flame length (m)": 20.0,
                # Threshold differs from the default 6.3 kW/m2 on purpose.
                "Distance downwind to intensity level 1 (4 kW/m2) (m)": 33.0,
            }
            for r in rows
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        discharge.to_excel(writer, sheet_name="Discharge", index=False)
        jet.to_excel(writer, sheet_name="Jet fire", index=False)
    return path


def test_join_matches_when_the_leak_is_in_the_scenario_column(tmp_path: Path):
    """A flat study reports the leak in Scenario, not as a path segment."""
    from nncm.phast.output_reader import build_training_table

    cases, _ = generate_cases(_sampling_config(n_vessels=2, n_leaks_per_vessel=2))
    rows = [
        {
            "path": f"Study\\{case.vessel_name}",
            "scenario": case.leak_name,
            "weather": "Category 5/D",
            "hole": case.orifice_mm,
            "rate": 10.0 + index,
        }
        for index, case in enumerate(cases.itertuples())
    ]
    workbook = _write_result_workbook(tmp_path / "results.xlsx", rows)

    frame, report = build_training_table(workbook, cases=cases)

    assert report.n_matched_cases == len(cases)
    assert report.n_unmatched == 0
    assert report.matched_by["leak"] == len(cases)
    # Sampled inputs win over the values echoed by Phast.
    for row in frame.itertuples():
        case = cases[cases["leak_name"] == row.leak_name].iloc[0]
        assert row.orifice_mm == pytest.approx(case["orifice_mm"])
        assert row.material == case["material"]
        assert row.vessel_id == case["vessel_id"]
    assert frame["vessel_id"].nunique() == 2


def test_join_falls_back_to_the_vessel_when_the_leak_is_unknown(tmp_path: Path):
    from nncm.phast.output_reader import build_training_table

    cases, _ = generate_cases(_sampling_config(n_vessels=2, n_leaks_per_vessel=2))
    rows = [
        {
            "path": f"Study\\{name}",
            "scenario": "RENAMED_IN_PHAST",
            "weather": "Category 5/D",
            "hole": 50.0,
            "rate": 5.0,
        }
        for name in cases["vessel_name"].unique()
    ]
    workbook = _write_result_workbook(tmp_path / "vessel_only.xlsx", rows)

    frame, report = build_training_table(workbook, cases=cases)

    assert report.matched_by["vessel"] == 2
    assert report.matched_by["leak"] == 0
    # Vessel-level attributes are recovered; the hole size comes from the sheet.
    assert frame["material"].notna().all()
    assert frame["vessel_id"].nunique() == 2
    assert frame["orifice_mm"].tolist() == [50.0, 50.0]


def test_result_columns_resolve_despite_project_specific_thresholds(tmp_path: Path):
    from nncm.phast.output_reader import build_training_table, resolve_column, threshold_of

    frame = pd.DataFrame(
        columns=[
            "Distance downwind to intensity level 1 (4 kW/m2) (m)",
            "Distance downwind to overpressure 1 (0.02068 bar) (m)",
            "Distance to LFL (m)",
            "Distance to LFL fraction (m)",
        ]
    )
    assert resolve_column(frame, "Distance downwind to intensity level 1") == (
        "Distance downwind to intensity level 1 (4 kW/m2) (m)"
    )
    assert resolve_column(frame, "Distance downwind to intensity level 1 (6.3 kW/m2) (m)") == (
        "Distance downwind to intensity level 1 (4 kW/m2) (m)"
    )
    # Similar names must not collide.
    assert resolve_column(frame, "Distance to LFL (m)") == "Distance to LFL (m)"
    assert resolve_column(frame, "Distance to LFL fraction (m)") == "Distance to LFL fraction (m)"
    assert resolve_column(frame, "Flame length (m)") is None

    assert threshold_of("Distance downwind to intensity level 1 (4 kW/m2) (m)") == "4 kW/m2"
    assert threshold_of("Distance to LFL (m)") == ""

    cases, _ = generate_cases(_sampling_config(n_vessels=1, n_leaks_per_vessel=1))
    case = cases.iloc[0]
    workbook = _write_result_workbook(
        tmp_path / "thresholds.xlsx",
        [
            {
                "path": f"Study\\{case['vessel_name']}",
                "scenario": case["leak_name"],
                "weather": "Category 5/D",
                "hole": case["orifice_mm"],
                "rate": 3.0,
            }
        ],
    )
    extracted, report = build_training_table(workbook, cases=cases)
    assert extracted["Jet_fire_distance_level1"].iloc[0] == 33.0
    assert report.thresholds["Jet_fire_distance_level1"] == "4 kW/m2"
    # The stub workbook omits most columns, but the one present under a
    # different threshold must not be reported as missing.
    assert not [w for w in report.warnings if "intensity level 1" in w]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature_degC": [20.0, -40.0, 150.0],
            "pressure_barg": [10.0, 1.0, 100.0],
            "orifice_mm": [5.0, 100.0, 250.0],
            "wind_speed_ms": [5.0, 2.0, 10.0],
            "stability_index": [4.0, 6.0, 4.0],
            "material": ["METHANE", "PROPANE", "METHANE"],
            "mat_molecular_weight": [16.04, 44.1, 16.04],
        }
    )


def test_feature_spec_round_trips_through_json():
    spec = fit_feature_spec(_feature_frame())
    restored = FeatureSpec.from_dict(spec.to_dict())
    pd.testing.assert_frame_equal(
        build_features(_feature_frame(), spec), build_features(_feature_frame(), restored)
    )


def test_features_are_identical_for_single_row_and_batch():
    """The GUI predicts one row at a time; it must match batch training exactly."""
    frame = _feature_frame()
    spec = fit_feature_spec(frame)
    batch = build_features(frame, spec)
    single = build_features(frame.iloc[[1]], spec)
    pd.testing.assert_frame_equal(single, batch.iloc[[1]])


def test_missing_optional_columns_are_imputed_not_crashed():
    spec = fit_feature_spec(_feature_frame())
    minimal = pd.DataFrame(
        {"temperature_degC": [30.0], "pressure_barg": [5.0], "orifice_mm": [12.0]}
    )
    built = build_features(minimal, spec)
    assert list(built.columns) == spec.feature_columns
    assert built.notna().all().all()


def test_domain_report_flags_extrapolation():
    spec = fit_feature_spec(_feature_frame())
    inside = domain_report({"temperature_degC": 20.0, "pressure_barg": 10.0, "orifice_mm": 50.0}, spec)
    outside = domain_report({"temperature_degC": 900.0, "pressure_barg": 10.0, "orifice_mm": 50.0}, spec)
    assert inside == {}
    assert "temperature_degC" in outside


def test_domain_report_flags_unknown_material_for_one_hot_models():
    frame = _feature_frame().drop(columns=["mat_molecular_weight"])
    spec = fit_feature_spec(frame)  # falls back to one-hot material identity
    assert spec.material_categories

    base = {"temperature_degC": 20.0, "pressure_barg": 10.0, "orifice_mm": 50.0}
    assert "material" not in domain_report({**base, "material": "METHANE"}, spec)
    assert "material" in domain_report({**base, "material": "XENON"}, spec)
    assert "material" in domain_report(base, spec)


# ---------------------------------------------------------------------------
# Project layout and training split
# ---------------------------------------------------------------------------
def test_project_config_round_trip(tmp_path: Path):
    project = Project.create(tmp_path / "proj")
    project.config.sampling.n_vessels = 123
    project.config.sampling.materials = [Material("ETHANE", properties={"molecular_weight": 30.07})]
    project.save_config()

    reopened = Project.open(tmp_path / "proj")
    assert reopened.config.sampling.n_vessels == 123
    assert reopened.config.sampling.materials[0].name == "ETHANE"
    assert reopened.config.sampling.materials[0].properties["molecular_weight"] == 30.07
    assert isinstance(reopened.config.sampling.pressure, Range)


def test_old_config_is_migrated_to_the_minimal_phast_footprint(tmp_path: Path):
    import json

    from nncm.config import CONFIG_VERSION

    legacy = {
        "version": 2,
        "phast": {
            "folder_name": "NNCM",
            "write_material_rows": True,
            "vessel_defaults": {
                "SpecifyVolumeInFlag": "0 No",
                "FlashFlag": "1 Pressure/temperature",
                "RiskEffects": "1 Flammable only",
                "TankType": "1 Vertical cylinder",
            },
            "leak_defaults": {"ReleaseDirection": "0 Horizontal", "EventFrequency": 1e-4},
        },
    }
    path = tmp_path / "nncm.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    config = NncmConfig.load(path)
    assert config.version == CONFIG_VERSION
    assert "TankType" not in config.phast.vessel_defaults
    assert "RiskEffects" not in config.phast.vessel_defaults
    assert "EventFrequency" not in config.phast.leak_defaults
    assert config.phast.folder_name == ""
    # v4: vessels must never reference a material the study does not define.
    assert config.phast.write_material_rows is True
    # Settings that are still wanted are untouched.
    assert config.phast.vessel_defaults["FlashFlag"] == "1 Pressure/temperature"
    assert config.phast.leak_defaults["ReleaseDirection"] == "0 Horizontal"


def test_migration_keeps_deliberate_customisation(tmp_path: Path):
    import json

    path = tmp_path / "nncm.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "phast": {
                    "folder_name": "Unit 42",
                    "vessel_defaults": {"TankType": "2 Horizontal cylinder"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = NncmConfig.load(path)
    assert config.phast.folder_name == "Unit 42"
    assert config.phast.vessel_defaults["TankType"] == "2 Horizontal cylinder"


def test_masked_scaler_ignores_missing_values():
    """Imputing before fitting would shrink the spread; the stats must be exact."""
    from nncm.training import _fit_masked_scaler

    values = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0], [np.nan, 50.0]])
    scaler = _fit_masked_scaler(values)

    assert scaler.mean_[0] == pytest.approx(3.0)
    assert scaler.scale_[0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert scaler.mean_[1] == pytest.approx(30.0)
    assert scaler.scale_[1] == pytest.approx(np.std([10.0, 30.0, 50.0]))


def test_constant_target_does_not_produce_a_zero_scale():
    from nncm.training import _fit_masked_scaler

    scaler = _fit_masked_scaler(np.array([[2.0], [2.0], [np.nan]]))
    assert scaler.scale_[0] == 1.0


def _partial_dataset(n_vessels: int = 40) -> pd.DataFrame:
    """Dataset where one target is only present for a third of the rows."""
    rng = np.random.default_rng(0)
    rows = []
    for vessel in range(n_vessels):
        temperature = rng.uniform(-20, 120)
        pressure = 10 ** rng.uniform(0, 2)
        for leak in range(6):
            orifice = 10 ** rng.uniform(0, 2.5)
            rate = pressure * orifice**2 * 1e-3
            rows.append(
                {
                    "vessel_id": vessel,
                    "material": "METHANE",
                    "temperature_degC": temperature,
                    "pressure_barg": pressure,
                    "orifice_mm": orifice,
                    "wind_speed_ms": 5.0,
                    "stability_index": 4.0,
                    "Release_rate": rate,
                    # Only large holes produce this one, as in a real study.
                    "Distance_to_LFL": 3.0 * rate**0.5 if orifice > 50 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_training_uses_rows_where_only_some_targets_are_present(tmp_path: Path):
    from nncm.training import train_model

    project = Project.create(tmp_path / "masked")
    frame = _partial_dataset()
    config = project.config.training
    config.targets = ["Release_rate", "Distance_to_LFL"]
    config.log_targets = ["Release_rate", "Distance_to_LFL"]
    config.epochs = 6
    config.hidden_units = [16, 16]
    config.head_units = [8]
    config.use_material_properties = False

    partial_coverage = frame["Distance_to_LFL"].notna().mean()
    assert 0.2 < partial_coverage < 0.8, "fixture should be genuinely partial"

    result = train_model(project, frame=frame, config=config, log=lambda _: None)

    # Every row with at least one target is used, not just the complete ones.
    assert result.n_train + result.n_val + result.n_test == len(frame)
    assert result.metrics["coverage"]["Release_rate"] == 1.0
    assert result.metrics["coverage"]["Distance_to_LFL"] == pytest.approx(partial_coverage)

    # The sparse target is still scored, on its own rows only.
    sparse = result.metrics["per_target"]["Distance_to_LFL"]
    dense = result.metrics["per_target"]["Release_rate"]
    assert sparse["n_test"] < dense["n_test"] == result.n_test

    # Masked rows must not leak into predictions as zeros, and a log target can
    # never invert to a negative rate or distance.
    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert predictions["pred_Distance_to_LFL"].notna().all()
    assert (predictions["pred_Release_rate"] >= 0).all()
    assert (predictions["pred_Distance_to_LFL"] >= 0).all()
    assert predictions["pred_Release_rate"].max() > 0


def test_grouped_split_never_shares_a_vessel_between_partitions():
    from nncm.training import _grouped_split

    groups = np.repeat(np.arange(50), 8).astype(str)
    train, val, test = _grouped_split(groups, test_size=0.2, valid_size=0.2, seed=7)

    assert len(train) + len(val) + len(test) == len(groups)
    train_groups = set(groups[train])
    val_groups = set(groups[val])
    test_groups = set(groups[test])
    assert not (train_groups & test_groups)
    assert not (train_groups & val_groups)
    assert not (val_groups & test_groups)


def test_grouped_split_refuses_impossible_splits():
    from nncm.training import _grouped_split

    with pytest.raises(ValueError):
        _grouped_split(np.array(["a", "a", "b"]), test_size=0.5, valid_size=0.5, seed=1)
