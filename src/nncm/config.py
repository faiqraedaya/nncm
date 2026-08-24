"""Project configuration and on-disk layout for NNCM.

A *project* is a directory holding everything for one modelling campaign:
config, sampled cases, Phast workbooks, extracted datasets and trained models.
Every module in the package takes a :class:`Project` rather than reaching for
module-level constants, so the GUI, the CLI and tests all operate on the same
state.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

TEMPLATE_NAME = "Safeti Template Input Sheet.xlsx"


def shipped_template() -> Path:
    """Where the template that ships with the application lives.

    Resolved on every call rather than frozen into a constant at import time,
    and never written into a project's configuration: a path to a file inside
    the installation is only true for the machine and the directory it was
    computed on. Stored in ``nncm.json`` it breaks as soon as the checkout
    moves, the project is opened elsewhere, or the app is frozen.
    """
    bundle_root = getattr(sys, "_MEIPASS", None)  # PyInstaller unpack directory
    if bundle_root:
        bundled = Path(bundle_root) / "templates" / TEMPLATE_NAME
        if bundled.is_file():
            return bundled
    return REPO_ROOT / "templates" / TEMPLATE_NAME


DEFAULT_TEMPLATE = shipped_template()
"""The shipped template, as resolved at import time.

Kept for callers that want the path itself. Anything reading a *project's*
template must go through :meth:`PhastConfig.template_path`, which falls back to
the shipped one when the project does not name a template of its own.
"""

CONFIG_FILENAME = "nncm.json"
CONFIG_VERSION = 5


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass
class Range:
    """Inclusive sampling range. ``log`` samples uniformly in log10 space."""

    min: float
    max: float
    log: bool = False

    def validate(self, name: str) -> None:
        if self.max < self.min:
            raise ValueError(f"{name}: max ({self.max}) < min ({self.min})")
        if self.log and self.min <= 0:
            raise ValueError(f"{name}: log sampling requires min > 0 (got {self.min})")


@dataclass
class MixtureComponent:
    """One component of a multi-component material.

    ``fraction`` is a percentage on the material's ``composition_basis``,
    matching how Safeti's MIXTURE sheet stores compositions (mole %, summing
    to 100). ``component`` must name a component in the Phast property system.
    """

    component: str
    fraction: float


@dataclass
class Material:
    """A material to sample over, and how it is declared to Phast.

    A material with no ``components`` is a pure component: it is declared on
    the COMPONENT sheet and referenced by name. A material with components is a
    mixture: it is declared on the MIXTURE sheet, one row per component, and
    vessels reference it by name in exactly the same way.

    Per-material range overrides keep each fluid inside a sensible envelope —
    the single biggest lever on how physically valid the generated study is.
    """

    name: str
    phase_hint: str = "any"          # any | vapour | liquid — used for reporting only
    components: list[MixtureComponent] = field(default_factory=list)
    composition_basis: str = "mole"  # mole | mass
    temperature: Range | None = None  # overrides SamplingConfig.temperature
    pressure: Range | None = None     # overrides SamplingConfig.pressure
    weight: float = 1.0               # relative share of vessels
    properties: dict[str, float] = field(default_factory=dict)
    """Optional numeric descriptors (molecular_weight, normal_boiling_point_K,
    critical_temperature_K, critical_pressure_bara, lfl_vol_frac, ...). When
    present they become model input features, which is what lets one network
    generalise across materials instead of memorising material IDs."""

    @property
    def is_mixture(self) -> bool:
        return bool(self.components)

    def normalised_components(self) -> list[MixtureComponent]:
        """Components rescaled to sum to 100 %, as the MIXTURE sheet expects."""
        total = sum(c.fraction for c in self.components)
        if total <= 0:
            raise ValueError(f"{self.name}: component fractions must sum to more than zero")
        return [
            MixtureComponent(c.component, c.fraction * 100.0 / total) for c in self.components
        ]

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("material name must not be empty")
        if self.composition_basis not in {"mole", "mass"}:
            raise ValueError(f"{self.name}: composition_basis must be 'mole' or 'mass'")
        for component in self.components:
            if not component.component.strip():
                raise ValueError(f"{self.name}: component name must not be empty")
            if component.fraction <= 0:
                raise ValueError(
                    f"{self.name}: fraction for '{component.component}' must be positive"
                )
        if self.temperature:
            self.temperature.validate(f"{self.name}.temperature")
        if self.pressure:
            self.pressure.validate(f"{self.name}.pressure")


def parse_composition(text: str) -> list[MixtureComponent]:
    """Parse ``"METHANE 90, ETHANE 7, PROPANE 3"`` into components.

    Accepts ``name value``, ``name: value`` or ``name=value``, separated by
    commas or semicolons. An empty string means the material is pure.
    """
    components: list[MixtureComponent] = []
    for chunk in re.split(r"[;,\n]", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"^(.*?)[\s:=]+([0-9]*\.?[0-9]+)\s*%?$", chunk)
        if not match:
            raise ValueError(f"cannot read composition entry '{chunk}' (expected 'NAME 12.5')")
        components.append(MixtureComponent(match.group(1).strip(), float(match.group(2))))
    return components


def format_composition(material: "Material") -> str:
    """Inverse of :func:`parse_composition`, for display and editing."""
    return ", ".join(f"{c.component} {c.fraction:g}" for c in material.components)


def default_materials() -> list["Material"]:
    """Starter catalogue: the fluids a hydrocarbon plant actually holds.

    Seven pure components and six process streams, each with the temperature
    and pressure envelope it is stored at. The envelopes are the point: a
    design that samples LNG at 200 degC or dense-phase CO2 at 1 barg spends
    solver time on states no plant ever sees, and Phast will often decline to
    converge on them anyway.

    Component names are spelled as the Phast property system spells them —
    ``NITROGEN (ASPHYXIATING)``, ``HYDROGEN SULFIDE`` with an f — and every one
    below was taken from a real Safeti study rather than guessed. A name Phast
    cannot resolve fails the import in a way that looks like a corrupt file.

    Compositions are representative, not authoritative: they are a plausible
    starting point to be replaced with the stream data for the plant being
    modelled.
    """
    return [
        # -- pure components ----------------------------------------------
        Material(
            "METHANE",
            "vapour",
            temperature=Range(-100.0, 60.0),
            pressure=Range(5.0, 150.0, log=True),
            properties={"molecular_weight": 16.04, "normal_boiling_point_K": 111.7,
                        "critical_temperature_K": 190.6, "critical_pressure_bara": 46.0},
        ),
        Material(
            "ETHANE",
            "any",
            temperature=Range(-88.0, 50.0),
            pressure=Range(2.0, 80.0, log=True),
            properties={"molecular_weight": 30.07, "normal_boiling_point_K": 184.6,
                        "critical_temperature_K": 305.3, "critical_pressure_bara": 48.7},
        ),
        Material(
            "PROPANE",
            "any",
            temperature=Range(-42.0, 55.0),
            pressure=Range(1.0, 25.0, log=True),
            properties={"molecular_weight": 44.10, "normal_boiling_point_K": 231.1,
                        "critical_temperature_K": 369.8, "critical_pressure_bara": 42.5},
        ),
        Material(
            "N-BUTANE",
            "any",
            temperature=Range(-10.0, 60.0),
            pressure=Range(0.5, 12.0, log=True),
            properties={"molecular_weight": 58.12, "normal_boiling_point_K": 272.7,
                        "critical_temperature_K": 425.1, "critical_pressure_bara": 38.0},
        ),
        Material(
            "AMMONIA",
            "any",
            temperature=Range(-33.0, 45.0),
            pressure=Range(0.5, 20.0, log=True),
            properties={"molecular_weight": 17.03, "normal_boiling_point_K": 239.8,
                        "critical_temperature_K": 405.5, "critical_pressure_bara": 113.3},
        ),
        Material(
            # Dense phase: below about 6 barg a release flashes to solid, which
            # is a different consequence problem from the one this samples.
            "CARBON DIOXIDE (TOXIC)",
            "any",
            temperature=Range(-30.0, 50.0),
            pressure=Range(6.0, 150.0, log=True),
            properties={"molecular_weight": 44.01, "normal_boiling_point_K": 194.7,
                        "critical_temperature_K": 304.1, "critical_pressure_bara": 73.8},
        ),
        Material(
            "HYDROGEN",
            "vapour",
            temperature=Range(-20.0, 150.0),
            pressure=Range(5.0, 200.0, log=True),
            properties={"molecular_weight": 2.02, "normal_boiling_point_K": 20.3,
                        "critical_temperature_K": 33.2, "critical_pressure_bara": 13.0},
        ),
        # -- process streams ----------------------------------------------
        Material(
            "NATURAL GAS",
            "vapour",
            components=[
                MixtureComponent("METHANE", 91.58),
                MixtureComponent("ETHANE", 4.87),
                MixtureComponent("PROPANE", 1.54),
                MixtureComponent("NITROGEN (ASPHYXIATING)", 1.18),
                MixtureComponent("N-BUTANE", 0.45),
                MixtureComponent("ISOBUTANE", 0.32),
                MixtureComponent("CARBON DIOXIDE (TOXIC)", 0.06),
            ],
            temperature=Range(-20.0, 60.0),
            pressure=Range(5.0, 150.0, log=True),
            properties={"molecular_weight": 17.36, "normal_boiling_point_K": 113.0,
                        "critical_temperature_K": 195.0, "critical_pressure_bara": 46.0},
        ),
        Material(
            "LNG",
            "liquid",
            components=[
                MixtureComponent("METHANE", 91.5),
                MixtureComponent("ETHANE", 5.5),
                MixtureComponent("PROPANE", 2.0),
                MixtureComponent("N-BUTANE", 0.6),
                MixtureComponent("NITROGEN (ASPHYXIATING)", 0.4),
            ],
            # Atmospheric storage at its bubble point: a few degrees of range,
            # and the tank pressure is inches of water, not bar.
            temperature=Range(-162.0, -150.0),
            pressure=Range(0.05, 6.0, log=True),
            properties={"molecular_weight": 17.9, "normal_boiling_point_K": 113.0,
                        "critical_temperature_K": 195.0, "critical_pressure_bara": 46.0},
        ),
        Material(
            "LPG",
            "any",
            components=[
                MixtureComponent("PROPANE", 60.0),
                MixtureComponent("N-BUTANE", 30.0),
                MixtureComponent("ISOBUTANE", 9.0),
                MixtureComponent("ETHANE", 1.0),
            ],
            temperature=Range(-42.0, 55.0),
            pressure=Range(1.0, 18.0, log=True),
            properties={"molecular_weight": 49.4, "normal_boiling_point_K": 245.0,
                        "critical_temperature_K": 390.0, "critical_pressure_bara": 40.0},
        ),
        Material(
            "SOUR FEED GAS",
            "vapour",
            components=[
                MixtureComponent("METHANE", 80.0),
                MixtureComponent("ETHANE", 6.0),
                MixtureComponent("CARBON DIOXIDE (TOXIC)", 4.0),
                MixtureComponent("PROPANE", 3.0),
                MixtureComponent("HYDROGEN SULFIDE", 3.0),
                MixtureComponent("NITROGEN (ASPHYXIATING)", 1.3),
                MixtureComponent("N-BUTANE", 1.2),
                MixtureComponent("ISOBUTANE", 0.8),
                MixtureComponent("N-PENTANE", 0.5),
                MixtureComponent("WATER", 0.2),
            ],
            temperature=Range(10.0, 80.0),
            pressure=Range(20.0, 120.0, log=True),
            properties={"molecular_weight": 20.7, "normal_boiling_point_K": 120.0,
                        "critical_temperature_K": 210.0, "critical_pressure_bara": 48.0},
        ),
        Material(
            "NGL",
            "any",
            components=[
                MixtureComponent("PROPANE", 25.0),
                MixtureComponent("N-BUTANE", 20.0),
                MixtureComponent("ISOBUTANE", 12.0),
                MixtureComponent("ISOPENTANE", 12.0),
                MixtureComponent("N-PENTANE", 10.0),
                MixtureComponent("N-HEXANE", 9.0),
                MixtureComponent("N-HEPTANE", 6.0),
                MixtureComponent("BENZENE", 3.0),
                MixtureComponent("TOLUENE", 3.0),
            ],
            temperature=Range(-20.0, 60.0),
            pressure=Range(5.0, 30.0, log=True),
            properties={"molecular_weight": 64.4, "normal_boiling_point_K": 290.0,
                        "critical_temperature_K": 450.0, "critical_pressure_bara": 35.0},
        ),
        Material(
            "STABILISED CONDENSATE",
            "liquid",
            components=[
                MixtureComponent("N-HEXANE", 22.0),
                MixtureComponent("N-HEPTANE", 20.0),
                MixtureComponent("N-OCTANE", 16.0),
                MixtureComponent("N-NONANE", 12.0),
                MixtureComponent("TOLUENE", 12.0),
                MixtureComponent("M-XYLENE", 10.0),
                MixtureComponent("BENZENE", 8.0),
            ],
            temperature=Range(10.0, 80.0),
            pressure=Range(0.5, 10.0, log=True),
            properties={"molecular_weight": 100.6, "normal_boiling_point_K": 385.0,
                        "critical_temperature_K": 570.0, "critical_pressure_bara": 28.0},
        ),
    ]


@dataclass
class SamplingConfig:
    n_vessels: int = 500
    n_leaks_per_vessel: int = 6
    temperature: Range = field(default_factory=lambda: Range(-160.0, 250.0))
    pressure: Range = field(default_factory=lambda: Range(0.5, 200.0, log=True))
    orifice: Range = field(default_factory=lambda: Range(1.0, 500.0, log=True))
    elevation: Range = field(default_factory=lambda: Range(1.0, 1.0))
    sampler: str = "lhs"              # lhs | sobol | random
    seed: int = 42
    materials: list[Material] = field(default_factory=lambda: default_materials())
    mass_inventory_kg: float = 50_000.0
    """Inventory given to every generated vessel. Large enough that steady-state
    releases are not inventory-limited; override for time-varying studies."""

    def validate(self) -> None:
        if self.n_vessels < 1 or self.n_leaks_per_vessel < 1:
            raise ValueError("n_vessels and n_leaks_per_vessel must be >= 1")
        if not self.materials:
            raise ValueError("at least one material is required")
        self.temperature.validate("temperature")
        self.pressure.validate("pressure")
        self.orifice.validate("orifice")
        names = [m.name.strip().casefold() for m in self.materials]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate material names: {sorted(duplicates)}")
        for material in self.materials:
            material.validate()
        if self.sampler not in {"lhs", "sobol", "random"}:
            raise ValueError(f"unknown sampler '{self.sampler}'")


# ---------------------------------------------------------------------------
# Phast workbook I/O
# ---------------------------------------------------------------------------
@dataclass
class PhastConfig:
    template: str = ""
    """A workbook to use instead of the shipped template.

    Empty — the default — means the template that ships with the application,
    resolved at use time by :meth:`template_path`. The shipped template's own
    path is deliberately never stored here: it is an absolute path into the
    installation directory, so a project carrying it stops working the moment
    the checkout moves or the project is opened on another machine.
    """
    study_name: str = "Study"
    folder_name: str = ""
    """Optional model folder for the generated equipment. Empty keeps the study
    flat, exactly as the template ships."""
    write_material_rows: bool = True
    """Declare every sampled material in the workbook — pure components on the
    COMPONENT sheet, mixtures on the MIXTURE sheet with one row per component.
    A vessel referencing a material the study does not define is an incomplete
    model, so leave this on unless the materials already exist in the target
    study."""
    vessel_defaults: dict[str, Any] = field(
        default_factory=lambda: {
            # The minimum that makes a sampled scenario unambiguous: inventory
            # given as mass, and temperature/pressure read as the specified
            # condition. Anything beyond this is another validation rule Phast
            # can trip over on import.
            "SpecifyVolumeInFlag": "0 No",
            "FlashFlag": "1 Pressure/temperature",
        }
    )
    leak_defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "SupplyDischargeCoefficient": "0 No",
            "ReleaseDirection": "0 Horizontal",
        }
    )
    max_rows_per_workbook: int = 0
    """Split the export into several workbooks of at most this many leak rows
    (0 = single file). Phast import slows down badly on very large sheets."""
    verify_after_write: bool = True
    """Re-open each written workbook and check the values read back. Catches a
    malformed export here rather than at the Phast import dialog."""

    def template_path(self) -> Path:
        """The workbook this project writes into: its own, or the shipped one."""
        return Path(self.template) if self.template else shipped_template()


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
@dataclass
class ExtractionConfig:
    weather_filter: list[str] = field(default_factory=list)
    """Keep only these weather categories (empty = keep all). Weather is a real
    model input, so keeping several categories is usually what you want."""
    require_targets: list[str] = field(
        default_factory=lambda: ["Release_rate", "Velocity"]
    )
    """Rows missing any of these are dropped as failed/unconverged scenarios."""
    drop_nonpositive: list[str] = field(
        default_factory=lambda: ["Release_rate", "Velocity"]
    )
    drop_unmatched: bool = False
    """Discard result rows with no matching sampled case. Off by default so
    results from existing Phast projects can be folded into the dataset."""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    targets: list[str] = field(
        default_factory=lambda: [
            "Release_rate",
            "Velocity",
            "Distance_to_LFL",
            "Flame_length",
        ]
    )
    log_targets: list[str] = field(
        default_factory=lambda: [
            "Release_rate",
            "Velocity",
            "Distance_to_LFL",
            "Flame_length",
        ]
    )
    """All four spans decades — log space is the right target space for each."""
    group_column: str = "vessel_id"
    """Rows sharing a vessel share T, P and material; splitting them across
    train/test leaks information and inflates the reported scores."""
    use_material_properties: bool = True
    use_weather: bool = True
    hidden_units: list[int] = field(default_factory=lambda: [128, 128, 96])
    head_units: list[int] = field(default_factory=lambda: [48, 24])
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    epochs: int = 400
    patience_early_stop: int = 40
    patience_reduce_lr: int = 15
    test_size: float = 0.15
    valid_size: float = 0.15
    mc_samples: int = 50
    seed: int = 42


@dataclass
class NncmConfig:
    version: int = CONFIG_VERSION
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    phast: PhastConfig = field(default_factory=PhastConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Not a field: set when loading upgraded an older file, so the caller can
    # write the upgraded version back to disk.
    migrated = False

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NncmConfig":
        upgraded = _migrate(data)
        config = _build(cls, upgraded)
        config.migrated = upgraded is not data
        return config

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "NncmConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def validate(self) -> None:
        self.sampling.validate()
        if not self.training.targets:
            raise ValueError("training.targets must not be empty")
        unknown = set(self.training.log_targets) - set(self.training.targets)
        if unknown:
            raise ValueError(f"log_targets not in targets: {sorted(unknown)}")


# Settings a previous version wrote by default and that we no longer want,
# because every extra column written is another thing Phast can object to on
# import. Only reverted when the stored value is still the old default, so a
# deliberate customisation survives.
_RETIRED_PHAST_DEFAULTS = {
    "vessel_defaults": {"TankType": "1 Vertical cylinder", "RiskEffects": "1 Flammable only"},
    "leak_defaults": {"EventFrequency": 1e-4},
}
_RETIRED_PHAST_VALUES = {"folder_name": ("NNCM", "")}


def _drop_stale_template(phast: dict[str, Any]) -> None:
    """Forget a stored path that is really just the shipped template.

    Versions up to 4 wrote the shipped template's absolute path into every
    project. That path is only true for the machine and the directory it was
    computed on, so moving the checkout left projects pointing at a file that
    was never theirs to name.

    Cleared only when the stored path no longer resolves *and* carries the
    shipped template's own filename. A project naming a genuinely custom
    workbook keeps it, and still reports it missing if it has moved — that is
    a path the user chose, and silently swapping it for a different template
    would change what gets exported.
    """
    stored = phast.get("template")
    if not isinstance(stored, str) or not stored:
        return
    path = Path(stored)
    if path.name != TEMPLATE_NAME:
        return          # a custom template: the user's to fix, not ours
    if path == shipped_template() or not path.is_file():
        phast["template"] = ""


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring an older ``nncm.json`` up to the current schema."""
    version = int(data.get("version", 1) or 1)
    if version >= CONFIG_VERSION:
        return data

    data = json.loads(json.dumps(data))  # don't mutate the caller's dict
    phast = data.get("phast")
    if isinstance(phast, dict):
        for section, retired in _RETIRED_PHAST_DEFAULTS.items():
            defaults = phast.get(section)
            if isinstance(defaults, dict):
                for key, old_value in retired.items():
                    if defaults.get(key) == old_value:
                        defaults.pop(key)
        for key, (old_value, new_value) in _RETIRED_PHAST_VALUES.items():
            if phast.get(key) == old_value:
                phast[key] = new_value
        if version < 4:
            # v3 briefly stopped declaring materials to keep the export small.
            # That leaves vessels pointing at materials the study never defines,
            # which is an incomplete model — always declare them.
            phast["write_material_rows"] = True
        if version < 5:
            _drop_stale_template(phast)

    data["version"] = CONFIG_VERSION
    return data


def _build(cls: type, data: Any) -> Any:
    """Recursively rebuild nested dataclasses from plain dicts."""
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type if not isinstance(f.type, str) else _resolve(f.type)
        kwargs[f.name] = _coerce(ftype, value)
    return cls(**kwargs)


def _resolve(name: str) -> Any:
    return {
        "Range": Range,
        "Range | None": Range,
        "Material": Material,
        "list[Material]": list,
        "SamplingConfig": SamplingConfig,
        "PhastConfig": PhastConfig,
        "ExtractionConfig": ExtractionConfig,
        "TrainingConfig": TrainingConfig,
    }.get(name, None)


def _material_from_dict(data: dict[str, Any]) -> Material:
    data = dict(data)
    for key in ("temperature", "pressure"):
        if isinstance(data.get(key), dict):
            data[key] = Range(**data[key])
    data["components"] = [
        MixtureComponent(**c) if isinstance(c, dict) else c
        for c in data.get("components", []) or []
    ]
    known = {f.name for f in fields(Material)}
    return Material(**{k: v for k, v in data.items() if k in known})


def _coerce(ftype: Any, value: Any) -> Any:
    if value is None:
        return None
    if ftype is Range and isinstance(value, dict):
        return Range(**value)
    if ftype is list and isinstance(value, list):
        return [_material_from_dict(v) if isinstance(v, dict) else v for v in value]
    if is_dataclass(ftype) and isinstance(value, dict):
        return _build(ftype, value)
    return value


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
@dataclass
class Project:
    """Directory layout for one modelling campaign."""

    root: Path
    config: NncmConfig = field(default_factory=NncmConfig)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    # -- paths ------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def cases_dir(self) -> Path:
        return self.root / "cases"

    @property
    def cases_path(self) -> Path:
        return self.cases_dir / "cases.csv"

    @property
    def phast_dir(self) -> Path:
        return self.root / "phast"

    @property
    def phast_input_dir(self) -> Path:
        return self.phast_dir / "input"

    @property
    def phast_output_dir(self) -> Path:
        return self.phast_dir / "output"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def training_data_path(self) -> Path:
        return self.datasets_dir / "training_data.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def registry_path(self) -> Path:
        return self.models_dir / "registry.json"

    def run_dir(self, run_id: str) -> Path:
        return self.models_dir / run_id

    def ensure_dirs(self) -> None:
        for directory in (
            self.cases_dir,
            self.phast_input_dir,
            self.phast_output_dir,
            self.datasets_dir,
            self.models_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def create(cls, root: Path, config: NncmConfig | None = None, overwrite: bool = False) -> "Project":
        project = cls(Path(root), config or NncmConfig())
        project.ensure_dirs()
        if project.config_path.exists() and not overwrite:
            project.config = NncmConfig.load(project.config_path)
            if project.config.migrated:
                # Persist the upgrade so the file on disk matches what runs.
                project.save_config()
                project.config.migrated = False
        else:
            project.config.save(project.config_path)
        return project

    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root).resolve()
        config_path = root / CONFIG_FILENAME
        config = NncmConfig.load(config_path) if config_path.exists() else NncmConfig()
        if config.migrated and config_path.exists():
            config.save(config_path)
            config.migrated = False
        return cls(root, config)

    def save_config(self) -> Path:
        return self.config.save(self.config_path)

    # -- model registry ---------------------------------------------------
    def registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"latest": None, "runs": []}

    def register_run(self, run_id: str, metrics: dict[str, Any]) -> None:
        registry = self.registry()
        registry["runs"] = [r for r in registry.get("runs", []) if r.get("run_id") != run_id]
        registry["runs"].append({"run_id": run_id, "metrics": metrics})
        registry["latest"] = run_id
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def latest_run_dir(self) -> Path | None:
        latest = self.registry().get("latest")
        if latest:
            candidate = self.run_dir(latest)
            if candidate.exists():
                return candidate
        runs = sorted((p for p in self.models_dir.glob("*") if (p / "model.keras").exists()))
        return runs[-1] if runs else None


def default_project_root() -> Path:
    """Repo-local workspace used when the user does not name a project."""
    return REPO_ROOT / "workspace"
