"""Feature engineering — the single source of truth for train *and* inference.

The previous code duplicated this logic between the trainer and the GUI, which
means any change silently invalidates every saved model. Here the spec is
fitted once during training, serialised alongside the model, and replayed at
prediction time, so the two can never drift.

Features are chosen for physical meaning rather than volume: the dominant
scaling for a choked gas release is ``m_dot ~ P_abs * A * sqrt(MW / T)``, so
that group (and its parts, in log space) is given to the network directly. That
is what lets a network extrapolate across a wide input range instead of
memorising a grid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any  # noqa: F401  (used in annotations below)

import numpy as np
import pandas as pd

ATMOSPHERIC_PRESSURE_BARA = 1.01325

# Canonical input columns understood by the feature builder.
BASE_INPUTS = ["temperature_degC", "pressure_barg", "orifice_mm"]
WEATHER_INPUTS = ["wind_speed_ms", "stability_index"]
PROPERTY_PREFIX = "mat_"


@dataclass
class FeatureSpec:
    """Everything needed to rebuild the exact training feature matrix."""

    feature_columns: list[str] = field(default_factory=list)
    property_columns: list[str] = field(default_factory=list)
    material_categories: list[str] = field(default_factory=list)
    use_weather: bool = True
    use_material_properties: bool = True
    medians: dict[str, float] = field(default_factory=dict)
    input_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Observed range of every raw input; inference flags out-of-domain queries
    instead of silently extrapolating."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        bounds = {k: tuple(v) for k, v in (data.get("input_bounds") or {}).items()}
        return cls(
            feature_columns=list(data.get("feature_columns", [])),
            property_columns=list(data.get("property_columns", [])),
            material_categories=list(data.get("material_categories", [])),
            use_weather=bool(data.get("use_weather", True)),
            use_material_properties=bool(data.get("use_material_properties", True)),
            medians={k: float(v) for k, v in (data.get("medians") or {}).items()},
            input_bounds=bounds,
        )


def fit_feature_spec(
    frame: pd.DataFrame,
    use_weather: bool = True,
    use_material_properties: bool = True,
) -> FeatureSpec:
    """Decide which features exist for this dataset and record their statistics."""
    property_columns = (
        sorted(c for c in frame.columns if c.startswith(PROPERTY_PREFIX))
        if use_material_properties
        else []
    )
    property_columns = [c for c in property_columns if frame[c].notna().any()]

    # Fall back to one-hot material identity only when no numeric descriptors
    # are available — one-hot cannot generalise to an unseen material.
    material_categories: list[str] = []
    if not property_columns and "material" in frame.columns:
        material_categories = sorted(str(m) for m in frame["material"].dropna().unique())

    has_weather = use_weather and all(c in frame.columns for c in WEATHER_INPUTS)

    spec = FeatureSpec(
        property_columns=property_columns,
        material_categories=material_categories,
        use_weather=has_weather,
        use_material_properties=bool(property_columns),
    )

    built = _build(frame, spec, impute=False)
    spec.feature_columns = list(built.columns)
    spec.medians = {
        column: float(np.nanmedian(built[column].to_numpy(dtype=float)))
        if np.isfinite(built[column].to_numpy(dtype=float)).any()
        else 0.0
        for column in built.columns
    }
    raw_inputs = BASE_INPUTS + ([*WEATHER_INPUTS] if has_weather else [])
    spec.input_bounds = {
        column: (float(frame[column].min()), float(frame[column].max()))
        for column in raw_inputs
        if column in frame.columns and frame[column].notna().any()
    }
    return spec


def build_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the feature matrix for ``frame`` exactly as fitted."""
    built = _build(frame, spec, impute=True)
    for column in spec.feature_columns:
        if column not in built.columns:
            built[column] = spec.medians.get(column, 0.0)
    return built[spec.feature_columns]


def _build(frame: pd.DataFrame, spec: FeatureSpec, impute: bool) -> pd.DataFrame:
    missing = [c for c in BASE_INPUTS if c not in frame.columns]
    if missing:
        raise KeyError(f"missing required input columns: {missing}")

    temperature_c = frame["temperature_degC"].to_numpy(dtype=float)
    pressure_g = frame["pressure_barg"].to_numpy(dtype=float)
    orifice_mm = np.clip(frame["orifice_mm"].to_numpy(dtype=float), 1e-3, None)

    temperature_k = np.clip(temperature_c + 273.15, 1.0, None)
    pressure_abs = np.clip(pressure_g + ATMOSPHERIC_PRESSURE_BARA, 1e-3, None)
    area_mm2 = math.pi / 4.0 * orifice_mm**2

    # Accumulate in a dict and build the frame once — hundreds of one-hot
    # columns inserted individually is pathologically slow in pandas.
    columns: dict[str, np.ndarray] = {
        "temperature_K": temperature_k,
        "log_pressure_abs": np.log10(pressure_abs),
        "log_orifice": np.log10(orifice_mm),
        "log_area": np.log10(area_mm2),
        "pressure_ratio": pressure_abs / ATMOSPHERIC_PRESSURE_BARA,
        "inv_sqrt_temperature": 1.0 / np.sqrt(temperature_k),
        # Choked-flow group: log10(P_abs * A / sqrt(T)). Mass flow tracks this
        # almost exactly for gas releases, so the network only has to learn the
        # departures from it rather than the whole surface.
        "log_choked_group": np.log10(pressure_abs * area_mm2 / np.sqrt(temperature_k)),
    }

    if spec.use_material_properties and spec.property_columns:
        for column in spec.property_columns:
            columns[column] = (
                frame[column].to_numpy(dtype=float)
                if column in frame.columns
                else np.full(len(frame), np.nan)
            )
        molecular_weight = _property(frame, "mat_molecular_weight")
        critical_temperature = _property(frame, "mat_critical_temperature_K")
        critical_pressure = _property(frame, "mat_critical_pressure_bara")
        boiling_point = _property(frame, "mat_normal_boiling_point_K")
        if molecular_weight is not None:
            columns["log_mass_flux_group"] = np.log10(
                pressure_abs * area_mm2 * np.sqrt(np.clip(molecular_weight, 1e-3, None) / temperature_k)
            )
        if critical_temperature is not None:
            columns["reduced_temperature"] = temperature_k / np.clip(critical_temperature, 1e-3, None)
        if critical_pressure is not None:
            columns["reduced_pressure"] = pressure_abs / np.clip(critical_pressure, 1e-3, None)
        if boiling_point is not None:
            # Superheat relative to the normal boiling point separates gas jets,
            # flashing releases and subcooled liquid spills.
            columns["superheat_ratio"] = temperature_k / np.clip(boiling_point, 1e-3, None)

    if spec.material_categories:
        material = (
            frame["material"].astype(str).to_numpy()
            if "material" in frame.columns
            else np.full(len(frame), "")
        )
        for category in spec.material_categories:
            columns[f"material_is_{_slug(category)}"] = (material == category).astype(float)

    if spec.use_weather:
        wind = (
            frame["wind_speed_ms"].to_numpy(dtype=float)
            if "wind_speed_ms" in frame.columns
            else np.full(len(frame), np.nan)
        )
        columns["log_wind_speed"] = np.log10(np.clip(wind, 0.1, None))
        columns["stability_index"] = (
            frame["stability_index"].to_numpy(dtype=float)
            if "stability_index" in frame.columns
            else np.full(len(frame), np.nan)
        )

    features = pd.DataFrame(columns, index=frame.index).replace([np.inf, -np.inf], np.nan)
    if impute:
        for column in features.columns:
            fill = spec.medians.get(column, 0.0)
            features[column] = features[column].fillna(fill)
    return features


def _property(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in frame.columns:
        return None
    values = frame[name].to_numpy(dtype=float)
    return values if np.isfinite(values).any() else None


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower()


def domain_report(row: dict[str, Any], spec: FeatureSpec) -> dict[str, str]:
    """Flag inputs outside the training envelope (predictions there are guesses)."""
    flags: dict[str, str] = {}
    for column, (low, high) in spec.input_bounds.items():
        value = row.get(column)
        if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
            continue
        if value < low:
            flags[column] = f"{value:.4g} below training range ({low:.4g} .. {high:.4g})"
        elif value > high:
            flags[column] = f"{value:.4g} above training range ({low:.4g} .. {high:.4g})"

    # A one-hot model has no representation for an unseen material: every
    # indicator is zero, which is a point the network was never trained on.
    if spec.material_categories:
        material = row.get("material")
        if material is None or str(material).strip() == "":
            flags["material"] = (
                "no material given — the model was trained per material, so this "
                "prediction sits outside its inputs"
            )
        elif str(material) not in spec.material_categories:
            flags["material"] = (
                f"'{material}' was not in the training set "
                f"({len(spec.material_categories)} known materials)"
            )
    return flags
