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
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

ATMOSPHERIC_PRESSURE_BARA = 1.01325

# Canonical input columns understood by the feature builder.
BASE_INPUTS = ["temperature_degC", "pressure_barg", "orifice_mm"]
WEATHER_INPUTS = ["wind_speed_ms", "stability_index"]
PROPERTY_PREFIX = "mat_"

# Inputs the design may or may not vary. A varied one becomes a feature; a
# constant one is recorded as a fixed setting the model is only valid for.
OPTIONAL_INPUTS = {"elevation_m": "elevation_m", "mass_inventory_kg": "log_inventory"}

# Fallback log offset for a target with none configured, as a fraction of the
# median positive training value.
DEFAULT_OFFSET_FRACTION = 1e-3


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
    material_bounds: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    """Observed range of the base inputs per material. Each material is sampled
    inside its own envelope, so a point inside the global range can still be
    far outside anything the model saw for that fluid."""
    material_properties: dict[str, dict[str, float | None]] = field(default_factory=dict)
    """Descriptor values each training material carried. Prediction looks them
    up by name, so a query naming a material gets the properties the model was
    trained with, not the current config's or the training median."""
    variable_inputs: list[str] = field(default_factory=list)
    fixed_inputs: dict[str, float] = field(default_factory=dict)
    """Inputs held constant across the training data. The model says nothing
    about any other value of them."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        def bounds(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
            return {k: (float(v[0]), float(v[1])) for k, v in (raw or {}).items()}

        return cls(
            feature_columns=list(data.get("feature_columns", [])),
            property_columns=list(data.get("property_columns", [])),
            material_categories=list(data.get("material_categories", [])),
            use_weather=bool(data.get("use_weather", True)),
            use_material_properties=bool(data.get("use_material_properties", True)),
            medians={k: float(v) for k, v in (data.get("medians") or {}).items()},
            input_bounds=bounds(data.get("input_bounds")),
            material_bounds={
                m: bounds(b) for m, b in (data.get("material_bounds") or {}).items()
            },
            material_properties={
                m: dict(p) for m, p in (data.get("material_properties") or {}).items()
            },
            variable_inputs=list(data.get("variable_inputs", [])),
            fixed_inputs={k: float(v) for k, v in (data.get("fixed_inputs") or {}).items()},
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

    variable_inputs: list[str] = []
    fixed_inputs: dict[str, float] = {}
    for column in OPTIONAL_INPUTS:
        if column not in frame.columns or not frame[column].notna().any():
            continue
        values = frame[column].dropna().astype(float)
        if values.nunique() > 1:
            variable_inputs.append(column)
        else:
            fixed_inputs[column] = float(values.iloc[0])

    spec = FeatureSpec(
        property_columns=property_columns,
        material_categories=material_categories,
        use_weather=has_weather,
        use_material_properties=bool(property_columns),
        variable_inputs=variable_inputs,
        fixed_inputs=fixed_inputs,
    )

    built = _build(frame, spec, impute=False)
    spec.feature_columns = list(built.columns)
    spec.medians = {
        column: float(np.nanmedian(built[column].to_numpy(dtype=float)))
        if np.isfinite(built[column].to_numpy(dtype=float)).any()
        else 0.0
        for column in built.columns
    }
    raw_inputs = (
        BASE_INPUTS
        + ([*WEATHER_INPUTS] if has_weather else [])
        + variable_inputs
        + property_columns
    )
    spec.input_bounds = _bounds(frame, raw_inputs)

    if "material" in frame.columns:
        materials = frame.dropna(subset=["material"])
        for name, group in materials.groupby(materials["material"].astype(str)):
            spec.material_bounds[name] = _bounds(group, BASE_INPUTS)
            if property_columns:
                first = group[property_columns].iloc[0]
                spec.material_properties[name] = {
                    column: (float(first[column]) if pd.notna(first[column]) else None)
                    for column in property_columns
                }
    return spec


def _bounds(frame: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    return {
        column: (float(frame[column].min()), float(frame[column].max()))
        for column in columns
        if column in frame.columns and frame[column].notna().any()
    }


def resolve_materials(frame: pd.DataFrame, spec: FeatureSpec, strict: bool = False) -> pd.DataFrame:
    """Fill each row's material descriptors from the table saved with the model.

    Values given explicitly in ``frame`` win; the table fills the gaps. With
    ``strict``, a row the model cannot place — no descriptors and a material it
    never saw — raises instead of being imputed with training medians, which
    would silently answer for an average fluid rather than the one named.
    """
    if not spec.property_columns:
        return frame
    frame = frame.copy()
    for column in spec.property_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    if "material" in frame.columns and spec.material_properties:
        names = frame["material"].astype(str).str.strip()
        for column in spec.property_columns:
            lookup = names.map(
                lambda name, c=column: (spec.material_properties.get(name) or {}).get(c)
            )
            frame[column] = frame[column].astype(float).fillna(lookup.astype(float))

    if strict:
        known = set(spec.material_properties)
        named = (
            frame["material"].astype(str).str.strip()
            if "material" in frame.columns
            else pd.Series("", index=frame.index)
        )
        no_properties = frame[spec.property_columns].isna().all(axis=1)
        unplaced = no_properties & ~named.isin(known)
        if unplaced.any():
            offenders = sorted(set(named[unplaced].replace({"nan": "", "None": ""})))
            listed = ", ".join(repr(n) for n in offenders if n) or "no material given"
            raise ValueError(
                f"cannot place {int(unplaced.sum())} row(s) ({listed}): the model needs "
                f"either a material it was trained on ({', '.join(sorted(known)) or 'none recorded'}) "
                f"or the descriptors {', '.join(spec.property_columns)} as columns"
            )
    return frame


def build_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the feature matrix for ``frame`` exactly as fitted."""
    built = _build(resolve_materials(frame, spec), spec, impute=True)
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

    for column in spec.variable_inputs:
        values = (
            frame[column].to_numpy(dtype=float)
            if column in frame.columns
            else np.full(len(frame), np.nan)
        )
        if OPTIONAL_INPUTS[column].startswith("log_"):
            values = np.log10(np.clip(values, 1e-6, None))
        columns[OPTIONAL_INPUTS[column]] = values

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


def _number(value: Any) -> float | None:
    """A finite real number, or None. Accepts NumPy scalars, which pandas rows carry."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _check_range(flags: dict[str, str], column: str, value: float, low: float, high: float, where: str) -> None:
    if value < low:
        flags[column] = f"{value:.4g} below {where} ({low:.4g} .. {high:.4g})"
    elif value > high:
        flags[column] = f"{value:.4g} above {where} ({low:.4g} .. {high:.4g})"


def domain_report(row: dict[str, Any], spec: FeatureSpec) -> dict[str, str]:
    """Flag inputs outside the training envelope (predictions there are guesses)."""
    flags: dict[str, str] = {}
    material = row.get("material")
    material = "" if material is None or (isinstance(material, float) and math.isnan(material)) else str(material).strip()

    envelope = spec.material_bounds.get(material, {})
    for column, (low, high) in spec.input_bounds.items():
        value = _number(row.get(column))
        if value is None:
            continue
        _check_range(flags, column, value, low, high, "training range")
        if column not in flags and column in envelope:
            low_m, high_m = envelope[column]
            _check_range(flags, column, value, low_m, high_m, f"the range trained for {material}")

    for column, fixed in spec.fixed_inputs.items():
        value = _number(row.get(column))
        if value is not None and not math.isclose(value, fixed, rel_tol=1e-6, abs_tol=1e-9):
            flags[column] = (
                f"{value:.4g} given, but every training case used {fixed:.4g} — "
                "the model does not respond to this input"
            )

    # A one-hot model has no representation for an unseen material: every
    # indicator is zero, which is a point the network was never trained on.
    if spec.material_categories:
        if not material:
            flags["material"] = (
                "no material given — the model was trained per material, so this "
                "prediction sits outside its inputs"
            )
        elif material not in spec.material_categories:
            flags["material"] = (
                f"'{material}' was not in the training set "
                f"({len(spec.material_categories)} known materials)"
            )
    elif spec.property_columns and material and material not in spec.material_properties:
        flags["material"] = (
            f"'{material}' was not in the training set; predicted from its descriptors alone"
        )
    return flags


def domain_flags(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """:func:`domain_report` for every row, joined into one readable string per row."""
    resolved = resolve_materials(frame, spec)
    return pd.Series(
        [
            "; ".join(f"{column}: {message}" for column, message in domain_report(row, spec).items())
            for row in resolved.to_dict("records")
        ],
        index=frame.index,
        dtype=object,
    )


# ---------------------------------------------------------------------------
# Target transform
# ---------------------------------------------------------------------------
def fit_log_offset(values: np.ndarray, configured: float | None) -> float:
    """The offset ``c`` in ``log10(y + c)``.

    ``log1p`` is linear below about 0.1, which erases the difference between a
    1 mm and a 5 mm release. A plain log keeps every decade, and ``c`` sets the
    smallest value worth resolving while keeping zeros finite. Configured per
    target in physical units; otherwise a small fraction of the median.
    """
    if configured is not None and configured > 0:
        return float(configured)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.median(positive) * DEFAULT_OFFSET_FRACTION) if positive.size else 1e-6


def to_log_space(values: np.ndarray, offset: float) -> np.ndarray:
    """Log targets are physically non-negative; negatives are clipped to zero."""
    return np.log10(np.clip(values, 0.0, None) + offset)


def from_log_space(values: np.ndarray, offset: float) -> np.ndarray:
    return np.maximum(10.0**values - offset, 0.0)
