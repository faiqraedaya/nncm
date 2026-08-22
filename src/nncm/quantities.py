"""Names, units and explanations for every quantity the interface shows.

One table, one entry per quantity, carrying all three things a reader needs:
the full name, the unit, and — where the term is not self-evident — one plain
sentence saying what it is. Labels kept apart from their explanations drift
apart, so they live next to each other here and every form, table header,
tooltip and chart axis reads from this module.

Keys are the identifiers the code already uses: a configuration field name, a
case-table column, a Phast target, or a metric. :func:`describe` degrades to a
readable label for anything not listed, so an unlisted target still renders.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    """What a figure is called, what it is measured in, and what it means."""

    label: str
    unit: str = ""
    help: str = ""

    def with_unit(self) -> str:
        """``"Release rate (kg/s)"`` — for a chart axis or a column head."""
        return f"{self.label} ({self.unit})" if self.unit else self.label


# ---------------------------------------------------------------------------
# Sampling design
# ---------------------------------------------------------------------------
_SAMPLING = {
    "n_vessels": Quantity(
        "Vessels", "",
        "Distinct vessels in the design. Each one draws its own material, "
        "temperature and pressure.",
    ),
    "n_leaks_per_vessel": Quantity(
        "Leaks per vessel", "",
        "Hole sizes sampled on every vessel. Scenarios = vessels x leaks.",
    ),
    "sampler": Quantity(
        "Sampler", "",
        "Latin hypercube spreads points evenly over the ranges; Sobol keeps "
        "filling the gaps as the count grows; random is the honest baseline.",
    ),
    "seed": Quantity(
        "Seed", "",
        "Fixes the draw, so the same design can be regenerated exactly.",
    ),
    "mass_inventory_kg": Quantity(
        "Inventory", "kg",
        "Mass given to every generated vessel. Large enough that steady-state "
        "releases are not limited by the inventory.",
    ),
    "temperature": Quantity(
        "Temperature", "degC",
        "Storage temperature range the vessels are drawn from.",
    ),
    "pressure": Quantity(
        "Pressure", "barg",
        "Storage gauge pressure range. Spans decades, so it is usually "
        "sampled in log space.",
    ),
    "orifice": Quantity(
        "Orifice diameter", "mm",
        "Hole size range. Spans decades, so it is usually sampled in log space.",
    ),
    "elevation": Quantity(
        "Elevation", "m",
        "Height of the release point above ground.",
    ),
    "log_spacing": Quantity(
        "Log-spaced", "",
        "Sample uniformly in log10 space — the right choice when the range "
        "spans decades, so small values are not swamped by large ones.",
    ),
}

# ---------------------------------------------------------------------------
# Phast exchange
# ---------------------------------------------------------------------------
_PHAST = {
    "template": Quantity(
        "Safeti template", "",
        "The workbook every export is built from. Its sheets and headers "
        "define what Phast will accept on import.",
    ),
    "study_name": Quantity("Study name", "", "Name given to the generated Phast study."),
    "max_rows_per_workbook": Quantity(
        "Leak rows per workbook", "",
        "Split the export into files of at most this many leak rows. Phast's "
        "import slows down badly on very large sheets. Blank keeps one file.",
    ),
    "export_path": Quantity(
        "Write to", "",
        "Where the Phast input workbook is written. Import this file into "
        "Phast, run the study, then export the results.",
    ),
    "result_path": Quantity(
        "Result workbook", "",
        "The workbook Phast exported after the run, holding the Discharge, "
        "Dispersion and Fire sheets.",
    ),
    "merge": Quantity(
        "Merge into the existing dataset", "",
        "Add these results to the dataset already in the project instead of "
        "replacing it. Rows for the same scenario are kept once.",
    ),
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
_TRAINING = {
    "epochs": Quantity(
        "Epochs", "",
        "Most passes over the training set. Early stopping usually ends the "
        "run before this.",
    ),
    "batch_size": Quantity("Batch size", "rows", "Rows per gradient step."),
    "learning_rate": Quantity(
        "Learning rate", "",
        "Step size for the optimiser. Too large diverges, too small crawls.",
    ),
    "dropout": Quantity(
        "Dropout", "fraction",
        "Share of units switched off during training. Also what the "
        "uncertainty estimate samples at prediction time.",
    ),
    "hidden_units": Quantity(
        "Hidden units", "",
        "Width of each shared layer, listed in order, e.g. 128, 128, 96.",
    ),
    "targets": Quantity(
        "Targets", "",
        "Consequence quantities the network learns to predict. Targets absent "
        "from the dataset are skipped with a note in the log.",
    ),
    "use_weather": Quantity(
        "Weather as input", "",
        "Give wind speed and stability class to the network. Needed whenever "
        "the dataset holds more than one weather category.",
    ),
    "use_material_properties": Quantity(
        "Material properties as input", "",
        "Give molecular weight, boiling point and the critical point to the "
        "network. This is what lets one model generalise across materials "
        "instead of memorising their names.",
    ),
    "mc_samples": Quantity(
        "Uncertainty samples", "",
        "Forward passes with dropout left on. Their spread is the plus-or-"
        "minus shown beside each prediction. Zero skips the estimate.",
    ),
}

# ---------------------------------------------------------------------------
# Case table and model inputs
# ---------------------------------------------------------------------------
_INPUTS = {
    "case_id": Quantity("Case", "", "Row identifier in the sampled design."),
    "vessel_id": Quantity(
        "Vessel", "",
        "Grouping key. Rows sharing a vessel share its material, temperature "
        "and pressure, so they are never split across train and test.",
    ),
    "vessel_name": Quantity("Vessel name", "", "Name written into the Phast study."),
    "leak_name": Quantity("Leak name", "", "Name of the leak scenario on its vessel."),
    "material": Quantity("Material", "", "Fluid released, as named in the Phast study."),
    "phase_hint": Quantity(
        "Phase", "",
        "Expected phase of the stored fluid. Reporting only — Phast decides "
        "the phase from the conditions.",
    ),
    "temperature_degC": Quantity("Temperature", "degC", "Storage temperature of the vessel."),
    "pressure_barg": Quantity("Pressure", "barg", "Storage gauge pressure of the vessel."),
    "orifice_mm": Quantity("Orifice diameter", "mm", "Diameter of the hole the fluid leaves through."),
    "elevation_m": Quantity("Elevation", "m", "Height of the release point above ground."),
    "wind_speed_ms": Quantity("Wind speed", "m/s", "Wind speed of the weather category."),
    "stability_index": Quantity(
        "Stability class", "",
        "Pasquill class A to F: A is strongly convective and disperses "
        "quickly, F is stable and holds a cloud together.",
    ),
    "weather": Quantity("Weather", "", "Weather category the result was computed for."),
}

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
_MATERIAL = {
    "name": Quantity("Material", "", "Name Phast will resolve; it must match the property system."),
    "composition": Quantity(
        "Composition", "mole %",
        "Components and their percentages, e.g. METHANE 90, ETHANE 7. Leave "
        "it empty for a pure component.",
    ),
    "weight": Quantity(
        "Weight", "",
        "Relative share of vessels given to this material, against the other "
        "weights in the table.",
    ),
    "composition_basis": Quantity("Composition basis", "", "Whether the percentages are by mole or by mass."),
    "material_phase_hint": Quantity("Phase", "", "Expected phase of the stored fluid. Reporting only."),
    "molecular_weight": Quantity("Molecular weight", "g/mol", "Mass of one mole of the fluid."),
    "normal_boiling_point_K": Quantity("Normal boiling point", "K", "Boiling point at atmospheric pressure."),
    "critical_temperature_K": Quantity("Critical temperature", "K", "Above this, no pressure will liquefy the fluid."),
    "critical_pressure_bara": Quantity("Critical pressure", "bara", "Pressure at the critical point."),
    "lfl_vol_frac": Quantity("Lower flammable limit", "fraction", "Leanest mixture in air that will still burn."),
}

# ---------------------------------------------------------------------------
# Predicted consequences
# ---------------------------------------------------------------------------
_TARGETS = {
    "Release_rate": Quantity("Release rate", "kg/s", "Peak mass leaving the hole each second."),
    "Velocity": Quantity("Release velocity", "m/s", "Speed of the fluid as it leaves the hole."),
    "Release_temperature": Quantity("Release temperature", "degC", "Temperature of the fluid at the hole."),
    "Liquid_mass_fraction": Quantity("Liquid mass fraction", "fraction", "Share of the release that leaves as liquid."),
    "Droplet_diameter": Quantity("Droplet diameter", "um", "Size of the droplets formed as the jet breaks up."),
    "Expanded_diameter": Quantity("Expanded diameter", "m", "Jet diameter once it has expanded to atmospheric pressure."),
    "Release_duration": Quantity("Release duration", "s", "Time from the start of the release to its end."),
    "Distance_to_LFL": Quantity(
        "Distance to LFL", "m",
        "Downwind distance at which the cloud has diluted to the lower "
        "flammable limit. Beyond it the cloud is too lean to ignite.",
    ),
    "Distance_to_LFL_fraction": Quantity("Distance to LFL fraction", "m", "Distance to a fraction of the lower flammable limit."),
    "Distance_to_UFL": Quantity("Distance to UFL", "m", "Distance at which the cloud has diluted to the upper flammable limit."),
    "Flash_fire_mass": Quantity("Flash fire mass", "kg", "Largest flammable mass held in the cloud at one time."),
    "Flame_length": Quantity("Flame length", "m", "Length of the jet fire, measured from the hole."),
    "Flame_emissive_power": Quantity("Flame emissive power", "kW/m2", "Heat radiated from the flame surface."),
    "Jet_fire_distance_level1": Quantity("Jet fire distance, level 1", "m", "Downwind distance to the first radiation threshold."),
    "Jet_fire_distance_level2": Quantity("Jet fire distance, level 2", "m", "Downwind distance to the second radiation threshold."),
    "Jet_fire_distance_level3": Quantity("Jet fire distance, level 3", "m", "Downwind distance to the third radiation threshold."),
    "Pool_diameter": Quantity("Pool diameter", "m", "Diameter of the liquid pool the release forms."),
    "Explosion_distance_level1": Quantity("Explosion distance, level 1", "m", "Downwind distance to the first overpressure threshold."),
    "Explosion_distance_level2": Quantity("Explosion distance, level 2", "m", "Downwind distance to the second overpressure threshold."),
    "Explosion_distance_level3": Quantity("Explosion distance, level 3", "m", "Downwind distance to the third overpressure threshold."),
}

# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
_METRICS = {
    "r2": Quantity(
        "R2", "",
        "Share of the variance the model reproduces on held-out vessels. "
        "1.0 is perfect; 0.0 is no better than always predicting the mean.",
    ),
    "mae": Quantity("MAE", "", "Mean absolute error, in the unit of the target."),
    "rmse": Quantity("RMSE", "", "Root mean squared error, in the unit of the target. Large errors count for more."),
    "median_ape": Quantity(
        "Median APE", "%",
        "Typical error as a percentage of the true value. The median, not the "
        "mean, so a handful of near-zero values cannot dominate it.",
    ),
    "log10_rmse": Quantity("log10 RMSE", "decades", "Error in log space — a factor rather than a difference."),
    "n_test": Quantity("Test rows", "rows", "Held-out rows the score was computed on."),
    "mean_r2": Quantity("Mean R2", "", "R2 averaged over every scored target."),
}

# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------
_RESULTS = {
    "quantity": Quantity("Quantity", "", "The consequence being reported."),
    "target": Quantity("Target", "", "Consequence quantity the score was computed for."),
    "prediction": Quantity("Prediction", "", "What the model predicts for these inputs."),
    "spread": Quantity(
        "Spread, 1 sigma", "",
        "One standard deviation across the dropout samples. It is the spread "
        "of the model's own opinion, not a validated confidence interval.",
    ),
    "value_unit": Quantity("Unit", "", "Unit the prediction is given in."),
}

QUANTITIES: dict[str, Quantity] = {
    **_SAMPLING, **_PHAST, **_TRAINING, **_INPUTS, **_MATERIAL, **_TARGETS, **_METRICS, **_RESULTS,
}


def describe(name: str) -> Quantity:
    """The registry entry for ``name``, or a readable fallback.

    An unlisted quantity is never a reason to show a blank or a raw identifier:
    the fallback turns ``Flame_length_2`` into ``Flame length 2`` and carries no
    unit, which is honest about what is known.
    """
    known = QUANTITIES.get(name)
    if known is not None:
        return known
    words = str(name).replace("_", " ").strip()
    return Quantity(words[:1].upper() + words[1:] if words else str(name))


def label(name: str) -> str:
    return describe(name).label


def unit(name: str) -> str:
    return describe(name).unit
