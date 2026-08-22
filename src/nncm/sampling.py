"""Case generation: space-filling sampling of vessel/leak scenarios.

Produces the canonical *case table* (one row per leak scenario) that every
later stage keys off: the Phast input writer turns it into a workbook, and the
result extractor joins Phast output back onto it by ``leak_name``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import Material, Range, SamplingConfig

MAX_MATERIAL_BARS = 10  # rows the vessels-per-material panel stays legible at

CASE_COLUMNS = [
    "case_id",
    "vessel_id",
    "vessel_name",
    "leak_name",
    "material",
    "phase_hint",
    "temperature_degC",
    "pressure_barg",
    "orifice_mm",
    "elevation_m",
    "mass_inventory_kg",
]


@dataclass
class SamplingReport:
    n_vessels: int
    n_leaks: int
    materials: dict[str, int]
    bounds: dict[str, tuple[float, float]]

    def summary(self) -> str:
        lines = [
            f"vessels: {self.n_vessels}",
            f"leak scenarios: {self.n_leaks}",
            "materials: " + ", ".join(f"{k}={v}" for k, v in sorted(self.materials.items())),
        ]
        lines += [f"{k}: {v[0]:.4g} .. {v[1]:.4g}" for k, v in self.bounds.items()]
        return "\n".join(lines)


def _engine(sampler: str, dim: int, seed: int) -> qmc.QMCEngine:
    if sampler == "sobol":
        return qmc.Sobol(d=dim, scramble=True, seed=seed)
    if sampler == "random":
        return qmc.Halton(d=dim, seed=seed)  # cheap stand-in, still reproducible
    return qmc.LatinHypercube(d=dim, seed=seed)


def _scale(unit_samples: np.ndarray, bounds: list[Range]) -> np.ndarray:
    """Map [0, 1) samples onto the requested ranges, honouring log scaling."""
    out = np.empty_like(unit_samples, dtype=float)
    for idx, rng in enumerate(bounds):
        column = unit_samples[:, idx]
        if rng.log:
            lo, hi = np.log10(rng.min), np.log10(rng.max)
            out[:, idx] = 10.0 ** (lo + column * (hi - lo))
        else:
            out[:, idx] = rng.min + column * (rng.max - rng.min)
    return out


def _assign_materials(config: SamplingConfig, rng: np.random.Generator) -> list[Material]:
    """Deterministic, weight-proportional allocation of vessels to materials."""
    weights = np.array([max(m.weight, 0.0) for m in config.materials], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(config.materials))
    shares = weights / weights.sum() * config.n_vessels
    counts = np.floor(shares).astype(int)
    remainder = config.n_vessels - counts.sum()
    if remainder > 0:  # hand the leftovers to the largest fractional parts
        order = np.argsort(-(shares - counts))
        counts[order[:remainder]] += 1
    assignment: list[Material] = []
    for material, count in zip(config.materials, counts):
        assignment.extend([material] * int(count))
    rng.shuffle(assignment)  # avoid material correlating with vessel index
    return assignment


def generate_cases(config: SamplingConfig) -> tuple[pd.DataFrame, SamplingReport]:
    """Generate the case table for the configured sampling plan."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    materials = _assign_materials(config, rng)

    # Vessel-level sample: temperature/pressure in the *unit cube*, rescaled per
    # material so each fluid stays inside its own physically sensible envelope.
    vessel_unit = _engine(config.sampler, 2, config.seed).random(config.n_vessels)

    temperatures = np.empty(config.n_vessels)
    pressures = np.empty(config.n_vessels)
    for i, material in enumerate(materials):
        t_range = material.temperature or config.temperature
        p_range = material.pressure or config.pressure
        scaled = _scale(vessel_unit[i : i + 1], [t_range, p_range])
        temperatures[i], pressures[i] = scaled[0, 0], scaled[0, 1]

    # Leak-level sample: one stratified draw per vessel so every vessel spans
    # the whole hole-size range instead of getting a random clump of sizes.
    n_leaks_each = config.n_leaks_per_vessel
    strata = np.linspace(0.0, 1.0, n_leaks_each + 1)
    leak_unit = np.empty((config.n_vessels, n_leaks_each))
    for i in range(config.n_vessels):
        offsets = rng.random(n_leaks_each)
        leak_unit[i] = strata[:-1] + offsets * (strata[1:] - strata[:-1])
        rng.shuffle(leak_unit[i])
    orifices = _scale(leak_unit.reshape(-1, 1), [config.orifice]).reshape(config.n_vessels, n_leaks_each)

    elevation_unit = rng.random((config.n_vessels, n_leaks_each, 1))
    elevations = _scale(elevation_unit.reshape(-1, 1), [config.elevation]).reshape(
        config.n_vessels, n_leaks_each
    )

    records = []
    case_id = 0
    for i, material in enumerate(materials):
        vessel_name = f"PV{i + 1:05d}"
        for j in range(n_leaks_each):
            case_id += 1
            records.append(
                {
                    "case_id": case_id,
                    "vessel_id": i + 1,
                    "vessel_name": vessel_name,
                    "leak_name": f"{vessel_name}_L{j + 1:02d}",
                    "material": material.name,
                    "phase_hint": material.phase_hint,
                    "temperature_degC": float(temperatures[i]),
                    "pressure_barg": float(pressures[i]),
                    "orifice_mm": float(orifices[i, j]),
                    "elevation_m": float(elevations[i, j]),
                    "mass_inventory_kg": float(config.mass_inventory_kg),
                }
            )

    cases = pd.DataFrame.from_records(records, columns=CASE_COLUMNS)

    # Material property descriptors ride along so the trainer can use them
    # without re-reading the config.
    property_names = sorted({k for m in config.materials for k in m.properties})
    if property_names:
        lookup = {m.name: m.properties for m in config.materials}
        for prop in property_names:
            cases[f"mat_{prop}"] = cases["material"].map(
                lambda name, p=prop: lookup.get(name, {}).get(p, np.nan)
            )

    report = SamplingReport(
        n_vessels=config.n_vessels,
        n_leaks=len(cases),
        materials=cases.groupby("material")["vessel_name"].nunique().to_dict(),
        bounds={
            "temperature_degC": (cases["temperature_degC"].min(), cases["temperature_degC"].max()),
            "pressure_barg": (cases["pressure_barg"].min(), cases["pressure_barg"].max()),
            "orifice_mm": (cases["orifice_mm"].min(), cases["orifice_mm"].max()),
        },
    )
    return cases, report


def save_cases(cases: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(path, index=False)
    return path


def load_cases(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_case_distributions(cases: pd.DataFrame, out_path: Path | None = None):
    """What the generated design covers. Returns the figure.

    Three questions, three forms. Where the vessels sit in the temperature and
    pressure envelope is a *relationship*, so it is a scatter; how the hole
    sizes are spread is a *distribution*, so it is a histogram in one hue; how
    many vessels each material got is a *magnitude* across nominal categories,
    so it is horizontal bars in one hue with no legend.

    There is deliberately no temperature or pressure marginal: the scatter
    already shows both along its own axes, and a panel that re-draws what the
    panel beside it has drawn costs room without answering anything.

    Colour carries material identity only while the categorical set can hold
    every material. Past that, colouring three of nine and leaving the rest
    grey would say more about the palette than about the design, so identity
    moves to the bar chart and the scatter drops to a single hue.

    The bar panel has a limit of its own — the rows it can hold before the
    names collide — and folds its smallest materials into a single "Other"
    bar rather than dropping them, so the panel still accounts for every
    vessel in the design. The subtitle says whenever either has happened.
    """
    # The figure API, not pyplot: this runs on a worker thread, and pyplot
    # would build the figure on whichever backend the window loaded — Qt
    # objects created off the UI thread, which is a crash waiting to happen.
    # It also keeps no global registry, so nothing leaks between runs.
    from matplotlib.figure import Figure

    from . import theme
    from .quantities import describe

    theme.apply_matplotlib_style()
    vessels = cases.drop_duplicates("vessel_name")
    per_material = vessels.groupby("material")["vessel_name"].nunique().sort_values()
    named = len(per_material) <= len(theme.SERIES)
    # The panel is a quarter of a figure that is itself only as tall as the
    # pane it is drawn into, so the row count is capped by what stays legible
    # rather than by what fits. The tail is folded, never dropped: the bars
    # still add up to every vessel in the design.
    if len(per_material) > MAX_MATERIAL_BARS:
        folded = len(per_material) - (MAX_MATERIAL_BARS - 1)
        rest = per_material.iloc[:folded]
        shown = pd.concat(
            [pd.Series({f"Other ({folded})": int(rest.sum())}), per_material.iloc[folded:]]
        )
    else:
        folded = 0
        shown = per_material

    fig = Figure(figsize=(10.4, 6.2))
    # The lower right panel gets the taller row: it holds one line per
    # material, and a plant with a dozen streams runs out of rows first.
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1.0, 1.3),
                            hspace=0.55, wspace=0.30)
    ax_conditions = fig.add_subplot(grid[:, 0])
    ax_orifice = fig.add_subplot(grid[0, 1])
    ax_materials = fig.add_subplot(grid[1, 1])

    caveats = [
        f"{len(cases):,} scenarios across {len(vessels):,} vessels; pressure and hole "
        "size on log axes, as they were sampled.",
    ]
    if not named:
        caveats.append(
            f"{len(per_material)} materials is past the {len(theme.SERIES)}-colour set, "
            "so the scatter is one hue and the bars below carry the names."
        )
    if folded:
        caveats.append(f"The {folded} smallest materials are counted together as Other.")

    # Lay the figure out before anything is drawn into it, so a mark sized in
    # screen units is sized against the axes it will actually occupy.
    top = theme.figure_header(fig, "Case design coverage", caveats)
    fig.subplots_adjust(top=top, left=0.07, right=0.965, bottom=0.10)

    # -- vessel conditions -------------------------------------------------
    if named:
        for colour, (material, group) in zip(
            theme.series_colors(len(per_material)), vessels.groupby("material")
        ):
            ax_conditions.scatter(
                group["temperature_degC"], group["pressure_barg"],
                s=14, alpha=0.75, color=colour, linewidths=0, label=material,
            )
        # Under the panel rather than over the points: a legend laid on a
        # scatter hides the data it is naming. markerscale lifts the handle to
        # a size a colour can actually be read at.
        ax_conditions.legend(
            loc="upper left", bbox_to_anchor=(0.0, -0.20), ncols=3,
            markerscale=2.4, handletextpad=0.3, columnspacing=1.4, borderaxespad=0.0,
        )
    else:
        ax_conditions.scatter(
            vessels["temperature_degC"], vessels["pressure_barg"],
            s=14, alpha=0.55, color=theme.SERIES[0], linewidths=0,
        )
    ax_conditions.set_yscale("log")
    ax_conditions.set_title("Vessel conditions")
    ax_conditions.set_xlabel(describe("temperature_degC").with_unit())
    ax_conditions.set_ylabel(describe("pressure_barg").with_unit())
    theme.style_axes(ax_conditions)

    # -- marginals ---------------------------------------------------------
    _histogram(ax_orifice, cases["orifice_mm"], "Hole size coverage",
               describe("orifice_mm").with_unit(), "Scenarios", theme)

    # -- vessels per material ---------------------------------------------
    positions = np.arange(len(shown))
    largest = float(shown.max()) if len(shown) else 1.0
    ax_materials.set_xlim(0, largest * 1.30)  # room for the count at the bar end
    ax_materials.set_ylim(-0.7, len(shown) - 0.3)
    # Many rows get thinner bars rather than a taller block, down to the floor
    # at which a fill stops reading as a bar.
    units_per_pixel = (len(shown) + 0.4) / max(ax_materials.bbox.height, 1.0)
    height = min(18.0 * units_per_pixel, 0.62)
    for position, (label, value) in enumerate(shown.items()):
        # Other is not an identity, so it does not take an identity's colour.
        colour = theme.SERIES_OTHER if str(label).startswith("Other (") else theme.SERIES[0]
        theme.rounded_barh(ax_materials, position, float(value), height, colour)
        ax_materials.text(
            float(value) + largest * 0.03, position, f"{int(value):,}",
            va="center", ha="left", fontsize=theme.FONT_CAPTION, color=theme.SUBTLE,
        )
    ax_materials.set_yticks(positions, list(shown.index))
    ax_materials.set_title("Vessels per material")
    theme.style_axes(ax_materials, grid="none")
    # The value sits at the end of every bar, so an axis repeating it would be
    # a second reading of the same figure.
    ax_materials.set_xticks([])
    ax_materials.spines["bottom"].set_visible(False)

    if out_path is not None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)  # a figure needs a canvas before it can be saved
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig


def _histogram(ax, values, title: str, label: str, counted: str, theme) -> None:
    """One series, one hue, no legend: bar length already carries the count.

    Binned in log space because the quantity was sampled in log space, but
    labelled in millimetres: an axis reading "log10 diameter" asks the reader
    to raise ten to a power before they can tell whether the design covers the
    holes they care about.
    """
    values = np.asarray(values, dtype=float)
    count = int(min(40, max(10, len(values) // 12)))
    low, high = float(values.min()), float(values.max())
    logged = low > 0 and high > low
    bins = np.logspace(np.log10(low), np.log10(high), count + 1) if logged else count
    ax.hist(values, bins=bins, color=theme.SERIES[0], edgecolor="none")
    if logged:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(counted)
    theme.style_axes(ax, grid="y")
