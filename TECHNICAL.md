# How nncm works

The parts of the pipeline that are decisions rather than plumbing — why each stage does what it
does, and where it will tell you it is unhappy. [README.md](README.md) covers installing and
running it.

## Sampling

Latin Hypercube (or Sobol) sampling over temperature, pressure and orifice diameter, with:

* **log-spaced pressure and hole size**, so the design covers 1 mm holes as densely as 500 mm
  ones — linear spacing puts 90 % of samples in the top decade and starves the small releases;
* **per-material envelopes**, so cryogenic and ambient fluids are each sampled where they are
  physically meaningful;
* **one design per material**, so each material's vessels are stratified across its own envelope
  (one design dealt out across all materials would leave each a random subset of it);
* **stratified hole sizes per vessel**, so every vessel spans the full size range;
* a fixed seed — the same config always produces the same design;
* **a design ID in every vessel name** (`D33C26_PV00001`), a hash of the sampling config. Results
  from two designs can be appended to one dataset without one scenario replacing another, and the
  grouped split never merges two different vessels that happen to share a number.

Samplers: `lhs` (default), `sobol`, `halton`, and `random` (plain uniform, the baseline).

Materials carry optional property descriptors (molecular weight, boiling point, critical point).
These become model inputs, which is what lets a single network cover several materials rather
than memorising one fluid.

### Materials are defined, not just named

A material with no components is a pure component; one with components is a mixture, given as
mole (or mass) percentages:

```json
{
  "name": "NATURAL GAS",
  "components": [
    {"component": "METHANE", "fraction": 91.58},
    {"component": "ETHANE", "fraction": 4.87},
    {"component": "NITROGEN (ASPHYXIATING)", "fraction": 1.18}
  ],
  "composition_basis": "mole"
}
```

A new project starts with a catalogue of thirteen — seven pure fluids and six process streams —
each with its own temperature and pressure envelope. The envelopes are the point: a design that
samples LNG at 200 degC or dense-phase CO2 at 1 barg spends solver time on states no plant ever
sees, and Phast will often decline to converge on them anyway.

Fractions are normalised to 100 % on export. Component names must match the Phast property system
exactly — `NITROGEN (ASPHYXIATING)`, not `NITROGEN`. In the application, the material editor
writes the composition as `METHANE 91.58, ETHANE 4.87, ...` alongside the material's ranges and
property descriptors.

## Writing the Phast input workbook

Cases are written into a copy of `templates/Safeti Template Input Sheet.xlsx` (a client workbook,
not in the repository — see the README) — a pre-configured
empty study carrying the weather set, parameter sets and terrain. Columns are addressed by
**Safeti attribute code**, read from the sheet's own header block rather than by position, and
values are converted into whatever unit the template declares, so a template configured in psi or
inches still receives correct numbers. `--split N` produces several workbooks of at most N leak
rows, never separating a leak from its vessel.

Each material used by a vessel is declared the way real project sheets do it: pure components as
a row on the **COMPONENT** sheet, mixtures as a block on the **MIXTURE** sheet whose first row
carries the name and property template and whose continuation rows add one component and fraction
each. Vessels then reference the material by name, and the export refuses to pass silently if a
vessel names a material the workbook never defines.

**The template is patched, not re-saved.** Loading the workbook with a spreadsheet library and
saving it produces a file Phast will not import: every string becomes an inline string, the
shared-string table is deleted, `fileVersion` is dropped and all 61 comment/VML parts are renamed.
Instead the writer edits the worksheet XML directly, so only the sheets that receive rows plus the
extended shared-string table change — five of the workbook's 322 parts for a study with mixtures,
four without. Everything else is byte-identical to the template.

The export keeps its footprint small — four sheets, about ten columns per equipment row — because
every extra cell is another rule Phast's importer can reject. Values written into enumeration
columns are checked against the template's own pick-lists, and each workbook is re-read after
writing to confirm the cells landed. Both surface in the export report.

If Phast still refuses a workbook, `nncm export --passthrough` writes an unmodified copy of the
template (verifiably byte-identical). If that is rejected too, the problem is the template or the
Phast version, not the generated rows.

## Extracting results

The result workbook's `Discharge`, `Flammable Dispersion`, `Jet fire`, `Explosions` and pool fire
sheets are joined on `path + scenario + weather`. Weather categories (`Category 5/D` → wind
5 m/s, stability D) are decoded into numeric features, and everything is merged back onto the
sampled cases so the true inputs — including material — travel with each target. Scenarios that
failed to converge in Phast are dropped and reported, not silently turned into zeros.

Three things make the join survive real projects:

* **Identity is matched, not assumed.** Where equipment sits in the study decides how Phast
  reports it: a flat study puts the leak name in the `Scenario` column and stops the path at the
  vessel, while a routed study runs the path down to the leak and puts the hole size in
  `Scenario`. Every candidate name is tried — the scenario label and each path segment — and a row
  that only identifies its vessel still inherits that vessel's material, conditions and grouping
  key. The report says how many rows matched by leak and how many by vessel.
* **Rows no case covers still get a grouping key.** Without a case table, or for results the
  table does not cover, the grouping key is the vessel state Phast echoes — material, temperature
  and pressure — which is exactly what the grouped split must keep on one side. Material
  descriptors are filled from the project's material catalogue.
* **Threshold columns are matched by level, not by value.** `Distance downwind to intensity level
  1 (4 kW/m2) (m)` and the same column at 6.3 kW/m² are the same quantity at a different study
  setting, so targets are named `Jet_fire_distance_level1`…`level3` and the threshold actually
  used is reported. Worth checking before merging datasets from different studies.

Roughly 20 quantities are extracted per scenario; the four trained by default are
`Release_rate`, `Velocity`, `Distance_to_LFL` and `Flame_length`.

## Training

A multi-output MLP — shared trunk, per-target heads, LayerNorm and swish — trained on
log-transformed targets, with:

* **`log10(y + c)` targets.** `log1p` is linear below about 0.1, so it cannot tell a 0.001 kg/s
  release from a 0.01 kg/s one; a plain log gives every decade the same weight. The offset `c`
  (`training.log_offsets`, in the target's own unit) is the smallest value worth resolving and
  keeps zeros finite. On a synthetic choked-flow set this cut the median error below 0.01 kg/s
  from 100 % to 18 %.
* **R² in log space** for log targets. On raw values a few large releases dominate the sum of
  squares, and a model that gets small releases badly wrong still scores about 0.9. The raw-value
  R² is kept as `r2_linear`.

* **grouped splitting by vessel.** Leaks from one vessel share temperature, pressure and material;
  splitting them at random puts near-copies in both train and test and inflates the score.
  Splitting by vessel measures generalisation to *new equipment*, which is what the model is for.
* **physically grouped features** — the choked-flow group `P·A/√T` (and `P·A·√(MW/T)` where
  molecular weight is known), reduced temperature and pressure, superheat ratio — so the network
  learns departures from the physics rather than the whole surface.
* **masked targets.** Phast does not compute every consequence for every scenario: a liquid
  release has no jet fire, a small one no explosion. A row is trained on wherever a target exists
  and masked out of that target's loss where it does not, so a dataset with 100 % release rates
  and 43 % LFL distances trains on all of it instead of only the complete rows. Coverage and
  per-target test counts are reported.
* **fixed settings recorded.** Inputs the design held constant (elevation, inventory) and the
  template's per-row defaults are saved with the run; a varied elevation or inventory becomes a
  feature instead. The model says nothing about any other value of a fixed setting, and
  prediction says so.
* **per-run artifacts.** Each run writes its own directory holding the model, scalers, feature
  spec, the descriptors of every training material, `metrics.json`, training history and test
  predictions. Runs never overwrite each other;
  `models/registry.json` records which one is current.

## Prediction

Single-point or batch (`--csv`) prediction, reported in real units. A material is looked up in
the run's own descriptor table, so later catalogue edits cannot change an old model's answer; an
unknown material without descriptors is refused. `--mc 50` runs MC-dropout passes, batched into
one call, to attach an uncertainty to each answer.

Inputs outside the training envelope are flagged as extrapolation rather than answered silently:
against the global range, against the range trained *for that material*, and for any fixed
setting given a different value. The model will still produce a number, and it is the flag that
tells you what it is worth. Batch predictions carry the flags in a `domain_warnings` column.

## Tests

```bash
uv run --with pytest python -m pytest tests -q
```

Covers the functions whose failure gives a wrong answer without an error: sampling envelopes and
stratification, unit conversion, workbook write and round-trip (needs the template), the result
join across study layouts, feature agreement between training and inference, material lookup and
domain checks, the grouped split, an end-to-end train and predict, and the GUI's task threading
and cancellation.
