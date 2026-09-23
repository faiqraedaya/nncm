# nncm

**Neural Network Consequence Modelling** — a surrogate model for consequence analysis.
Sample release scenarios, run them through Phast/Safeti once, train a neural network on the
results, then predict consequences in milliseconds instead of hours of solver time.

```
sample  ──▶  export  ──▶  [ Phast ]  ──▶  import  ──▶  train  ──▶  predict
 design      workbook      you run it      dataset     model      app / CLI / CSV
```

Only the two Phast steps — importing the workbook and exporting the results — are manual.
Everything else lives in one application and one CLI, around a *project directory* that holds the
config, the sampled cases, the workbooks, the dataset and every trained model.

## Install

```bash
git clone https://github.com/faiqraedaya/nncm
cd nncm
uv sync
```

The Safeti input template is a client workbook and is not in the repository. Copy it to
`templates/Safeti Template Input Sheet.xlsx`, or set `phast.template` in a project's `nncm.json`.

## Use it

```bash
uv run nncm gui
```

The window works through the five stages in order — **Project · Sample · Phast · Train ·
Predict**. Each stage says what it needs and what it produced, and nothing runs before the stage
it depends on has finished.

The same workflow from the command line:

```bash
uv run nncm init                        # create ./workspace with a default config
uv run nncm sample --vessels 500        # sampling design      -> cases/cases.csv
uv run nncm export                      # Phast input workbook -> phast/input/*.xlsx

#   ... import that workbook into Phast, run the study, export the results workbook ...

uv run nncm import path/to/results.xlsx # training data        -> datasets/training_data.csv
uv run nncm train                       # trained model        -> models/run_<stamp>/
uv run nncm predict --temperature 25 --pressure 60 --orifice 25 --material METHANE --mc 50
uv run nncm predict --csv inputs.csv     # adds predictions and a domain_warnings column
```

Every command takes `--project/-p <dir>`; the default is `./workspace` in the current directory.

A material named at prediction time gets the property descriptors the model was trained with. A
material the model has not seen needs its descriptors as `mat_*` columns, or the prediction is
refused rather than answered for an average fluid.

## What each stage does

| Stage | What it does |
| --- | --- |
| **sample** | Builds the design of experiments — a Latin hypercube per material over temperature and pressure inside its own envelope, with stratified hole sizes. Vessel names carry a design ID, so two designs never collide. |
| **export** | Writes the cases into a copy of the Safeti template, as a workbook Phast will import. |
| **import** | Reads the Phast results workbook, joins it back onto the cases, and reports the scenarios that failed to converge instead of scoring them as zero. |
| **train** | Fits one network across all materials and consequences, split by vessel so the score measures generalisation to new equipment. |
| **predict** | Answers a single point or a CSV, with optional uncertainty, and flags inputs outside the training envelope of the named material, and settings the model never saw vary. |

Four consequences are trained by default — release rate, velocity, distance to LFL and flame
length. Around twenty are extracted from the results workbook, so others can be trained by naming
them in the config.

## Project layout

```
workspace/
├── nncm.json                  # single source of configuration
├── cases/cases.csv            # the sampling design (one row per leak scenario)
├── phast/input/*.xlsx         # workbooks to import into Phast
├── phast/output/*.xlsx        # where to drop Phast result workbooks
├── datasets/training_data.csv # extracted, joined training data
├── models/run_<stamp>/        # model.keras, scalers, meta.json, metrics, history
├── models/registry.json       # which run is current
└── plots/                     # design and diagnostic plots
```

## Configuration

Everything lives in `nncm.json`, editable in the app's **Project** stage: sampling ranges,
materials, the Phast template path and its per-column defaults, extraction filters and training
hyperparameters.

A material is either a pure component or a mixture — double-click a row of the Materials table to
edit one. Component names must match the Phast property system exactly: `NITROGEN
(ASPHYXIATING)`, not `NITROGEN`.

A new project starts with thirteen: seven pure fluids (methane, ethane, propane, n-butane,
ammonia, CO2 and hydrogen) and six process streams (natural gas, LNG, LPG, sour feed gas, NGL and
stabilised condensate), each carrying the temperature and pressure envelope it is stored at.
Compositions are representative — replace them with the stream data for the plant you are
modelling.

## Tests

```bash
uv run --with pytest python -m pytest tests -q
```

The workbook tests skip unless the template is present.

## How it works

[TECHNICAL.md](TECHNICAL.md) covers the parts worth knowing when a result looks wrong, or when
you need to trust a number: why the design is sampled the way it is, the mixture format, why the
workbook is patched rather than re-saved, how results are matched back to cases, and what the
model is actually fitting.

## License

MIT.
