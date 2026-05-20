# nncm

Neural Network Consequence Modelling — predicts release rate, velocity, distance to LFL, and flame
length from process pressure, temperature, and orifice diameter.

## Features

- Generates synthetic training scenarios via Latin Hypercube Sampling (LHS) across configurable
  pressure, temperature, and orifice diameter ranges.
- Produces 1 000 pressure vessels × 10 leaks each (10 000 total scenarios) in PHAST-compatible
  Excel format.
- Trains a multi-output residual neural network with four separate prediction heads.
- Applies log transform and StandardScaler normalisation for skewed targets; all transforms are
  stored alongside the model for consistent inference.
- Provides interactive consequence prediction through a PySide6 desktop GUI (nncm).
- Supports optional SHAP explainability plots for per-feature attribution.

## Requirements

- **Python** ≥ 3.10
- **tensorflow** ≥ 2.13 — GPU is optional; CPU inference works out of the box.
- **PySide6** ≥ 6.5 — required only for the `gui` subcommand; Qt runtime must be present.
- All other dependencies are resolved automatically during installation (see below).

## Installation

Clone the repository and install in editable mode with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/faiqraedaya/nncm
cd nncm
uv sync
```

Or with plain pip:

```bash
git clone https://github.com/faiqraedaya/nncm
cd nncm
pip install -e .
```

To include SHAP:

```bash
uv sync --extra shap
# or
pip install -e ".[shap]"
```

## Quick start

```bash
# 1. Generate LHS training data  →  data/generated/
nncm generate

# 2. Train the model             →  models/best_model.h5 + scalers
nncm train

# 3. Launch the prediction GUI
nncm gui
```

The same commands work via `python main.py` or `python -m nncm` from the repository root.

## Usage

### `nncm generate`

Runs Latin Hypercube Sampling to produce two Excel files:

| Output file | Contents |
|---|---|
| `data/generated/pressure_vessel_input.xlsx` | 1 000 vessels with sampled temperature and pressure |
| `data/generated/leak_input.xlsx` | 10 000 leaks linked to vessels, with sampled orifice diameters |

Also saves distribution plots to `data/plots/`.

Default sampling bounds (edit `src/nncm/data_generator.py` to change):

| Parameter | Min | Max | Unit |
|---|---|---|---|
| Temperature | −40 | 200 | °C |
| Pressure | 1 | 100 | barg |
| Orifice diameter | 1 | 1 000 | mm |

### `nncm train`

Reads `data/generated/training_data.csv` (columns: `Pressure`, `Temperature`, `Orifice_diameter`,
`Release_rate`, `Velocity`, `Distance_to_LFL`, `Flame_length`), trains the model, and writes
artifacts to `models/`:

```
models/
├── best_model.h5        # Keras model weights (best validation loss)
├── scaler_X.joblib      # StandardScaler for input features
├── scaler_y.joblib      # StandardScaler for targets (when SCALE_TARGETS=True)
└── target_meta.joblib   # Feature engineering config and log-transform metadata
```

Key hyperparameters (top of `src/nncm/neural_network.py`):

| Name | Default | Purpose |
|---|---|---|
| `EPOCHS` | 200 | Maximum training epochs |
| `BATCH_SIZE` | 256 | Mini-batch size |
| `LEARNING_RATE` | 3e-4 | Initial Adam learning rate |
| `PATIENCE_ES` | 30 | Early stopping patience |
| `PATIENCE_RLR` | 12 | ReduceLROnPlateau patience |
| `USE_LOG_TARGET` | `True` | Apply log1p to `Release_rate` before training |
| `SHAP_ENABLED` | `True` | Run SHAP summary plot after evaluation |

### `nncm gui`

Opens the nncm window. Enter pressure, temperature, and orifice diameter; click **Predict
Consequences** to see the four predicted outputs. The GUI loads model artifacts from `models/`
automatically.

## Configuration

No environment variables or config files are required. Behaviour is controlled by module-level
constants at the top of each source file:

| Constant | File | Purpose |
|---|---|---|
| `N_PRESSURE_VESSELS` / `N_LEAKS_PER_VESSEL` | `data_generator.py` | Dataset size |
| `T_MIN`, `T_MAX`, `P_MIN`, `P_MAX`, `D_MIN`, `D_MAX` | `data_generator.py` | LHS sampling bounds |
| `CSV_PATH` | `neural_network.py` | Path to training CSV |
| `MODEL_SAVE_PATH` | `neural_network.py` | Path to save the trained model |
| `USE_ENGINEERED_FEATURES` | `neural_network.py` | Enable log/interaction feature engineering |

## Project structure

```
2507P_NNCM/
├── main.py                   # Root entry point (delegates to nncm.__main__)
├── pyproject.toml            # Project metadata and dependencies
├── src/nncm/
│   ├── __init__.py
│   ├── __main__.py           # CLI (generate / train / gui subcommands)
│   ├── data_generator.py     # LHS sampling and Excel export
│   ├── neural_network.py     # Model training, evaluation, and artifact saving
│   └── gui_predictor.py      # PySide6 prediction GUI (nncm)
├── models/                   # Trained artifacts — gitignored
├── data/
│   ├── raw/                  # Reference NNCM output files — tracked
│   ├── generated/            # LHS outputs and training CSV — gitignored
│   └── plots/                # Distribution plots — gitignored
└── LICENSE
```

## License

This project is provided under the MIT License.