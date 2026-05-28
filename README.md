# nncm

Neural Network Consequence Modelling — predicts release rate, velocity, distance to LFL, and flame
length from process pressure, temperature, and orifice diameter.

## Features

### `nncm generate`

Latin Hypercube Sampling over configurable ranges (defaults below). Writes
`data/generated/pressure_vessel_input.xlsx` and `leak_input.xlsx` for use with external
consequence-modelling software.

| Parameter | Min | Max | Unit |
|---|---|---|---|
| Temperature | −40 | 200 | °C |
| Pressure | 1 | 100 | barg |
| Orifice diameter | 1 | 1 000 | mm |

### `nncm train`

Reads `data/generated/training_data.csv` (columns: `Pressure`, `Temperature`, `Orifice_diameter`,
`Release_rate`, `Velocity`, `Distance_to_LFL`, `Flame_length`) and writes:

```
models/
├── best_model.keras    # Trained model (Keras 3 native format)
├── scaler_X.joblib     # Input feature scaler
├── scaler_y.joblib     # Target scaler
└── target_meta.joblib  # Feature engineering + log-transform metadata
```

The trained network is a plain MLP (96 → 96 → 64) with LayerNorm, swish activations, and four
per-target heads. `Release_rate` and `Distance_to_LFL` are log1p-transformed before training to
handle wide dynamic ranges; targets are StandardScaled. MC Dropout uncertainty estimates are
reported per target after evaluation.

Key hyperparameters live at the top of `src/nncm/neural_network.py` (`EPOCHS`, `BATCH_SIZE`,
`LEARNING_RATE`, `DROPOUT_RATE`, `LOG_TARGET_COLUMNS`, etc.).

### `nncm gui`

PySide6 window that loads `models/best_model.keras` plus the scalers and metadata, then predicts
all four outputs from user-entered pressure, temperature, and orifice diameter.

## Installation

```bash
git clone https://github.com/faiqraedaya/nncm
cd nncm
uv sync
```

To include SHAP explainability:

```bash
uv sync --extra shap
```

## Usage

```bash
uv run nncm generate    # Generate LHS sampling data → data/generated/
uv run nncm train       # Train model → models/best_model.keras
uv run nncm gui         # Launch prediction GUI
```

## License

MIT.
