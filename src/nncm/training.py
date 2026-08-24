"""Model training.

Two things differ substantially from the original script:

* **Grouped splitting.** Leaks from one vessel share temperature, pressure and
  material. Splitting them at random puts near-identical rows in train *and*
  test, so the reported score measures interpolation between siblings rather
  than generalisation. Splitting by vessel gives an honest number.
* **Everything is logged to disk.** Each run gets its own directory with the
  model, scalers, feature spec, metrics and history, and the project registry
  records which run is current. Runs stop overwriting each other.
"""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import Project, TrainingConfig
from .features import FeatureSpec, build_features, fit_feature_spec

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


@dataclass
class TrainingResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    targets: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}  ({self.n_train} train / {self.n_val} val / {self.n_test} test rows)",
            f"{'target':<28}{'R2':>9}{'MAE':>12}{'RMSE':>12}{'MedAPE %':>11}",
        ]
        for target in self.targets:
            m = self.metrics.get("per_target", {}).get(target, {})
            if "r2" not in m:
                lines.append(f"{target:<28}{'too few test rows':>44}")
                continue
            lines.append(
                f"{target:<28}{m.get('r2', float('nan')):>9.4f}"
                f"{m.get('mae', float('nan')):>12.4g}"
                f"{m.get('rmse', float('nan')):>12.4g}"
                f"{m.get('median_ape', float('nan')):>11.2f}"
            )
        return "\n".join(lines)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def _grouped_split(
    groups: np.ndarray, test_size: float, valid_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices by group so no group spans two partitions."""
    unique = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_size))) if test_size > 0 else 0
    n_valid = max(1, int(round(len(unique) * valid_size))) if valid_size > 0 else 0
    if n_test + n_valid >= len(unique):
        raise ValueError(
            f"not enough groups ({len(unique)}) for the requested split; "
            "reduce test_size/valid_size or gather more vessels"
        )
    test_groups = set(unique[:n_test].tolist())
    valid_groups = set(unique[n_test : n_test + n_valid].tolist())
    is_test = np.array([g in test_groups for g in groups])
    is_valid = np.array([g in valid_groups for g in groups])
    is_train = ~(is_test | is_valid)
    return np.where(is_train)[0], np.where(is_valid)[0], np.where(is_test)[0]


def _fit_masked_scaler(values: np.ndarray):
    """StandardScaler fitted while ignoring missing target values.

    ``StandardScaler.fit`` cannot see past NaNs, and imputing before fitting
    would shrink the standard deviations; the statistics are computed
    NaN-aware and installed directly instead.
    """
    from sklearn.preprocessing import StandardScaler

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = std**2
    scaler.n_features_in_ = values.shape[1]
    scaler.n_samples_seen_ = int(np.isfinite(values).sum(axis=0).max())
    return scaler


def _build_model(input_dim: int, targets: list[str], config: TrainingConfig):
    from tensorflow.keras import Input, Model, layers, regularizers

    l2 = regularizers.l2(config.weight_decay)

    def block(x, units: int, name: str, dropout: bool = True):
        x = layers.Dense(units, kernel_regularizer=l2, name=f"{name}_dense")(x)
        # LayerNorm, not BatchNorm: identical behaviour in train and eval mode.
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        x = layers.Activation("swish", name=f"{name}_act")(x)
        if dropout and config.dropout > 0:
            x = layers.Dropout(config.dropout, name=f"{name}_drop")(x)
        return x

    inputs = Input(shape=(input_dim,), name="features")
    x = inputs
    for idx, units in enumerate(config.hidden_units, start=1):
        x = block(x, units, f"trunk{idx}")

    outputs = []
    for target in targets:
        h = x
        for idx, units in enumerate(config.head_units, start=1):
            h = block(h, units, f"{target}_h{idx}", dropout=False)
        outputs.append(layers.Dense(1, activation="linear", name=f"{target}_output")(h))
    return Model(inputs=inputs, outputs=outputs, name="consequence_model")


def train_model(
    project: Project,
    frame: pd.DataFrame | None = None,
    config: TrainingConfig | None = None,
    log: LogFn | None = None,
    run_id: str | None = None,
    progress: ProgressFn | None = None,
) -> TrainingResult:
    """Train on the project's dataset and save a self-contained run directory."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    import tensorflow as tf
    from tensorflow.keras import Model, callbacks, layers

    config = config or project.config.training
    emit: LogFn = log or (lambda message: print(message))
    _seed_everything(config.seed)
    tf.random.set_seed(config.seed)

    if frame is None:
        if not project.training_data_path.exists():
            raise FileNotFoundError(
                f"no training data at {project.training_data_path}; run the Phast import first"
            )
        frame = pd.read_csv(project.training_data_path)

    targets = [t for t in config.targets if t in frame.columns]
    missing = [t for t in config.targets if t not in frame.columns]
    if missing:
        emit(f"! targets absent from dataset, skipped: {', '.join(missing)}")
    if not targets:
        raise ValueError("none of the configured targets are present in the dataset")

    # A row is usable when *any* target is present. Phast does not produce every
    # consequence for every scenario — a liquid release has no jet fire, a small
    # one no explosion — so requiring all targets would throw away most of the
    # study. Missing targets are masked out of the loss instead.
    present = frame[targets].notna()
    usable = frame[present.any(axis=1)].reset_index(drop=True)
    present = usable[targets].notna()
    dropped = len(frame) - len(usable)
    if dropped:
        emit(f"dropped {dropped} rows with no target values at all ({len(usable)} remain)")
    for target in targets:
        coverage = float(present[target].mean())
        emit(f"  {target:<28} {int(present[target].sum()):>6} rows ({coverage * 100:.1f} %)")
        if present[target].sum() < 30:
            emit(f"! {target} has too few values to learn from — consider dropping it")
    if len(usable) < 50:
        raise ValueError(f"only {len(usable)} usable rows — not enough to train")

    spec = fit_feature_spec(
        usable,
        use_weather=config.use_weather,
        use_material_properties=config.use_material_properties,
    )
    X = build_features(usable, spec).to_numpy(dtype=float)
    y = usable[targets].to_numpy(dtype=float)
    emit(f"features ({len(spec.feature_columns)}): {', '.join(spec.feature_columns)}")

    log_targets = [t for t in config.log_targets if t in targets]
    mask = present.to_numpy(dtype=float)  # 1 where the target was computed
    y_transformed = y.copy()
    shifts: dict[str, float] = {}
    # A log-transformed target is non-negative by construction, so inverting the
    # transform must not produce a negative rate or distance. Where a shift was
    # needed the data did contain non-positive values, and the observed minimum
    # is the honest floor.
    floors: dict[str, float] = {}
    for idx, target in enumerate(targets):
        if target in log_targets:
            column = y[:, idx]
            shift = 0.0
            minimum = np.nanmin(column)
            if minimum <= 0:
                shift = abs(minimum) + 1e-6
            shifts[target] = shift
            floors[target] = 0.0 if shift == 0.0 else float(minimum)
            y_transformed[:, idx] = np.log1p(column + shift)

    group_column = config.group_column if config.group_column in usable.columns else None
    if group_column is None:
        emit(
            f"! group column '{config.group_column}' not in dataset — falling back to a random split; "
            "reported scores will be optimistic"
        )
        groups = np.arange(len(usable))
    else:
        groups = usable[group_column].astype(str).to_numpy()
    train_idx, val_idx, test_idx = _grouped_split(
        groups, config.test_size, config.valid_size, config.seed
    )
    emit(
        f"split by {group_column or 'row'}: "
        f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test rows"
    )

    scaler_X = StandardScaler().fit(X[train_idx])
    X_train, X_val, X_test = (scaler_X.transform(X[i]) for i in (train_idx, val_idx, test_idx))
    scaler_y = _fit_masked_scaler(y_transformed[train_idx])
    # Masked entries are scaled too, then zero-filled; their loss weight is 0,
    # so the value never reaches the gradient.
    y_train = np.nan_to_num(scaler_y.transform(y_transformed[train_idx]))
    y_val = np.nan_to_num(scaler_y.transform(y_transformed[val_idx]))
    mask_train, mask_val, mask_test = mask[train_idx], mask[val_idx], mask[test_idx]

    model = _build_model(X.shape[1], targets, config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={f"{t}_output": "mse" for t in targets},
        metrics={f"{t}_output": ["mae"] for t in targets},
    )

    class _Progress(callbacks.Callback):
        """Reports epochs that have finished — never a rate, never a guess.

        Early stopping can end the run before ``config.epochs``, so the total
        handed out is an upper bound and the interface says so.
        """

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if progress is not None:
                progress(epoch + 1, config.epochs)
            if epoch % 10 == 0 or epoch < 3:
                emit(
                    f"epoch {epoch + 1:>4}  loss={logs.get('loss', float('nan')):.4f}  "
                    f"val_loss={logs.get('val_loss', float('nan')):.4f}"
                )

    history = model.fit(
        X_train,
        [y_train[:, i] for i in range(len(targets))],
        sample_weight=[mask_train[:, i] for i in range(len(targets))],
        validation_data=(
            X_val,
            [y_val[:, i] for i in range(len(targets))],
            [mask_val[:, i] for i in range(len(targets))],
        ),
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=0,
        shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience_early_stop,
                restore_best_weights=True,
                start_from_epoch=3,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=config.patience_reduce_lr, min_lr=1e-7
            ),
            _Progress(),
        ],
    )

    # Single-output inference model so downstream code never has to reassemble
    # a list of heads.
    prediction_model = Model(
        inputs=model.input,
        outputs=layers.Concatenate(name="consequence_outputs")(model.outputs),
    )

    def _to_original(scaled: np.ndarray) -> np.ndarray:
        processed = scaler_y.inverse_transform(scaled)
        out = processed.copy()
        for idx, target in enumerate(targets):
            if target in log_targets:
                out[:, idx] = np.maximum(
                    np.expm1(processed[:, idx]) - shifts.get(target, 0.0),
                    floors.get(target, 0.0),
                )
        return out

    y_pred = _to_original(prediction_model.predict(X_test, verbose=0))
    y_true = y[test_idx]

    per_target: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(targets):
        true_col, pred_col = y_true[:, idx], y_pred[:, idx]
        # Score each target only where Phast actually produced a value.
        finite = (mask_test[:, idx] > 0) & np.isfinite(true_col) & np.isfinite(pred_col)
        true_col, pred_col = true_col[finite], pred_col[finite]
        denominator = np.maximum(np.abs(true_col), np.percentile(np.abs(true_col), 5) or 1e-6)
        if finite.sum() < 5:
            per_target[target] = {"n_test": int(finite.sum())}
            continue
        per_target[target] = {
            "r2": float(r2_score(true_col, pred_col)),
            "mae": float(mean_absolute_error(true_col, pred_col)),
            "rmse": float(np.sqrt(mean_squared_error(true_col, pred_col))),
            # Median APE, not mean: a handful of near-zero true values otherwise
            # dominate the mean and make the metric meaningless.
            "median_ape": float(np.median(np.abs((true_col - pred_col) / denominator)) * 100.0),
            "log10_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            np.log10(np.clip(true_col, 1e-9, None))
                            - np.log10(np.clip(pred_col, 1e-9, None))
                        )
                        ** 2
                    )
                )
            ),
            "n_test": int(finite.sum()),
        }

    scored = [m["r2"] for m in per_target.values() if "r2" in m]
    metrics: dict[str, Any] = {
        "per_target": per_target,
        "coverage": {t: float(present[t].mean()) for t in targets},
        "mean_r2": float(np.mean(scored)) if scored else float("nan"),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "group_column": group_column,
        "epochs_run": int(len(history.history.get("loss", []))),
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }

    run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = project.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_model.save(run_dir / "model.keras")
    joblib.dump(scaler_X, run_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, run_dir / "scaler_y.joblib")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "targets": targets,
                "log_targets": log_targets,
                "shifts": shifts,
                "floors": floors,
                "feature_spec": spec.to_dict(),
                "training_config": {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                },
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(
        {
            **{f"true_{t}": y_true[:, i] for i, t in enumerate(targets)},
            **{f"pred_{t}": y_pred[:, i] for i, t in enumerate(targets)},
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    project.register_run(run_id, metrics)

    emit(f"saved run to {run_dir}")
    result = TrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        metrics=metrics,
        history={k: [float(x) for x in v] for k, v in history.history.items()},
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        targets=targets,
    )
    emit(result.summary())
    return result


def plot_parity(run_dir: Path, out_path: Path | None = None):
    """How closely the model reproduces Phast, one panel per target.

    A parity plot answers one question — does the prediction land on the true
    value — so it gets one hue and one reference line. The line is the thing
    the points are read against, so it is drawn a step firmer than a gridline,
    exactly as a zero baseline would be. R2 is direct-labelled in each panel;
    everything else about the run is in the scores table beside the plot.
    """
    # The figure API, not pyplot — see plot_case_distributions: a figure built
    # by pyplot picks up the window's Qt backend, and this can be called from a
    # worker thread.
    from matplotlib.figure import Figure

    from . import theme
    from .quantities import describe

    theme.apply_matplotlib_style()
    run_dir = Path(run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    targets = [c[5:] for c in predictions.columns if c.startswith("true_")]
    # Two columns up to four targets, three past that: a panel narrower than
    # about three inches cannot hold a log axis and its labels.
    cols = 2 if len(targets) <= 4 else 3
    cols = max(min(len(targets), cols), 1)
    rows = math_ceil(len(targets), cols)

    fig = Figure(figsize=(4.2 * cols, 3.3 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    dropped = 0
    for index, target in enumerate(targets):
        ax = axes[index // cols][index % cols]
        quantity = describe(target)
        true = predictions[f"true_{target}"].to_numpy(dtype=float)
        pred = predictions[f"pred_{target}"].to_numpy(dtype=float)
        usable = np.isfinite(true) & np.isfinite(pred) & (true > 0) & (pred > 0)
        dropped += int((~usable).sum())
        ax.set_title(quantity.with_unit())
        if not usable.any():
            # An empty panel says what is absent rather than showing a blank box.
            ax.text(0.5, 0.5, "No positive pair to plot.", ha="center", va="center",
                    fontsize=theme.pt(theme.FONT_CAPTION),
                    color=theme.ink_hex(theme.INK_SECONDARY), transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            theme.style_axes(ax, grid="none")
            continue

        low = float(min(true[usable].min(), pred[usable].min()))
        high = float(max(true[usable].max(), pred[usable].max()))
        # The 1:1 line is the reference the marks are read against, so it is
        # ordered chrome in ink, not a second data series in a hue.
        ax.plot([low, high], [low, high], color=theme.ink_hex(theme.INK_GLYPH),
                linewidth=1.0, zorder=1)
        ax.scatter(true[usable], pred[usable], s=12, alpha=0.45,
                   color=theme.SERIES[0], linewidths=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Phast")
        ax.set_ylabel("Predicted")
        theme.style_axes(ax)

        r2 = _r2(true[usable], pred[usable])
        ax.text(0.03, 0.97, f"R2 {r2:.3f}" if np.isfinite(r2) else "R2 unavailable",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=theme.pt(theme.FONT_CAPTION),
                color=theme.ink_hex(theme.INK_SECONDARY))

    for index in range(len(targets), rows * cols):
        axes[index // cols][index % cols].axis("off")

    caveats = [
        f"{len(predictions):,} held-out rows, on vessels the model never saw in training.",
        "Both axes are logarithmic; the line is where a prediction equals Phast.",
    ]
    if dropped:
        caveats.append(
            f"{dropped:,} points are not drawn: a log axis has no place for a "
            "missing or non-positive value."
        )
    top = theme.figure_header(fig, "Predicted against Phast", caveats)
    fig.subplots_adjust(top=top, left=0.085, right=0.98, bottom=0.11, hspace=0.55, wspace=0.26)

    if out_path is not None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)  # a figure needs a canvas before it can be saved
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig


def _r2(true: np.ndarray, predicted: np.ndarray) -> float:
    """R2 of the drawn points, so the label matches what is on screen."""
    spread = float(((true - true.mean()) ** 2).sum())
    if spread <= 0:
        return float("nan")
    return 1.0 - float(((true - predicted) ** 2).sum()) / spread


def math_ceil(value: int, divisor: int) -> int:
    return -(-value // divisor)
