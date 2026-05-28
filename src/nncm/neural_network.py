"""
Neural network to predict multiple consequence metrics (Release rate, Velocity,
Distance to LFL, Flame length) from Pressure, Temperature, Orifice_Diameter.
"""

import os
import random
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# Config / Hyperparameters
# ---------------------------
CSV_PATH = PROJECT_ROOT / "data" / "generated" / "training_data.csv"
RAW_INPUT_COLUMNS = ["Pressure", "Temperature", "Orifice_diameter"]
TARGET_COLUMNS = ["Release_rate", "Velocity", "Distance_to_LFL", "Flame_length"]
LOG_TARGET_COLUMNS = {"Release_rate", "Distance_to_LFL"}
USE_ENGINEERED_FEATURES = True
SCALE_TARGETS = True

TEST_SIZE = 0.10
VALID_SIZE = 0.10
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6
PATIENCE_ES = 30
PATIENCE_RLR = 12
DROPOUT_RATE = 0.20
MC_SAMPLES = 50
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "best_model.keras"
USE_LOG_TARGET = True

# ---------------------------
# Utilities
# ---------------------------
def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def inverse_transform_targets(values, log_columns, shifts):
    """Inverse log1p transformation for specified columns."""
    arr = np.asarray(values, dtype=float)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr_inv = arr.copy()
    for idx, col in enumerate(TARGET_COLUMNS):
        if col in log_columns:
            arr_inv[:, idx] = np.expm1(arr[:, idx]) - shifts.get(col, 0.0)
    return arr_inv.reshape(original_shape)


def build_feature_dataframe(df):
    """Create engineered feature set from raw columns."""
    feat = pd.DataFrame()
    feat["Pressure"] = df["Pressure"].astype(float)
    feat["Temperature"] = df["Temperature"].astype(float)
    feat["Orifice_diameter"] = df["Orifice_diameter"].astype(float)

    if USE_ENGINEERED_FEATURES:
        pressure = feat["Pressure"].values
        temperature = feat["Temperature"].values
        orifice = np.maximum(feat["Orifice_diameter"].values, 1e-3)
        temp_k = temperature + 273.15

        feat["log_pressure"] = np.log1p(np.clip(pressure, 0, None) + 1e-6)
        feat["log_temperature"] = np.log1p(np.clip(temp_k, 0, None) + 1e-6)
        feat["log_orifice"] = np.log1p(orifice)
        feat["pressure_temperature"] = pressure * temperature
        feat["pressure_orifice"] = pressure * orifice
        feat["temperature_orifice"] = temperature * orifice
        feat["pressure_over_orifice"] = pressure / orifice
        feat["temperature_sq"] = temperature ** 2

    return feat


def to_target_dict(values):
    """Convert (n_samples, n_targets) matrix to dict for multi-output training."""
    return {col: values[:, idx] for idx, col in enumerate(TARGET_COLUMNS)}


def concatenate_outputs(predictions):
    """Convert list of per-target outputs to single matrix."""
    if isinstance(predictions, list):
        return np.hstack([np.asarray(p).reshape(-1, 1) for p in predictions])
    return np.asarray(predictions)


def dense_block(x, units, l2, name_prefix, dropout=True):
    # LayerNorm (not BatchNorm) — normalizes per-sample, so train and eval modes
    # behave identically. Avoids the EMA-running-stats divergence that makes
    # val_loss NaN in early epochs.
    from tensorflow.keras import layers
    x = layers.Dense(units, kernel_regularizer=l2, name=f"{name_prefix}_d")(x)
    x = layers.LayerNormalization(name=f"{name_prefix}_ln")(x)
    x = layers.Activation("swish", name=f"{name_prefix}_act")(x)
    if dropout:
        x = layers.Dropout(DROPOUT_RATE, name=f"{name_prefix}_drop")(x)
    return x


def build_model(input_dim, target_names):
    from tensorflow.keras import layers, regularizers, Model, Input
    l2 = regularizers.l2(WEIGHT_DECAY)
    inp = Input(shape=(input_dim,), name="inputs")

    # Shared trunk: plain MLP, 3 layers is shallow enough that residuals aren't needed
    x = dense_block(inp, 96, l2, "trunk1")
    x = dense_block(x,   96, l2, "trunk2")
    x = dense_block(x,   64, l2, "trunk3")

    # Per-target heads — each target gets its own 2-layer specialization
    outputs = []
    for name in target_names:
        h = dense_block(x, 32, l2, f"{name}_h1", dropout=False)
        h = dense_block(h, 16, l2, f"{name}_h2", dropout=False)
        out = layers.Dense(1, activation="linear", name=f"{name}_output")(h)
        outputs.append(out)

    model = Model(inputs=inp, outputs=outputs, name="consequence_model")
    return model


def main():
    from tensorflow.keras import layers, callbacks, Model

    # ---------------------------
    # Load data
    # ---------------------------
    # Ensure output directory exists BEFORE training so ModelCheckpoint can write to it
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    required_columns = RAW_INPUT_COLUMNS + TARGET_COLUMNS
    assert set(required_columns).issubset(df.columns), \
        f"CSV must contain columns: {required_columns}"

    feature_df = build_feature_dataframe(df[RAW_INPUT_COLUMNS])
    feature_columns = feature_df.columns.tolist()
    X = feature_df.astype(float).values
    y = df[TARGET_COLUMNS].astype(float).values

    log_target_columns = set(LOG_TARGET_COLUMNS if USE_LOG_TARGET else [])
    target_shifts = {col: 0.0 for col in TARGET_COLUMNS}
    y_transformed = y.copy()
    if log_target_columns:
        for idx, col in enumerate(TARGET_COLUMNS):
            if col in log_target_columns:
                col_values = y[:, idx]
                shift = 0.0
                min_val = col_values.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1e-6
                target_shifts[col] = shift
                y_transformed[:, idx] = np.log1p(col_values + shift)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_transformed, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    valid_fraction_of_temp = VALID_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=valid_fraction_of_temp, random_state=SEED, shuffle=True
    )

    # StandardScaler equalizes target variances to 1, so explicit per-target
    # loss weighting is unnecessary and was previously miscomputed in the
    # unscaled space — letting it default to equal weights is correct here.

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    if SCALE_TARGETS:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        scaler_y = None
        y_train_scaled = y_train
        y_val_scaled = y_val
        y_test_scaled = y_test

    y_train_dict = to_target_dict(y_train_scaled)
    y_val_dict = to_target_dict(y_val_scaled)
    y_train_list = [y_train_dict[col] for col in TARGET_COLUMNS]
    y_val_list = [y_val_dict[col] for col in TARGET_COLUMNS]

    # ---------------------------
    # Model architecture
    # ---------------------------
    model = build_model(X_train_scaled.shape[1], TARGET_COLUMNS)
    # clipnorm guards against rare exploding-gradient steps in early training
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    losses = {f"{col}_output": "mse" for col in TARGET_COLUMNS}
    metrics = {f"{col}_output": ["mae"] for col in TARGET_COLUMNS}
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    model.summary()
    prediction_model = Model(
        inputs=model.input,
        outputs=layers.Concatenate(name="consequence_outputs_concat")(model.outputs)
    )

    # ---------------------------
    # Callbacks
    # ---------------------------
    # start_from_epoch=3 prevents EarlyStopping from latching onto a NaN epoch-1
    # val_loss as the "best" weights to restore.
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_ES,
        restore_best_weights=True,
        start_from_epoch=3,
        verbose=1,
    )
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_RLR, verbose=1, min_lr=1e-7)
    mc = callbacks.ModelCheckpoint(
        str(MODEL_SAVE_PATH),
        monitor="val_loss",
        save_best_only=True,
        initial_value_threshold=1e6,
        verbose=1,
    )

    # ---------------------------
    # Training
    # ---------------------------
    model.fit(
        X_train_scaled, y_train_list,
        validation_data=(X_val_scaled, y_val_list),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[es, rlr, mc],
        verbose=2,
        shuffle=True
    )

    # ---------------------------
    # Save final model (inference-ready with concatenated output)
    # EarlyStopping(restore_best_weights=True) has already restored the best
    # in-memory weights, so we save the prediction model directly.
    # ---------------------------
    prediction_model.save(str(MODEL_SAVE_PATH))
    print(f"\nSaved trained model to {MODEL_SAVE_PATH}")

    # ---------------------------
    # Evaluation (test set)
    # ---------------------------
    y_test_pred_scaled = prediction_model.predict(X_test_scaled)
    if SCALE_TARGETS and scaler_y is not None:
        y_test_pred_processed = scaler_y.inverse_transform(y_test_pred_scaled)
        y_test_processed = scaler_y.inverse_transform(y_test_scaled)
    else:
        y_test_pred_processed = y_test_pred_scaled
        y_test_processed = y_test

    y_test_true = inverse_transform_targets(y_test_processed, log_target_columns, target_shifts)
    y_test_pred_inv = np.clip(
        inverse_transform_targets(y_test_pred_processed, log_target_columns, target_shifts),
        0, None
    )

    overall_mse = mean_squared_error(y_test_true, y_test_pred_inv)
    overall_rmse = sqrt(overall_mse)
    overall_mae = mean_absolute_error(y_test_true, y_test_pred_inv)
    overall_r2 = r2_score(y_test_true, y_test_pred_inv, multioutput="uniform_average")

    print("\n----- Test set performance (final) -----")
    print(f"Overall MSE  : {overall_mse:.6f}")
    print(f"Overall RMSE : {overall_rmse:.6f}")
    print(f"Overall MAE  : {overall_mae:.6f}")
    print(f"Overall R^2  : {overall_r2:.5f}")

    print("\nPer-target metrics:")
    for idx, col in enumerate(TARGET_COLUMNS):
        true_col = y_test_true[:, idx]
        pred_col = y_test_pred_inv[:, idx]
        mse = mean_squared_error(true_col, pred_col)
        rmse = sqrt(mse)
        mae = mean_absolute_error(true_col, pred_col)
        r2 = r2_score(true_col, pred_col)
        mape = np.mean(np.abs((true_col - pred_col) / np.maximum(np.abs(true_col), 1e-6))) * 100.0
        print(f"  {col}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.3f} %, R^2={r2:.5f}")

    residuals = y_test_true - y_test_pred_inv
    residual_stds = residuals.std(axis=0)
    for idx, col in enumerate(TARGET_COLUMNS):
        print(f"Residual std for {col}: {residual_stds[idx]:.6f}")

    # ---------------------------
    # MC Dropout uncertainty (training=True keeps dropout active)
    # ---------------------------
    print(f"\n----- MC Dropout uncertainty ({MC_SAMPLES} samples) -----")
    mc_preds_scaled = np.stack([
        prediction_model(X_test_scaled, training=True).numpy()
        for _ in range(MC_SAMPLES)
    ], axis=0)  # (MC_SAMPLES, n_test, n_targets)
    mc_mean_scaled = mc_preds_scaled.mean(axis=0)
    mc_std_scaled = mc_preds_scaled.std(axis=0)

    if SCALE_TARGETS and scaler_y is not None:
        mc_mean_proc = scaler_y.inverse_transform(mc_mean_scaled)
        # Propagate std through the linear scaler (multiply by scale, not shift)
        mc_std_proc = mc_std_scaled * scaler_y.scale_
    else:
        mc_mean_proc = mc_mean_scaled
        mc_std_proc = mc_std_scaled

    mc_mean_orig = np.clip(
        inverse_transform_targets(mc_mean_proc, log_target_columns, target_shifts),
        0, None
    )
    for idx, col in enumerate(TARGET_COLUMNS):
        mean_unc = mc_std_proc[:, idx].mean()
        print(f"  {col}: mean epistemic std = {mean_unc:.6f}")

    # ---------------------------
    # Quick plots (optional)
    # ---------------------------
    try:
        import matplotlib.pyplot as plt
        num_targets = len(TARGET_COLUMNS)
        fig, axes = plt.subplots(1, num_targets, figsize=(5 * num_targets, 5), squeeze=False)
        for idx, col in enumerate(TARGET_COLUMNS):
            ax = axes[0, idx]
            true_col = y_test_true[:, idx]
            pred_col = y_test_pred_inv[:, idx]
            ax.scatter(true_col, pred_col, s=8, alpha=0.6)
            mn = min(true_col.min(), pred_col.min())
            mx = max(true_col.max(), pred_col.max())
            ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.8)
            ax.set_xlabel(f"True {col}")
            ax.set_ylabel(f"Predicted {col}")
            ax.set_title(f"Parity: {col}")
            ax.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    # ---------------------------
    # SHAP explainability (optional; can be heavy)
    # ---------------------------
    SHAP_ENABLED = False
    if SHAP_ENABLED:
        try:
            import shap
            n_bg = min(200, X_train_scaled.shape[0])
            background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], n_bg, replace=False)]
            shap_target_idx = 0
            n_explain = min(200, X_test_scaled.shape[0])
            X_explain = X_test_scaled[:n_explain]
            try:
                explainer = shap.DeepExplainer(prediction_model, background)
                shap_values = explainer.shap_values(X_explain)
            except Exception:
                explainer = shap.KernelExplainer(lambda x: prediction_model.predict(x)[:, shap_target_idx], background)
                shap_values = explainer.shap_values(X_test_scaled[:50], nsamples=100)
                X_explain = X_test_scaled[:50]

            # Normalize SHAP output shape to (n_samples, n_features) for the chosen target.
            # SHAP DeepExplainer with multi-output Keras can return either:
            #   - list of (n_samples, n_features) arrays, one per output (older SHAP), or
            #   - a single (n_samples, n_features, n_outputs) array (newer SHAP)
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[shap_target_idx]
            else:
                sv = np.asarray(shap_values)
                if sv.ndim == 3:
                    shap_values_to_plot = sv[:, :, shap_target_idx]
                else:
                    shap_values_to_plot = sv

            shap.summary_plot(
                shap_values_to_plot,
                X_explain,
                feature_names=feature_columns,
                show=False,
                title=f"SHAP summary for {TARGET_COLUMNS[shap_target_idx]}"
            )
        except Exception as e:
            print("SHAP explanation skipped or failed:", str(e))

    # ---------------------------
    # Save scalers and artifacts
    # ---------------------------
    import joblib

    joblib.dump(scaler_X, models_dir / "scaler_X.joblib")
    if SCALE_TARGETS and scaler_y is not None:
        joblib.dump(scaler_y, models_dir / "scaler_y.joblib")
    joblib.dump(
        {
            "use_log_target": bool(log_target_columns),
            "log_columns": list(log_target_columns),
            "shifts": target_shifts,
            "target_columns": TARGET_COLUMNS,
            "scale_targets": SCALE_TARGETS,
            "feature_columns": feature_columns,
            "use_engineered_features": USE_ENGINEERED_FEATURES,
            "raw_input_columns": RAW_INPUT_COLUMNS,
        },
        models_dir / "target_meta.joblib"
    )

    print(f"Scalers and metadata saved to {models_dir}")


if __name__ == "__main__":
    main()
