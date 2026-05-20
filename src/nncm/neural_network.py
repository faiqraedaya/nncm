"""
Neural network to predict multiple consequence metrics (Release rate, Velocity,
Distance to LFL, Flame length) from Pressure, Temperature, Orifice_Diameter.

Requirements:
  pip install pandas numpy scikit-learn tensorflow matplotlib shap
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
LOG_TARGET_COLUMNS = {"Release_rate"}
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
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "best_model.h5"
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


def build_model(input_dim, target_names):
    from tensorflow.keras import layers, regularizers, Model, Input
    l2 = regularizers.l2(WEIGHT_DECAY)
    inp = Input(shape=(input_dim,), name="inputs")
    x = layers.Dense(192, kernel_regularizer=l2)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.05)(x)

    r = layers.Dense(192, kernel_regularizer=l2)(x)
    r = layers.BatchNormalization()(r)
    r = layers.Activation("swish")(r)
    r = layers.Dropout(0.05)(r)
    x = layers.Add()([x, r])

    x = layers.Dense(128, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.05)(x)

    r = layers.Dense(128, kernel_regularizer=l2)(x)
    r = layers.BatchNormalization()(r)
    r = layers.Activation("swish")(r)
    r = layers.Dropout(0.05)(r)
    x = layers.Add()([x, r])

    x = layers.Dense(64, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.05)(x)

    r = layers.Dense(64, kernel_regularizer=l2)(x)
    r = layers.BatchNormalization()(r)
    r = layers.Activation("swish")(r)
    r = layers.Dropout(0.05)(r)
    x = layers.Add()([x, r])

    outputs = []
    for name in target_names:
        head = layers.Dense(48, kernel_regularizer=l2)(x)
        head = layers.BatchNormalization()(head)
        head = layers.Activation("swish")(head)
        head = layers.Dropout(0.05)(head)
        out = layers.Dense(1, activation="linear", name=f"{name}_output")(head)
        outputs.append(out)

    model = Model(inputs=inp, outputs=outputs, name="consequence_model")
    return model


def main():
    from tensorflow.keras import layers, callbacks, Model

    # ---------------------------
    # Load data
    # ---------------------------
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

    target_variances = np.var(y_train, axis=0)
    inv_var = 1.0 / np.maximum(target_variances, 1e-6)
    inv_var_norm = inv_var / inv_var.mean()
    loss_weights = {col: float(inv_var_norm[idx]) for idx, col in enumerate(TARGET_COLUMNS)}

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
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.8
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    losses = {f"{col}_output": "mse" for col in TARGET_COLUMNS}
    metrics = {f"{col}_output": ["mae"] for col in TARGET_COLUMNS}
    loss_weights_named = {f"{col}_output": loss_weights[col] for col in TARGET_COLUMNS}
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights_named, metrics=metrics)
    model.summary()
    prediction_model = Model(
        inputs=model.input,
        outputs=layers.Concatenate(name="consequence_outputs_concat")(model.outputs)
    )

    # ---------------------------
    # Callbacks
    # ---------------------------
    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True, verbose=1)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_RLR, verbose=1, min_lr=1e-7)
    mc = callbacks.ModelCheckpoint(str(MODEL_SAVE_PATH), monitor="val_loss", save_best_only=True, verbose=1)
    lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr: float(lr_schedule(epoch)), verbose=0)

    # ---------------------------
    # Training
    # ---------------------------
    model.fit(
        X_train_scaled, y_train_list,
        validation_data=(X_val_scaled, y_val_list),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[es, rlr, mc, lr_scheduler],
        verbose=2,
        shuffle=True
    )

    # ---------------------------
    # Evaluation (test set)
    # ---------------------------
    if MODEL_SAVE_PATH.exists():
        model.load_weights(str(MODEL_SAVE_PATH))

    y_test_pred_scaled = prediction_model.predict(X_test_scaled)
    if SCALE_TARGETS and scaler_y is not None:
        y_test_pred_processed = scaler_y.inverse_transform(y_test_pred_scaled)
        y_test_processed = scaler_y.inverse_transform(y_test_scaled)
    else:
        y_test_pred_processed = y_test_pred_scaled
        y_test_processed = y_test

    y_test_true = inverse_transform_targets(y_test_processed, log_target_columns, target_shifts)
    y_test_pred_inv = inverse_transform_targets(y_test_pred_processed, log_target_columns, target_shifts)

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
    SHAP_ENABLED = True
    if SHAP_ENABLED:
        try:
            import shap
            background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], min(200, X_train_scaled.shape[0]), replace=False)]
            shap_target_idx = 0
            try:
                explainer = shap.DeepExplainer(prediction_model, background)
                shap_values = explainer.shap_values(X_test_scaled[:200])
            except Exception:
                explainer = shap.KernelExplainer(lambda x: prediction_model.predict(x)[:, shap_target_idx], background)
                shap_values = explainer.shap_values(X_test_scaled[:50], nsamples=100)
            shap_values_to_plot = shap_values[shap_target_idx] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(
                shap_values_to_plot,
                X_test_scaled[:min(200, X_test_scaled.shape[0])],
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
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

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

    print("Model, scaler and metadata saved.")


if __name__ == "__main__":
    main()
