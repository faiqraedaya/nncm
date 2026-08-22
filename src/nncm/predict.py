"""Inference: load a saved run and predict, with uncertainty and domain checks.

Used by the GUI, the CLI and any script — there is exactly one code path from
raw inputs to predictions, so the GUI can never disagree with the trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import FeatureSpec, build_features, domain_report


@dataclass
class Prediction:
    values: dict[str, float]
    uncertainty: dict[str, float]
    domain_warnings: dict[str, str]

    def format_value(self, target: str) -> str:
        value = self.values.get(target, float("nan"))
        return format_quantity(value)


def format_quantity(value: float) -> str:
    """Significant-figure formatting — release rates span 1e-3 to 1e3 kg/s."""
    if value is None or not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.3e}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


class ModelBundle:
    """A trained run: model, scalers, feature spec and target transforms."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        meta_path = self.run_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no meta.json in {self.run_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = meta
        self.run_id: str = meta.get("run_id", self.run_dir.name)
        self.targets: list[str] = list(meta["targets"])
        self.log_targets: set[str] = set(meta.get("log_targets", []))
        self.shifts: dict[str, float] = {k: float(v) for k, v in (meta.get("shifts") or {}).items()}
        # Physical floor for each log-transformed target, recorded at training
        # time; inverting the transform must not yield a negative quantity.
        self.floors: dict[str, float] = {k: float(v) for k, v in (meta.get("floors") or {}).items()}
        self.spec = FeatureSpec.from_dict(meta["feature_spec"])
        self.metrics: dict[str, Any] = meta.get("metrics", {})

        import joblib
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.run_dir / "model.keras", compile=False)
        self.scaler_X = joblib.load(self.run_dir / "scaler_X.joblib")
        self.scaler_y = joblib.load(self.run_dir / "scaler_y.joblib")

    # -- core ---------------------------------------------------------------
    def _inverse(self, scaled: np.ndarray) -> np.ndarray:
        processed = self.scaler_y.inverse_transform(scaled)
        out = processed.copy()
        for idx, target in enumerate(self.targets):
            if target in self.log_targets:
                out[:, idx] = np.maximum(
                    np.expm1(processed[:, idx]) - self.shifts.get(target, 0.0),
                    self.floors.get(target, 0.0),
                )
        return out

    def predict_frame(self, frame: pd.DataFrame, mc_samples: int = 0) -> pd.DataFrame:
        """Predict for every row; optional MC-dropout spread as ``<target>_std``."""
        features = build_features(frame, self.spec).to_numpy(dtype=float)
        scaled = self.scaler_X.transform(features)
        predictions = self._inverse(np.asarray(self.model.predict(scaled, verbose=0)))
        result = pd.DataFrame(predictions, columns=self.targets, index=frame.index)

        if mc_samples > 0:
            samples = np.stack(
                [
                    self._inverse(np.asarray(self.model(scaled, training=True)))
                    for _ in range(mc_samples)
                ],
                axis=0,
            )
            # Spread is computed after inverting the transform, so the number is
            # in the target's real units rather than log space.
            for idx, target in enumerate(self.targets):
                result[f"{target}_std"] = samples[:, :, idx].std(axis=0)
        return result

    def predict_one(self, inputs: dict[str, float], mc_samples: int = 0) -> Prediction:
        frame = pd.DataFrame([inputs])
        predicted = self.predict_frame(frame, mc_samples=mc_samples)
        row = predicted.iloc[0]
        return Prediction(
            values={t: float(row[t]) for t in self.targets},
            uncertainty={
                t: float(row[f"{t}_std"]) for t in self.targets if f"{t}_std" in predicted.columns
            },
            domain_warnings=domain_report(inputs, self.spec),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def required_inputs(self) -> list[str]:
        return list(self.spec.input_bounds)

    def describe(self) -> str:
        per_target = self.metrics.get("per_target", {})
        lines = [f"run {self.run_id} — targets: {', '.join(self.targets)}"]
        for target, m in per_target.items():
            lines.append(
                f"  {target:<28} R2={m.get('r2', float('nan')):.4f}  "
                f"MedAPE={m.get('median_ape', float('nan')):.1f} %"
            )
        for column, (low, high) in self.spec.input_bounds.items():
            lines.append(f"  domain {column}: {low:.4g} .. {high:.4g}")
        return "\n".join(lines)


def load_latest(project) -> ModelBundle:
    run_dir = project.latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError(f"no trained runs in {project.models_dir}")
    return ModelBundle(run_dir)
