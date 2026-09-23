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

from .features import FeatureSpec, build_features, domain_flags, domain_report, from_log_space, resolve_materials


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


# Below this many rows a direct call beats ``model.predict``, which sets up a
# dataset pipeline on every call (about 60 ms against 20 ms for one row).
DIRECT_CALL_ROWS = 1024
MC_TILED_ROWS = 65_536  # rows per tiled MC-dropout block


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
        # Runs written before the log10 transform used log1p with a shift and
        # a floor; both are kept so those runs still load and answer the same.
        self.legacy_transform = meta.get("target_transform") != "log10_offset"
        self.log_offsets: dict[str, float] = {
            k: float(v) for k, v in (meta.get("log_offsets") or {}).items()
        }
        self.shifts: dict[str, float] = {k: float(v) for k, v in (meta.get("shifts") or {}).items()}
        self.floors: dict[str, float] = {k: float(v) for k, v in (meta.get("floors") or {}).items()}
        self.spec = FeatureSpec.from_dict(meta["feature_spec"])
        self.metrics: dict[str, Any] = meta.get("metrics", {})
        self.study_settings: dict[str, Any] = meta.get("study_settings", {})

        import joblib
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.run_dir / "model.keras", compile=False)
        self.scaler_X = joblib.load(self.run_dir / "scaler_X.joblib")
        self.scaler_y = joblib.load(self.run_dir / "scaler_y.joblib")

    # -- core ---------------------------------------------------------------
    def _inverse(self, scaled: np.ndarray) -> np.ndarray:
        """Scaled network output -> real units. Works on any leading shape."""
        shape = scaled.shape
        processed = self.scaler_y.inverse_transform(scaled.reshape(-1, shape[-1]))
        out = processed.copy()
        for idx, target in enumerate(self.targets):
            if target not in self.log_targets:
                continue
            if self.legacy_transform:
                out[:, idx] = np.maximum(
                    np.expm1(processed[:, idx]) - self.shifts.get(target, 0.0),
                    self.floors.get(target, 0.0),
                )
            else:
                out[:, idx] = from_log_space(processed[:, idx], self.log_offsets[target])
        return out.reshape(shape)

    def _forward(self, scaled: np.ndarray, training: bool = False) -> np.ndarray:
        if len(scaled) <= DIRECT_CALL_ROWS:
            return np.asarray(self.model(scaled, training=training))
        if training:
            return np.concatenate(
                [
                    np.asarray(self.model(scaled[i : i + MC_TILED_ROWS], training=True))
                    for i in range(0, len(scaled), MC_TILED_ROWS)
                ]
            )
        return np.asarray(self.model.predict(scaled, verbose=0))

    def predict_frame(self, frame: pd.DataFrame, mc_samples: int = 0) -> pd.DataFrame:
        """Predict for every row; optional MC-dropout spread as ``<target>_std``.

        Raises ``ValueError`` for a row the model cannot place: a material it
        was not trained on, given without its descriptors.
        """
        resolved = resolve_materials(frame, self.spec, strict=True)
        features = build_features(resolved, self.spec).to_numpy(dtype=float)
        scaled = self.scaler_X.transform(features)
        predictions = self._inverse(self._forward(scaled))
        result = pd.DataFrame(predictions, columns=self.targets, index=frame.index)

        if mc_samples > 0:
            # All passes in one call: the input is tiled, each copy draws its
            # own dropout mask. Fifty separate calls cost fifty times the
            # per-call overhead for the same arithmetic.
            # Rows are taken in chunks so a large CSV does not tile into
            # millions of rows at once.
            chunk = max(1, MC_TILED_ROWS // mc_samples)
            parts = []
            for start in range(0, len(scaled), chunk):
                block = scaled[start : start + chunk]
                tiled = np.tile(block, (mc_samples, 1))
                parts.append(
                    self._forward(tiled, training=True).reshape(mc_samples, len(block), -1)
                )
            samples = self._inverse(np.concatenate(parts, axis=1))
            # Spread is computed after inverting the transform, so the number is
            # in the target's real units rather than log space.
            for idx, target in enumerate(self.targets):
                result[f"{target}_std"] = samples[:, :, idx].std(axis=0)
        return result

    def predict_batch(self, frame: pd.DataFrame, mc_samples: int = 0) -> pd.DataFrame:
        """Inputs, predictions and a per-row extrapolation note, side by side."""
        predictions = self.predict_frame(frame, mc_samples=mc_samples)
        out = pd.concat([frame, predictions], axis=1)
        out["domain_warnings"] = domain_flags(frame, self.spec)
        return out

    def predict_one(self, inputs: dict[str, float], mc_samples: int = 0) -> Prediction:
        frame = pd.DataFrame([inputs])
        predicted = self.predict_frame(frame, mc_samples=mc_samples)
        row = predicted.iloc[0]
        resolved = resolve_materials(frame, self.spec).iloc[0].to_dict()
        return Prediction(
            values={t: float(row[t]) for t in self.targets},
            uncertainty={
                t: float(row[f"{t}_std"]) for t in self.targets if f"{t}_std" in predicted.columns
            },
            domain_warnings=domain_report(resolved, self.spec),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def required_inputs(self) -> list[str]:
        return [c for c in self.spec.input_bounds if not c.startswith("mat_")]

    def validity_note(self) -> str:
        """The settings every training case shared, which the model cannot vary."""
        fixed = dict(self.spec.fixed_inputs)
        parts = [f"{k} = {v:g}" for k, v in fixed.items()]
        settings = self.study_settings
        if settings.get("template"):
            parts.append(f"template '{settings['template']}' (its weather and parameter sets)")
        for key in ("vessel_defaults", "leak_defaults"):
            parts += [f"{k} = {v}" for k, v in (settings.get(key) or {}).items()]
        if not parts:
            return ""
        return "Valid only for the fixed study settings: " + "; ".join(parts) + "."

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
        note = self.validity_note()
        if note:
            lines.append(f"  {note}")
        return "\n".join(lines)


def load_latest(project) -> ModelBundle:
    run_dir = project.latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError(f"no trained runs in {project.models_dir}")
    return ModelBundle(run_dir)
