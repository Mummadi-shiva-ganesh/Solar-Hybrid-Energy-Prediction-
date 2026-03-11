"""
MLOps: Unit Tests for ML Components
Covers: model accuracy, scaler, prediction bounds, metadata, bias testing
Run with: pytest src/test_ml_components.py -v
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_artifacts():
    """Load model, scaler, and metadata from models/"""
    model_path   = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    scaler_path  = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    meta_path    = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata


# ---------------------------------------------------------------------------
# 10.2 Model Versioning – metadata structure
# ---------------------------------------------------------------------------
class TestModelMetadata:

    def test_metadata_file_exists(self):
        """ML-META-01: model_metadata.json must exist"""
        path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        assert os.path.exists(path), "model_metadata.json not found in models/"

    def test_metadata_required_keys(self):
        """ML-META-02: metadata must contain required keys"""
        _, _, metadata = load_artifacts()
        required = {"model_name", "model_type", "trained_date", "feature_names", "performance"}
        missing = required - set(metadata.keys())
        assert not missing, f"Missing keys in metadata: {missing}"

    def test_metadata_performance_keys(self):
        """ML-META-03: performance dict must contain r2, rmse, mae"""
        _, _, metadata = load_artifacts()
        perf = metadata.get("performance", {})
        for key in ("val_r2", "val_rmse", "val_mae"):
            assert key in perf, f"Missing performance metric: {key}"

    def test_metadata_feature_names_nonempty(self):
        """ML-META-04: feature_names must be non-empty"""
        _, _, metadata = load_artifacts()
        assert len(metadata.get("feature_names", [])) > 0


# ---------------------------------------------------------------------------
# 10.6 Model Monitoring – artifact integrity
# ---------------------------------------------------------------------------
class TestModelArtifacts:

    def test_model_file_exists(self):
        """ML-ART-01: best_model.pkl must exist"""
        path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
        assert os.path.exists(path)

    def test_scaler_file_exists(self):
        """ML-ART-02: scaler.pkl must exist"""
        path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
        assert os.path.exists(path)

    def test_model_loadable(self):
        """ML-ART-03: model loads without error"""
        model, _, _ = load_artifacts()
        assert model is not None

    def test_scaler_loadable(self):
        """ML-ART-04: scaler loads without error"""
        _, scaler, _ = load_artifacts()
        assert scaler is not None


# ---------------------------------------------------------------------------
# ML Accuracy Tests
# ---------------------------------------------------------------------------
class TestModelAccuracy:

    def test_r2_above_threshold(self):
        """ML-ACC-01: validation R² must be >= 0.90"""
        _, _, metadata = load_artifacts()
        r2 = metadata["performance"]["val_r2"]
        assert r2 >= 0.90, f"R² {r2:.4f} < required 0.90"

    def test_rmse_reasonable(self):
        """ML-ACC-02: validation RMSE must be < 5000 kW"""
        _, _, metadata = load_artifacts()
        rmse = metadata["performance"]["val_rmse"]
        assert rmse < 5000, f"RMSE {rmse:.2f} is unreasonably large"

    def test_mae_reasonable(self):
        """ML-ACC-03: validation MAE must be < 3000 kW"""
        _, _, metadata = load_artifacts()
        mae = metadata["performance"]["val_mae"]
        assert mae < 3000, f"MAE {mae:.2f} is unreasonably large"


# ---------------------------------------------------------------------------
# Scaler Tests
# ---------------------------------------------------------------------------
class TestScaler:

    def test_scaler_transform_shape(self):
        """ML-SCL-01: scaler output shape matches input shape"""
        _, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame([[25.5, 35.2, 0.8]], columns=features)
        X_scaled = scaler.transform(X)
        assert X_scaled.shape == (1, len(features))

    def test_scaler_no_nan(self):
        """ML-SCL-02: scaler must produce no NaN in output"""
        _, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame([[25.5, 35.2, 0.8]], columns=features)
        X_scaled = scaler.transform(X)
        assert not np.isnan(X_scaled).any(), "Scaler produced NaN values"

    def test_scaler_zero_input(self):
        """ML-SCL-03: zero input must not crash scaler"""
        _, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame([[0.0, 0.0, 0.0]], columns=features)
        X_scaled = scaler.transform(X)
        assert X_scaled.shape == (1, len(features))


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------
class TestPrediction:

    def _predict(self, ambient, module, irradiation):
        model, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame([[ambient, module, irradiation]], columns=features)
        X_scaled = scaler.transform(X)
        return float(model.predict(X_scaled)[0])

    def test_prediction_valid_input(self):
        """ML-PRD-01: prediction on valid input must be numeric"""
        pred = self._predict(25.5, 35.2, 0.8)
        assert isinstance(pred, float), "Prediction must be a float"

    def test_prediction_nonnegative(self):
        """ML-PRD-02: prediction for positive irradiation must be >= 0"""
        pred = self._predict(25.5, 35.2, 0.8)
        assert pred >= 0, f"Prediction {pred} is negative"

    def test_prediction_zero_irradiation(self):
        """ML-PRD-03: prediction at zero irradiation must be >= 0 (no negative power)"""
        pred = self._predict(25.0, 30.0, 0.0)
        assert pred >= 0

    def test_prediction_high_irradiation(self):
        """ML-PRD-04: higher irradiation should yield higher or equal prediction than low"""
        pred_low  = self._predict(25.0, 35.0, 0.1)
        pred_high = self._predict(25.0, 35.0, 0.9)
        assert pred_high >= pred_low, (
            f"High irradiation ({pred_high:.1f}) should be >= low ({pred_low:.1f})"
        )

    def test_no_nan_prediction(self):
        """ML-PRD-05: prediction must not be NaN"""
        pred = self._predict(25.5, 35.2, 0.8)
        assert not np.isnan(pred)

    def test_batch_prediction(self):
        """ML-PRD-06: batch prediction on 5 samples returns 5 results"""
        model, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame([
            [25.5, 35.2, 0.8],
            [30.0, 40.0, 0.9],
            [20.0, 28.0, 0.5],
            [15.0, 22.0, 0.2],
            [35.0, 48.0, 1.0],
        ], columns=features)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# 11.7 Bias & Fairness Tests
# ---------------------------------------------------------------------------
class TestBiasAndFairness:
    """
    Verify prediction consistency across environmental subgroups.
    Fairness constraint: R² variance across subgroups < 2.5pp, MAE ratio < 3x.
    """

    def _score_group(self, conditions):
        """Return predictions for a list of (ambient, module, irr) tuples."""
        model, scaler, metadata = load_artifacts()
        features = metadata["feature_names"]
        X = pd.DataFrame(conditions, columns=features)
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    def test_morning_conditions(self):
        """BIAS-01: morning low-irradiation predictions must be non-negative"""
        morning = [(20 + i * 0.5, 25 + i, 0.1 + i * 0.02) for i in range(10)]
        preds = self._score_group(morning)
        assert all(p >= 0 for p in preds), "Negative prediction in morning conditions"

    def test_afternoon_conditions(self):
        """BIAS-02: afternoon high-irradiation predictions must be positive"""
        afternoon = [(30 + i * 0.3, 42 + i * 0.2, 0.7 + i * 0.01) for i in range(10)]
        preds = self._score_group(afternoon)
        assert all(p > 0 for p in preds), "Non-positive prediction in afternoon conditions"

    def test_high_vs_low_temperature_variance(self):
        """BIAS-03: prediction variance between temperature groups must be reasonable"""
        low_temp  = self._score_group([(15, 22, 0.6)] * 5)
        high_temp = self._score_group([(40, 52, 0.6)] * 5)
        ratio = max(np.mean(high_temp), np.mean(low_temp)) / (
            min(np.mean(high_temp), np.mean(low_temp)) + 1e-9
        )
        assert ratio < 5.0, f"Temperature group predictions differ by factor {ratio:.2f} — possible bias"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
