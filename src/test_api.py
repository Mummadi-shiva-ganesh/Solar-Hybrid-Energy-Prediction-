"""
MLOps: API Integration Tests (Section 11.3)
Tests all Flask endpoints for correct responses, authentication, and error handling.
Run with: pytest src/test_api.py -v
"""

import os
import sys
import json
import pytest

# Add project root so imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Change working dir so model paths (models/best_model.pkl) resolve correctly
os.chdir(PROJECT_ROOT)

from api import app  # noqa: E402  – import after chdir

VALID_API_KEY = "solar-yield-secret-2026"
VALID_FEATURES = {
    "AMBIENT_TEMPERATURE": 25.5,
    "MODULE_TEMPERATURE": 35.2,
    "IRRADIATION": 0.8
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# 10.1 Health / Infra
# ---------------------------------------------------------------------------
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """API-001: GET /api/health must return 200"""
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_status_healthy(self, client):
        """API-002: health status must be 'healthy' when model is loaded"""
        r = client.get("/api/health")
        data = r.get_json()
        assert data["status"] == "healthy"

    def test_health_model_loaded(self, client):
        """API-003: health must report model_loaded=true"""
        data = client.get("/api/health").get_json()
        assert data["model_loaded"] is True

    def test_health_has_timestamp(self, client):
        """API-004: health response must include timestamp"""
        data = client.get("/api/health").get_json()
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------
class TestModelInfoEndpoint:

    def test_model_info_200(self, client):
        """API-005: GET /api/model/info must return 200"""
        assert client.get("/api/model/info").status_code == 200

    def test_model_info_has_fields(self, client):
        """API-006: model/info must contain key fields"""
        data = client.get("/api/model/info").get_json()
        for field in ("model_name", "model_type", "version",
                      "trained_date", "performance", "feature_names"):
            assert field in data, f"Missing field: {field}"

    def test_model_info_r2_positive(self, client):
        """API-007: val_r2 must be positive"""
        data = client.get("/api/model/info").get_json()
        assert data["performance"]["val_r2"] > 0


# ---------------------------------------------------------------------------
# Predict Endpoint – happy path
# ---------------------------------------------------------------------------
class TestPredictEndpoint:

    def _post(self, client, payload, key=VALID_API_KEY):
        return client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-API-KEY": key}
        )

    def test_predict_valid_returns_200(self, client):
        """API-008: valid predict request returns 200"""
        r = self._post(client, {"features": VALID_FEATURES})
        assert r.status_code == 200

    def test_predict_has_prediction_field(self, client):
        """API-009: response must contain 'prediction' key"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert "prediction" in data

    def test_predict_is_numeric(self, client):
        """API-010: prediction value must be numeric"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert isinstance(data["prediction"], (int, float))

    def test_predict_non_negative(self, client):
        """API-011: prediction must be >= 0"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert data["prediction"] >= 0

    def test_predict_status_success(self, client):
        """API-012: status field must be 'success'"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert data["status"] == "success"

    def test_predict_has_confidence(self, client):
        """API-013: response must include confidence"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_has_model_name(self, client):
        """API-014: response must include model_name"""
        data = self._post(client, {"features": VALID_FEATURES}).get_json()
        assert "model_name" in data


# ---------------------------------------------------------------------------
# Predict Endpoint – authentication
# ---------------------------------------------------------------------------
class TestPredictAuthentication:

    def _post(self, client, payload, key=None):
        headers = {"X-API-KEY": key} if key else {}
        return client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
            headers=headers
        )

    def test_missing_api_key_returns_401(self, client):
        """API-015: missing X-API-KEY must return 401"""
        r = self._post(client, {"features": VALID_FEATURES})
        assert r.status_code == 401

    def test_wrong_api_key_returns_401(self, client):
        """API-016: wrong API key must return 401"""
        r = self._post(client, {"features": VALID_FEATURES}, key="wrong-key")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Predict Endpoint – error handling (OWASP ML)
# ---------------------------------------------------------------------------
class TestPredictErrorHandling:

    def _post(self, client, payload):
        return client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-API-KEY": VALID_API_KEY}
        )

    def test_missing_features_key_returns_400(self, client):
        """API-017: request without 'features' key returns 400"""
        r = self._post(client, {})
        assert r.status_code == 400

    def test_non_numeric_feature_returns_400(self, client):
        """API-018: non-numeric feature value returns 400"""
        r = self._post(client, {"features": {
            "AMBIENT_TEMPERATURE": "hot",
            "MODULE_TEMPERATURE": 35.2,
            "IRRADIATION": 0.8
        }})
        assert r.status_code == 400

    def test_extreme_values_returns_400(self, client):
        """API-019: feature value > 1,000,000 returns 400"""
        r = self._post(client, {"features": {
            "AMBIENT_TEMPERATURE": 9_999_999,
            "MODULE_TEMPERATURE": 35.2,
            "IRRADIATION": 0.8
        }})
        assert r.status_code == 400

    def test_empty_body_returns_400(self, client):
        """API-020: completely empty request body returns 400"""
        r = client.post(
            "/api/predict",
            data="",
            content_type="application/json",
            headers={"X-API-KEY": VALID_API_KEY}
        )
        assert r.status_code in (400, 415, 500)


# ---------------------------------------------------------------------------
# Batch Predict
# ---------------------------------------------------------------------------
class TestBatchPredictEndpoint:

    def _batch(self, client, data_list):
        return client.post(
            "/api/predict/batch",
            data=json.dumps({"data": data_list}),
            content_type="application/json",
            headers={"X-API-KEY": VALID_API_KEY}
        )

    def test_batch_returns_200(self, client):
        """API-021: batch predict returns 200"""
        r = self._batch(client, [VALID_FEATURES, VALID_FEATURES])
        assert r.status_code == 200

    def test_batch_count_matches(self, client):
        """API-022: batch predict count matches input length"""
        samples = [VALID_FEATURES] * 3
        data = self._batch(client, samples).get_json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_missing_data_key_returns_400(self, client):
        """API-023: batch without 'data' key returns 400"""
        r = client.post(
            "/api/predict/batch",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-API-KEY": VALID_API_KEY}
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Utility Endpoints
# ---------------------------------------------------------------------------
class TestUtilityEndpoints:

    def test_features_endpoint_200(self, client):
        """API-024: GET /api/features returns 200"""
        assert client.get("/api/features").status_code == 200

    def test_features_returns_list(self, client):
        """API-025: /api/features returns a non-empty feature list"""
        data = client.get("/api/features").get_json()
        assert "features" in data
        assert len(data["features"]) > 0

    def test_metrics_endpoint_200(self, client):
        """API-026: GET /api/metrics returns 200"""
        assert client.get("/api/metrics").status_code == 200

    def test_spec_endpoint_200(self, client):
        """API-027: GET /api/spec returns 200"""
        assert client.get("/api/spec").status_code == 200

    def test_dashboard_returns_html(self, client):
        """API-028: GET / returns HTML dashboard"""
        r = client.get("/")
        assert r.status_code == 200
        assert b"Sopanel" in r.data


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
