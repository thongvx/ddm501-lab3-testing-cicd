"""
Integration tests for API endpoints (FastAPI app + real trained model).

Run tests:
    pytest tests/integration/test_api.py -v
"""

import time
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import app.main as main

pytestmark = pytest.mark.integration


class TestHealthEndpoint:
    """Integration tests for /health endpoint."""

    def test_health_returns_200(self, test_client: TestClient) -> None:
        """Test that health endpoint returns 200 status code."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client: TestClient) -> None:
        """Test that health response has status field."""
        assert "status" in test_client.get("/health").json()

    def test_health_response_has_model_loaded_field(self, test_client: TestClient) -> None:
        """Test that health response has model_loaded field."""
        assert "model_loaded" in test_client.get("/health").json()

    def test_health_model_loaded_is_boolean(self, test_client: TestClient) -> None:
        """Test that model_loaded is a boolean value."""
        assert isinstance(test_client.get("/health").json()["model_loaded"], bool)

    def test_health_reports_healthy_with_model(self, test_client: TestClient) -> None:
        """With the model loaded at startup the service is healthy."""
        assert test_client.get("/health").json() == {"status": "healthy", "model_loaded": True}


class TestRootEndpoint:
    """Integration tests for / endpoint."""

    def test_root_returns_200(self, test_client: TestClient) -> None:
        """Test that root endpoint returns 200 status code."""
        assert test_client.get("/").status_code == 200

    def test_root_contains_api_info(self, test_client: TestClient) -> None:
        """Test that root response contains name, version and docs fields."""
        data = test_client.get("/").json()
        for field in ("name", "version", "docs", "health"):
            assert field in data

    def test_openapi_docs_available(self, test_client: TestClient) -> None:
        """The OpenAPI schema advertised by / exists and lists /predict."""
        schema = test_client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]


class TestPredictEndpoint:
    """Integration tests for /predict endpoint."""

    def test_predict_valid_request_returns_200(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that valid prediction request returns 200."""
        assert test_client.post("/predict", json=sample_prediction_request).status_code == 200

    def test_predict_response_has_predicted_rating(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that response contains predicted_rating field."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert "predicted_rating" in data

    def test_predict_response_has_user_id(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that response echoes the user_id."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that response echoes the movie_id."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_has_model_version(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Responses carry the model version for traceability."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["model_version"] == main.MODEL_VERSION

    def test_predict_response_rating_in_valid_range(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that predicted_rating is between 1.0 and 5.0."""
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_matches_model_directly(
        self, test_client: TestClient, trained_model: Any
    ) -> None:
        """The API returns exactly what the model predicts (no hidden transformation)."""
        data = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"}).json()
        assert data["predicted_rating"] == trained_model.predict("196", "242")

    def test_predict_strips_whitespace_in_ids(self, test_client: TestClient) -> None:
        """IDs are normalised by the schema before reaching the model."""
        a = test_client.post("/predict", json={"user_id": " 196 ", "movie_id": "242"}).json()
        b = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"}).json()
        assert a == b

    # Validation errors -----------------------------------------------------------------

    def test_predict_missing_user_id_returns_422(self, test_client: TestClient) -> None:
        """Test that missing user_id returns 422 Unprocessable Entity."""
        assert test_client.post("/predict", json={"movie_id": "242"}).status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client: TestClient) -> None:
        """Test that missing movie_id returns 422."""
        assert test_client.post("/predict", json={"user_id": "196"}).status_code == 422

    def test_predict_empty_body_returns_422(self, test_client: TestClient) -> None:
        """Test that empty request body returns 422."""
        assert test_client.post("/predict", json={}).status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client: TestClient) -> None:
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/predict", content="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_all_invalid_requests_return_422(
        self, test_client: TestClient, invalid_prediction_requests: List[Dict[str, str]]
    ) -> None:
        """Every malformed request in the fixture is rejected by validation."""
        for payload in invalid_prediction_requests:
            assert test_client.post("/predict", json=payload).status_code == 422, payload

    def test_integer_ids_return_422(self, test_client: TestClient) -> None:
        """JSON numbers are not accepted as IDs."""
        assert (
            test_client.post("/predict", json={"user_id": 196, "movie_id": 242}).status_code == 422
        )

    def test_validation_error_explains_field(self, test_client: TestClient) -> None:
        """422 responses tell the client which field is wrong."""
        detail = test_client.post("/predict", json={"movie_id": "242"}).json()["detail"]
        assert any("user_id" in err["loc"] for err in detail)

    # Multiple requests -----------------------------------------------------------------

    def test_predict_multiple_valid_requests(
        self, test_client: TestClient, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Test multiple prediction requests all succeed."""
        for pair in known_user_movie_pairs:
            payload = {"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            assert test_client.post("/predict", json=payload).status_code == 200

    def test_predict_unknown_user_returns_fallback(self, test_client: TestClient) -> None:
        """Cold-start users get a valid fallback prediction, not an error."""
        response = test_client.post("/predict", json={"user_id": "new_user", "movie_id": "242"})
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0


class TestBatchPredictEndpoint:
    """Integration tests for /predict/batch endpoint."""

    def test_batch_predict_returns_200(
        self, test_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Test that batch prediction returns 200."""
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 200

    def test_batch_predict_returns_correct_count(
        self, test_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Test that batch prediction returns correct number of results."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        expected = len(sample_batch_request["predictions"])
        assert data["total_count"] == expected
        assert len(data["predictions"]) == expected

    def test_batch_predict_all_ratings_in_range(
        self, test_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Test that all batch predictions are in valid range."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert all(1.0 <= p["predicted_rating"] <= 5.0 for p in data["predictions"])

    def test_batch_preserves_request_order(
        self, test_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Results are returned in the same order as requested."""
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        requested = [(p["user_id"], p["movie_id"]) for p in sample_batch_request["predictions"]]
        returned = [(p["user_id"], p["movie_id"]) for p in data["predictions"]]
        assert returned == requested

    def test_batch_consistent_with_single_predict(
        self, test_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Batch and single endpoints agree for the same pairs."""
        batch = test_client.post("/predict/batch", json=sample_batch_request).json()["predictions"]
        for item in batch:
            single = test_client.post(
                "/predict", json={"user_id": item["user_id"], "movie_id": item["movie_id"]}
            ).json()
            assert single["predicted_rating"] == item["predicted_rating"]

    def test_batch_empty_list_returns_422(self, test_client: TestClient) -> None:
        """An empty batch is rejected."""
        assert test_client.post("/predict/batch", json={"predictions": []}).status_code == 422

    def test_batch_over_limit_returns_422(self, test_client: TestClient) -> None:
        """More than 100 items is rejected."""
        items = [{"user_id": str(i), "movie_id": "1"} for i in range(101)]
        assert test_client.post("/predict/batch", json={"predictions": items}).status_code == 422


class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, test_client: TestClient) -> None:
        """Test that unknown endpoint returns 404."""
        assert test_client.get("/unknown").status_code == 404

    def test_method_not_allowed_get_predict(self, test_client: TestClient) -> None:
        """Test that GET /predict returns 405 Method Not Allowed."""
        assert test_client.get("/predict").status_code == 405

    def test_method_not_allowed_post_health(self, test_client: TestClient) -> None:
        """Test that POST /health returns 405."""
        assert test_client.post("/health").status_code == 405

    def test_large_payload_rejected(self, test_client: TestClient) -> None:
        """Extremely long IDs are rejected by validation."""
        response = test_client.post("/predict", json={"user_id": "1" * 10000, "movie_id": "242"})
        assert response.status_code in (400, 422)

    def test_503_when_model_not_loaded(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """Without a model the service degrades to 503 and reports unhealthy."""
        monkeypatch.setattr(main, "model", None)
        assert test_client.post("/predict", json=sample_prediction_request).status_code == 503
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 503
        assert test_client.get("/health").json() == {"status": "unhealthy", "model_loaded": False}
        assert test_client.get("/model/info").json()["is_loaded"] is False

    def test_500_when_prediction_fails(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """Unexpected model errors become a 500 without leaking internals."""

        class Broken:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise ValueError("secret internal detail")

        monkeypatch.setattr(main, "model", Broken())
        single = test_client.post("/predict", json=sample_prediction_request)
        batch = test_client.post("/predict/batch", json=sample_batch_request)
        assert (single.status_code, batch.status_code) == (500, 500)
        assert "secret" not in single.text and "secret" not in batch.text

    def test_cors_headers_present(self, test_client: TestClient) -> None:
        """CORS is enabled for browser clients."""
        response = test_client.get("/health", headers={"Origin": "http://example.com"})
        assert response.headers.get("access-control-allow-origin") in ("*", "http://example.com")


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, test_client: TestClient) -> None:
        """Test that model info endpoint returns 200."""
        assert test_client.get("/model/info").status_code == 200

    def test_model_info_has_version(self, test_client: TestClient) -> None:
        """Test that model info has version field."""
        assert test_client.get("/model/info").json()["model_version"] == main.MODEL_VERSION

    def test_model_info_has_is_loaded(self, test_client: TestClient) -> None:
        """Test that model info has is_loaded field."""
        assert test_client.get("/model/info").json()["is_loaded"] is True

    def test_model_info_has_model_type(self, test_client: TestClient) -> None:
        """The algorithm family is reported."""
        assert "SVD" in test_client.get("/model/info").json()["model_type"]


class TestPerformance:
    """Simple non-functional checks (system level)."""

    def test_single_prediction_latency(
        self, test_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """p95 latency of /predict through the full HTTP stack stays under 50 ms."""
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            test_client.post("/predict", json=sample_prediction_request)
            timings.append(time.perf_counter() - start)
        timings.sort()
        assert timings[int(0.95 * len(timings)) - 1] < 0.05
