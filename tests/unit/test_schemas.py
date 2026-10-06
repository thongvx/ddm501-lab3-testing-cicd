"""
Unit tests for Pydantic schemas.

Run tests:
    pytest tests/unit/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self) -> None:
        """Test that valid request passes validation."""
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self) -> None:
        """Test numeric string IDs are valid."""
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    # Missing fields --------------------------------------------------------------

    def test_missing_user_id_raises_error(self) -> None:
        """Test that missing user_id raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(movie_id="242")  # type: ignore[call-arg]

    def test_missing_movie_id_raises_error(self) -> None:
        """Test that missing movie_id raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196")  # type: ignore[call-arg]

    def test_missing_both_fields_raises_error(self) -> None:
        """Test that missing both fields raises ValidationError with two errors."""
        with pytest.raises(ValidationError) as exc:
            PredictionRequest()  # type: ignore[call-arg]
        assert exc.value.error_count() == 2

    # Empty / invalid input ---------------------------------------------------------

    def test_empty_user_id_raises_error(self) -> None:
        """Test that empty user_id raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_empty_movie_id_raises_error(self) -> None:
        """Test that empty movie_id raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id="")

    def test_whitespace_only_user_id_raises_error(self) -> None:
        """Test that whitespace-only user_id raises ValidationError."""
        with pytest.raises(ValidationError, match="whitespace"):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """IDs are normalised by stripping surrounding whitespace."""
        request = PredictionRequest(user_id="  196 ", movie_id="\t242\n")
        assert (request.user_id, request.movie_id) == ("196", "242")

    def test_none_values_raise_error(self) -> None:
        """Test that None values raise ValidationError."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["user_id", "movie_id"])
    def test_id_longer_than_50_chars_raises_error(self, field: str) -> None:
        """IDs are capped at 50 characters (protects against huge payloads)."""
        data = {"user_id": "196", "movie_id": "242", field: "1" * 51}
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_id_of_exactly_50_chars_is_valid(self) -> None:
        """Boundary: 50 characters is still accepted."""
        assert len(PredictionRequest(user_id="1" * 50, movie_id="242").user_id) == 50

    # Type validation -----------------------------------------------------------------

    def test_integer_user_id_converted_to_string(self) -> None:
        """Pydantic v2 does not coerce int -> str: integer IDs are rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=196, movie_id=242)  # type: ignore[arg-type]


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self) -> None:
        """Test that valid response passes validation."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        assert response.predicted_rating == 3.5

    def test_rating_below_minimum_raises_error(self) -> None:
        """Test that rating below 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=0.5, model_version="1.0.0"
            )

    def test_rating_above_maximum_raises_error(self) -> None:
        """Test that rating above 5.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=5.5, model_version="1.0.0"
            )

    @pytest.mark.parametrize("rating", [1.0, 5.0])
    def test_rating_at_boundaries(self, rating: float) -> None:
        """Test ratings at exact boundaries (1.0 and 5.0) are accepted."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=rating, model_version="1.0.0"
        )
        assert response.predicted_rating == rating

    def test_response_serialises_to_json(self) -> None:
        """The API contract fields are all present in the JSON output."""
        payload = PredictionResponse(
            user_id="1", movie_id="2", predicted_rating=4.0, model_version="1.0.0"
        ).model_dump()
        assert set(payload) == {"user_id", "movie_id", "predicted_rating", "model_version"}


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self) -> None:
        """Test that valid health response passes validation."""
        health = HealthResponse(status="healthy", model_loaded=True)
        assert health.status == "healthy"
        assert health.model_loaded is True

    @pytest.mark.parametrize("status,loaded", [("healthy", True), ("unhealthy", False)])
    def test_health_response_status_types(self, status: str, loaded: bool) -> None:
        """Test the two status values the API emits."""
        health = HealthResponse(status=status, model_loaded=loaded)
        assert (health.status, health.model_loaded) == (status, loaded)

    def test_health_response_requires_model_loaded(self) -> None:
        """model_loaded is mandatory."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy")  # type: ignore[call-arg]


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_batch_request(self) -> None:
        """Test that valid batch request passes validation."""
        batch = BatchPredictionRequest(
            predictions=[
                PredictionItem(user_id="196", movie_id="242"),
                PredictionItem(user_id="186", movie_id="302"),
            ]
        )
        assert len(batch.predictions) == 2

    def test_batch_request_from_dicts(self) -> None:
        """Nested dicts are parsed into PredictionItem objects."""
        batch = BatchPredictionRequest.model_validate(
            {"predictions": [{"user_id": "1", "movie_id": "2"}]}
        )
        assert isinstance(batch.predictions[0], PredictionItem)

    def test_empty_predictions_list_raises_error(self) -> None:
        """Test that empty predictions list raises ValidationError."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_too_many_predictions_raises_error(self) -> None:
        """Test that more than 100 predictions raises ValidationError."""
        items = [PredictionItem(user_id=str(i), movie_id="1") for i in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)

    def test_exactly_100_predictions_is_valid(self) -> None:
        """Boundary: 100 items is the maximum accepted."""
        items = [PredictionItem(user_id=str(i), movie_id="1") for i in range(100)]
        assert len(BatchPredictionRequest(predictions=items).predictions) == 100

    def test_invalid_item_in_batch_raises_error(self) -> None:
        """One invalid item invalidates the whole batch."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest.model_validate(
                {"predictions": [{"user_id": "1", "movie_id": "2"}, {"user_id": ""}]}
            )


class TestOtherSchemas:
    """Batch response and error schema."""

    def test_batch_response(self) -> None:
        """Batch response carries the items and their count."""
        item = PredictionResponse(
            user_id="1", movie_id="2", predicted_rating=3.0, model_version="1"
        )
        response = BatchPredictionResponse(predictions=[item], total_count=1)
        assert response.total_count == len(response.predictions) == 1

    def test_error_response_default_code(self) -> None:
        """Error responses default to UNKNOWN_ERROR."""
        assert ErrorResponse(detail="boom").error_code == "UNKNOWN_ERROR"
