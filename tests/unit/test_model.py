"""
Unit tests for MovieRatingModel class.

Two groups:
* tests against the real trained SVD model (``trained_model`` fixture);
* isolated tests against a tiny stub model pickled to ``tmp_path`` - they need no
  training and cover branches the real model never hits (clipping, unloaded model).

Run tests:
    pytest tests/unit/test_model.py -v
"""

import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app import model as model_module
from app.model import MovieRatingModel

Prediction = namedtuple("Prediction", ["uid", "iid", "est"])


class StubAlgo:
    """Minimal stand-in for a Surprise algorithm: returns a fixed estimate."""

    def __init__(self, est: float) -> None:
        self.est = est

    def predict(self, uid: str, iid: str) -> Prediction:
        return Prediction(uid, iid, self.est)


def make_model(tmp_path: Path, est: float) -> MovieRatingModel:
    """Pickle a stub algorithm and load it through the real wrapper."""
    path = tmp_path / f"stub_{est}.pkl"
    path.write_bytes(pickle.dumps(StubAlgo(est)))
    return MovieRatingModel(model_path=str(path))


class TestMovieRatingModel:
    """Unit tests for MovieRatingModel class."""

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, trained_model: MovieRatingModel) -> None:
        """Test that model loads without errors."""
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model: MovieRatingModel) -> None:
        """Test that model instance has the model attribute."""
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    # =========================================================================
    # Prediction Return Type Tests
    # =========================================================================

    def test_predict_returns_float(self, trained_model: MovieRatingModel) -> None:
        """Test that predict() returns a float value."""
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    def test_predict_rounded_to_two_decimals(self, trained_model: MovieRatingModel) -> None:
        """Predictions are rounded to 2 decimals for a stable API contract."""
        result = trained_model.predict("196", "242")
        assert result == round(result, 2)

    # =========================================================================
    # Rating Range Tests
    # =========================================================================

    def test_predict_returns_value_in_valid_range(self, trained_model: MovieRatingModel) -> None:
        """Test that predictions are within 1-5 range."""
        result = trained_model.predict("196", "242")
        assert 1.0 <= result <= 5.0

    def test_predict_multiple_pairs_all_in_range(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Test that all predictions are in valid range."""
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= result <= 5.0

    # =========================================================================
    # Batch Prediction Tests
    # =========================================================================

    def test_predict_batch_returns_list(self, trained_model: MovieRatingModel) -> None:
        """Test that predict_batch() returns a list."""
        results = trained_model.predict_batch([("196", "242"), ("186", "302")])
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model: MovieRatingModel) -> None:
        """Test that predict_batch() returns correct number of results."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model: MovieRatingModel) -> None:
        """Test that all batch predictions are in valid range."""
        results = trained_model.predict_batch([("196", "242"), ("186", "302"), ("22", "377")])
        assert all(isinstance(r, float) and 1.0 <= r <= 5.0 for r in results)

    def test_predict_batch_empty_list(self, trained_model: MovieRatingModel) -> None:
        """An empty batch returns an empty list (no error)."""
        assert trained_model.predict_batch([]) == []

    # =========================================================================
    # is_loaded() Tests
    # =========================================================================

    def test_is_loaded_returns_bool(self, trained_model: MovieRatingModel) -> None:
        """Test that is_loaded() returns a boolean."""
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model: MovieRatingModel) -> None:
        """Test that is_loaded() returns True for loaded model."""
        assert trained_model.is_loaded() is True

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_predict_with_none_user_id(self, trained_model: MovieRatingModel) -> None:
        """None IDs are rejected with TypeError instead of producing a silent prediction."""
        with pytest.raises(TypeError):
            trained_model.predict(None, "242")  # type: ignore[arg-type]

    def test_predict_with_empty_string(self, trained_model: MovieRatingModel) -> None:
        """Empty IDs are unknown to the model -> global-mean fallback, still in range."""
        result = trained_model.predict("", "")
        assert 1.0 <= result <= 5.0


class TestModelWithStub:
    """Isolated unit tests (no trained model needed)."""

    def test_prediction_above_max_is_clipped(self, tmp_path: Path) -> None:
        """Raw estimates above 5 are clipped to 5.0."""
        assert make_model(tmp_path, 7.3).predict("u", "m") == 5.0

    def test_prediction_below_min_is_clipped(self, tmp_path: Path) -> None:
        """Raw estimates below 1 are clipped to 1.0."""
        assert make_model(tmp_path, -2.0).predict("u", "m") == 1.0

    def test_prediction_is_rounded(self, tmp_path: Path) -> None:
        """Estimates are rounded to two decimals."""
        assert make_model(tmp_path, 3.14159).predict("u", "m") == 3.14

    def test_batch_preserves_order(self, tmp_path: Path) -> None:
        """predict_batch returns one value per pair, in order."""
        model = make_model(tmp_path, 4.0)
        assert model.predict_batch([("a", "1"), ("b", "2")]) == [4.0, 4.0]

    def test_predict_raises_when_model_unloaded(self, tmp_path: Path) -> None:
        """predict() fails clearly if the underlying model was dropped."""
        model = make_model(tmp_path, 3.0)
        model.model = None
        assert model.is_loaded() is False
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict("u", "m")

    def test_predict_batch_raises_when_model_unloaded(self, tmp_path: Path) -> None:
        """predict_batch() fails clearly if the underlying model was dropped."""
        model = make_model(tmp_path, 3.0)
        model.model = None
        with pytest.raises(RuntimeError):
            model.predict_batch([("u", "m")])

    def test_integer_ids_rejected(self, tmp_path: Path) -> None:
        """The wrapper enforces string IDs (Surprise uses raw string IDs)."""
        with pytest.raises(TypeError):
            make_model(tmp_path, 3.0).predict(196, 242)  # type: ignore[arg-type]


class TestModelFileHandling:
    """Tests for model file handling."""

    def test_model_raises_error_for_missing_file(self) -> None:
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")

    def test_model_raises_error_for_corrupt_file(self, tmp_path: Path) -> None:
        """A file that is not a pickle raises and is logged, not silently ignored."""
        bad = tmp_path / "corrupt.pkl"
        bad.write_bytes(b"this is not a pickle")
        with pytest.raises(Exception):
            MovieRatingModel(model_path=str(bad))

    def test_model_path_is_stored(self, tmp_path: Path) -> None:
        """The wrapper remembers where it loaded from (useful for /model/info)."""
        model = make_model(tmp_path, 3.0)
        assert model.model_path.endswith("stub_3.0.pkl")


class TestModelSingleton:
    """Tests for get_model() / reset_model()."""

    def test_get_model_returns_same_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_model() creates the model once and then reuses it."""
        path = tmp_path / "stub.pkl"
        path.write_bytes(pickle.dumps(StubAlgo(3.0)))
        monkeypatch.setattr(model_module.MovieRatingModel.__init__, "__defaults__", (str(path),))
        model_module.reset_model()
        first = model_module.get_model()
        assert model_module.get_model() is first
        model_module.reset_model()
        assert model_module._model_instance is None
