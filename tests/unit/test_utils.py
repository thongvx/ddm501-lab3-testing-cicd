"""
Unit tests for configuration and application helpers.

Run tests:
    pytest tests/unit/test_utils.py -v
"""

import importlib
from pathlib import Path

import pytest
from fastapi import HTTPException

import app.config as config
import app.main as main


class TestConfig:
    """Tests for app.config."""

    def test_rating_bounds(self) -> None:
        """The API contract uses the MovieLens 1-5 scale."""
        assert (config.MIN_RATING, config.MAX_RATING) == (1.0, 5.0)

    def test_default_model_path_inside_project(self) -> None:
        """By default the model is read from <project>/models/svd_model.pkl."""
        assert Path(config.MODEL_PATH).name == "svd_model.pkl"
        assert Path(config.MODEL_PATH).parent.name == "models"

    def test_environment_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MODEL_PATH, MODEL_VERSION, PORT and DEBUG come from the environment."""
        monkeypatch.setenv("MODEL_PATH", "/tmp/other.pkl")
        monkeypatch.setenv("MODEL_VERSION", "9.9.9")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("DEBUG", "TRUE")
        reloaded = importlib.reload(config)
        try:
            assert reloaded.MODEL_PATH == "/tmp/other.pkl"
            assert reloaded.MODEL_VERSION == "9.9.9"
            assert reloaded.PORT == 9000
            assert reloaded.DEBUG is True
        finally:
            monkeypatch.undo()
            importlib.reload(config)

    def test_debug_defaults_to_false(self) -> None:
        """DEBUG is off unless explicitly enabled."""
        assert config.DEBUG is False


class TestMainHelpers:
    """Tests for helper functions in app.main."""

    def test_require_model_raises_503_without_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_require_model() turns a missing model into HTTP 503."""
        monkeypatch.setattr(main, "model", None)
        with pytest.raises(HTTPException) as exc:
            main._require_model()
        assert exc.value.status_code == 503

    def test_load_model_failure_leaves_model_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If loading fails at startup the API stays up but reports unhealthy."""

        def boom() -> None:
            raise FileNotFoundError("no model")

        monkeypatch.setattr(main, "MovieRatingModel", boom)
        monkeypatch.setattr(main, "model", object())
        main.load_model()
        assert main.model is None
