"""
Shared pytest fixtures for all tests.

This file is automatically loaded by pytest and provides
fixtures that can be used across all test modules.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from app.config import BASE_DIR, MODEL_PATH
from app.main import app
from app.model import MovieRatingModel

RATINGS_FILE = Path.home() / ".surprise_data" / "ml-100k" / "ml-100k" / "u.data"


# =============================================================================
# API Client Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_client() -> Iterator[TestClient]:
    """
    Create a test client for API tests.

    The ``with`` block runs the application lifespan, so the model is loaded
    exactly as in production (a bare ``TestClient(app)`` would skip startup).
    Scope: session - created once for all tests
    """
    if not Path(MODEL_PATH).exists():
        pytest.skip("Model file not found. Run scripts/train_model.py first.")
    with TestClient(app) as client:
        yield client


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def trained_model() -> MovieRatingModel:
    """
    Load model once for all tests.

    Scope: session - model is loaded once and reused
    """
    try:
        return MovieRatingModel()
    except FileNotFoundError:
        pytest.skip("Model file not found. Run scripts/train_model.py first.")


@pytest.fixture(scope="session")
def model_metrics() -> Dict[str, Any]:
    """Hold-out metrics written by scripts/train_model.py."""
    path = BASE_DIR / "models" / "metrics.json"
    if not path.exists():
        pytest.skip("models/metrics.json not found. Run scripts/train_model.py first.")
    return dict(json.loads(path.read_text()))


# =============================================================================
# Real Dataset Fixture (MovieLens 100K, downloaded by the training script)
# =============================================================================


@pytest.fixture(scope="session")
def movielens_ratings() -> List[Dict[str, Any]]:
    """
    All 100,000 MovieLens ratings as dicts (user_id, movie_id, rating, timestamp).

    Skips when the dataset has not been downloaded (it is fetched by train_model.py).
    """
    if not RATINGS_FILE.exists():
        pytest.skip("MovieLens 100K not downloaded. Run scripts/train_model.py first.")
    with open(RATINGS_FILE, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        return [
            {"user_id": u, "movie_id": m, "rating": float(r), "timestamp": int(t)}
            for u, m, r, t in reader
        ]


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prediction_request() -> Dict[str, str]:
    """Sample valid prediction request."""
    return {"user_id": "196", "movie_id": "242"}


@pytest.fixture
def sample_batch_request() -> Dict[str, List[Dict[str, str]]]:
    """Sample batch prediction request."""
    return {
        "predictions": [
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "186", "movie_id": "302"},
            {"user_id": "22", "movie_id": "377"},
        ]
    }


@pytest.fixture
def sample_ratings() -> List[Dict[str, Any]]:
    """Sample ratings data for data quality tests."""
    return [
        {"user_id": "1", "movie_id": "10", "rating": 4.0},
        {"user_id": "1", "movie_id": "20", "rating": 3.5},
        {"user_id": "2", "movie_id": "10", "rating": 5.0},
        {"user_id": "2", "movie_id": "30", "rating": 2.0},
        {"user_id": "3", "movie_id": "10", "rating": 3.0},
        {"user_id": "3", "movie_id": "20", "rating": 4.5},
        {"user_id": "3", "movie_id": "30", "rating": 1.0},
    ]


@pytest.fixture
def invalid_prediction_requests() -> List[Dict[str, str]]:
    """Collection of invalid prediction requests for testing validation."""
    return [
        {},  # Empty
        {"user_id": "196"},  # Missing movie_id
        {"movie_id": "242"},  # Missing user_id
        {"user_id": "", "movie_id": "242"},  # Empty user_id
        {"user_id": "196", "movie_id": ""},  # Empty movie_id
        {"user_id": "   ", "movie_id": "242"},  # Whitespace user_id
    ]


# =============================================================================
# Known Test Cases Fixtures
# =============================================================================


@pytest.fixture
def known_user_movie_pairs() -> List[Dict[str, Any]]:
    """
    Known user-movie pairs from MovieLens 100K dataset.
    These are actual ratings that exist in the training data.
    """
    return [
        {"user_id": "196", "movie_id": "242", "actual_rating": 3.0},
        {"user_id": "186", "movie_id": "302", "actual_rating": 3.0},
        {"user_id": "22", "movie_id": "377", "actual_rating": 1.0},
        {"user_id": "244", "movie_id": "51", "actual_rating": 2.0},
        {"user_id": "166", "movie_id": "346", "actual_rating": 1.0},
    ]


@pytest.fixture
def unknown_users() -> List[str]:
    """User IDs that are unlikely to exist in the dataset."""
    return ["99999", "999999", "0", "-1", "new_user"]


@pytest.fixture
def unknown_movies() -> List[str]:
    """Movie IDs that are unlikely to exist in the dataset."""
    return ["99999", "999999", "0", "-1", "new_movie"]
