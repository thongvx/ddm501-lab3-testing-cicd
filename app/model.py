"""
ML Model wrapper for movie rating prediction.
"""

import logging
import pickle  # nosec B403 - the model file is produced by our own training script
from typing import Any, List, Optional, Tuple

from app.config import MAX_RATING, MIN_RATING, MODEL_PATH

logger = logging.getLogger(__name__)


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model_path = model_path
        self.model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)  # nosec B301 - trusted, self-produced artifact
            logger.info("Model loaded successfully from %s", self.model_path)
        except FileNotFoundError:
            logger.error("Model file not found: %s", self.model_path)
            raise
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Unknown users or movies fall back to the global mean rating learned by SVD,
        so the method never fails for valid string IDs.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)

        Raises:
            RuntimeError: If the model is not loaded.
            TypeError: If an ID is not a string.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not isinstance(user_id, str) or not isinstance(movie_id, str):
            raise TypeError("user_id and movie_id must be strings")

        prediction = self.model.predict(user_id, movie_id)
        rating = round(float(prediction.est), 2)

        # Clip to valid range
        return max(MIN_RATING, min(MAX_RATING, rating))

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings, in the same order as ``pairs``
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Singleton instance
_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance


def reset_model() -> None:
    """Reset the model instance (useful for testing)."""
    global _model_instance
    _model_instance = None
