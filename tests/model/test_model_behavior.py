"""
Model behavioral tests (CheckList approach).

- Invariance: output must not change for perturbations that should not matter
- Directional: output must move in the expected direction when the input changes
- Minimum functionality: simple cases the model must get right
- Performance / robustness: quality thresholds and unusual inputs

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""

import random
from statistics import mean
from typing import Any, Dict, List

import pytest

from app.model import MovieRatingModel

Records = List[Dict[str, Any]]


class TestModelInvariance:
    """Invariance tests - output shouldn't change for certain perturbations."""

    def test_same_input_same_output(self, trained_model: MovieRatingModel) -> None:
        """Test that same input always produces same output."""
        assert trained_model.predict("196", "242") == trained_model.predict("196", "242")

    def test_multiple_calls_consistent(self, trained_model: MovieRatingModel) -> None:
        """Test that 5 calls with the same input give identical results."""
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_batch_order_independent(self, trained_model: MovieRatingModel) -> None:
        """Test that batch predictions are independent of input order."""
        pairs1 = [("196", "242"), ("186", "302"), ("22", "377")]
        pairs2 = list(reversed(pairs1))
        results1 = dict(zip(pairs1, trained_model.predict_batch(pairs1)))
        results2 = dict(zip(pairs2, trained_model.predict_batch(pairs2)))
        assert results1 == results2

    def test_individual_vs_batch_same_results(self, trained_model: MovieRatingModel) -> None:
        """Test that individual and batch predictions match."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        individual = [trained_model.predict(u, m) for u, m in pairs]
        assert trained_model.predict_batch(pairs) == individual

    def test_batch_context_does_not_leak(self, trained_model: MovieRatingModel) -> None:
        """A pair's prediction does not depend on which other pairs share the batch."""
        alone = trained_model.predict_batch([("196", "242")])[0]
        with_others = trained_model.predict_batch([("1", "1"), ("196", "242"), ("2", "2")])[1]
        assert alone == with_others


class TestModelDirectional:
    """Directional tests - output should change in expected direction."""

    def test_predictions_are_reasonable(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: Records
    ) -> None:
        """
        Predictions for known ratings are close to the truth.

        SVD smooths towards user/item means, so a single extreme rating cannot always be
        reproduced (user 166 gave movie 346 one star; the model predicts ~2.8). We require
        >= 80% of pairs within 1.5 stars and none worse than 2.0 stars.
        """
        errors = [
            abs(trained_model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"])
            for p in known_user_movie_pairs
        ]
        assert sum(e < 1.5 for e in errors) / len(errors) >= 0.8, errors
        assert max(errors) < 2.0, errors

    def test_different_movies_different_predictions(self, trained_model: MovieRatingModel) -> None:
        """Same user, different movies -> predictions differ (model is personalised by item)."""
        predictions = {trained_model.predict("196", m) for m in ["242", "302", "377", "51", "346"]}
        assert len(predictions) > 1

    def test_different_users_different_predictions(self, trained_model: MovieRatingModel) -> None:
        """Different users, same movie -> predictions differ (model is personalised by user)."""
        predictions = {trained_model.predict(u, "242") for u in ["196", "186", "22", "244", "166"]}
        assert len(predictions) > 1

    def test_loved_movie_scores_higher_than_hated_movie(
        self, trained_model: MovieRatingModel, movielens_ratings: Records
    ) -> None:
        """For users who gave both 5 and 1 stars, the 5-star movie is predicted higher (>= 95%)."""
        by_user: Dict[str, Dict[float, List[str]]] = {}
        for r in movielens_ratings:
            by_user.setdefault(r["user_id"], {}).setdefault(r["rating"], []).append(r["movie_id"])
        checks = [
            trained_model.predict(user, stars[5.0][0]) > trained_model.predict(user, stars[1.0][0])
            for user, stars in by_user.items()
            if 5.0 in stars and 1.0 in stars
        ]
        assert len(checks) > 100
        assert sum(checks) / len(checks) >= 0.95

    def test_cold_start_follows_movie_popularity(self, trained_model: MovieRatingModel) -> None:
        """For an unknown user, acclaimed movies score higher than poorly rated ones."""
        acclaimed = ["408", "318", "169"]  # A Close Shave, Schindler's List, The Wrong Trousers
        poorly_rated = ["122", "243", "325"]  # among the lowest-rated movies with > 100 ratings
        low = max(trained_model.predict("new_user", m) for m in poorly_rated)
        high = min(trained_model.predict("new_user", m) for m in acclaimed)
        assert high > low


class TestMinimumFunctionality:
    """Minimum functionality tests - basic cases the model must handle."""

    def test_can_predict_for_known_user(self, trained_model: MovieRatingModel) -> None:
        """Test that model can make prediction for known user."""
        prediction = trained_model.predict("196", "242")
        assert prediction is not None
        assert 1.0 <= prediction <= 5.0

    def test_can_predict_for_multiple_users(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: Records
    ) -> None:
        """Test that model can make predictions for multiple known users."""
        for pair in known_user_movie_pairs:
            assert 1.0 <= trained_model.predict(pair["user_id"], pair["movie_id"]) <= 5.0

    def test_predictions_not_all_same(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: Records
    ) -> None:
        """If all predictions are identical, the model might be broken."""
        predictions = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, "All predictions are identical"

    def test_handles_unknown_user_gracefully(
        self, trained_model: MovieRatingModel, unknown_users: List[str]
    ) -> None:
        """Unknown users get a valid prediction (baseline fallback) instead of a crash."""
        for user_id in unknown_users:
            assert 1.0 <= trained_model.predict(user_id, "242") <= 5.0

    def test_handles_unknown_movie_gracefully(
        self, trained_model: MovieRatingModel, unknown_movies: List[str]
    ) -> None:
        """Unknown movies get a valid prediction (baseline fallback) instead of a crash."""
        for movie_id in unknown_movies:
            assert 1.0 <= trained_model.predict("196", movie_id) <= 5.0

    def test_unknown_user_and_movie_returns_global_mean(
        self, trained_model: MovieRatingModel
    ) -> None:
        """With no information at all the model falls back to the global mean (~3.53)."""
        global_mean = round(trained_model.model.trainset.global_mean, 2)  # type: ignore[union-attr]
        assert trained_model.predict("new_user", "new_movie") == pytest.approx(
            global_mean, abs=0.01
        )


class TestModelPerformance:
    """Performance-related behavioral tests."""

    def test_average_error_acceptable(
        self, trained_model: MovieRatingModel, movielens_ratings: Records
    ) -> None:
        """MAE on a 1,000-rating random sample of the training data is below 1.0."""
        sample = random.Random(42).sample(movielens_ratings, 1000)
        mae = mean(
            abs(trained_model.predict(r["user_id"], r["movie_id"]) - r["rating"]) for r in sample
        )
        assert mae < 1.0

    def test_no_extreme_errors(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: Records
    ) -> None:
        """No prediction for a known pair is off by more than 3.0."""
        for pair in known_user_movie_pairs:
            error = abs(
                trained_model.predict(pair["user_id"], pair["movie_id"]) - pair["actual_rating"]
            )
            assert error <= 3.0

    def test_holdout_metrics_meet_thresholds(self, model_metrics: Dict[str, Any]) -> None:
        """Generalisation on unseen ratings: hold-out RMSE < 0.95 and MAE < 0.75."""
        assert model_metrics["rmse"] < 0.95
        assert model_metrics["mae"] < 0.75

    def test_beats_global_mean_baseline(
        self, trained_model: MovieRatingModel, movielens_ratings: Records
    ) -> None:
        """The model is better than always predicting the global mean."""
        sample = random.Random(7).sample(movielens_ratings, 1000)
        global_mean = mean(r["rating"] for r in movielens_ratings)
        model_mae = mean(
            abs(trained_model.predict(r["user_id"], r["movie_id"]) - r["rating"]) for r in sample
        )
        baseline_mae = mean(abs(global_mean - r["rating"]) for r in sample)
        assert model_mae < baseline_mae


class TestModelRobustness:
    """Robustness tests - model behavior under unusual conditions."""

    def test_handles_string_numeric_ids(self, trained_model: MovieRatingModel) -> None:
        """Numeric-looking string IDs are the normal case and work."""
        assert 1.0 <= trained_model.predict("1", "1") <= 5.0

    def test_handles_leading_zeros_in_ids(self, trained_model: MovieRatingModel) -> None:
        """
        IDs are opaque strings: "001" is NOT user "1".

        "001" is unknown, so it gets the same cold-start prediction as any other new user.
        This documents the behaviour so a future normalisation change is a conscious decision.
        """
        assert trained_model.predict("001", "242") == trained_model.predict("new_user", "242")
        assert 1.0 <= trained_model.predict("001", "242") <= 5.0

    def test_handles_unicode_and_long_ids(self, trained_model: MovieRatingModel) -> None:
        """Arbitrary strings never crash the model."""
        for user_id in ["người_dùng", "🙂", "x" * 500]:
            assert 1.0 <= trained_model.predict(user_id, "242") <= 5.0
