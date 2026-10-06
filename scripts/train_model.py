"""
Train and save the movie rating prediction model (SVD on MovieLens 100K).

Usage:
    python scripts/train_model.py              # 5-fold CV + hold-out metrics + final model
    python scripts/train_model.py --skip-cv    # faster (used by CI)

Outputs:
    models/svd_model.pkl     trained model used by the API
    models/metrics.json      hold-out RMSE/MAE used by the model-validation gate
"""

import argparse
import json
import pickle  # nosec B403
from pathlib import Path
from typing import Dict

from surprise import SVD, Dataset, accuracy
from surprise.model_selection import cross_validate, train_test_split

SEED = 42
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HYPERPARAMS: Dict[str, float] = {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}


def build_model() -> SVD:
    """Create the SVD model with fixed hyper-parameters and seed (reproducible)."""
    return SVD(random_state=SEED, **HYPERPARAMS)


def main() -> None:
    """Train, evaluate and save the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cv", action="store_true", help="skip 5-fold cross-validation")
    args = parser.parse_args()

    print("=" * 60)
    print("Movie Rating Prediction Model Training")
    print("=" * 60)
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n[1/5] Loading MovieLens 100K dataset...")
    # prompt=False: download without asking (non-interactive CI / Docker builds)
    data = Dataset.load_builtin("ml-100k", prompt=False)

    if not args.skip_cv:
        print("\n[2/5] 5-fold cross-validation...")
        cv = cross_validate(build_model(), data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        print(
            f"      Mean RMSE: {cv['test_rmse'].mean():.4f}  Mean MAE: {cv['test_mae'].mean():.4f}"
        )
    else:
        print("\n[2/5] Cross-validation skipped (--skip-cv)")

    print("\n[3/5] Hold-out evaluation (80/20 split)...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)
    holdout_model = build_model()
    holdout_model.fit(trainset)
    predictions = holdout_model.test(testset)
    metrics = {
        "rmse": round(accuracy.rmse(predictions, verbose=False), 4),
        "mae": round(accuracy.mae(predictions, verbose=False), 4),
        "n_test": len(testset),
        "seed": SEED,
        **HYPERPARAMS,
    }
    print(f"      Hold-out RMSE: {metrics['rmse']}  MAE: {metrics['mae']}")

    print("\n[4/5] Training final model on the full dataset...")
    model = build_model()
    model.fit(data.build_full_trainset())

    print(f"\n[5/5] Saving model and metrics to {MODELS_DIR}...")
    with open(MODELS_DIR / "svd_model.pkl", "wb") as f:
        pickle.dump(model, f)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sample = model.predict("196", "242")
    print(f"\nSample prediction for user 196, movie 242: {sample.est:.2f}")
    print("Training complete!")


if __name__ == "__main__":
    main()
