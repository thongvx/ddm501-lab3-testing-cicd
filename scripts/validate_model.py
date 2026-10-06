"""
Model validation gate used in CI (model-validation workflow).

Fails (exit code 1) if the trained model does not load, predicts outside 1-5,
or misses the hold-out quality thresholds.

Usage:
    python scripts/validate_model.py [--max-rmse 0.95] [--max-mae 0.75]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import MovieRatingModel  # noqa: E402


def main() -> int:
    """Run all model checks and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rmse", type=float, default=0.95)
    parser.add_argument("--max-mae", type=float, default=0.75)
    args = parser.parse_args()

    failures = []
    model = MovieRatingModel()
    if not model.is_loaded():
        failures.append("model failed to load")

    sample = model.predict_batch([("196", "242"), ("186", "302"), ("99999", "242")])
    if not all(1.0 <= r <= 5.0 for r in sample):
        failures.append(f"prediction out of range: {sample}")

    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    if metrics["rmse"] > args.max_rmse:
        failures.append(f"RMSE {metrics['rmse']} > {args.max_rmse}")
    if metrics["mae"] > args.max_mae:
        failures.append(f"MAE {metrics['mae']} > {args.max_mae}")

    print(f"Hold-out RMSE={metrics['rmse']} (max {args.max_rmse})")
    print(f"Hold-out MAE={metrics['mae']} (max {args.max_mae})")
    print(f"Sample predictions: {sample}")
    if failures:
        print("MODEL VALIDATION FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("Model validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
