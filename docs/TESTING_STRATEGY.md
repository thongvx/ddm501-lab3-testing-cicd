# Testing Strategy – Movie Rating Prediction API

**Course:** DDM501 – AI in Production · **Lab 3:** Testing & CI/CD for ML Systems
**Author:** Vương Xuân Thong (25MS13306)

This document explains *what* we test, *why*, *how* the tests are organised, and how the CI/CD
pipeline uses them as quality gates before code or a model reaches production.

---

## 1. System under test

| Component | File | Responsibility |
|---|---|---|
| Model wrapper | `app/model.py` | Load the pickled Surprise **SVD** model, predict one pair or a batch, clip to 1–5 |
| API | `app/main.py` | FastAPI app: `/`, `/health`, `/predict`, `/predict/batch`, `/model/info`; model loaded in `lifespan` |
| Schemas | `app/schemas.py` | Pydantic v2 request/response contracts (ID length 1–50, rating 1–5, batch 1–100) |
| Config | `app/config.py` | Paths, versions and server settings, overridable via environment variables |
| Training | `scripts/train_model.py` | MovieLens 100K → SVD (seed 42); writes `models/svd_model.pkl` + hold-out `metrics.json` |
| Model gate | `scripts/validate_model.py` | Fails if the model does not load, predicts outside 1–5, or misses RMSE/MAE thresholds |

## 2. Quality goals and risks

| Risk in an ML service | Example failure | Guarded by |
|---|---|---|
| Code regressions | refactor breaks clipping or batch order | unit tests |
| Broken API contract | field renamed, wrong status code, internals leaked in errors | integration tests |
| Bad training data | duplicated ratings, ratings outside 1–5, missing IDs | data tests |
| Silently degraded model | model predicts the mean for everyone; worse than baseline | behavioural tests + model gate |
| Environment drift | dependency cannot be installed, image does not start | CI on Python 3.10/3.11, Docker smoke test |
| Unsafe release | untested tag shipped to production | CD re-runs CI, staging smoke test before release |

## 3. Test pyramid

```
                    ▲  System / E2E        Docker smoke tests (CI) + staging smoke tests (CD)
                   ▲▲▲ Model behaviour     23 tests  – invariance, directional, MFT, performance, robustness
                 ▲▲▲▲▲ Data quality        32 tests  – schema, ranges, completeness, uniqueness, distribution
              ▲▲▲▲▲▲▲▲ Integration         44 tests  – every endpoint, validation, error paths, latency
          ▲▲▲▲▲▲▲▲▲▲▲▲ Unit                63 tests  – model wrapper, schemas, config, helpers
```

**Total: 162 tests, 100% line coverage of `app/`** (gate: ≥ 80%). The full suite runs in about 2 s
once the model is trained, so there is no reason to skip it locally.

### 3.1 Unit tests – `tests/unit/` (63)

| File | Classes | What is verified |
|---|---|---|
| `test_model.py` | `TestMovieRatingModel` (14) | load, return type `float`, 2-decimal rounding, 1–5 range, batch type/length/range/empty, `is_loaded`, `None`/empty IDs |
| | `TestModelWithStub` (7) | **isolated** tests with a stub model pickled to `tmp_path`: clipping above 5 and below 1, rounding, order, `RuntimeError` when unloaded, `TypeError` for non-string IDs |
| | `TestModelFileHandling` (3), `TestModelSingleton` (1) | missing / corrupt file, stored path, `get_model()` singleton and `reset_model()` |
| `test_schemas.py` | 5 classes (32) | required fields, empty / whitespace / `None`, stripping, 50-char limit and boundary, no int→str coercion, rating bounds 1.0/5.0, batch 1–100 and boundaries, nested validation |
| `test_utils.py` | `TestConfig` (4), `TestMainHelpers` (2) | env-var overrides, defaults, 503 helper, startup failure leaves API alive but unhealthy |

The stub-model tests make the unit layer independent of training and cover branches the real model
never reaches (e.g. a raw estimate of 7.3 must be clipped to 5.0).

### 3.2 Integration tests – `tests/integration/test_api.py` (44)

The FastAPI app is exercised through `TestClient` **inside a `with` block**, so the `lifespan` handler
loads the real model exactly as in production (the starter's bare `TestClient(app)` skipped startup and
every `/predict` returned 503).

* Happy paths for all 5 endpoints, response structure and echoes, `model_version`, OpenAPI schema.
* API result equals a direct model call (no hidden transformation); batch equals single predictions;
  batch preserves order.
* Validation: missing fields, empty body, invalid JSON, integer IDs, every invalid fixture, 10 000-char
  IDs, empty / >100 batches → **422** with the offending field in `detail`.
* Error handling: 404, 405 (`GET /predict`, `POST /health`), **503** when the model is missing
  (also `/health` → unhealthy), **500** when prediction raises – and the internal exception text is
  *not* leaked to the client. CORS header present.
* Non-functional: p95 latency of `/predict` through the HTTP stack < 50 ms.

### 3.3 Data tests – `tests/data/test_data_quality.py` (32)

Two data sources:

1. `sample_ratings` (fixture) – the lab's required checks: range, negatives, maximum, missing IDs,
   ID types, nulls/NaN, required fields, mean 2.0–4.5, 0 < std < 2.0, distinct values, unique
   (user, movie) pairs, multiple users/movies, numeric types.
2. **The real MovieLens 100K file the model is trained on** – a data contract: exactly 100 000 rows,
   schema, ratings ∈ {1,2,3,4,5}, no missing IDs, no duplicate pairs, 943 users / 1 682 movies,
   every user ≥ 20 ratings, mean 3.0–4.0, std 0.8–1.4, no star value > 50%, timestamps inside the
   collection period, and the "known pairs" used by model tests really exist with those ratings.

`TestDataValidationCatchesBadData` feeds corrupted records to the rules to prove the checks can fail
(a test that can never fail protects nothing).

### 3.4 Model behavioural tests – `tests/model/test_model_behavior.py` (23)

Following the CheckList methodology:

| Type | Tests |
|---|---|
| **Invariance** | same input → same output (×2, ×5); batch order does not change per-pair results; batch = individual; a pair's score does not depend on its batch neighbours |
| **Directional** | different users / movies → different predictions; for > 100 users who gave both 5★ and 1★, the 5★ movie is predicted higher in ≥ 95% of cases; for a new user, acclaimed movies (A Close Shave, Schindler's List, The Wrong Trousers) score above poorly rated ones |
| **Minimum functionality** | known users predicted in range; predictions not all identical; unknown users / movies get a valid fallback; unknown user + movie → global mean (≈ 3.53) |
| **Performance** | MAE < 1.0 on 1 000 real ratings; no error > 3.0 on known pairs; hold-out RMSE < 0.95 and MAE < 0.75; better than the global-mean baseline |
| **Robustness** | numeric strings, leading zeros (`"001"` is a different, unknown user – documented), Unicode and 500-char IDs |

**Why "≥ 80% within 1.5 stars" instead of "every pair within 1.5":** the starter suggested every known
pair should be within 1.5 stars. User 166 rated movie 346 one star, but SVD predicts ≈ 2.8 because
it regularises towards user and item means. Loosening the tolerance until it passes would hide
information, so the test asserts what a good collaborative-filtering model can actually guarantee
(most pairs close, none absurd) and the real accuracy requirement is moved to a proper hold-out set
(`test_holdout_metrics_meet_thresholds`).

### 3.5 System / end-to-end tests (in the pipelines)

* **CI `docker` job:** builds the multi-stage image, runs it, checks `/health` reports
  `model_loaded: true`, a valid `/predict`, a 422 for a bad request, and waits for Docker's own
  `HEALTHCHECK` to report `healthy`.
* **CD `deploy-staging` job:** pulls the image that was just pushed (by digest), runs it and smoke-tests
  `/health`, `/predict`, `/predict/batch`, `/model/info` before a release is created.

## 4. Fixtures and test data

All shared fixtures live in `tests/conftest.py`:

| Fixture | Scope | Purpose |
|---|---|---|
| `test_client` | session | `TestClient` with lifespan → real model loaded once |
| `trained_model` | session | `MovieRatingModel()`; skips with a clear message if not trained |
| `model_metrics` | session | hold-out metrics from `models/metrics.json` |
| `movielens_ratings` | session | the 100 000 real ratings (skips if the dataset is not downloaded) |
| `sample_*`, `invalid_prediction_requests`, `known_user_movie_pairs`, `unknown_users/movies` | function | small deterministic inputs |

Determinism: the model is trained with `random_state=42`, samples use `random.Random(seed)`, so the
suite gives identical results on every run and platform.

## 5. Quality gates in CI/CD

```
push / PR ─► Lint (pre-commit: black, isort, flake8, mypy, hygiene) ─┐
          ─► Type check (mypy, disallow-untyped-defs) ───────────────┤
                                                                     ▼
             Tests on Python 3.10 & 3.11: train ► model gate ► 162 tests ► coverage ≥ 80%
                                                                     ▼
             Docker: build multi-stage image ► smoke test ► HEALTHCHECK healthy

tag v* ─► CI (reused) ► build & push GHCR ► staging smoke test ► GitHub Release ► production

model/script/dependency change, weekly ─► Model Validation: 5-fold CV ► RMSE ≤ 0.95, MAE ≤ 0.75
```

| Gate | Threshold | Where |
|---|---|---|
| Formatting / lint | black, isort, flake8 clean | pre-commit hook + CI `lint` |
| Types | mypy, no untyped defs in `app/`, `scripts/` | pre-commit + CI `type-check` |
| Tests | 162/162 pass on 3.10 and 3.11 | CI `test` |
| Coverage | ≥ 80% (`--cov-fail-under=80`, `fail_under` in `pyproject.toml`) | CI `test` |
| Model quality | hold-out RMSE ≤ 0.95, MAE ≤ 0.75 | CI `test`, Model Validation workflow |
| Runtime | container healthy, endpoints respond | CI `docker`, CD `deploy-staging` |

## 6. Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/train_model.py            # once: downloads MovieLens 100K, trains, writes metrics
pytest tests/ --cov=app --cov-report=html  # full suite + HTML report in htmlcov/
pytest tests/unit/ -v                     # one layer
pytest -m integration                     # by marker
pre-commit run --all-files                # everything CI's lint job runs
```

## 7. Policies

* **New code ships with tests**; coverage may not drop below 80% (the build fails).
* **Flaky tests are bugs:** tests use fixed seeds and no wall-clock assumptions except the generous
  50 ms latency check; a flaky test is fixed or quarantined with an issue, never retried silently.
* **Tests must be able to fail:** every rule-based check has a negative test.
* **Model changes** go through the Model Validation workflow; thresholds change only in a reviewed PR.

## 8. Limitations and next steps

* Load testing (Locust/k6) against the container to validate throughput, not just single-request latency.
* Mutation testing (`mutmut`) to measure test strength beyond line coverage.
* Fairness/slice tests (e.g. error by user activity level) and drift tests on new rating logs.
* Contract tests for API consumers (schemathesis on the OpenAPI schema).
