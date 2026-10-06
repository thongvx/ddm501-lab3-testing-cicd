# Test Coverage Report

**Line coverage of `app/`: 100%** (162 tests passed) · CI gate: `--cov-fail-under=80` and
`fail_under = 80` in `pyproject.toml`.

| Name                |    Stmts |     Miss |    Cover |   Missing |
|-------------------- | -------: | -------: | -------: | --------: |
| app/\_\_init\_\_.py |        1 |        0 |     100% |           |
| app/config.py       |       13 |        0 |     100% |           |
| app/main.py         |       56 |        0 |     100% |           |
| app/model.py        |       42 |        0 |     100% |           |
| app/schemas.py      |       30 |        0 |     100% |           |
| **TOTAL**           |  **142** |    **0** | **100%** |           |

Generated with:

```bash
python scripts/train_model.py --skip-cv
pytest tests/ --cov=app --cov-report=term-missing --cov-report=html --cov-fail-under=80
```

* Every CI run (Python 3.10 and 3.11) prints this table in the job summary and uploads the full
  **HTML report** (`htmlcov/`), `coverage.xml` and `junit.xml` as the artifact
  **`coverage-report`** (Actions → CI Pipeline → run → *Artifacts*).
* Tests per layer: unit 63 · integration 44 · data 32 · model behaviour 23.
* The only excluded line is `if __name__ == "__main__":` in `app/main.py` (local dev server entry point).
