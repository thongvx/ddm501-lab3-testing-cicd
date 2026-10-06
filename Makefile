.PHONY: install train test test-unit test-integration test-data test-model coverage lint format typecheck precommit docker-build docker-run validate

install:          ## install runtime + dev dependencies and git hooks
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

train:            ## train the SVD model (writes models/svd_model.pkl + metrics.json)
	python scripts/train_model.py

validate:         ## model quality gate
	python scripts/validate_model.py

test:             ## full suite with coverage gate
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html --cov-fail-under=80

test-unit:
	pytest tests/unit/ -v
test-integration:
	pytest tests/integration/ -v
test-data:
	pytest tests/data/ -v
test-model:
	pytest tests/model/ -v

lint:             ## flake8 + formatting checks
	flake8 app/ tests/ scripts/
	black --check app/ tests/ scripts/
	isort --check-only app/ tests/ scripts/

format:
	black app/ tests/ scripts/
	isort app/ tests/ scripts/

typecheck:
	mypy app/ scripts/

precommit:
	pre-commit run --all-files

docker-build:
	docker build -t movie-rating-api:local .

docker-run:
	docker run --rm -p 8000:8000 movie-rating-api:local
