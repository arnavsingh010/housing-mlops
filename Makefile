.PHONY: install hooks lint format test train evaluate predict docker clean all

install:       ## Install package + dev tools
	pip install -e ".[dev]"

hooks:         ## Install git hooks (commit + push)
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint:          ## Lint + format check
	ruff check . && ruff format --check .

format:        ## Auto-fix style issues
	ruff check --fix . && ruff format .

test:          ## Run tests with coverage
	pytest --cov

train:         ## Train model (enforces quality gate)
	housing-mlops train

evaluate:      ## Re-evaluate saved model
	housing-mlops evaluate

predict:       ## Predict 5 sample rows
	housing-mlops predict --sample 5

docker:        ## Build docker image
	docker build -t housing-mlops .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml models/*.joblib models/*.json data/*.csv

all: lint test train
