.PHONY: help install demo-data demo pipeline ingest features eda stats train explain robustness generators api app start mlflow test lint format retrain monitor dvc clean docker-up docker-down zip

PY ?= python3
MOD ?= all

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:         ## Install the package and dependencies
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

demo-data:       ## Generate SYNTHETIC demo data (not DeepSafe) in data/raw
	$(PY) scripts/make_demo_data.py

demo: demo-data pipeline  ## Offline end-to-end run on synthetic data

dataset:         ## Download + arrange the DeepSafe benchmark from HuggingFace (needs internet)
	$(PY) scripts/download_dataset.py

pipeline:        ## Run all stages: ingest -> features -> eda -> stats -> train -> explain -> robustness -> generators
	$(PY) -m forensix.pipeline --modality $(MOD)

ingest:          ## Stage 1-2: scan, validate, deduplicate, split, version
	$(PY) -m forensix.pipeline ingest
features:        ## Stage 4: feature engineering
	$(PY) -m forensix.pipeline features --modality $(MOD)
eda:             ## Stage 3: exploratory analysis figures
	$(PY) -m forensix.pipeline eda --modality $(MOD)
stats:           ## Stage 5: statistical validation
	$(PY) -m forensix.pipeline stats --modality $(MOD)
train:           ## Stage 6: train + compare LR / SVM / RF / XGBoost, track in MLflow, register
	$(PY) -m forensix.pipeline train --modality $(MOD) summary
explain:         ## Stage 7: global SHAP + evaluation figures
	$(PY) -m forensix.pipeline explain --modality $(MOD)
robustness:      ## Stage 9: clean vs degraded F1
	$(PY) -m forensix.pipeline robustness --modality $(MOD)
generators:      ## Stage 10: per-generator consistency
	$(PY) -m forensix.pipeline generators --modality $(MOD)

api:             ## Start the FastAPI backend on :8000
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
app:             ## Start the Streamlit app on :8501 (uses the API if running, else in-process)
	$(PY) -m streamlit run frontend/app.py
start:           ## Start API and app together
	$(PY) -m uvicorn api.main:app --port 8000 & sleep 3 && $(PY) -m streamlit run frontend/app.py
mlflow:          ## Open the MLflow UI on :5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000

test:            ## Run the test suite
	$(PY) -m pytest
lint:            ## flake8 + black --check
	$(PY) -m flake8 forensix api frontend sdk scripts --max-line-length 140 --extend-ignore E203,W503,E731 && $(PY) -m black --check -l 120 forensix api frontend sdk scripts
format:          ## Format with black
	$(PY) -m black -l 120 forensix api frontend sdk scripts tests

monitor:         ## Check drift / degradation and report
	$(PY) scripts/monitor_and_retrain.py --check
retrain:         ## Retrain when triggered; promote only if better than production
	$(PY) scripts/monitor_and_retrain.py
dvc:             ## Reproduce the DVC pipeline
	dvc repro

docker-up:       ## Build and start API + app with docker compose
	docker compose up --build -d
docker-down:     ## Stop the containers
	docker compose down

clean:           ## Remove caches (keeps data and models)
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache *.egg-info

zip:             ## Package the project (without data/artifacts) as ForensiX.zip
	cd .. && zip -r ForensiX.zip ForensiX -x '*/__pycache__/*' '*/data/raw/*' '*/data/processed/*' '*/data/features/*' '*/artifacts/*' '*/reports/*' '*.egg-info/*'
