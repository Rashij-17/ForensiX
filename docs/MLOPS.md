# MLOps lifecycle

| Concern | Where | Notes |
|---------|-------|-------|
| Dataset versioning | `forensix/mlops/versioning.py`, `artifacts/dataset_versions.json`, `dvc.yaml` | A new version (`v1`, `v2`, ...) is appended only when the content fingerprint changes; records sample counts per modality, preprocessing version, feature version, date, and verified production samples added |
| Experiment tracking | `forensix/mlops/tracking.py` | MLflow, local SQLite (`artifacts/mlflow.db`); one run per modality x model x cycle; logs params (modality, model, feature set, hyper-parameters, dataset version) and validation / test accuracy, precision, recall, F1, ROC AUC, CV F1 |
| Model registry | `forensix/mlops/registry.py`, `artifacts/registry.json` | Source of truth for names like *Image XGBoost v1* and their stage; mirrored into the MLflow Model Registry (best effort). Exactly one `production` version per modality |
| Monitoring | `forensix/mlops/monitoring.py` | Every analysis appends features + probability to `artifacts/monitoring/predictions.jsonl` (never the media). Input drift = PSI per feature vs the training baseline; also prediction / confidence distributions; with verified labels: production F1, precision, recall |
| Retraining trigger | `retrain_decision` | Drift share >= `drift_share_trigger` (30%) of features above PSI `psi_threshold` (0.2), **or** production F1 more than `f1_drop_trigger` (0.05) below baseline. Requires `min_samples` (30) logged predictions for drift and `min_labeled` (20) verified labels for performance, so it never fires on noise |
| Gated promotion | `run_retraining` | Trains all four models on train + verified samples, scores candidate **and** production model on the same test split, promotes only if candidate F1 > production F1 + `min_improvement`; otherwise candidates are marked `rejected` and production is unchanged. The previous production model is archived, never deleted |

## Typical loop
1. Users analyse media in the app; predictions are logged.
2. Analysts verify some results (*Verify this result* on the Analyze page → `/feedback`).
3. `make monitor` (or the MLOps page) shows drift and production performance.
4. `make retrain` runs the gated cycle; `scripts/registry_cli.py` or the MLOps page can approve versions manually.

## DVC
`dvc init && dvc add data/raw && dvc repro` tracks the raw data and re-runs only the stages whose inputs or parameters changed.
