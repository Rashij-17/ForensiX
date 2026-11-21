# Trained models

Versioned model folders are written here by `make train` (e.g. `models/image/xgboost-v1/`):
`model.joblib`, `meta.json`, `background.csv`, `reference_stats.json`. They are excluded from git;
the registry (`artifacts/registry.json`) records which version is approved for production.
