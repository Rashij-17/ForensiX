# API reference

Interactive docs: `http://localhost:8000/docs`. Uploads are processed in memory / a temp directory and deleted.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/status` | Service health, available modalities, production models, dataset versions, demo flag |
| POST | `/analyze` | multipart `file` (+ `case_id`, `explain`, `note`) → probability, verdict, model results, SHAP |
| POST | `/evidence` | `file` → base64 PNG figures (FFT/DCT, noise, spectrogram/MFCC, frames, temporal) |
| POST | `/compare` | `file_a`, `file_b` (same modality) → scores, similarity metrics, figure |
| GET | `/lab/options/{modality}` | Adjustable degradations |
| POST | `/lab` | `file`, `kind`, optional `strengths` (comma list) → confidence curve (+ image preview) |
| POST | `/report` | `file` (+ `case_id`, `analyst`) → `{markdown, html}` forensic report |
| POST/GET | `/cases`, `/cases/{id}` | Create / list / read cases |
| PATCH | `/cases/{id}/status` | `open`, `under review`, `closed` |
| POST | `/cases/{id}/notes` | Add a note |
| GET | `/cases/{id}/report` | Case report |
| DELETE | `/cases/{id}` | Delete a case |
| GET | `/benchmarks`, `/benchmarks/stats/{modality}` | Model comparison, statistics, SHAP, robustness, generators |
| GET | `/registry` | All registered model versions |
| POST | `/registry/promote` | `{modality, algorithm, version}` → approve for production |
| GET | `/monitoring/{modality}` | Drift, prediction distribution, production metrics |
| GET | `/predictions` | Recent prediction log (no media) |
| POST | `/feedback` | `{record_id, label}` attach a verified label |
| GET | `/retrain/{modality}/check` | Retraining decision with reasons |
| POST | `/retrain/{modality}` | `{force}` run a gated retraining cycle |
| GET | `/experiments`, `/dataset/versions` | MLflow runs, dataset versions |

Errors: `422` invalid input (unsupported / corrupt / too short), `404` unknown case, `409` no trained model, `400` empty file.
