# Contributing

1. Fork, create a branch, `make install`.
2. Make your change with tests (`make test`) and keep `make lint` clean (`make format` applies black).
3. Open a pull request describing what changed and why.

**Purpose limitation.** ForensiX is a defensive detector. Contributions that help *create* deepfakes, evade
detection, or weaken the honesty of reported results (for example removing limitations from reports) will not be accepted.

## Adding a feature
Add the extractor in `forensix/features/<modality>.py`, add a human-readable entry in `forensix/features/catalog.py`
(a test fails if one is missing), bump `versions.feature` in `config/params.yaml`, then `make features train`.

## Adding a model
Register a constructor in `forensix/models/train.py::make_model`, add its name to `MODEL_NAMES` / `MODEL_LABELS` in
`forensix/config.py` and hyper-parameters under `models:` in `config/params.yaml`. It is then trained, tracked in MLflow,
registered and compared automatically.

## Adding a transformation
Add the function to `forensix/robustness/transforms.py`, expose it in the `*_transforms` registry and in
`forensix/lab.py::lab_options` so it shows up in the benchmark and the Forensic Lab.
