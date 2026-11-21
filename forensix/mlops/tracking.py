"""MLflow experiment tracking (local SQLite backend, no server required)."""

from __future__ import annotations

import contextlib

from ..utils import get_logger

log = get_logger()


def setup_mlflow(cfg):
    import mlflow

    uri = f"sqlite:///{(cfg.artifacts_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    return mlflow


@contextlib.contextmanager
def tracked_run(cfg, modality: str, run_name: str, params: dict, tags: dict | None = None):
    """Context manager yielding a ``log(metrics=..., artifacts=...)`` callable, or a no-op if MLflow is missing.

    Records modality, model, feature set, hyper-parameters and the accuracy / precision / recall / F1 / ROC AUC
    of every training run, as required by the experiment-tracking part of the lifecycle.
    """
    try:
        mlflow = setup_mlflow(cfg)
        exp_dir = cfg.artifacts_dir / "mlartifacts"
        exp_dir.mkdir(parents=True, exist_ok=True)
        name = f"forensix-{modality}"
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=exp_dir.as_uri())
        mlflow.set_experiment(name)
        run = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # pragma: no cover
        log.warning("MLflow unavailable (%s); continuing without tracking", exc)
        yield lambda **kw: None, None
        return
    try:
        mlflow.log_params({k: ("None" if v is None else v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)

        def _log(metrics: dict | None = None, artifacts: list[str] | None = None):
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            for a in artifacts or []:
                try:
                    mlflow.log_artifact(a)
                except Exception:
                    pass

        yield _log, run.info.run_id
    finally:
        mlflow.end_run()
