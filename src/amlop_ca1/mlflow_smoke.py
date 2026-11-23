"""Check that a team MLflow server works end to end from a client machine.

Logs a run with a tiny model, reloads the model from the server, registers it,
loads it back through the registry, then deletes what it created. Run it after
every deploy, restart or config change:

    MLFLOW_TRACKING_URI=http://<server> PYTHONPATH=src python -m amlop_ca1.mlflow_smoke

Pass --keep to leave the run and model in place, for example to check after a
restart or redeploy that they survived (then run with --check-kept). Exits with
status 1 if any step fails.
"""

import argparse
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression


EXPERIMENT = "smoke-test"
MODEL_NAME = "smoke-test-model"
RUN_NAME = "smoke-test"


def tiny_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    return LogisticRegression().fit(X, y), X


def run_checks(keep=False):
    client = MlflowClient()
    print("tracking uri:", mlflow.get_tracking_uri())

    client.search_experiments(max_results=1)
    print("ok   server reachable")

    model, X = tiny_model()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("check", "smoke")
        mlflow.log_metric("accuracy", model.score(X, [0, 0, 1, 1]))
        info = mlflow.sklearn.log_model(model, name="model", input_example=X[:1])
    print("ok   run logged:", run.info.run_id)

    # A local path here means artifacts were written on this machine, not the server.
    artifact_uri = client.get_run(run.info.run_id).info.artifact_uri
    if not artifact_uri.startswith(("mlflow-artifacts:", "s3://", "http")):
        raise RuntimeError(f"artifacts are not stored on the server: {artifact_uri}")
    print("ok   artifacts on server:", artifact_uri)

    reloaded = mlflow.sklearn.load_model(info.model_uri)
    if not (reloaded.predict(X) == model.predict(X)).all():
        raise RuntimeError("reloaded model predicts differently")
    print("ok   model reloaded from run")

    version = mlflow.register_model(info.model_uri, MODEL_NAME).version
    registered = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    if not (registered.predict(X) == model.predict(X)).all():
        raise RuntimeError("registered model predicts differently")
    print("ok   model registered and loaded: version", version)

    if keep:
        print("kept run and model for --check-kept")
    else:
        client.delete_registered_model(MODEL_NAME)
        client.delete_run(run.info.run_id)
        print("ok   cleaned up")


def check_kept():
    client = MlflowClient()
    print("tracking uri:", mlflow.get_tracking_uri())
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string=f"tags.mlflow.runName = '{RUN_NAME}'")
    if runs.empty:
        raise RuntimeError("kept smoke-test run is gone")
    print("ok   kept run still listed:", len(runs))

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    if not versions:
        raise RuntimeError("kept registered model is gone")
    latest = max(versions, key=lambda v: int(v.version))
    mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    print("ok   kept model still loads: version", latest.version)

    client.delete_registered_model(MODEL_NAME)
    for run_id in runs["run_id"]:
        client.delete_run(run_id)
    print("ok   cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--check-kept", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_kept:
            check_kept()
        else:
            run_checks(keep=args.keep)
    except Exception as error:
        print("FAIL", type(error).__name__, error)
        sys.exit(1)
