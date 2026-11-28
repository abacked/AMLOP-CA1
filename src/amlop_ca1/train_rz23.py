"""Train r-z23's fraud models and log them to MLflow.

Both versions use XGBoost on the order features from features.py:

- v1: baseline with default settings, no class weighting and a 0.5 threshold.
- v2: weights the fraud class, uses shallower trees and picks the threshold with
  the best validation F1, which the saved model's predict() uses.

order_month was dropped from features.py after the v1 diagnostics in
notebooks/model_rz23.ipynb showed it does not help on unseen months.

The ETL calls train() with the cleaned, merged table from
cleaning.clean_and_merge; run this file directly to train on the base files:

    PYTHONPATH=src python -m amlop_ca1.train_rz23 --version v2

Rows are split by order date (70% train, 15% validation, 15% test) so models
are always scored on later orders than they were trained on. Compare models on
the validation metrics; test metrics are reported but never used for tuning.

Set MLFLOW_TRACKING_URI to log to the team server, otherwise MLflow logs
locally. MLFLOW_EXPERIMENT_NAME picks the experiment (default amlop-ca1-fraud,
as in .env.example). XGBoost uses CUDA when an NVIDIA GPU is found, or pass
--device.
"""

import argparse
import os
import shutil

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from amlop_ca1 import cleaning
from amlop_ca1.features import REQUIRED_COLUMNS, make_feature_pipeline


CONTRIBUTOR = "r-z23"
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "amlop-ca1-fraud")
TARGET = "is_fraud"
PARAMS = {
    "v1": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    "v2": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
}


def pick_device(device="auto"):
    if device == "auto":
        return "cuda" if shutil.which("nvidia-smi") else "cpu"
    return device


def split_by_date(df, train_share=0.7, valid_share=0.15):
    # Cut on whole days so one day's orders never land in two sets.
    dates = df["order_date"]
    train_end = dates.quantile(train_share).normalize()
    valid_end = dates.quantile(train_share + valid_share).normalize()
    train = df[dates < train_end]
    valid = df[(dates >= train_end) & (dates < valid_end)]
    test = df[dates >= valid_end]
    return train, valid, test, train_end, valid_end


def build_model(version, y_train, device):
    features = make_feature_pipeline()
    params = dict(PARAMS[version])
    if version == "v2":
        params["scale_pos_weight"] = (y_train == 0).sum() / y_train.sum()
    model = XGBClassifier(**params, tree_method="hist", device=device, random_state=42)
    return Pipeline([("features", features), ("model", model)])


def best_threshold(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def scores(y, prob, threshold, prefix):
    pred = (prob >= threshold).astype(int)
    return {
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_pr_auc": average_precision_score(y, prob),
        f"{prefix}_roc_auc": roc_auc_score(y, prob),
    }


def train(df, version="v2", run_name=None, batch="base", device="auto"):
    """Fit on the cleaned, merged table, log one MLflow run and return its run id.

    Rows without an order date cannot be placed in time and are left out.
    """
    df = df.dropna(subset=["order_date"])
    train_df, valid_df, test_df, train_end, valid_end = split_by_date(df)
    columns = list(REQUIRED_COLUMNS)
    device = pick_device(device)

    model = build_model(version, train_df[TARGET], device)
    model.fit(train_df[columns], train_df[TARGET])
    valid_prob = model.predict_proba(valid_df[columns])[:, 1]
    test_prob = model.predict_proba(test_df[columns])[:, 1]
    scale_pos_weight = model["model"].scale_pos_weight
    threshold = 0.5
    if version == "v2":
        threshold = best_threshold(valid_df[TARGET], valid_prob)
        # Save the threshold inside the model so predict() matches the logged metrics.
        model = FixedThresholdClassifier(FrozenEstimator(model), threshold=threshold)
        model.fit(valid_df[columns], valid_df[TARGET])

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name or f"{batch}-{CONTRIBUTOR}-{version}") as run:
        mlflow.set_tags({"contributor": CONTRIBUTOR, "version": version, "batch": batch})
        mlflow.log_params(PARAMS[version])
        mlflow.log_params({
            "model": "XGBClassifier",
            "device": device,
            "scale_pos_weight": scale_pos_weight,
            "threshold": round(threshold, 4),
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
        })
        mlflow.log_metrics(scores(valid_df[TARGET], valid_prob, threshold, "valid"))
        mlflow.log_metrics(scores(test_df[TARGET], test_prob, threshold, "test"))
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle",
            input_example=train_df[columns].head(5),
        )
    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch", default="base")
    parser.add_argument("--version", default="v2", choices=list(PARAMS))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    customers, orders, labels = cleaning.load_batch(args.data_dir, args.batch)
    df = cleaning.clean_and_merge(customers, orders, labels)
    print("run id:", train(df, version=args.version, batch=args.batch, device=args.device))
