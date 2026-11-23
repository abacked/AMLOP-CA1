"""Run from the repo root: python -m pytest -q tests/test_train_rz23.py.

Trains both versions on synthetic merged rows and logs to a temporary local
MLflow store, so no base CSVs or team MLflow server are required.
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlop_ca1 import train_rz23  # noqa: E402
from amlop_ca1.features import REQUIRED_COLUMNS  # noqa: E402

# Warnings raised inside MLflow itself, so tests still pass under pytest -W error.
pytestmark = [
    pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy is deprecated"),
    pytest.mark.filterwarnings("ignore:Operation with range is deprecated"),
    pytest.mark.filterwarnings("ignore:Hint.*Inferred schema contains integer column"),
    pytest.mark.filterwarnings("ignore:codecs.open\\(\\) is deprecated"),
    pytest.mark.filterwarnings("ignore:.*Any type hint is inferred as AnyType"),
]


@pytest.fixture
def merged():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "country_code": rng.choice(["MY", "PH", "BD"], n),
        "order_id": [f"{i:010d}" for i in range(n)],
        "order_value": rng.uniform(5, 200, n),
        "num_items_ordered": rng.integers(1, 6, n).astype(float),
        "collect_type": rng.choice(["delivery", "pickup"], n),
        "payment_method": rng.choice(["credit card", "cash", "unknown"], n),
        "order_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
        "refund_value": rng.uniform(0, 50, n),
    })
    df["is_fraud"] = ((df["country_code"] == "BD") & (rng.random(n) < 0.6)).astype("int8")
    return df


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(None)


def test_split_is_by_date_and_keeps_every_row(merged):
    train, valid, test, train_end, valid_end = train_rz23.split_by_date(merged)
    assert len(train) + len(valid) + len(test) == len(merged)
    assert train["order_date"].max() < train_end <= valid["order_date"].min()
    assert valid["order_date"].max() < valid_end <= test["order_date"].min()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_train_logs_run_and_model_reloads(merged, version):
    run_id = train_rz23.train(merged, version=version, batch="test", device="cpu")

    run = mlflow.get_run(run_id)
    assert run.info.run_name == f"test-r-z23-{version}"
    assert run.data.tags["contributor"] == "r-z23"
    for name in ["valid_precision", "valid_recall", "valid_pr_auc", "test_pr_auc"]:
        assert 0 <= run.data.metrics[name] <= 1

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    rows = merged[list(REQUIRED_COLUMNS)].head(10)
    assert model.predict_proba(rows).shape == (10, 2)


def test_only_v2_weights_fraud(merged):
    assert train_rz23.build_model("v1", merged["is_fraud"], "cpu")["model"].scale_pos_weight is None
    assert train_rz23.build_model("v2", merged["is_fraud"], "cpu")["model"].scale_pos_weight > 1


def test_model_ignores_label_and_refund(merged):
    model = train_rz23.build_model("v2", merged["is_fraud"], "cpu")
    model.fit(merged[list(REQUIRED_COLUMNS)], merged["is_fraud"])
    names = model["features"].get_feature_names_out()
    assert not any("is_fraud" in name or "refund" in name or "order_month" in name for name in names)


def test_missing_order_dates_are_left_out(merged):
    merged.loc[:9, "order_date"] = pd.NaT
    run_id = train_rz23.train(merged, version="v1", device="cpu")
    params = mlflow.get_run(run_id).data.params
    total = int(params["train_rows"]) + int(params["valid_rows"]) + int(params["test_rows"])
    assert total == len(merged) - 10


def test_v2_saved_model_predicts_with_logged_threshold(merged):
    run_id = train_rz23.train(merged, version="v2", device="cpu")
    threshold = float(mlflow.get_run(run_id).data.params["threshold"])
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    rows = merged[list(REQUIRED_COLUMNS)]
    prob = model.predict_proba(rows)[:, 1]
    expected = (prob >= threshold).astype(int)
    # The logged threshold is rounded to 4 places, so skip rows sitting right on it.
    close = np.isclose(prob, threshold, atol=1e-4)
    assert (model.predict(rows)[~close] == expected[~close]).all()


def test_failed_logging_marks_run_failed(merged, monkeypatch):
    def broken_log_model(*args, **kwargs):
        raise OSError("artifact store unavailable")

    monkeypatch.setattr(mlflow.sklearn, "log_model", broken_log_model)
    with pytest.raises(OSError):
        train_rz23.train(merged, version="v1", run_name="broken", device="cpu")
    run = mlflow.search_runs(experiment_names=[train_rz23.EXPERIMENT], filter_string="tags.mlflow.runName = 'broken'")
    assert run["status"].tolist() == ["FAILED"]
