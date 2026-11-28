"""Run from the repo root: python -m pytest -q tests/test_compare_models.py.

Logs small v1/v2 runs to a temporary local MLflow store, then checks the
ranking.
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlop_ca1 import compare_models, train_rz23  # noqa: E402

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


def test_ranks_runs_from_one_batch_only(merged):
    train_rz23.train(merged, "v1", batch="base", device="cpu")
    train_rz23.train(merged, "v2", batch="base", device="cpu")
    train_rz23.train(merged, "v2", batch="other", device="cpu")

    table = compare_models.compare(compare_models.load_runs("base"))

    assert len(table) == 2
    assert set(table["run_name"]) == {"base-r-z23-v1", "base-r-z23-v2"}
    assert table["valid_pr_auc"].is_monotonic_decreasing


def test_missing_batch_raises():
    mlflow.set_experiment(compare_models.EXPERIMENT)
    with pytest.raises(RuntimeError, match="no finished runs"):
        compare_models.load_runs("nope")


def test_other_contributors_ranked_and_unscored_runs_dropped(merged):
    train_rz23.train(merged, "v1", batch="base", device="cpu")
    mlflow.set_experiment(compare_models.EXPERIMENT)
    with mlflow.start_run(run_name="base-abacked-v1"):
        mlflow.set_tags({"contributor": "abacked", "version": "v1", "batch": "base"})
        mlflow.log_metrics({"valid_pr_auc": 0.99, "valid_f1": 0.9})
    with mlflow.start_run(run_name="base-abacked-v2"):
        mlflow.set_tags({"contributor": "abacked", "version": "v2", "batch": "base"})

    table = compare_models.compare(compare_models.load_runs("base"))

    assert list(table["run_name"]) == ["base-abacked-v1", "base-r-z23-v1"]
    assert table["contributor"][0] == "abacked"
