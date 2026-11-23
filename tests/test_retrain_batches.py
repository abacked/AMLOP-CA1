"""Run from the repo root: python -m pytest -q tests/test_retrain_batches.py.

Simulates a future batch from the base CSVs: a sample of orders is split by date
into an "old" and a "new" batch, each written as its own three files. Each batch
is cleaned on its own, as the ETL will, then both are combined to retrain. Uses
only the supplied base data; skipped when the base CSVs are not available.
"""

import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amlop_ca1 import cleaning, train_rz23  # noqa: E402

DATA_DIR = Path(os.environ.get("SPBUY_DATA_DIR", REPO / "data"))
SAMPLE_LABELS = 30000
NEW_BATCH_START = "2023-04-01"

# Warnings raised inside MLflow itself, so tests still pass under pytest -W error.
pytestmark = [
    pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy is deprecated"),
    pytest.mark.filterwarnings("ignore:Operation with range is deprecated"),
    pytest.mark.filterwarnings("ignore:Hint.*Inferred schema contains integer column"),
    pytest.mark.filterwarnings("ignore:codecs.open\\(\\) is deprecated"),
    pytest.mark.filterwarnings("ignore:.*Any type hint is inferred as AnyType"),
]


@pytest.fixture(scope="module")
def batch_dir(tmp_path_factory):
    if not (DATA_DIR / "labels_base.csv").exists():
        pytest.skip(f"_base CSVs not found in {DATA_DIR}")
    logging.getLogger("amlop_ca1").setLevel(logging.ERROR)
    customers, orders, labels = cleaning.load_batch(DATA_DIR, "base")
    labels = labels.sample(SAMPLE_LABELS, random_state=0)

    order_key = orders["country_code"].str.upper() + "|" + orders["order_id"]
    label_key = labels["country_code"].str.upper() + "|" + labels["order_id"]
    orders = orders[order_key.isin(label_key)]
    order_key = order_key[orders.index]
    dates = pd.to_datetime(orders["order_date"], format="mixed", errors="coerce")

    out = tmp_path_factory.mktemp("batches")
    for batch, in_batch in [("old", dates < NEW_BATCH_START), ("new", dates >= NEW_BATCH_START)]:
        batch_orders = orders[in_batch]
        batch_labels = labels[label_key.isin(order_key[in_batch])]
        customer_key = customers["country_code"].str.upper() + "|" + customers["customer_id"]
        wanted = batch_labels["country_code"].str.upper() + "|" + batch_labels["customer_id"]
        batch_orders.to_csv(out / f"orders_{batch}.csv", index=False)
        batch_labels.to_csv(out / f"labels_{batch}.csv", index=False)
        customers[customer_key.isin(wanted)].to_csv(out / f"customers_{batch}.csv", index=False)
    return out


@pytest.fixture(scope="module")
def cleaned(batch_dir):
    return {batch: cleaning.clean_and_merge(*cleaning.load_batch(batch_dir, batch)) for batch in ["old", "new"]}


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(None)


def test_batches_do_not_overlap(cleaned):
    old, new = cleaned["old"], cleaned["new"]
    assert len(old) > 0 and len(new) > 0
    assert old["order_date"].max() < new["order_date"].min()
    combined = pd.concat([old, new])
    assert not combined.duplicated(cleaning.ORDER_KEY).any()


def test_retraining_scores_on_the_new_batch(cleaned):
    combined = pd.concat([cleaned["old"], cleaned["new"]], ignore_index=True)
    train, valid, test, train_end, valid_end = train_rz23.split_by_date(combined)
    assert test["order_date"].min() >= cleaned["new"]["order_date"].min()

    old_only = train_rz23.split_by_date(cleaned["old"])
    assert valid_end > old_only[4]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_retrain_logs_named_run_for_new_batch(cleaned, version):
    combined = pd.concat([cleaned["old"], cleaned["new"]], ignore_index=True)
    run_id = train_rz23.train(combined, version=version, batch="new", device="cpu")

    run = mlflow.get_run(run_id)
    assert run.info.run_name == f"new-r-z23-{version}"
    assert run.data.tags["batch"] == "new"
    rows = int(run.data.params["train_rows"]) + int(run.data.params["valid_rows"]) + int(run.data.params["test_rows"])
    assert rows == len(combined)
    assert 0 < run.data.metrics["test_pr_auc"] <= 1
