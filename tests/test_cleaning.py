"""
Tests for cleaning.py.

Unit tests use small hand-made frames with dirty values the base data does
not contain, to prove normalisation is generic. The batch test splits a small
fake batch (customers_test.csv, orders_test.csv, labels_test.csv) out of the
_base files into a temp folder and runs the full clean -> merge path on it.

Run from the repo root:  python -m pytest -q
Set SPBUY_DATA_DIR if the _base CSVs are not in ./data.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from amlop_ca1 import checks  # noqa: E402
from amlop_ca1 import cleaning as cl  # noqa: E402

DATA_DIR = Path(os.environ.get("SPBUY_DATA_DIR", REPO / "data"))


# ---------------------------------------------------------------------------
# Unit tests: unseen dirty values
# ---------------------------------------------------------------------------

def raw_customers():
    return pd.DataFrame({
        "country_code": [" ph", "MY", "my", "TH"],
        "customer_id": ["c1 ", "c2", "c3", "c4"],
        "mobile_verified": ["YES", True, "false", "maybe"],
        "num_orders_last_50days": [1, 2, 3, "n/a"],
        "num_cancelled_orders_last_50days": [0, 0, 0, 0],
        "num_refund_orders_last_50days": [0, 0, 0, 0],
        "total_payment_last_50days": [1.0, 2.0, 3.0, 4.0],
        "num_associated_customers": [1, 1, 1, 1],
        "first_order_datetime": ["2021-03-23 09:16:29", None, "2020/01/02 10:00:00", "not a date"],
    })


def raw_orders():
    return pd.DataFrame({
        "country_code": ["PH", "my", "MY", "TH", "TH"],
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "collect_type": ["Self-Pick-Up", "DELIVERY ", "pick_up", "Delivery", "drone"],
        "payment_method": ["Cash_On_Delivery", "ONLINE  BANKING", None, "on Delivery", "Credit Card"],
        "order_value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "num_items_ordered": [1, 2, None, 4, 5],
        "refund_value": [0.0, 0.0, 0.0, 0.0, 0.0],
        "order_date": ["2023/04/09", "2023-03-05", "2023/Feb/09", "2023-MAR-05", "05.03.2023"],
    })


def raw_labels():
    return pd.DataFrame({
        "country_code": ["ph", "MY", "MY", "TH", "TH", "TH"],
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o9"],
        "customer_id": ["c1", "c2", "c3", "c4", "c4", "c4"],
        "is_fraud": [1, "0", 0, 1, "x", 0],
    })


def test_clean_customers_normalises_unseen_values():
    out = cl.clean_customers(raw_customers())
    assert out["country_code"].tolist() == ["PH", "MY", "MY", "TH"]
    assert out["customer_id"].tolist() == ["c1", "c2", "c3", "c4"]
    assert out["mobile_verified"].tolist()[:3] == [True, True, False]
    assert pd.isna(out["mobile_verified"].iloc[3])  # "maybe" is not guessed
    assert pd.isna(out["num_orders_last_50days"].iloc[3])
    assert pd.api.types.is_datetime64_any_dtype(out["first_order_datetime"])
    assert out["first_order_datetime"].iloc[2] == pd.Timestamp("2020-01-02 10:00:00")
    assert out["first_order_missing"].tolist() == [0, 1, 0, 1]


def test_clean_orders_normalises_unseen_values():
    out = cl.clean_orders(raw_orders())
    # o3 is dropped: num_items_ordered is missing
    assert out["order_id"].tolist() == ["o1", "o2", "o4", "o5"]
    assert out["country_code"].tolist() == ["PH", "MY", "TH", "TH"]
    assert out["collect_type"].tolist() == ["pickup", "delivery", "delivery", "drone"]
    assert out["payment_method"].tolist() == [
        "payment on delivery", "online banking", "payment on delivery", "credit card",
    ]
    assert out["order_date"].iloc[:3].tolist() == [
        pd.Timestamp("2023-04-09"), pd.Timestamp("2023-03-05"), pd.Timestamp("2023-03-05"),
    ]
    assert pd.isna(out["order_date"].iloc[3])  # unknown format -> NaT, not a guess


def test_missing_payment_method_becomes_unknown():
    orders = raw_orders()
    orders.loc[2, "num_items_ordered"] = 3  # keep the row with the null payment method
    out = cl.clean_orders(orders)
    assert out.loc[out["order_id"] == "o3", "payment_method"].item() == "unknown"


def test_clean_labels_drops_invalid_targets():
    out = cl.clean_labels(raw_labels())
    assert "o5" not in out["order_id"].tolist()
    assert set(out["is_fraud"].unique()) <= {0, 1}
    assert out["is_fraud"].dtype == np.int8
    assert out["country_code"].iloc[0] == "PH"


def test_merge_uses_composite_keys_and_logs_losses():
    customers = cl.clean_customers(raw_customers())
    orders = cl.clean_orders(raw_orders())
    labels = cl.clean_labels(raw_labels())
    merged = cl.merge_all(customers, orders, labels)
    assert merged["order_id"].tolist() == ["o1", "o2", "o4"]
    stats = merged.attrs["join_stats"]
    assert stats["labels_without_order"] == 2  # o3 (dropped order) and o9 (never existed)
    assert stats["rows_out"] == 3


def test_same_order_id_in_two_countries_is_not_a_duplicate():
    orders = pd.concat([raw_orders().iloc[[0]], raw_orders().iloc[[0]].assign(country_code="MY")])
    out = cl.clean_orders(orders)
    assert len(out) == 2


def test_merge_rejects_duplicate_keys():
    labels = cl.clean_labels(raw_labels())
    with pytest.raises(ValueError, match="duplicate"):
        cl.merge_all(cl.clean_customers(raw_customers()), cl.clean_orders(raw_orders()),
                     pd.concat([labels, labels]))


def test_cleaners_do_not_mutate_input():
    raw = raw_orders()
    snapshot = raw.copy()
    cl.clean_orders(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_missing_column_is_a_clear_error():
    with pytest.raises(ValueError, match="payment_method"):
        cl.clean_orders(raw_orders().drop(columns="payment_method"))


def test_add_log_features():
    df = pd.DataFrame({"num_items_ordered": [0, 9, 1600]})
    out = cl.add_log_features(df)
    np.testing.assert_allclose(out["log_num_items_ordered"], np.log1p([0, 9, 1600]))
    assert "num_items_ordered" in out  # original kept


# ---------------------------------------------------------------------------
# Batch test: a small fake batch split out of _base
# ---------------------------------------------------------------------------

def load_batch(folder, suffix):
    """Read customers_<suffix>.csv, orders_<suffix>.csv and labels_<suffix>.csv."""
    customers = pd.read_csv(folder / f"customers_{suffix}.csv", dtype=cl.CSV_DTYPES)
    orders = pd.read_csv(folder / f"orders_{suffix}.csv", dtype=cl.CSV_DTYPES)
    labels = pd.read_csv(folder / f"labels_{suffix}.csv", dtype=cl.CSV_DTYPES)
    return customers, orders, labels


def composite_key(df, id_column):
    """e.g. "MY|431b5e1fcd". Uppercased so "my" rows still match."""
    return df["country_code"].str.upper() + "|" + df[id_column]


@pytest.fixture(scope="module")
def fake_batch_dir(tmp_path_factory):
    """Write customers_test.csv / orders_test.csv / labels_test.csv to a temp dir."""
    if not (DATA_DIR / "labels_base.csv").exists():
        pytest.skip(f"_base CSVs not found in {DATA_DIR}")

    customers, orders, labels = load_batch(DATA_DIR, "base")

    # Take 5,000 random labels, then keep only their orders and customers.
    # The rows are not cleaned, so lowercase country codes etc. come along too.
    labels = labels.sample(5000, random_state=0)
    wanted_orders = composite_key(labels, "order_id")
    wanted_customers = composite_key(labels, "customer_id")
    orders = orders[composite_key(orders, "order_id").isin(wanted_orders)]
    customers = customers[composite_key(customers, "customer_id").isin(wanted_customers)]

    # Inject dirty values the base data never contains.
    orders = orders.copy()
    orders.iloc[:3, orders.columns.get_loc("collect_type")] = ["SELF-PICK-UP", " Pickup ", "DELIVERY"]
    orders.iloc[:2, orders.columns.get_loc("payment_method")] = ["Payment_On_Delivery", "PAYPAL"]
    orders.iloc[:2, orders.columns.get_loc("order_date")] = ["2023-Jan-05", "2023/MAY/01"]

    out = tmp_path_factory.mktemp("batch")
    customers.to_csv(out / "customers_test.csv", index=False)
    orders.to_csv(out / "orders_test.csv", index=False)
    labels.to_csv(out / "labels_test.csv", index=False)
    return out


def test_fake_batch_end_to_end(fake_batch_dir):
    customers, orders, labels = load_batch(fake_batch_dir, "test")

    clean_customers = cl.clean_customers(customers)
    clean_orders = cl.clean_orders(orders)
    clean_labels = cl.clean_labels(labels)
    merged = cl.merge_all(clean_customers, clean_orders, clean_labels)

    assert len(merged) > 0.95 * len(labels)
    assert not merged.duplicated(cl.ORDER_KEY).any()
    assert set(merged["collect_type"].unique()) <= cl.KNOWN_COLLECT_TYPES
    assert set(merged["payment_method"].unique()) <= cl.KNOWN_PAYMENT_METHODS
    assert merged["order_date"].notna().all()
    assert set(merged["country_code"].unique()) <= cl.KNOWN_COUNTRIES

    # Every fixable issue is zero after cleaning.
    issues = checks.check_issues(clean_customers, clean_orders, clean_labels)
    counts = issues.set_index(["table", "column", "issue"])["count"]
    fixable = [
        ("customers", "country_code", "not uppercase"),
        ("labels", "country_code", "not uppercase"),
        ("customers", "mobile_verified", "not True/False"),
        ("orders", "payment_method", "missing"),
        ("orders", "num_items_ordered", "missing"),
        ("orders", "collect_type", "minority spellings"),
        ("orders", "payment_method", "minority spellings"),
        ("orders", "order_date", "not YYYY/MM/DD text or datetime"),
        ("orders", "order_date", "unparseable"),
    ]
    for issue in fixable:
        assert counts[issue] == 0, f"still found after cleaning: {issue}"

    # Missing first_order_datetime is flagged, never dropped.
    assert merged["first_order_missing"].sum() == merged["first_order_datetime"].isna().sum()
