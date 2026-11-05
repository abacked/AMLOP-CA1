"""
One data issue at a time: before and after cleaning.

Each test starts from the same clean sample of the base data (2,000 labels and
their orders and customers, with every row that has a known issue left out), so
cleaning it changes nothing and rejects nothing. It then adds exactly ONE issue,
prints what the affected table looks like before cleaning, runs cleaning.py,
prints it again, and checks that the issue is gone from the merged table.

Nothing prints the rows themselves; the profile shows rows, columns, dtypes,
nulls, distinct values and the formats of the values (e.g. "9999/99/99").

Run from the repo root and show the printed reports:
    python -m pytest tests/test_issues.py -s
One issue only, e.g.:
    python -m pytest tests/test_issues.py -s -k conflicting_is_fraud
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from amlop_ca1 import checks  # noqa: E402
from amlop_ca1 import cleaning as cl  # noqa: E402

DATA_DIR = Path(os.environ.get("SPBUY_DATA_DIR", REPO / "data"))
SAMPLE_LABELS = 2000
N = 5  # rows given the issue in each test

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)


# ---------------------------------------------------------------------------
# The clean starting sample
# ---------------------------------------------------------------------------

def is_main_spelling(column, aliases):
    """true where the value is the most common spelling of what it means."""
    counts = column.value_counts()
    meaning = cl.normalise_text(pd.Series(counts.index)).replace(aliases)
    main = counts.groupby(meaning.to_numpy()).idxmax()
    return column.isin(main.to_numpy())


@pytest.fixture(scope="module")
def clean_sample():
    """raw base rows with none of the known issues, as loaded by load_csv."""
    if not (DATA_DIR / "labels_base.csv").exists():
        pytest.skip(f"_base CSVs not found in {DATA_DIR}")
    customers, orders, labels = cl.load_batch(DATA_DIR, "base")

    customers = customers[
        (customers["country_code"] == customers["country_code"].str.upper())
        & customers["mobile_verified"].isin(["True", "False"])
        & customers["first_order_datetime"].notna()
    ]
    orders = orders[
        orders["payment_method"].notna()
        & orders["num_items_ordered"].notna()
        & orders["order_date"].str.fullmatch(r"\d{4}/\d{2}/\d{2}")
        & is_main_spelling(orders["collect_type"], cl.COLLECT_TYPE_ALIASES)
        & is_main_spelling(orders["payment_method"], cl.PAYMENT_METHOD_ALIASES)
    ]
    labels = labels[labels["country_code"] == labels["country_code"].str.upper()]

    # labels whose order and customer are both clean, then only their orders and customers.
    labels = labels[cl.key_isin(labels, orders, cl.ORDER_KEY) & cl.key_isin(labels, customers, cl.CUSTOMER_KEY)]
    labels = labels.sample(SAMPLE_LABELS, random_state=0)
    orders = orders[cl.key_isin(orders, labels, cl.ORDER_KEY)]
    customers = customers[cl.key_isin(customers, labels, cl.CUSTOMER_KEY)]
    return tuple(table.reset_index(drop=True) for table in (customers, orders, labels))


# ---------------------------------------------------------------------------
# Running one case and printing the report
# ---------------------------------------------------------------------------

def run_cleaning(tables, folder, reader):
    """write the raw tables as a batch, read them back like a new batch, clean and merge."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in zip(("customers", "orders", "labels"), tables):
        table.to_csv(folder / f"{name}_issue.csv", index=False)
    if reader == "load_csv":
        raw = cl.load_batch(folder, "issue")
    else:  # someone reading the files without cleaning.load_csv
        raw = tuple(pd.read_csv(folder / f"{name}_issue.csv") for name in ("customers", "orders", "labels"))

    rejects = cl.RejectLog()
    cleaned = (cl.clean_customers(raw[0], rejects), cl.clean_orders(raw[1], rejects), cl.clean_labels(raw[2], rejects))
    merged = cl.merge_all(*cleaned, rejects)
    return raw, cleaned, merged, rejects


def print_profile(title, table):
    print(f"\n  {title}: {len(table):,} rows x {table.shape[1]} columns")
    print("    " + checks.profile_table(table).to_string(index=False).replace("\n", "\n    "))


TABLE_INDEX = {"customers": 0, "orders": 1, "labels": 2}


def report(case, raw, cleaned, merged, rejects, before, after, baseline_rows, messages):
    table = case["table"]
    print(f"\n{'=' * 100}\nISSUE: {case['id']} - {case['what']}\n{'=' * 100}")
    print_profile(f"BEFORE  raw {table}", raw[TABLE_INDEX[table]])
    print(f"\n  issue count before: {before}")
    print("\n  what cleaning logged:")
    for message in messages or ["(nothing)"]:
        print(f"    {message}")
    print_profile(f"AFTER   cleaned {table}", cleaned[TABLE_INDEX[table]])
    print(f"\n  issue count after (in the merged table): {after}")
    print(f"  merged rows: {baseline_rows:,} without the issue -> {len(merged):,} with it")
    summary = rejects.summary()
    print("  rejected rows:" + (" none" if summary.empty else "\n    " + summary.to_string(index=False).replace("\n", "\n    ")))


# ---------------------------------------------------------------------------
# The issues. Each one: change N rows (inject), count the issue in the raw
# tables (before) and in the merged table (after).
#   inject(customers, orders, labels) -> (customers, orders, labels, info)
#   info["lost"] = labels that can no longer be used, so merged shrinks by that.
#   info["keys"] = the order_ids (or customer_ids) that were given the issue.
# ---------------------------------------------------------------------------

def first(table, n=N):
    return table.index[:n]


def labels_of_customers(labels, customers):
    return int(cl.key_isin(labels, customers, cl.CUSTOMER_KEY).sum())


def inject_lowercase_country(c, o, l):
    l.loc[first(l), "country_code"] = l.loc[first(l), "country_code"].str.lower()
    return c, o, l, {"lost": 0}


def inject_padded_ids(c, o, l):
    o.loc[first(o), "order_id"] = "  " + o.loc[first(o), "order_id"] + " "
    return c, o, l, {"lost": 0}


def inject_mobile_yes(c, o, l):
    rows = c.index[c["mobile_verified"] == "True"][:N]
    c.loc[rows, "mobile_verified"] = ["yes", "YES", "Y", " true ", "1"]
    return c, o, l, {"lost": 0, "keys": c.loc[rows, "customer_id"].tolist()}


def inject_collect_type_spelling(c, o, l):
    o.loc[first(o), "collect_type"] = ["SELF-PICK-UP", " Pick_Up", "self pickup", "DELIVERY ", "delivery"]
    return c, o, l, {"lost": 0}


def inject_payment_spelling(c, o, l):
    o.loc[first(o), "payment_method"] = ["Cash_On_Delivery", "ON-DELIVERY", " Credit  Card", "PAYPAL", "online_banking"]
    return c, o, l, {"lost": 0}


def inject_missing_payment(c, o, l):
    o.loc[first(o), "payment_method"] = None
    return c, o, l, {"lost": 0, "keys": o.loc[first(o), "order_id"].tolist()}


def inject_date_formats(c, o, l):
    dates = pd.to_datetime(o.loc[first(o), "order_date"], format="%Y/%m/%d")
    o.loc[first(o), "order_date"] = [
        dates.iloc[0].strftime("%Y-%m-%d"), dates.iloc[1].strftime("%Y/%b/%d"),
        dates.iloc[2].strftime("%Y-%b-%d").upper(), dates.iloc[3].strftime("%Y/%B/%d"),
        " " + dates.iloc[4].strftime("%Y/%m/%d") + " ",
    ]
    expected = dict(zip(o.loc[first(o), "order_id"], dates))
    return c, o, l, {"lost": 0, "expected_dates": expected}


def inject_unknown_date_format(c, o, l):
    dates = pd.to_datetime(o.loc[first(o), "order_date"], format="%Y/%m/%d")
    o.loc[first(o), "order_date"] = [
        dates.iloc[0].strftime("%d.%m.%Y"), dates.iloc[1].strftime("%m/%d/%Y"),
        "2023-13-45", "yesterday", dates.iloc[4].strftime("%d %B %Y"),
    ]
    return c, o, l, {"lost": 0, "keys": o.loc[first(o), "order_id"].tolist()}


def inject_null_order_id(c, o, l):
    l.loc[first(l), "order_id"] = None
    return c, o, l, {"lost": N, "reason": "missing order_id"}


def inject_blank_customer_id(c, o, l):
    lost = labels_of_customers(l, c.loc[first(c)])
    c.loc[first(c), "customer_id"] = ["", " ", "   ", "\t", "  "]
    return c, o, l, {"lost": lost, "reason": "missing customer_id"}


def inject_exact_duplicate(c, o, l):
    l = pd.concat([l, l.loc[first(l)]], ignore_index=True)
    return c, o, l, {"lost": 0, "reason": "exact duplicate"}


def inject_conflicting_is_fraud(c, o, l):
    flipped = l.loc[first(l)].assign(is_fraud=1 - l.loc[first(l), "is_fraud"])
    keys = l.loc[first(l), "order_id"].tolist()
    l = pd.concat([l, flipped], ignore_index=True)
    return c, o, l, {"lost": N, "keys": keys, "reason": "conflicting duplicate: is_fraud"}


def inject_conflicting_customer(c, o, l):
    changed = c.loc[first(c)].assign(num_orders_last_50days=c.loc[first(c), "num_orders_last_50days"] + 1)
    lost = labels_of_customers(l, c.loc[first(c)])
    keys = c.loc[first(c), "customer_id"].tolist()
    c = pd.concat([c, changed], ignore_index=True)
    return c, o, l, {"lost": lost, "keys": keys, "reason": "conflicting duplicate: num_orders_last_50days"}


def inject_orphan_labels(c, o, l):
    orphans = l.loc[first(l)].assign(order_id=[f"zz{i:08d}" for i in range(N)])
    l = pd.concat([l, orphans], ignore_index=True)
    return c, o, l, {"lost": 0, "keys": orphans["order_id"].tolist(), "reason": "no matching order"}


def inject_invalid_is_fraud(c, o, l):
    l["is_fraud"] = l["is_fraud"].astype(object)
    l.loc[first(l), "is_fraud"] = ["x", "2", "-1", "yes", None]
    return c, o, l, {"lost": N, "keys": l.loc[first(l), "order_id"].tolist(), "reason": "is_fraud not 0 or 1"}


def inject_missing_num_items(c, o, l):
    o.loc[first(o), "num_items_ordered"] = None
    return c, o, l, {"lost": N, "keys": o.loc[first(o), "order_id"].tolist(),
                     "reason": "missing order_value, num_items_ordered"}


def inject_non_numeric_order_value(c, o, l):
    o["order_value"] = o["order_value"].astype(object)
    o.loc[first(o), "order_value"] = ["abc", "12,50", "$5", "n/a", "--"]
    return c, o, l, {"lost": N, "keys": o.loc[first(o), "order_id"].tolist(),
                     "reason": "missing order_value, num_items_ordered"}


def inject_numeric_ids(c, o, l):
    # every order_id becomes all digits, many with leading zeros ("0000000042").
    # read with a plain pd.read_csv, they turn into numbers and lose the zeros.
    keys = o[cl.ORDER_KEY].drop_duplicates()
    new_ids = dict(zip(map(tuple, keys.to_numpy()), (f"{i * 37:010d}" for i in range(1, len(keys) + 1))))
    for table in (o, l):
        table["order_id"] = [new_ids[key] for key in map(tuple, table[cl.ORDER_KEY].to_numpy())]
    return c, o, l, {"lost": 0, "expected_ids": set(new_ids.values())}


def lost_zeros(ids):
    return int((ids.astype("string").str.len() != 10).sum())


ISSUES = [
    dict(id="lowercase_country", table="labels", inject=inject_lowercase_country,
         what="country_code in lowercase, so labels don't match their orders",
         before=lambda c, o, l: checks.issue_count(l["country_code"] != l["country_code"].str.upper()),
         after=lambda m, info: checks.issue_count(m["country_code"] != m["country_code"].str.upper())),
    dict(id="padded_ids", table="orders", inject=inject_padded_ids,
         what="spaces around order_id, so orders don't match their labels",
         before=lambda c, o, l: checks.issue_count(o["order_id"] != o["order_id"].str.strip()),
         after=lambda m, info: checks.issue_count(m["order_id"] != m["order_id"].str.strip())),
    dict(id="mobile_verified_text", table="customers", inject=inject_mobile_yes,
         what='mobile_verified written as "yes", "Y", "1" instead of True/False',
         before=lambda c, o, l: checks.issue_count(~c["mobile_verified"].isin(["True", "False"])),
         after=lambda m, info: checks.issue_count(m.loc[m["customer_id"].isin(info["keys"]), "mobile_verified"] != True)),  # noqa: E712
    dict(id="collect_type_spelling", table="orders", inject=inject_collect_type_spelling,
         what="collect_type spelled in different ways",
         before=lambda c, o, l: checks.count_minority_spellings(o["collect_type"], cl.COLLECT_TYPE_ALIASES),
         after=lambda m, info: checks.issue_count(~m["collect_type"].isin(cl.KNOWN_COLLECT_TYPES))),
    dict(id="payment_method_spelling", table="orders", inject=inject_payment_spelling,
         what="payment_method spelled in different ways",
         before=lambda c, o, l: checks.count_minority_spellings(o["payment_method"], cl.PAYMENT_METHOD_ALIASES),
         after=lambda m, info: checks.issue_count(~m["payment_method"].isin(cl.KNOWN_PAYMENT_METHODS))),
    dict(id="missing_payment_method", table="orders", inject=inject_missing_payment,
         what="payment_method missing",
         before=lambda c, o, l: o["payment_method"].isna().sum(),
         after=lambda m, info: checks.issue_count(m.loc[m["order_id"].isin(info["keys"]), "payment_method"] != "unknown")),
    dict(id="date_formats", table="orders", inject=inject_date_formats,
         what="order_date in other known formats (2023-04-09, 2023/Apr/09, 2023-APR-09, ...)",
         before=lambda c, o, l: checks.issue_count(~o["order_date"].str.fullmatch(r"\d{4}/\d{2}/\d{2}").fillna(False)),
         after=lambda m, info: sum(
             m.loc[m["order_id"] == order_id, "order_date"].item() != date
             for order_id, date in info["expected_dates"].items())),
    dict(id="unknown_date_format", table="orders", inject=inject_unknown_date_format,
         what="order_date in a format we can't read (09.04.2023, 2023-13-45, yesterday)",
         # the issue was dates silently becoming NaT; after, every unreadable date must be flagged.
         before=lambda c, o, l: checks.issue_count(
             o["order_date"].notna() & cl.parse_dates(o["order_date"], cl.ORDER_DATE_FORMATS).isna()),
         after=lambda m, info: checks.issue_count(m["order_date"].isna() & (m["order_date_invalid"] == 0))),
    dict(id="null_order_id", table="labels", inject=inject_null_order_id,
         what="order_id missing in labels (missing ids would match each other in a join)",
         before=lambda c, o, l: checks.count_missing_keys(l, cl.LABEL_REQUIRED_KEYS),
         after=lambda m, info: checks.count_missing_keys(m, cl.LABEL_REQUIRED_KEYS)),
    dict(id="blank_customer_id", table="customers", inject=inject_blank_customer_id,
         what="customer_id blank or only spaces",
         before=lambda c, o, l: checks.count_missing_keys(c, cl.CUSTOMER_KEY),
         after=lambda m, info: checks.count_missing_keys(m, cl.CUSTOMER_KEY)),
    dict(id="exact_duplicate", table="labels", inject=inject_exact_duplicate,
         what="the same label row twice",
         before=lambda c, o, l: l.duplicated(cl.ORDER_KEY).sum(),
         after=lambda m, info: m.duplicated(cl.ORDER_KEY).sum()),
    dict(id="conflicting_is_fraud", table="labels", inject=inject_conflicting_is_fraud,
         what="the same order labelled both fraud and not fraud",
         before=lambda c, o, l: checks.count_conflicting_duplicates(l, cl.ORDER_KEY),
         after=lambda m, info: m["order_id"].isin(info["keys"]).sum()),
    dict(id="conflicting_customer", table="customers", inject=inject_conflicting_customer,
         what="the same customer twice with different num_orders_last_50days",
         before=lambda c, o, l: checks.count_conflicting_duplicates(c, cl.CUSTOMER_KEY),
         after=lambda m, info: m["customer_id"].isin(info["keys"]).sum()),
    dict(id="orphan_labels", table="labels", inject=inject_orphan_labels,
         what="labels pointing at orders that don't exist",
         before=lambda c, o, l: checks.issue_count(~cl.key_isin(l, o, cl.ORDER_KEY)),
         after=lambda m, info: m["order_id"].isin(info["keys"]).sum()),
    dict(id="invalid_is_fraud", table="labels", inject=inject_invalid_is_fraud,
         what='is_fraud not 0 or 1 ("x", "2", "-1", "yes", missing)',
         before=lambda c, o, l: checks.issue_count(~pd.to_numeric(l["is_fraud"], errors="coerce").isin([0, 1])),
         after=lambda m, info: checks.issue_count(~m["is_fraud"].isin([0, 1])) + m["order_id"].isin(info["keys"]).sum()),
    dict(id="missing_num_items", table="orders", inject=inject_missing_num_items,
         what="num_items_ordered missing",
         before=lambda c, o, l: o["num_items_ordered"].isna().sum(),
         after=lambda m, info: m["num_items_ordered"].isna().sum() + m["order_id"].isin(info["keys"]).sum()),
    dict(id="non_numeric_order_value", table="orders", inject=inject_non_numeric_order_value,
         what='order_value is text ("abc", "12,50", "$5")',
         before=lambda c, o, l: checks.issue_count(pd.to_numeric(o["order_value"], errors="coerce").isna()),
         after=lambda m, info: m["order_value"].isna().sum() + m["order_id"].isin(info["keys"]).sum()),
    dict(id="ids_read_as_numbers", table="orders", inject=inject_numeric_ids, reader="read_csv",
         what="all-digit order_ids read with a plain pd.read_csv, so leading zeros are lost",
         before=lambda c, o, l: lost_zeros(o["order_id"]),
         after=lambda m, info: lost_zeros(m["order_id"]) + checks.issue_count(~m["order_id"].isin(info["expected_ids"]))),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def cleaning_messages(caplog):
    return [record.getMessage() for record in caplog.records
            if record.name.startswith("amlop_ca1") and record.levelno >= logging.WARNING]


def test_clean_sample_has_no_issues(clean_sample, tmp_path, caplog):
    """the starting point: cleaning changes nothing and rejects nothing."""
    with caplog.at_level(logging.WARNING):
        raw, cleaned, merged, rejects = run_cleaning(clean_sample, tmp_path, "load_csv")
    print(f"\n{'=' * 100}\nSTARTING SAMPLE (no issues)\n{'=' * 100}")
    for name, index in TABLE_INDEX.items():
        print_profile(f"raw {name}", raw[index])
    print(f"\n  merged rows: {len(merged):,}; rejected rows: {len(rejects)}")

    assert len(rejects) == 0, rejects.summary()
    assert len(merged) == SAMPLE_LABELS
    assert cleaning_messages(caplog) == []


@pytest.mark.parametrize("case", ISSUES, ids=[case["id"] for case in ISSUES])
def test_one_issue(case, clean_sample, tmp_path, caplog):
    customers, orders, labels = (table.copy() for table in clean_sample)
    customers, orders, labels, info = case["inject"](customers, orders, labels)

    with caplog.at_level(logging.WARNING):
        raw, cleaned, merged, rejects = run_cleaning(
            (customers, orders, labels), tmp_path, case.get("reader", "load_csv"))
    before = int(case["before"](*raw))
    after = int(case["after"](merged, info))
    report(case, raw, cleaned, merged, rejects, before, after, SAMPLE_LABELS, cleaning_messages(caplog))

    # the issue was really there in the raw data, cleaning said so, and it's gone afterwards.
    assert before > 0, "the issue wasn't injected"
    assert cleaning_messages(caplog) or info.get("reason") is None, "cleaning didn't log anything"
    assert after == 0, f"{after} rows still have the issue after cleaning"

    # only the rows given the issue are affected.
    assert len(merged) == SAMPLE_LABELS - info["lost"]
    if "reason" in info:
        assert info["reason"] in set(rejects.to_frame()["reason"]), rejects.summary()

    # every label is either in the merged table or in the reject log.
    rejected_labels = (rejects.to_frame()["table"] == "labels").sum()
    assert len(merged) + rejected_labels == len(raw[TABLE_INDEX["labels"]])
