"""count known data issues in raw or cleaned tables.

run on raw tables to find issues, and on cleaned tables to show they're fixed.
"""

import pandas as pd

from amlop_ca1.cleaning import (
    COLLECT_TYPE_ALIASES, CUSTOMER_KEY, ORDER_DATE_FORMATS, ORDER_KEY,
    PAYMENT_METHOD_ALIASES, normalise_text, parse_dates,
)


def check_issues(customers, orders, labels):
    """return a table with one row per issue and its count."""
    results = []

    def add(table, column, issue, count):
        results.append({"table": table, "column": column, "issue": issue, "count": int(count)})

    # country codes that aren't uppercase, e.g. "pk".
    for name, table in [("customers", customers), ("orders", orders), ("labels", labels)]:
        codes = table["country_code"].astype("string")
        add(name, "country_code", "not uppercase", (codes != codes.str.upper()).sum())

    # mobile_verified should only be true or false.
    verified = customers["mobile_verified"].astype("string")
    add("customers", "mobile_verified", "not True/False", (~verified.isin(["True", "False"])).sum())

    # missing values.
    add("customers", "first_order_datetime", "missing", customers["first_order_datetime"].isna().sum())
    add("orders", "payment_method", "missing", orders["payment_method"].isna().sum())
    add("orders", "num_items_ordered", "missing", orders["num_items_ordered"].isna().sum())

    # the same value spelled in different ways.
    add("orders", "collect_type", "minority spellings",
        count_minority_spellings(orders["collect_type"], COLLECT_TYPE_ALIASES))
    add("orders", "payment_method", "minority spellings",
        count_minority_spellings(orders["payment_method"], PAYMENT_METHOD_ALIASES))

    # dates: cleaned ones are real datetimes, raw ones are mixed text.
    order_date = orders["order_date"]
    if pd.api.types.is_datetime64_any_dtype(order_date):
        add("orders", "order_date", "not YYYY/MM/DD text or datetime", 0)
        add("orders", "order_date", "unparseable", order_date.isna().sum())
    else:
        text = order_date.astype("string")
        standard = text.str.fullmatch(r"\d{4}/\d{2}/\d{2}").fillna(False)
        add("orders", "order_date", "not YYYY/MM/DD text or datetime", (text.notna() & ~standard).sum())
        parsed = parse_dates(text, ORDER_DATE_FORMATS)
        add("orders", "order_date", "unparseable", (text.notna() & parsed.isna()).sum())

    # keys.
    add("orders", "order_id", "reused across countries", orders["order_id"].duplicated().sum())
    add("customers", "(country_code, customer_id)", "duplicate key", customers.duplicated(CUSTOMER_KEY).sum())
    add("orders", "(country_code, order_id)", "duplicate key", orders.duplicated(ORDER_KEY).sum())
    add("labels", "(country_code, order_id)", "duplicate key", labels.duplicated(ORDER_KEY).sum())

    # rows that won't find a partner when joining.
    add("labels", "(country_code, order_id)", "no matching order",
        (~key_isin(labels, orders, ORDER_KEY)).sum())
    add("labels", "(country_code, customer_id)", "no matching customer",
        (~key_isin(labels, customers, CUSTOMER_KEY)).sum())
    add("orders", "(country_code, order_id)", "no matching label",
        (~key_isin(orders, labels, ORDER_KEY)).sum())

    # refund bigger than the order itself.
    refund = pd.to_numeric(orders["refund_value"], errors="coerce")
    order_value = pd.to_numeric(orders["order_value"], errors="coerce")
    add("orders", "refund_value", "greater than order_value", (refund > order_value).sum())

    return pd.DataFrame(results)


def count_minority_spellings(column, aliases):
    """rows using a less common spelling, e.g. "delivery" (85) next to "Delivery" (937,730) -> 85."""
    spellings = column.dropna().astype("string").value_counts().reset_index()
    spellings.columns = ["spelling", "rows"]
    spellings["meaning"] = normalise_text(spellings["spelling"]).replace(aliases)

    rows_per_meaning = spellings.groupby("meaning")["rows"].sum()
    rows_in_top_spelling = spellings.groupby("meaning")["rows"].max()
    return int((rows_per_meaning - rows_in_top_spelling).sum())


def key_isin(left, right, key):
    """for each row in left, true if the same key exists in right."""
    right_keys = right[key].astype("string").drop_duplicates()
    left_keys = left[key].astype("string")
    matched = left_keys.merge(right_keys, on=key, how="left", indicator=True)
    return (matched["_merge"] == "both").to_numpy()
