"""count known data issues in raw or cleaned tables.

run on raw tables to find issues, and on cleaned tables to show they're fixed.
"""

import pandas as pd

from amlop_ca1.cleaning import (
    COLLECT_TYPE_ALIASES, CUSTOMER_KEY, CUSTOMER_REQUIRED_KEYS, LABEL_REQUIRED_KEYS,
    ORDER_DATE_FORMATS, ORDER_KEY, ORDER_REQUIRED_KEYS, PAYMENT_METHOD_ALIASES,
    key_isin, normalise_text, parse_dates,
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
    for name, table, key, required in [
        ("customers", customers, CUSTOMER_KEY, CUSTOMER_REQUIRED_KEYS),
        ("orders", orders, ORDER_KEY, ORDER_REQUIRED_KEYS),
        ("labels", labels, ORDER_KEY, LABEL_REQUIRED_KEYS),
    ]:
        add(name, f"({', '.join(required)})", "missing key", count_missing_keys(table, required))
        add(name, f"({', '.join(key)})", "duplicate key", table.duplicated(key).sum())
        add(name, f"({', '.join(key)})", "conflicting duplicate key", count_conflicting_duplicates(table, key))

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


def count_missing_keys(table, columns):
    """rows with any key column missing or blank."""
    keys = table[columns].astype("string").apply(lambda column: column.str.strip())
    return int((keys.isna() | (keys == "")).any(axis=1).sum())


def count_conflicting_duplicates(table, key):
    """rows sharing a key with another row that has different values (exact copies don't count)."""
    same_key = table[table.duplicated(key, keep=False)]
    distinct = same_key.drop_duplicates()
    return int(distinct.duplicated(key, keep=False).sum())


# profiles: the shape of a table without printing its rows.

def value_pattern(values):
    """the shape of each value: digits become 9, letters a or A. "2023/04/09" -> "9999/99/99", "PK" -> "AA"."""
    pattern = values.astype("string").str.replace(r"[0-9]", "9", regex=True)
    pattern = pattern.str.replace(r"[a-z]", "a", regex=True)
    return pattern.str.replace(r"[A-Z]", "A", regex=True)


def profile_table(df, max_formats=3, max_values=4, category_limit=30):
    """one row per column: dtype, nulls, distinct values and what the values look like.

    shows formats or a range, never the rows themselves:
      text with <= category_limit values -> the values, e.g. "delivery (1,990), pickup (10)"
      other text           -> the formats, e.g. "9999/99/99 (1,995), 9999-99-99 (5)"
      numbers and dates    -> the range, e.g. "0.0 .. 662.9"
    """
    rows = []
    for column in df.columns:
        values = df[column]
        present = values.dropna()
        rows.append({
            "column": column,
            "dtype": str(values.dtype),
            "nulls": int(values.isna().sum()),
            "distinct": int(present.nunique()),
            "looks like": describe_values(values, present, max_formats, max_values, category_limit),
        })
    return pd.DataFrame(rows)


def describe_values(values, present, max_formats, max_values, category_limit):
    if present.empty:
        return "(all missing)"
    if pd.api.types.is_bool_dtype(values) or present.nunique() <= max_values:
        return counts_text(present.astype("string").value_counts(), max_values)
    if pd.api.types.is_numeric_dtype(values):
        return f"{present.min():.4g} .. {present.max():.4g}"
    if pd.api.types.is_datetime64_any_dtype(values):
        return f"{present.min():%Y-%m-%d} .. {present.max():%Y-%m-%d}"

    if present.nunique() <= category_limit:
        return counts_text(present.astype("string").value_counts(), max_values)

    patterns = value_pattern(present).value_counts()
    if len(patterns) > 10:
        # ids: too many formats to list, so show their lengths instead.
        lengths = present.astype("string").str.len().value_counts()
        return f"{len(patterns):,} formats; length " + counts_text(lengths, max_formats)
    text = counts_text(patterns, max_formats)
    # show a few odd values too: whitespace and case don't show up in a pattern.
    padded = (present.astype("string") != present.astype("string").str.strip()).sum()
    return text + (f"; {padded:,} with spaces around" if padded else "")


def counts_text(counts, limit):
    shown = ", ".join(f"{value!r} ({count:,})" if isinstance(value, str) else f"{value} ({count:,})"
                      for value, count in counts.head(limit).items())
    return shown + (f", +{len(counts) - limit} more" if len(counts) > limit else "")


def issue_count(mask):
    """how many rows a true/false check flags. missing counts as false."""
    return int(pd.Series(mask).fillna(False).astype(bool).sum())
