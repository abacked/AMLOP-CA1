"""clean and merge the sp-buy customers, orders and labels files.

works on any batch with the same columns as the _base files. the notebook and
the etl both import this, so they clean data exactly the same way.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# join keys. order_id and customer_id repeat across countries, so always join on both columns.
ORDER_KEY = ["country_code", "order_id"]
CUSTOMER_KEY = ["country_code", "customer_id"]



# pass to pd.read_csv so ids stay text and mobile_verified isn't a mix of bool and text.
CSV_DTYPES = {
    "country_code": "string",
    "customer_id": "string",
    "order_id": "string",
    "mobile_verified": "string",
}

# columns each file must have.
CUSTOMER_COLUMNS = [
    "country_code", "customer_id", "mobile_verified",
    "num_orders_last_50days", "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days", "total_payment_last_50days",
    "num_associated_customers", "first_order_datetime",
]


ORDER_COLUMNS = [
    "country_code", "order_id", "collect_type", "payment_method",
    "order_value", "num_items_ordered", "refund_value", "order_date",
]


LABEL_COLUMNS = ["country_code", "order_id", "customer_id", "is_fraud"]

# columns that should be numbers.
CUSTOMER_NUMERIC = [
    "num_orders_last_50days", "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days", "total_payment_last_50days",
    "num_associated_customers",
]


ORDER_NUMERIC = ["order_value", "num_items_ordered", "refund_value"]

# orders missing orders these are dropped.
ORDER_REQUIRED = ["order_value", "num_items_ordered"]

# values seen in _base. new values are kept but logged as a warning.
KNOWN_COUNTRIES = {"BD", "MY", "PH", "PK", "TH"}
KNOWN_COLLECT_TYPES = {"delivery", "pickup"}
KNOWN_PAYMENT_METHODS = {
    "payment on delivery", "cybersource", "online banking", "credit card",
    "antfinancial gcash", "generic creditcard", "balance", "antfinancial bkash",
    "antfinancial truemoney", "boost wallet", "jazzcash wallet", "invoice",
    "adyen hpp molpay", "no payment", "cybersource applepay", "paypal",
    "xendit directdebit", "unknown",
}

# different wordings of the same value (case, spaces, "-" and "_" are already handled).
BOOL_VALUES = {
    "true": True, "yes": True, "y": True, "t": True, "1": True,
    "false": False, "no": False, "n": False, "f": False, "0": False,
}


COLLECT_TYPE_ALIASES = {
    "pick up": "pickup",
    "self pickup": "pickup",
    "self pick up": "pickup",
}


PAYMENT_METHOD_ALIASES = {
    "on delivery": "payment on delivery",
    "cash on delivery": "payment on delivery",
}


MISSING_PAYMENT_METHOD = "unknown"



# date formats to try, in order
ORDER_DATE_FORMATS = [
    "%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d", "%Y-%b-%d", "%Y/%B/%d", "%Y-%B-%d",
]
CUSTOMER_DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
]

# columns with a long right tail, given a log version by add_log_features.
SKEWED_COLUMNS = [
    "order_value", "num_items_ordered",
    "num_orders_last_50days", "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days", "total_payment_last_50days",
    "num_associated_customers",
]


# text helpers: tidy up text values so different spellings match.

def normalise_text(values):
    """lowercase, trim, and turn spaces/_/- into one space: " Payment_on-Delivery" -> "payment on delivery"."""
    text = values.astype("string")
    text = text.str.strip().str.lower()
    text = text.str.replace(r"[\s_\-]+", " ", regex=True)
    return text


def uppercase_code(values):
    """trim and uppercase: " pk" -> "PK"."""
    return values.astype("string").str.strip().str.upper()


def clean_distinct_values(column, clean_function):
    """clean each distinct value once, then map back. much faster than cleaning every row."""
    distinct_values = pd.Series(column.dropna().unique())
    cleaned_values = clean_function(distinct_values)
    lookup = dict(zip(distinct_values, cleaned_values))
    return column.map(lookup)


def canonicalise(column, aliases, known_values, name):
    """normalise text, apply aliases, and warn about values we haven't seen."""
    cleaned = clean_distinct_values(column, normalise_text)
    cleaned = cleaned.replace(aliases).astype("string")
    warn_about_new_values(cleaned, known_values, name)
    return cleaned


def warn_about_new_values(column, known_values, name):
    new_values = set(column.dropna().unique()) - known_values
    if new_values:
        logger.warning("%s: new values kept as they are: %s", name, sorted(new_values))


# date and number helpers.

def parse_dates(column, formats):
    """try every format on the values still unparsed. anything left becomes nat."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return column.copy()

    text = column.astype("string").str.strip()
    parsed = pd.to_datetime(text, format=formats[0], errors="coerce")

    for date_format in formats[1:]:
        still_missing = parsed.isna() & text.notna()
        if not still_missing.any():
            break
        parsed[still_missing] = pd.to_datetime(
            text[still_missing], format=date_format, errors="coerce"
        )
    return parsed


def convert_to_numbers(df, columns, table):
    """convert columns to numbers in place. anything that isn't a number becomes nan."""
    for column in columns:
        was_present = df[column].notna()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        broken = (was_present & df[column].isna()).sum()
        if broken:
            logger.warning("%s.%s: %d non-numeric values set to NaN", table, column, broken)


# row and column checks.

def check_columns(df, columns, table):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def drop_duplicate_keys(df, key, table):
    duplicated = df.duplicated(subset=key, keep="first")
    if duplicated.any():
        logger.warning("%s: dropped %d rows with a duplicate %s", table, duplicated.sum(), key)
    return df[~duplicated]


def check_unique(df, key, table):
    # faster than merge(validate=...) on 1m rows.
    if df.duplicated(subset=key).any():
        raise ValueError(f"{table} has duplicate {key}; clean it before merging")


# customers: uppercase country, real true/false, parse dates, flag missing first order.

def clean_customers(df):
    """return a cleaned copy of a customers table."""
    check_columns(df, CUSTOMER_COLUMNS, "customers")
    customers = df.copy()

    customers["country_code"] = clean_distinct_values(customers["country_code"], uppercase_code).astype("string")
    warn_about_new_values(customers["country_code"], KNOWN_COUNTRIES, "customers.country_code")
    customers["customer_id"] = customers["customer_id"].astype("string").str.strip()

    # "yes", "True", true etc. become a real boolean; anything unrecognised becomes na.
    verified_text = clean_distinct_values(customers["mobile_verified"], normalise_text)
    customers["mobile_verified"] = verified_text.map(BOOL_VALUES).astype("boolean")
    unrecognised = (verified_text.notna() & customers["mobile_verified"].isna()).sum()
    if unrecognised:
        logger.warning("customers.mobile_verified: %d unrecognised values set to NA", unrecognised)

    convert_to_numbers(customers, CUSTOMER_NUMERIC, "customers")

    raw_first_order = customers["first_order_datetime"]
    customers["first_order_datetime"] = parse_dates(raw_first_order, CUSTOMER_DATETIME_FORMATS)
    unparsed = (raw_first_order.notna() & customers["first_order_datetime"].isna()).sum()
    if unparsed:
        logger.warning("customers.first_order_datetime: %d values could not be parsed", unparsed)

    # missing first order date is a fraud signal (24% vs 11%), so keep the rows and flag them.
    customers["first_order_missing"] = customers["first_order_datetime"].isna().astype("int8")

    customers = drop_duplicate_keys(customers, CUSTOMER_KEY, "customers")
    logger.info("customers: %d rows, %d with no first_order_datetime",
                len(customers), customers["first_order_missing"].sum())
    return customers.reset_index(drop=True)


# orders: one spelling per value, fill missing payment, drop incomplete rows, parse dates.
# outliers and refund_value are kept on purpose.

def clean_orders(df):
    """return a cleaned copy of an orders table."""
    check_columns(df, ORDER_COLUMNS, "orders")
    orders = df.copy()

    orders["country_code"] = clean_distinct_values(orders["country_code"], uppercase_code).astype("string")
    warn_about_new_values(orders["country_code"], KNOWN_COUNTRIES, "orders.country_code")
    orders["order_id"] = orders["order_id"].astype("string").str.strip()

    orders["collect_type"] = canonicalise(
        orders["collect_type"], COLLECT_TYPE_ALIASES, KNOWN_COLLECT_TYPES, "orders.collect_type"
    )

    payment = canonicalise(
        orders["payment_method"], PAYMENT_METHOD_ALIASES, KNOWN_PAYMENT_METHODS, "orders.payment_method"
    )
    logger.info("orders.payment_method: %d missing values filled with %r",
                payment.isna().sum(), MISSING_PAYMENT_METHOD)
    orders["payment_method"] = payment.fillna(MISSING_PAYMENT_METHOD)

    convert_to_numbers(orders, ORDER_NUMERIC, "orders")
    incomplete = orders[ORDER_REQUIRED].isna().any(axis=1)
    if incomplete.any():
        logger.warning("orders: dropped %d rows missing %s", incomplete.sum(), ORDER_REQUIRED)
        orders = orders[~incomplete]

    raw_date = orders["order_date"]
    orders["order_date"] = parse_dates(raw_date, ORDER_DATE_FORMATS)
    unparsed = (raw_date.notna() & orders["order_date"].isna()).sum()
    if unparsed:
        logger.warning("orders.order_date: %d values could not be parsed", unparsed)

    orders = drop_duplicate_keys(orders, ORDER_KEY, "orders")
    logger.info("orders: %d rows", len(orders))
    return orders.reset_index(drop=True)


# labels: uppercase country, keep only is_fraud 0 or 1.

def clean_labels(df):
    """return a cleaned copy of a labels table."""
    check_columns(df, LABEL_COLUMNS, "labels")
    labels = df.copy()

    labels["country_code"] = clean_distinct_values(labels["country_code"], uppercase_code).astype("string")
    warn_about_new_values(labels["country_code"], KNOWN_COUNTRIES, "labels.country_code")
    labels["order_id"] = labels["order_id"].astype("string").str.strip()
    labels["customer_id"] = labels["customer_id"].astype("string").str.strip()

    labels["is_fraud"] = pd.to_numeric(labels["is_fraud"], errors="coerce")
    bad_label = ~labels["is_fraud"].isin([0, 1])
    if bad_label.any():
        logger.warning("labels: dropped %d rows where is_fraud is not 0 or 1", bad_label.sum())
        labels = labels[~bad_label]
    labels["is_fraud"] = labels["is_fraud"].astype("int8")

    labels = drop_duplicate_keys(labels, ORDER_KEY, "labels")
    logger.info("labels: %d rows, fraud rate %.2f%%", len(labels), 100 * labels["is_fraud"].mean())
    return labels.reset_index(drop=True)


# merge: one row per labelled order. rows that can't be matched are dropped and counted.

def merge_all(customers, orders, labels):
    """join labels -> orders -> customers. counts of dropped rows go in merged.attrs["join_stats"]."""
    check_unique(labels, ORDER_KEY, "labels")
    check_unique(orders, ORDER_KEY, "orders")
    check_unique(customers, CUSTOMER_KEY, "customers")

    labels_with_orders = labels.merge(orders, on=ORDER_KEY, how="inner")
    merged = labels_with_orders.merge(customers, on=CUSTOMER_KEY, how="inner")

    stats = {
        "labels_in": len(labels),
        "labels_without_order": len(labels) - len(labels_with_orders),
        "labels_without_customer": len(labels_with_orders) - len(merged),
        "orders_without_label": len(orders) - len(labels_with_orders),
        "rows_out": len(merged),
    }
    logger.info(
        "merge: %d labels -> %d rows (dropped %d with no order and %d with no customer; "
        "%d orders had no label)",
        stats["labels_in"], stats["rows_out"], stats["labels_without_order"],
        stats["labels_without_customer"], stats["orders_without_label"],
    )
    merged.attrs["join_stats"] = stats
    return merged


def clean_and_merge(customers, orders, labels):
    """clean the three raw tables and merge them in one go."""
    return merge_all(clean_customers(customers), clean_orders(orders), clean_labels(labels))


# transform: log versions of skewed columns for models that need them.

def add_log_features(df, columns=None):
    """return a copy with a log_<column> for each skewed column. originals are kept."""
    if columns is None:
        columns = SKEWED_COLUMNS

    result = df.copy()
    for column in columns:
        if column in result.columns:
            # counts and amounts should never be negative.
            result[f"log_{column}"] = np.log1p(result[column].clip(lower=0))
    return result
