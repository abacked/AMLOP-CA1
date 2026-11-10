"""clean and merge the sp-buy customers, orders and labels files.

works on any batch with the same columns as the _base files. the notebook and
the etl both import this, so they clean data exactly the same way.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# join keys. order_id and customer_id repeat across countries, so always join on both columns.
ORDER_KEY = ["country_code", "order_id"]
CUSTOMER_KEY = ["country_code", "customer_id"]



# read with load_csv so ids stay text ("00123" keeps its zeros) and mobile_verified
# isn't a mix of bool and text.
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

# every order_id and customer_id in _base is 10 characters. used to put back leading
# zeros if an id column was read as numbers ("0000123456" -> 123456 -> "0000123456").
ID_LENGTH = 10

# every row must have these to be joined. nulls would match each other in a pandas merge.
CUSTOMER_REQUIRED_KEYS = CUSTOMER_KEY
ORDER_REQUIRED_KEYS = ORDER_KEY
LABEL_REQUIRED_KEYS = ["country_code", "order_id", "customer_id"]

# columns kept in the reject log so a rejected row can be found again.
REJECT_COLUMNS = ["table", "reason", "source_row", "country_code", "order_id", "customer_id"]

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



# log file: everything cleaning and checks report, with a timestamp, saved to logs/.

PACKAGE_LOGGER = logging.getLogger("amlop_ca1")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def log_to_file(log_dir="logs", batch="base"):
    """also write every cleaning message to <log_dir>/cleaning_<batch>_<time>.log and return its path.

    calling it again (e.g. re-running a notebook cell) switches to a new file instead of
    writing every line twice.
    """
    stop_log_file()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"cleaning_{batch}_{datetime.now():%Y%m%d_%H%M%S}.log"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.amlop_log_file = True  # marks the handlers this module added
    PACKAGE_LOGGER.addHandler(handler)
    if PACKAGE_LOGGER.getEffectiveLevel() > logging.INFO:
        PACKAGE_LOGGER.setLevel(logging.INFO)

    logger.info("batch %s: logging to %s", batch, path)
    return path


def stop_log_file():
    """stop writing to the log file started by log_to_file, and close it."""
    for handler in list(PACKAGE_LOGGER.handlers):
        if getattr(handler, "amlop_log_file", False):
            PACKAGE_LOGGER.removeHandler(handler)
            handler.close()


def load_csv(path):
    """read one sp-buy csv with ids as text. low_memory=False looks at the whole column before picking types."""
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def load_batch(data_dir, batch):
    """read customers_<batch>.csv, orders_<batch>.csv and labels_<batch>.csv from data_dir."""
    data_dir = Path(data_dir)
    return tuple(load_csv(data_dir / f"{name}_{batch}.csv") for name in ("customers", "orders", "labels"))


# log severy row dropped by cleaning or merging with reasoning
class RejectLog:

    def __init__(self):
        self._parts = []

    def add(self, table, rows, reason):
        """record rows (a dataframe) as rejected. reason is a string or one string per row."""
        if rows.empty:
            return
        part = rows.reindex(columns=REJECT_COLUMNS[3:]).astype("string").reset_index(drop=True)
        part.insert(0, "source_row", rows.index.to_numpy())
        part.insert(0, "reason", pd.array(np.broadcast_to(np.asarray(reason, dtype=object), len(rows)), dtype="string"))
        part.insert(0, "table", table)
        self._parts.append(part)

    def to_frame(self):
        """one row per rejected record."""
        if not self._parts:
            return pd.DataFrame({column: pd.Series(dtype="string") for column in REJECT_COLUMNS})
        return pd.concat(self._parts, ignore_index=True)

    def summary(self):
        """rejected rows counted by table and reason."""
        frame = self.to_frame()
        return frame.groupby(["table", "reason"], sort=False).size().rename("rows").reset_index()
    def __len__(self):
        return sum(len(part) for part in self._parts)


def reject(rejects, table, rows, reason, log=True):
    """log rejected rows and add them to the reject log, if there is one."""
    if rows.empty:
        return
    if log:
        reasons = pd.Series(np.broadcast_to(np.asarray(reason, dtype=object), len(rows)))
        for text, count in reasons.value_counts().items():
            logger.warning("%s: rejected %d rows: %s", table, count, text)
    if rejects is not None:
        rejects.add(table, rows, reason)


# text helpers: tidy up text values so different spellings match.

def normalise_text(values):
    """lowercase, trim, and turn spaces/_/- into one space: " Payment_on-Delivery" -> "payment on delivery"."""
    text = values.astype("string")
    text = text.str.strip().str.lower()
    text = text.str.replace(r"[\s_\-]+", " ", regex=True)
    return text


def uppercase_code(values):
    """trim and uppercase: " pk" -> "PK". blank becomes na."""
    code = values.astype("string").str.strip().str.upper()
    return code.mask(code == "")


def clean_country(column, name):
    country = clean_distinct_values(column, uppercase_code).astype("string")
    warn_about_new_values(country, KNOWN_COUNTRIES, name)
    return country


def clean_id(column, name):
    """trim an id column; blank becomes na. ids read as numbers get their leading zeros back."""
    if pd.api.types.is_numeric_dtype(column):
        return restore_numeric_ids(column, name)
    ids = column.astype("string").str.strip()
    return ids.mask(ids == "")


def restore_numeric_ids(column, name):
    """turn ids that were read as numbers back into text, padded to ID_LENGTH.

    123456 (or 123456.0) -> "0000123456". a value that can't be an id (a decimal,
    infinity, more than ID_LENGTH digits) becomes na, so the row is rejected as a missing key.
    """
    numbers = pd.to_numeric(column, errors="coerce").astype("Float64")
    whole = numbers.notna() & np.isfinite(numbers.fillna(0)) & (numbers % 1 == 0) & (numbers >= 0)
    digits = numbers.where(whole).astype("Int64").astype("string")
    restorable = digits.notna() & (digits.str.len() <= ID_LENGTH)

    ids = digits.where(restorable).str.zfill(ID_LENGTH)
    padded = (restorable & (digits.str.len() < ID_LENGTH)).sum()
    logger.warning("%s was read as numbers (%s); restored %d ids, %d needed leading zeros put back. "
                   "load csvs with cleaning.load_csv to avoid this", name, column.dtype, restorable.sum(), padded)
    lost = column.notna() & ~restorable
    if lost.any():
        logger.warning("%s: %d values can't be restored as ids, e.g. %s; set to missing",
                       name, lost.sum(), column[lost].head(3).tolist())
    return ids.astype("string")


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


def parse_and_flag_dates(df, column, formats, flag, table):
    """parse a date column in place and add a 0/1 flag for values that were there but couldn't be parsed.

    missing values are not flagged; only values in a format we don't know.
    """
    raw = df[column]
    df[column] = parse_dates(raw, formats)
    invalid = raw.notna() & df[column].isna()
    df[flag] = invalid.astype("int8")
    if invalid.any():
        examples = raw[invalid].astype("string").drop_duplicates().head(5).tolist()
        logger.warning("%s.%s: %d values in an unknown format flagged in %s, e.g. %s",
                       table, column, invalid.sum(), flag, examples)


def convert_to_numbers(df, columns, table):
    """Coerce invalid, nonfinite and negative measurements to missing.

    Counts must be whole numbers. Orders missing required measurements are
    rejected by clean_orders; optional fields retain missing values. This is
    validation, not learned imputation or feature engineering.
    """
    for column in columns:
        was_present = df[column].notna()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(df[column]) | (df[column] < 0)
        if column.startswith("num_"):
            invalid |= df[column].notna() & (df[column] % 1 != 0)
        df[column] = df[column].mask(invalid)
        broken = (was_present & df[column].isna()).sum()
        if broken:
            logger.warning("%s.%s: %d invalid numeric measurements set to NaN", table, column, broken)


# row and column checks.

def check_columns(df, columns, table):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def drop_missing_keys(df, columns, table, rejects):
    """reject rows missing any key column. reason names the missing columns."""
    missing = df[columns].isna()
    has_missing = missing.any(axis=1).to_numpy()
    if not has_missing.any():
        return df
    reasons = [
        "missing " + ", ".join(np.array(columns)[row]) for row in missing.to_numpy()[has_missing]
    ]
    reject(rejects, table, df[has_missing], reasons)
    return df[~has_missing]


def resolve_duplicate_keys(df, key, table, rejects):
    """exact copies of a row become one row. copies that disagree are all rejected,
    since there's no way to tell which one is right (e.g. is_fraud 0 and 1 for one order).
    """
    same_key = df.duplicated(subset=key, keep=False).to_numpy()
    if not same_key.any():
        return df

    positions = np.flatnonzero(same_key)
    copies = df.iloc[positions]
    key_name = f"({', '.join(key)})"

    exact = copies.duplicated(keep="first").to_numpy()
    if exact.any():
        extra = copies[exact]
        logger.warning(
            "%s: detected %d %s key(s) with exact duplicate rows, e.g. %s; dropped %d extra copies, kept one of each",
            table, len(extra[key].drop_duplicates()), key_name, examples(extra, key), len(extra),
        )
        reject(rejects, table, extra, "exact duplicate", log=False)

    remaining = copies[~exact]
    conflicting = remaining.duplicated(subset=key, keep=False).to_numpy()
    conflicts = remaining[conflicting]
    if not conflicts.empty:
        # which columns differ within each key, e.g. "is_fraud".
        value_columns = [column for column in df.columns if column not in key]
        differs = conflicts.groupby(key)[value_columns].nunique(dropna=False).gt(1)
        differing = differs.apply(lambda row: ", ".join(row.index[row]), axis=1).rename("differs")
        per_row = conflicts[key].merge(differing.reset_index(), on=key, how="left")["differs"]
        for columns, count in differing.value_counts().items():
            group = conflicts[(per_row == columns).to_numpy()]
            logger.warning(
                "%s: detected %d %s key(s) with conflicting values in %s, e.g. %s; dropped all %d rows",
                table, count, key_name, columns, examples(group, key), len(group),
            )
        reject(rejects, table, conflicts, ("conflicting duplicate: " + per_row).tolist(), log=False)

    drop = np.zeros(len(df), dtype=bool)
    drop[positions[exact]] = True
    drop[positions[~exact][conflicting]] = True
    return df[~drop]


def examples(rows, key, limit=3):
    """a few keys as text for log messages, e.g. "PH/o1 (row 0), MY/o7 (row 12)"."""
    shown = rows.drop_duplicates(subset=key).head(limit)
    keys = shown[key].astype("string").agg("/".join, axis=1)
    return ", ".join(f"{text} (row {row})" for text, row in zip(keys, shown.index))


def check_no_missing_keys(df, required, table):
    """Reject null or blank keys even when callers bypass clean_* functions."""
    missing = pd.concat([
        df[column].isna() | df[column].astype("string").str.strip().eq("").fillna(False)
        for column in required
    ], axis=1).any(axis=1).sum()
    if missing:
        raise ValueError(f"{table} has {missing} rows with a missing {required}; clean it before merging")


def drop_repeated_keys(df, key, table, rejects):
    """reject every row whose key appears more than once, so a join can't pick one of them.

    clean_* already removes these, so this only fires for tables that skipped cleaning.
    """
    # faster than merge(validate=...) on 1m rows.
    repeated = df.duplicated(subset=key, keep=False).to_numpy()
    if not repeated.any():
        return df
    rows = df[repeated]
    logger.warning(
        "%s: detected %d (%s) key(s) repeated at merge, e.g. %s; dropped all %d rows",
        table, len(rows[key].drop_duplicates()), ", ".join(key), examples(rows, key), len(rows),
    )
    reject(rejects, table, rows, "repeated key at merge", log=False)
    return df[~repeated]


def key_isin(left, right, key):
    """for each row in left, true if the same key exists in right. a missing key never matches."""
    right_keys = right[key].astype("string").dropna().drop_duplicates()
    left_keys = left[key].astype("string")
    matched = left_keys.merge(right_keys, on=key, how="left", indicator=True)
    return (matched["_merge"] == "both").to_numpy() & left_keys.notna().all(axis=1).to_numpy()


# customers: uppercase country, real true/false, parse dates, flag missing first order.

def clean_customers(df, rejects=None):
    """return a cleaned copy of a customers table. dropped rows go to rejects (a RejectLog).

    the index is kept, so it still points at the row in the raw table.
    """
    check_columns(df, CUSTOMER_COLUMNS, "customers")
    customers = df.copy()

    customers["country_code"] = clean_country(customers["country_code"], "customers.country_code")
    customers["customer_id"] = clean_id(customers["customer_id"], "customers.customer_id")
    customers = drop_missing_keys(customers, CUSTOMER_REQUIRED_KEYS, "customers", rejects)

    # "yes", "True", true etc. become a real boolean; anything unrecognised becomes na.
    verified_text = clean_distinct_values(customers["mobile_verified"], normalise_text)
    customers["mobile_verified"] = verified_text.map(BOOL_VALUES).astype("boolean")
    unrecognised = (verified_text.notna() & customers["mobile_verified"].isna()).sum()
    if unrecognised:
        logger.warning("customers.mobile_verified: %d unrecognised values set to NA", unrecognised)

    convert_to_numbers(customers, CUSTOMER_NUMERIC, "customers")

    # missing first order date is a fraud signal (24% vs 11%), so keep the rows and flag them.
    customers["first_order_missing"] = customers["first_order_datetime"].isna().astype("int8")
    # a date in a format we don't know is kept as nat and flagged separately.
    parse_and_flag_dates(customers, "first_order_datetime", CUSTOMER_DATETIME_FORMATS,
                         "first_order_invalid", "customers")

    customers = resolve_duplicate_keys(customers, CUSTOMER_KEY, "customers", rejects)
    logger.info("customers: %d rows, %d with no first_order_datetime, %d with an invalid one",
                len(customers), customers["first_order_missing"].sum(), customers["first_order_invalid"].sum())
    return customers


# orders: one spelling per value, fill missing payment, drop incomplete rows, parse dates.
# outliers and refund_value are kept on purpose.

def clean_orders(df, rejects=None):
    """return a cleaned copy of an orders table. dropped rows go to rejects (a RejectLog).

    the index is kept, so it still points at the row in the raw table.
    """
    check_columns(df, ORDER_COLUMNS, "orders")
    orders = df.copy()

    orders["country_code"] = clean_country(orders["country_code"], "orders.country_code")
    orders["order_id"] = clean_id(orders["order_id"], "orders.order_id")
    orders = drop_missing_keys(orders, ORDER_REQUIRED_KEYS, "orders", rejects)

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
    reject(rejects, "orders", orders[incomplete], f"missing {', '.join(ORDER_REQUIRED)}")
    orders = orders[~incomplete]

    parse_and_flag_dates(orders, "order_date", ORDER_DATE_FORMATS, "order_date_invalid", "orders")

    orders = resolve_duplicate_keys(orders, ORDER_KEY, "orders", rejects)
    logger.info("orders: %d rows, %d with an invalid order_date", len(orders), orders["order_date_invalid"].sum())
    return orders


# labels: uppercase country, keep only is_fraud 0 or 1.

def clean_labels(df, rejects=None):
    """return a cleaned copy of a labels table. dropped rows go to rejects (a RejectLog).

    the index is kept, so it still points at the row in the raw table.
    """
    check_columns(df, LABEL_COLUMNS, "labels")
    labels = df.copy()

    labels["country_code"] = clean_country(labels["country_code"], "labels.country_code")
    labels["order_id"] = clean_id(labels["order_id"], "labels.order_id")
    labels["customer_id"] = clean_id(labels["customer_id"], "labels.customer_id")
    labels = drop_missing_keys(labels, LABEL_REQUIRED_KEYS, "labels", rejects)

    is_fraud = pd.to_numeric(labels["is_fraud"], errors="coerce")
    bad_label = ~is_fraud.isin([0, 1])
    reject(rejects, "labels", labels[bad_label], "is_fraud not 0 or 1")
    labels = labels[~bad_label].assign(is_fraud=is_fraud[~bad_label].astype("int8"))

    # one order with is_fraud 0 and 1 (or two customers) is rejected, not guessed.
    labels = resolve_duplicate_keys(labels, ORDER_KEY, "labels", rejects)
    logger.info("labels: %d rows, fraud rate %.2f%%", len(labels), 100 * labels["is_fraud"].mean())
    return labels


# merge: one row per labelled order. rows that can't be matched go to the reject log.

def merge_all(customers, orders, labels, rejects=None):
    """join labels -> orders -> customers.

    rows with a repeated key, labels with no order or no customer, and orders or
    customers that don't end up in the result are added to rejects (a RejectLog).
    counts go in merged.attrs["join_stats"].
    """
    check_no_missing_keys(labels, LABEL_REQUIRED_KEYS, "labels")
    check_no_missing_keys(orders, ORDER_REQUIRED_KEYS, "orders")
    check_no_missing_keys(customers, CUSTOMER_REQUIRED_KEYS, "customers")

    labels = drop_repeated_keys(labels, ORDER_KEY, "labels", rejects)
    orders = drop_repeated_keys(orders, ORDER_KEY, "orders", rejects)
    customers = drop_repeated_keys(customers, CUSTOMER_KEY, "customers", rejects)

    has_order = key_isin(labels, orders, ORDER_KEY)
    reject(rejects, "labels", labels[~has_order], "no matching order")
    labels_with_orders = labels[has_order]

    has_customer = key_isin(labels_with_orders, customers, CUSTOMER_KEY)
    reject(rejects, "labels", labels_with_orders[~has_customer], "no matching customer")

    used_labels = labels_with_orders[has_customer]

    # orders and customers that don't end up in the merged table, and why.
    order_has_label = key_isin(orders, labels, ORDER_KEY)
    order_used = key_isin(orders, used_labels, ORDER_KEY)
    reject(rejects, "orders", orders[~order_has_label], "no matching label")
    reject(rejects, "orders", orders[order_has_label & ~order_used], "label has no matching customer")

    customer_has_label = key_isin(customers, labels, CUSTOMER_KEY)
    customer_used = key_isin(customers, used_labels, CUSTOMER_KEY)
    reject(rejects, "customers", customers[~customer_has_label], "no matching label")
    reject(rejects, "customers", customers[customer_has_label & ~customer_used], "labels have no matching order")

    merged = (
        used_labels
        .merge(orders, on=ORDER_KEY, how="inner")
        .merge(customers, on=CUSTOMER_KEY, how="inner")
    )

    stats = {
        "labels_in": len(labels),
        "labels_without_order": int((~has_order).sum()),
        "labels_without_customer": int((~has_customer).sum()),
        "orders_without_label": int((~order_has_label).sum()),
        "customers_without_label": int((~customer_has_label).sum()),
        "rows_out": len(merged),
    }
    logger.info(
        "merge: %d labels -> %d rows (dropped %d with no order and %d with no customer; "
        "%d orders and %d customers had no label)",
        stats["labels_in"], stats["rows_out"], stats["labels_without_order"],
        stats["labels_without_customer"], stats["orders_without_label"], stats["customers_without_label"],
    )
    merged.attrs["join_stats"] = stats
    return merged


def clean_and_merge(customers, orders, labels, rejects=None):
    """clean the three raw tables and merge them in one go. pass a RejectLog to see what was dropped."""
    return merge_all(
        clean_customers(customers, rejects), clean_orders(orders, rejects), clean_labels(labels, rejects),
        rejects,
    )


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
