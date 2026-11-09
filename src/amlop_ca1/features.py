"""Order features available before shipment, independent of the cleaning module.

Requires pandas >= 2.0, numpy and scikit-learn >= 1.2. Pass a DataFrame with
REQUIRED_COLUMNS; extra columns are ignored. Category aliases and joins belong
to cleaning.py. This module performs only feature-level validation.

Split rows before calling ``make_feature_pipeline().fit(X_train)``. Call only
``transform`` on validation, test and inference rows, and save the fitted
pipeline alongside the model. No labels are required, and y is never inspected.
Missing numeric values use training medians (zero for entirely empty training
columns); missing categories use a fixed sentinel. Output is usually sparse;
row positions match the input, while OrderFeatures also preserves its index.
Callers must skip empty batches after cleaning. Very large finite numbers above
MAX_NUMERIC_VALUE raise a validation error before imputation or scaling; this
is a numerical safety limit, not a business rule for legitimate order values.
"""

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


REQUIRED_COLUMNS = (
    "order_value", "num_items_ordered", "country_code", "collect_type",
    "payment_method", "order_date",
)
NUMERIC_FEATURES = (
    "order_value", "num_items_ordered", "value_per_item", "order_weekday",
    "order_month",
)
CATEGORICAL_FEATURES = ("country_code", "collect_type", "payment_method")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DATE_FORMATS = ("%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d", "%Y/%B/%d")
# Leave ample float64 headroom for sums, medians and squared scaling terms.
MAX_NUMERIC_VALUE = 1e100


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OrderFeatures requires a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Input has duplicate column names")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required order columns: {missing}")
    if len(frame) == 0:
        raise ValueError("OrderFeatures requires at least one row; skip empty batches after cleaning")


def _nonnegative_numbers(values, *, whole=False):
    # Avoid interpreting timestamps as epoch numbers or discarding imaginary parts.
    invalid_types = (
        bool, np.bool_, complex, np.complexfloating, date, time, timedelta,
        np.datetime64, np.timedelta64,
    )
    values = values.astype(object)
    values = values.mask(values.map(lambda value: isinstance(value, invalid_types)))
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    if (np.isfinite(numbers) & (numbers > MAX_NUMERIC_VALUE)).any():
        raise ValueError(
            f"{values.name} exceeds the numerical safety limit {MAX_NUMERIC_VALUE:g}; "
            "validate this batch before preprocessing"
        )
    invalid = ~np.isfinite(numbers) | (numbers < 0)
    if whole:
        invalid |= np.isfinite(numbers) & (numbers != np.floor(numbers))
    numbers[invalid] = np.nan
    return numbers


def _dates(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format=DATE_FORMATS[0], errors="coerce")
    for date_format in DATE_FORMATS[1:]:
        parsed = parsed.fillna(pd.to_datetime(text, format=date_format, errors="coerce"))
    return parsed


class OrderFeatures(TransformerMixin, BaseEstimator):
    """Build the explicit predictor allowlist without learning from rows or y.

    Invalid numbers/dates become missing for downstream imputation. Zero item
    counts are retained, but their value_per_item is missing. Weekday is 0=Monday
    through 6=Sunday; month is 1..12. Ambiguous or unsupported text dates become
    missing rather than being guessed. Numeric date values are not epoch times.
    """

    def fit(self, X, y=None):
        _validate_frame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        _validate_frame(X)
        result = pd.DataFrame(index=X.index)
        amounts = _nonnegative_numbers(X["order_value"])
        items = _nonnegative_numbers(X["num_items_ordered"], whole=True)
        result["order_value"] = amounts
        result["num_items_ordered"] = items
        ratio = np.full(len(X), np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(amounts, items, out=ratio, where=items > 0)
        ratio[~np.isfinite(ratio)] = np.nan
        result["value_per_item"] = ratio
        dates = _dates(X["order_date"])
        result["order_weekday"] = dates.dt.day_of_week.to_numpy(dtype=float, na_value=np.nan)
        result["order_month"] = dates.dt.month.to_numpy(dtype=float, na_value=np.nan)
        for column in CATEGORICAL_FEATURES:
            # object + np.nan is compatible with sklearn's categorical imputer.
            values = X[column].map(
                lambda value: value.strip() if isinstance(value, str) and value.strip() else np.nan
            ).to_numpy(dtype=object)
            if column == "country_code":
                values = np.asarray([
                    value.upper() if isinstance(value, str) and len(value) == 2 and value.isalpha()
                    else np.nan for value in values
                ], dtype=object)
            result[column] = values
        return result.loc[:, list(FEATURE_COLUMNS)]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(FEATURE_COLUMNS, dtype=object)


def make_feature_pipeline(*, scale_numeric=False):
    """Return an unfitted preprocessing Pipeline; fit on training rows only.

    Missing indicators are fitted with the numeric imputer. Unseen categories
    produce an all-zero category block without changing the fitted schema.
    Scaling is optional for model families that benefit from standardisation.
    Class weighting, data splitting and model fitting belong to training code.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return Pipeline([
        ("features", OrderFeatures()),
        ("preprocess", ColumnTransformer([
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ], remainder="drop", sparse_threshold=1.0)),
    ])
