"""Run from the repo root: python -m pytest -q tests/test_features.py.

Requires pytest plus the dependencies documented in amlop_ca1.features.
Uses synthetic orders; no base CSVs or unmerged cleaning code are required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlop_ca1.features import (  # noqa: E402
    FEATURE_COLUMNS, MAX_NUMERIC_VALUE, OrderFeatures, make_feature_pipeline,
)


@pytest.fixture
def orders():
    return pd.DataFrame({
        "order_value": [10.0, 20.0, 30.0],
        "num_items_ordered": [1, 2, 3],
        "country_code": ["MY", "PH", "MY"],
        "collect_type": ["delivery", "pickup", "delivery"],
        "payment_method": ["card", "cash", "card"],
        "order_date": ["2023/01/02", "2023-02-07", "2023/Mar/08"],
    })


def dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def test_allowlist_and_derived_features(orders):
    result = OrderFeatures().fit_transform(orders)
    assert tuple(result.columns) == FEATURE_COLUMNS
    np.testing.assert_allclose(result["value_per_item"], [10, 10, 10])
    assert result["order_weekday"].tolist() == [0, 1, 2]
    assert result["order_month"].tolist() == [1, 2, 3]


def test_leakage_columns_cannot_affect_predictors(orders):
    extra = orders.assign(
        refund_value=[0, 999, 0], is_fraud=[0, 1, 1],
        order_id=["o1", "o2", "o3"], customer_id=["c1", "c2", "c3"],
        total_payment_last_50days=[100, 200, 300],
        num_orders_last_50days=[9, 8, 7], first_order_datetime="2020-01-01",
        mobile_verified=True, num_associated_customers=50,
    )
    builder = OrderFeatures().fit(extra, [0, 1, 1])
    pd.testing.assert_frame_equal(builder.transform(extra), builder.transform(orders))
    pipeline = make_feature_pipeline().fit(extra, [0, 1, 1])
    np.testing.assert_allclose(dense(pipeline.transform(extra)), dense(pipeline.transform(orders)))
    assert set(builder.get_feature_names_out()) == set(FEATURE_COLUMNS)


def test_row_order_index_and_input_are_preserved(orders):
    frame = orders.iloc[[2, 0, 1]].copy()
    frame.index = [9, 9, 2]
    original = frame.copy(deep=True)
    result = OrderFeatures().fit_transform(frame)
    assert result.index.tolist() == [9, 9, 2]
    assert result["order_value"].tolist() == [30, 10, 20]
    matrix = dense(make_feature_pipeline().fit_transform(frame))
    np.testing.assert_allclose(matrix[:, 0], [30, 10, 20])
    pd.testing.assert_frame_equal(frame, original)


def test_missing_values_and_all_missing_training_columns(orders):
    frame = orders.copy()
    frame["order_value"] = np.nan
    frame["order_date"] = None
    frame["payment_method"] = None
    frame.loc[0, "country_code"] = None
    frame.loc[1, "collect_type"] = " "
    pipeline = make_feature_pipeline().fit(frame)
    result = dense(pipeline.transform(frame))
    assert len(result) == len(frame)
    assert np.isfinite(result).all()
    inferred = dense(pipeline.transform(orders))
    assert inferred.shape == result.shape
    assert np.isfinite(inferred).all()


def test_unknown_categories_use_fitted_schema(orders):
    pipeline = make_feature_pipeline().fit(orders)
    unseen = orders.iloc[[0]].assign(country_code="SG", collect_type="drone", payment_method="wallet")
    result = dense(pipeline.transform(unseen))
    names = pipeline.get_feature_names_out()
    categorical = [i for i, name in enumerate(names) if name.startswith("categorical__")]
    assert result.shape[1] == len(names)
    assert np.isfinite(result).all()
    assert (result[:, categorical] == 0).all()


def test_zero_items_does_not_divide_by_zero(orders):
    frame = orders.copy()
    frame["num_items_ordered"] = 0
    result = OrderFeatures().fit_transform(frame)
    assert (result["num_items_ordered"] == 0).all()
    assert result["value_per_item"].isna().all()
    assert np.isfinite(dense(make_feature_pipeline().fit_transform(frame))).all()


def test_invalid_numbers_become_missing(orders):
    frame = orders.copy()
    frame["order_value"] = [-1, np.inf, "broken"]
    frame["num_items_ordered"] = [-2, 1.5, np.inf]
    result = OrderFeatures().fit_transform(frame)
    assert result[["order_value", "num_items_ordered", "value_per_item"]].isna().all().all()
    assert np.isfinite(dense(make_feature_pipeline().fit_transform(frame))).all()


def test_invalid_dates_are_missing_without_guessing(orders):
    frame = orders.copy()
    frame["order_date"] = ["2023-02-30", "02/03/2023", None]
    result = OrderFeatures().fit_transform(frame)
    assert result[["order_weekday", "order_month"]].isna().all().all()
    assert np.isfinite(dense(make_feature_pipeline().fit_transform(frame))).all()


def test_already_parsed_dates_are_supported(orders):
    frame = orders.copy()
    frame["order_date"] = pd.to_datetime(["2023-01-02", None, "2023-03-08"])
    result = OrderFeatures().fit_transform(frame)
    assert result.loc[0, "order_weekday"] == 0
    assert pd.isna(result.loc[1, "order_month"])


def test_nullable_values_and_invalid_categories_are_safe(orders):
    frame = orders.copy()
    frame["order_value"] = pd.Series([pd.NA, False, "2.5"], dtype="object")
    frame["num_items_ordered"] = pd.Series([pd.NA, True, "2"], dtype="object")
    frame["country_code"] = [pd.NA, 12, "???"]
    frame["collect_type"] = [pd.NA, 13, " "]
    frame["payment_method"] = [None, True, " "]
    frame["order_date"] = [pd.NA, 123, None]
    result = OrderFeatures().fit_transform(frame)
    assert result.loc[:1, ["order_value", "num_items_ordered"]].isna().all().all()
    assert result.loc[2, "value_per_item"] == pytest.approx(1.25)
    assert result[["country_code", "collect_type", "payment_method", "order_month"]].isna().all().all()
    assert np.isfinite(dense(make_feature_pipeline().fit_transform(frame))).all()


@pytest.mark.parametrize("scale", [False, True])
def test_preprocessing_is_fitted_only_on_training_rows(orders, scale):
    pipeline = make_feature_pipeline(scale_numeric=scale).fit(orders)
    numeric = pipeline["preprocess"].named_transformers_["numeric"]
    before = numeric["impute"].statistics_.copy()
    categories = [x.copy() for x in pipeline["preprocess"].named_transformers_["categorical"]["encode"].categories_]
    held_out = orders.iloc[[0, 1]].copy()
    held_out["order_value"] = [10000.0, np.nan]
    held_out["payment_method"] = ["new", None]
    result = dense(pipeline.transform(held_out))
    np.testing.assert_allclose(numeric["impute"].statistics_, before)
    assert before[0] == 20.0
    for old, new in zip(categories, pipeline["preprocess"].named_transformers_["categorical"]["encode"].categories_):
        np.testing.assert_array_equal(old, new)
    if scale:
        assert numeric["scale"].mean_[0] == 20.0
        assert result[1, 0] == pytest.approx(0)
    else:
        assert result[1, 0] == pytest.approx(20)


def test_repeated_transform_is_batch_independent(orders):
    pipeline = make_feature_pipeline().fit(orders)
    alone = dense(pipeline.transform(orders.iloc[[1]]))
    batch = dense(pipeline.transform(orders))
    np.testing.assert_allclose(alone[0], batch[1])


def test_schema_errors_are_clear_and_transform_requires_fit(orders):
    with pytest.raises(NotFittedError):
        OrderFeatures().transform(orders)
    with pytest.raises(ValueError, match="order_date"):
        OrderFeatures().fit(orders.drop(columns="order_date"))
    with pytest.raises(TypeError, match="DataFrame"):
        OrderFeatures().fit(orders.to_numpy())
    with pytest.raises(ValueError, match="duplicate column"):
        OrderFeatures().fit(pd.concat([orders, orders[["order_date"]]], axis=1))


def test_pipeline_can_be_cloned_and_serialized(orders, tmp_path):
    import joblib

    pipeline = clone(make_feature_pipeline(scale_numeric=True)).fit(orders)
    target = tmp_path / "features.joblib"
    joblib.dump(pipeline, target)
    loaded = joblib.load(target)
    np.testing.assert_allclose(dense(loaded.transform(orders)), dense(pipeline.transform(orders)))


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("column", ["order_value", "num_items_ordered"])
def test_overflowing_magnitudes_are_rejected_at_fit_and_inference(orders, scale, column):
    huge = orders.copy()
    huge[column] = [1e308, 1e308, np.nan]
    with pytest.raises(ValueError, match=f"{column} exceeds the numerical safety limit"):
        make_feature_pipeline(scale_numeric=scale).fit(huge)
    pipeline = make_feature_pipeline(scale_numeric=scale).fit(orders)
    with pytest.raises(ValueError, match="numerical safety limit"):
        pipeline.transform(huge)


@pytest.mark.parametrize("scale", [False, True])
def test_values_at_safety_limit_remain_finite(orders, scale):
    frame = orders.copy()
    frame["order_value"] = [MAX_NUMERIC_VALUE, MAX_NUMERIC_VALUE, np.nan]
    result = dense(make_feature_pipeline(scale_numeric=scale).fit_transform(frame))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("column", ["order_value", "num_items_ordered"])
@pytest.mark.parametrize("values", [
    pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    pd.to_timedelta([1, 2, 3], unit="D"),
    [1+2j, 2+0j, 3+0j],
])
def test_datetime_and_complex_numbers_are_missing(orders, column, values):
    frame = orders.copy()
    frame[column] = values
    result = OrderFeatures().fit_transform(frame)
    assert result[column].isna().all()
    assert np.isfinite(dense(make_feature_pipeline().fit_transform(frame))).all()


def test_empty_batches_have_an_actionable_error(orders):
    with pytest.raises(ValueError, match="skip empty batches"):
        make_feature_pipeline().fit(orders.iloc[:0])
    pipeline = make_feature_pipeline().fit(orders)
    with pytest.raises(ValueError, match="skip empty batches"):
        pipeline.transform(orders.iloc[:0])


def test_cleaning_style_nullable_strings_and_datetimes(orders):
    frame = orders.copy()
    for column in ["country_code", "collect_type", "payment_method"]:
        frame[column] = frame[column].astype("string")
    frame.loc[1, "payment_method"] = pd.NA
    frame["order_date"] = pd.to_datetime(["2023-01-02", "2023-02-07", "2023-03-08"])
    frame["order_value"] = frame["order_value"].astype("Float64")
    frame["num_items_ordered"] = frame["num_items_ordered"].astype("Int64")
    snapshot = frame.copy(deep=True)
    result = dense(make_feature_pipeline().fit_transform(frame))
    assert result.shape[0] == len(frame)
    assert np.isfinite(result).all()
    pd.testing.assert_frame_equal(frame, snapshot)
