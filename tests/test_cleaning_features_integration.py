"""Synthetic CSV batches through the merged cleaner and feature preprocessing."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amlop_ca1.features import FEATURE_COLUMNS, make_feature_pipeline
from amlop_ca1 import cleaning as cleaner

@pytest.fixture
def batch(tmp_path):
    c = pd.DataFrame({
        'country_code':['MY','PH'], 'customer_id':['0000000001','0000000001'],
        'mobile_verified':['yes','False'], 'num_orders_last_50days':[1,2],
        'num_cancelled_orders_last_50days':[0,0], 'num_refund_orders_last_50days':[0,0],
        'total_payment_last_50days':[100,200], 'num_associated_customers':[0,0],
        'first_order_datetime':[None,'2023-01-01 00:00:00']})
    o = pd.DataFrame({'country_code':['MY','PH'], 'order_id':['0000000002','0000000002'],
        'collect_type':['SELF-PICK-UP',None], 'payment_method':[None,'Credit_Card'],
        'order_value':[10.,20.], 'num_items_ordered':[0,2], 'refund_value':[999,0],
        'order_date':['2023/Jan/02','invalid']})
    l = pd.DataFrame({'country_code':['PH','MY'], 'order_id':['0000000002']*2,
        'customer_id':['0000000001']*2, 'is_fraud':[1,0]})
    for name, frame in zip(('customers','orders','labels'), (c,o,l)):
        frame.to_csv(tmp_path / f'{name}_synthetic.csv', index=False)
    return cleaner.load_batch(tmp_path, 'synthetic')

def dense(matrix):
    return matrix.toarray() if hasattr(matrix, 'toarray') else matrix

def test_three_file_batch_keys_nullables_alignment_and_label_free_inference(batch):
    snapshots = [x.copy(deep=True) for x in batch]
    rejects = cleaner.RejectLog()
    merged = cleaner.clean_and_merge(*batch, rejects)
    assert len(merged) == 2 and len(rejects) == 0
    assert merged.country_code.tolist() == ['PH','MY']
    assert merged.is_fraud.tolist() == [1,0]
    assert str(merged.order_date.dtype).startswith('datetime64')
    assert str(merged.country_code.dtype) == 'string'
    assert merged.order_date.isna().tolist() == [True,False]
    pipeline = make_feature_pipeline().fit(merged.iloc[[0]])
    transformed = pipeline['features'].transform(merged.drop(columns=['is_fraud']))
    assert tuple(transformed.columns) == FEATURE_COLUMNS
    assert transformed.order_value.tolist() == [20,10]
    assert pd.isna(transformed.value_per_item.iloc[1])
    matrix = dense(pipeline.transform(merged.drop(columns=['is_fraud'])))
    assert len(matrix) == 2 and np.isfinite(matrix).all()
    changed = merged.assign(refund_value=1e9,is_fraud=1, num_orders_last_50days=1e6)
    np.testing.assert_allclose(dense(pipeline.transform(changed)),matrix)
    for frame, snapshot in zip(batch,snapshots):
        pd.testing.assert_frame_equal(frame,snapshot)

def test_orphan_conflicting_and_missing_keys_report_rejections(batch):
    c,o,l = [x.copy(deep=True) for x in batch]
    l = pd.concat([l,l.iloc[[0]].assign(is_fraud=0),
        l.iloc[[1]].assign(order_id='orphan'),l.iloc[[1]].assign(order_id=' ')],ignore_index=True)
    rejects = cleaner.RejectLog()
    merged = cleaner.clean_and_merge(c,o,l,rejects)
    assert merged.country_code.tolist() == ['MY']
    reasons = rejects.to_frame().reason.tolist()
    assert any('conflicting duplicate' in r for r in reasons)
    assert 'no matching order' in reasons
    assert any('missing order_id' in r for r in reasons)
    assert len(merged) + int((rejects.to_frame().table == 'labels').sum()) == len(l)

def test_missing_batch_schema_fails_explicitly(batch):
    c,o,l = batch
    with pytest.raises(ValueError, match='order_value'):
        cleaner.clean_and_merge(c,o.drop(columns='order_value'),l)

def test_unknown_categories_and_training_state_are_isolated(batch):
    merged = cleaner.clean_and_merge(*batch)
    train = merged.iloc[[0]].copy()
    test = merged.iloc[[1]].assign(payment_method='future wallet',order_value=100000)
    pipeline = make_feature_pipeline(scale_numeric=True).fit(train)
    imputer = pipeline['preprocess'].named_transformers_['numeric']['impute']
    before = imputer.statistics_.copy()
    matrix = dense(pipeline.transform(test.drop(columns='is_fraud')))
    assert matrix.shape[0] == 1 and np.isfinite(matrix).all()
    np.testing.assert_array_equal(imputer.statistics_,before)


def test_orders_only_inference_uses_the_same_predictors_as_training(batch):
    customers, orders, labels = batch
    merged = cleaner.clean_and_merge(customers, orders, labels)
    pipeline = make_feature_pipeline().fit(merged)
    snapshot = orders.copy(deep=True)
    cleaned_orders = cleaner.clean_orders(orders)
    assert 'is_fraud' not in cleaned_orders and 'customer_id' not in cleaned_orders
    assert cleaned_orders.index.tolist() == orders.index.tolist()
    # Match by composite key, rather than assuming the label-driven merge order.
    aligned = merged.set_index(cleaner.ORDER_KEY).loc[
        pd.MultiIndex.from_frame(cleaned_orders[cleaner.ORDER_KEY])
    ].reset_index()
    actual = dense(pipeline.transform(cleaned_orders))
    expected = dense(pipeline.transform(aligned))
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()
    pd.testing.assert_frame_equal(orders, snapshot)
