"""Synthetic batch contract for PR #13 and feature preprocessing.

Once cleaning.py is merged these tests run automatically. Until then set
AMLOP_CLEANING_MODULE to the absolute path of a reviewed cleaning.py checkout.
An explicit invalid path fails rather than silently skipping integration.
"""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amlop_ca1.features import FEATURE_COLUMNS, make_feature_pipeline

@pytest.fixture(scope='module')
def cleaner():
    configured = os.environ.get('AMLOP_CLEANING_MODULE')
    path = Path(configured) if configured else Path(__file__).resolve().parents[1] / 'src/amlop_ca1/cleaning.py'
    if not path.is_file():
        if configured:
            pytest.fail(f'Configured cleaning module does not exist: {path}')
        pytest.skip('PR #13 is unmerged; set AMLOP_CLEANING_MODULE to test its checkout')
    spec = importlib.util.spec_from_file_location('reviewed_cleaning', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def batch(cleaner, tmp_path):
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

def test_three_file_batch_keys_nullables_alignment_and_label_free_inference(cleaner,batch):
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

def test_orphan_conflicting_and_missing_keys_report_rejections(cleaner,batch):
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

def test_missing_batch_schema_fails_explicitly(cleaner,batch):
    c,o,l = batch
    with pytest.raises(ValueError, match='order_value'):
        cleaner.clean_and_merge(c,o.drop(columns='order_value'),l)

def test_unknown_categories_and_training_state_are_isolated(cleaner,batch):
    merged = cleaner.clean_and_merge(*batch)
    train = merged.iloc[[0]].copy()
    test = merged.iloc[[1]].assign(payment_method='future wallet',order_value=100000)
    pipeline = make_feature_pipeline(scale_numeric=True).fit(train)
    imputer = pipeline['preprocess'].named_transformers_['numeric']['impute']
    before = imputer.statistics_.copy()
    matrix = dense(pipeline.transform(test.drop(columns='is_fraud')))
    assert matrix.shape[0] == 1 and np.isfinite(matrix).all()
    np.testing.assert_array_equal(imputer.statistics_,before)
