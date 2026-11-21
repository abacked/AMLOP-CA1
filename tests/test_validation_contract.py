"""Synthetic edge cases independent of base files."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amlop_ca1 import cleaning as cl

@pytest.mark.parametrize('key', ['country_code', 'order_id', 'customer_id'])
@pytest.mark.parametrize('value', [None, '', '  ', '\t'])
def test_merge_guard_rejects_null_and_blank_keys(key, value):
    customers = pd.DataFrame({'country_code':['MY'], 'customer_id':['c1']})
    orders = pd.DataFrame({'country_code':['MY'], 'order_id':['o1']})
    labels = pd.DataFrame({'country_code':['MY'], 'order_id':['o1'], 'customer_id':['c1'], 'is_fraud':[0]})
    target = {'country_code': labels, 'order_id': orders, 'customer_id': customers}[key]
    target[key] = value
    with pytest.raises(ValueError, match='missing'):
        cl.merge_all(customers, orders, labels)

@pytest.mark.parametrize('value', [-1, np.inf, -np.inf, 'broken', 1.5])
def test_invalid_item_counts_are_missing(value):
    frame = pd.DataFrame({'num_items_ordered': [value, 0, 2]})
    cl.convert_to_numbers(frame, ['num_items_ordered'], 'orders')
    assert pd.isna(frame.loc[0, 'num_items_ordered'])
    assert frame.loc[1:, 'num_items_ordered'].tolist() == [0, 2]

def test_optional_amounts_allow_missing_and_fractional_positive_values():
    frame = pd.DataFrame({'refund_value':[-1, np.inf, 2.5, None]})
    cl.convert_to_numbers(frame, ['refund_value'], 'orders')
    assert frame.refund_value.isna().tolist() == [True, True, False, True]
    assert frame.refund_value.iloc[2] == 2.5
