'''
Test sparseIndex.py with nose
to run tests: nosetests test_sparseIndex.py
Verbose (-v): nosetests -v test_sparsIndex.py
'''

from sparseIndex import *
from nose.tools import assert_equals
import numpy as np

def test_pricesToReturns_size():
    prices = np.random.randn(20, 10)
    returns = pricesToReturns(prices)
    assert_equals(returns.shape,(prices.shape[0]-1,prices.shape[1]))

def test_pricesToReturns_values():
    prices = np.array([[100.0,110.0,121.0],[100.0,90.0,81.0],]).T
    returns = pricesToReturns(prices)
    assert np.array_equal(returns, np.array([[0.1,0.1],[-0.1,-0.1]]).T)

def test_pricesToReturns_works_with_int_arrays():
    prices = np.array([[100,110,121],[100,90,81],]).T
    returns = pricesToReturns(prices)
    assert np.array_equal(returns, np.array([[0.1,0.1],[-0.1,-0.1]]).T)


def test_pricesToReturnsForIndex_size():
    prices = np.random.randn(20, 10)
    returns = pricesToReturnsForIndex(prices)
    assert_equals(returns.shape,(prices.shape[0]-1,))

def test_pricesToReturnsForIndex_values():
    prices = np.array([[100.0,110.0,121.0],[50.0,55.0,60.5],]).T
    returns = pricesToReturnsForIndex(prices)
    assert np.array_equal(returns, np.array([0.10,0.10]).T)

