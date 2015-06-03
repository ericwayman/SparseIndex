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


#test findSolutionCardinality()
def test_findSolutionCardinality_1():
    w = [1,2,3,4]
    threshold = 2.5
    assert_equals(findSolutionCardinality(w,threshold),2)


def test_findSolutionCardinality_2():
    w = [1,1,1,1,1,1]
    threshold = 0.5
    assert_equals(findSolutionCardinality(w,threshold),6)

def test_findSolutionCardinality_3():
    w = np.array([0,0,0,0])
    threshold = 0.5
    assert_equals(findSolutionCardinality(w,threshold),0)

