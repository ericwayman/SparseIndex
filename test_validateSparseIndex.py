'''
Test validateSparseIndex.py with nose
to run tests: nosetests test_validateSparseIndex.py
Verbose (-v): nosetests -v test_validateSparseIndex.py
'''
from validateSparseIndex import *
from nose.tools import assert_equals
import numpy as np
from math import sqrt


def test_TrackingError_1():
    R = np.array([[1,1],[1,1]])
    x = [1,1]
    y = [5,-2]
    #Rx - y =[-3,4]
    assert_equals(trackingError(R,x,y),5.0)


def test_TrackingError_2():
    R = np.array([[1,0],[0,1]])
    x = [1,1]
    y = [2,2]
    #Rx - y =[-1,-1]
    assert_equals(trackingError(R,x,y),sqrt(2)) 

def test_relativeTrackingError_1():
    R = np.array([[2,0],[1,1]])
    x = [1,1]
    y = [1,1]
    #Rx - y = [1,1]
    assert_equals(relativeTrackingError(R,x,y),1)

def test_relativeTrackingError_int_y_norm():
    R = np.array([[3,0],[3,0]])
    x = [1,1]
    y = [3,4] #||y|| = 5
    #Rx-y = [0,-1]
    assert_equals(relativeTrackingError(R,x,y),.2)