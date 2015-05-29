'''
Test multiProcessingSparseIndex.py with nose
to run tests: nosetests test_multiProcessingSparseIndex.py
Verbose (-v): nosetests -v test_multiProcessingSparseIndex.py
'''
from multiProcessingSparseIndex import *
import numpy as np
import sparseIndex as si
from nose.tools import assert_equals
from nose import with_setup

def set_up():
    return


def tear_down():
    return

@with_setup(set_up,tear_down)
def test():
    print "useless test"
    return

class Test_Task:

    def setup(self):
        print ("TestUM:setup() before each test method")
 
    def teardown(self):
        print ("TestUM:teardown() after each test method")
 
    @classmethod
    def setup_class(cls):
            #make data to feed into problems
        cls.symbols = ['COP', 'AXP', 'RTN', 'BA', 'AAPL', 'PEP', 'NAV', 'GSK', 'MSFT',
           'KMB', 'R', 'SAP', 'GS', 'CL', 'WMT', 'GE', 'SNE', 'PFE', 'AMZN',
           'MAR', 'NVS', 'KO', 'MMM', 'CMCSA', 'SNY', 'IBM', 'CVX', 'WFC',
           'DD', 'CVS', 'TOT', 'CAT', 'CAJ', 'BAC', 'WBA', 'AIG', 'TWX', 'HD',
           'TXN', 'VLO', 'F', 'CVC', 'TM', 'PG', 'LMT', 'K', 'HMC', 'GD',
           'HPQ', 'MTU', 'XRX', 'YHOO', 'XOM', 'JPM', 'MCD', 'CSCO',
           'NOC', 'MDLZ', 'UN']
        startDate = datetime.datetime(2012, 1, 1)
        endDate = datetime.datetime(2013, 1, 1)

        cls.prices = si.symbolsToPrices(cls.symbols,startDate,endDate)
        cls.returns = si.pricesToReturns(cls.prices)
        cls.indexReturns = si.pricesToReturnsForIndex(cls.prices)
        cls.numAssets = cls.returns.shape[1]
        cls.regularizer = 0.2
 
    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_Task_x_variable_length(self):
        task = Task(self.returns, self.indexReturns,self.regularizer,i=1)
        task()
        assert_equals(len(task.x.value),len(self.symbols))

    


