#! /usr/bin/env python
import sys
import numpy as np
import cvxpy as cvx
import datetime
from matplotlib import finance
from math import sqrt
import sparseIndex as si
import re

def solveCardinalityObjective(prices,error):
    """
    Solves the problem 
    for i=1:n
    max: x[i]
    subject to: ||Rx - y||_2 <= error, x>=0, sum(x)==1

    which is a lower bound approximation to the problem with objective: 
    card(x)
    R is a number of time periods by number of assets matrix of returns
    for each asset in the index.  
    y is the time series for the index returns
    (The index is assumed to be one share of each asset in the index)
    error is the treshold for the tracking error.
    """
    numAssets = prices.shape[1]
    x = cvx.Variable(numAssets)
    returns = si.pricesToReturns(prices)
    indexReturns = si.pricesToReturnsForIndex(prices)
    TrackingErrorSquared = cvx.sum_squares(returns*x -indexReturns)
    obj = cvx.Maximize(x[0])
    constraints = [x>=0,sum(x)==1,TrackingErrorSquared <=error**2]
    prob = cvx.Problem(obj,constraints)

    optimalVal = 0
    optimalSol = np.array([])
    for i in range(numAssets):
        prob.obj = cvx.Maximize(x[i])
        prob.solve(warm_start=True)
        value = prob.value
        if value > optimalVal:
            optimalVal = value
            optimalSol = x.value
    return optimalVal, np.array(optimalSol).reshape(-1,)


symbols = ['COP', 'AXP', 'RTN', 'BA', 'AAPL', 'PEP', 'NAV', 'GSK', 'MSFT',
           'KMB', 'R', 'SAP', 'GS', 'CL', 'WMT', 'GE', 'SNE', 'PFE', 'AMZN',
           'MAR', 'NVS', 'KO', 'MMM', 'CMCSA', 'SNY', 'IBM', 'CVX', 'WFC',
           'DD', 'CVS', 'TOT', 'CAT', 'CAJ', 'BAC', 'WBA', 'AIG', 'TWX', 'HD',
           'TXN', 'VLO', 'F', 'CVC', 'TM', 'PG', 'LMT', 'K', 'HMC', 'GD',
           'HPQ', 'MTU', 'XRX', 'YHOO', 'XOM', 'JPM', 'MCD', 'CSCO',
           'NOC', 'MDLZ', 'UN']
#symbols = ['AAPL','SNY']
startDate = datetime.datetime(2012, 1, 1)
endDate = datetime.datetime(2013, 1, 1)
#Each mapper has to have a copy of the entire data set?
prices = si.symbolsToPrices(symbols,startDate,endDate)
returns = si.pricesToReturns(prices)
indexReturns = si.pricesToReturnsForIndex(prices)
numAssets = returns.shape[1]
threshold = 1/(float(numAssets*10000))
regularizer = 0.02

for line in sys.stdin:
    #Haven't decided how to split the data yet, so I'll just copy this part over from sparseIndex
    #One idea is to assign an index to each symbol and have the input data be a list of indices.
    line = line.strip()
    try:
        i = symbols.index(line)
    except:
        continue
    #Solve problem
    x = cvx.Variable(numAssets)
    t = cvx.Variable()
    TrackingErrorSquared = cvx.sum_squares(returns*x - indexReturns)
    obj = cvx.Minimize( TrackingErrorSquared+t)
    baseConstraints = [x>=0,t>=0,sum(x)==1]
    constraints = baseConstraints+ [cvx.log(x[i])+cvx.log(t)>= cvx.log(regularizer)]
    prob = cvx.Problem(obj,constraints)
    prob.solve()
    card = np.sum([np.array(x.value).reshape(-1,) > threshold])
    upperBound = prob.value - t.value + regularizer*card
    optimalWeights = str(np.array(x.value).reshape(-1,))
    optimalWeights = re.sub('\n','',optimalWeights)
    #optimalVal,optimalT,card,upperBound,optimalWeights
    optimalWeights = re.sub('\[ {1,}','[',optimalWeights)
    optimalWeights = re.sub('\s+',',',optimalWeights)
    print "{0}\t{1}\t{2}\t{3}\t{4}".format(prob.value,t.value,card,upperBound,optimalWeights)
