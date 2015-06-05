#! /usr/bin/env python
import sys
import numpy as np
import cvxpy as cvx
import datetime
from matplotlib import finance
from math import sqrt
import sparseIndex as si
import ast
import re

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
#Each reducer has to have a copy of the entire data set?
prices = si.symbolsToPrices(symbols,startDate,endDate)
returns = si.pricesToReturns(prices)
indexReturns = si.pricesToReturnsForIndex(prices)
numAssets = returns.shape[1]
threshold = 1/(float(numAssets*10000))
optimalVal = np.float('inf')
optimalT = np.float('inf')
optimalWeights = ""

for line in sys.stdin:
    line = line.strip()
    #optimalVal,optimalT,card,upperBound
    parts = line.split('\t')
    [val,t,card,bound] = [float(x) for x in parts[:-1]]
    weights = parts[-1]
#    weights = re.sub('\[ {1,}','[',weights)
#    weights = re.sub('\s+',',',weights)
    weights = ast.literal_eval(weights)
    weights = np.array(weights)
    if val < optimalVal:
        optimalVal = val
        optimalT = t
        optimalWeights = weights
        
TrackingErrorSquared = optimalVal - optimalT
print np.array(symbols)[optimalWeights > threshold]
cardSol = si.solveCardinalityObjective(prices, sqrt(TrackingErrorSquared))
print "For the cardinality objective function:"
print("The optimal value is given by %s:" % str(cardSol[0]))
print("The optimal solution is given by: ")
print  str(cardSol[1])

