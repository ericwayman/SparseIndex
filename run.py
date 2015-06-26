#! /usr/bin/env python
from bottle import route, run, template, request
import numpy as np
import cvxpy as cvx
import pandas as pd
import multiprocessing as mp
import datetime
from sklearn.cluster import KMeans
from matplotlib import finance
from math import sqrt
from sparseIndex import *
#python run.py
#http://localhost:9000/?symbols=AAPL,SNY
#http://localhost:9000/?symbols=COP,AXP,RTN,BA,AAPL,PEP,NAV,GSK,MSFT,KMB,R,SAP,GS,CL,WMT,GE,SNE,PFE,AMZN,MAR,NVS,KO,MMM,CMCSA,SNY,IBM,CVX,WFC,DD,CVS,TOT,CAT,CAJ,BAC,WBA,AIG,TWX,HD,TXN,VLO,F,CVC,TM,PG,LMT,K,HMC,GD,HPQ,MTU,XRX,YHOO,XOM,JPM,MCD,CSCO,NOC,MDLZ,UN


@route('/')
def index():
    symbols = request.query.symbols.split(',')
    startDate = datetime.datetime(2012, 1, 1)
    endDate = datetime.datetime(2013, 1, 1)
    prices = symbolsToPrices(symbols,startDate,endDate)
    returns = pricesToReturns(prices)
    indexReturns = pricesToReturnsForIndex(prices)
    problem = SparseIndexProblem(returns,indexReturns,.02)
    problem.solveProblem()
    problem.printSummary()
    results = np.array(symbols)[problem.OptimalWeights > problem.threshold]
    d = {}
    template_str = '<b>Stocks chosen:<b>'
    for i,stock in enumerate(results):
        d['s'+str(i)] = stock
        template_str += ' {{s' + str(i) + '}}'    
    print template_str
    if len(d) == 0:
        return 'No stocks chosen'
    else:
        return template(template_str,d)

run(host='localhost', port=9000)
