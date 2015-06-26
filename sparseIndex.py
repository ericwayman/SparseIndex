import numpy as np
import cvxpy as cvx
import pandas as pd
import multiprocessing as mp
import datetime
from sklearn.cluster import KMeans
from matplotlib import finance
from math import sqrt

# a better idea might be to inherit from the cvxpy.Problem class
#classes upper case methods lower case
class SparseIndexProblem(object):

    
    def __init__(self,returns,indexReturns,regularizer=1):
        self.returns = returns
        self.indexReturns = indexReturns
        self.numAssets = returns.shape[1]
        #any asset in a quantity below the threshold we consider 0
        self.threshold = 1/(float(self.numAssets*10000))
        self.regularizer = regularizer
        self.OptimalValue = None
        self.TrackingErrorSquared = None
        self.Optimalweights = None
        self.t = None
        self.UpperBound = np.float('inf')

    def solveProblem(self):
        x = cvx.Variable(self.numAssets)
        t = cvx.Variable()
        
        #optimalIndex = -1
        optimalVal = np.float('inf')
        optimalT = np.float('inf')
        optimalSol = np.array([])
        
       


        TrackingErrorSquared = cvx.sum_squares(self.returns*x -self.indexReturns )
        obj = cvx.Minimize( TrackingErrorSquared+t)
        baseConstraints = [x>=0,t>=0,sum(x)==1]
        for i in range(self.numAssets):

            constraints = baseConstraints+ [cvx.log(x[i])+cvx.log(t)>= cvx.log(self.regularizer)]
            prob = cvx.Problem(obj,constraints)
            prob.solve()
            print("solved Problem %i.  Optimal value: %s" % (i,prob.value))
            card = np.sum([np.array(x.value).reshape(-1,) > self.threshold])
            print("the cardinality of the solution is %s" % str(card))
            upperBound = prob.value - t.value + self.regularizer*card
            print("the upper bound of the solution is %s" % str(upperBound))

            #update the upper bound
            if upperBound < self.UpperBound:
                self.UpperBound = upperBound

            #update the lower bound
            if prob.value < optimalVal:
                optimalVal = prob.value
                optimalSol = x.value
                optimalT = t.value
        self.OptimalValue = optimalVal
        self.OptimalWeights = np.array(optimalSol).reshape(-1,)
        self.t = optimalT
        self.TrackingErrorSquared = self.OptimalValue - optimalT

    def _clusterReturns(self,n_clusters):
        """
        Runs k-means 
        """
        zscore = lambda x: (x - x.mean()) / x.std()
        normalizedReturns = np.apply_along_axis(zscore,0, self.returns).T
        kmeans = KMeans(init='k-means++',n_clusters = n_clusters)
        kmeans.fit(normalizedReturns)
        return kmeans.labels_

    def solveProblemWithClustering(self,n_clusters):
        x = cvx.Variable(self.numAssets)
        t = cvx.Variable()

        optimalVal = np.float('inf')
        optimalT = np.float('inf')
        optimalSol = np.array([])
        
        TrackingErrorSquared = cvx.sum_squares(self.returns*x -self.indexReturns )
        obj = cvx.Minimize( TrackingErrorSquared+t)
        constraints = [x>=0,t>=0,sum(x)==1,cvx.log(x[0])+cvx.log(t)>= cvx.log(self.regularizer)]
        prob = cvx.Problem(obj,constraints)
        labels = self._clusterReturns(n_clusters)
        #loop through clusters, solve problem in each cluster
        for index in range(n_clusters):
            print("In cluster %i" % index)
            cluster = np.where(labels==index)[0]
            for i in cluster:
                #warm start solve remaining problems in cluster
                #update constraints
                prob.constraints[-1] = cvx.log(x[i])+cvx.log(t)>= cvx.log(self.regularizer)
                prob.solve(warm_start= True)
                print("solved Problem %i.  Optimal value: %s" % (i,prob.value))
                optimalVal = prob.value
                optimalSol = x.value
                optimalT = t.value

            if prob.value < optimalVal:
                #optimalIndex = i
                optimalVal = prob.value
                optimalSol = x.value
                optimalT = t.value
        self.OptimalValue = optimalVal
        self.OptimalWeights = np.array(optimalSol).reshape(-1,)
        self.t = optimalT
        self.TrackingErrorSquared = self.OptimalValue - optimalT


    def printSummary(self):
        print "with regularizer %f: " % self.regularizer
        print "The optimal weights are given by: \n %s" % str(self.OptimalWeights)
        print "The tracking error is given by %s" % str(sqrt(self.TrackingErrorSquared)) 
        print "The optimal Solution (lower bound) is given by %s" % str(self.OptimalValue)
        print "The upper bound for the solution is given by:"
        print str(self.UpperBound)

#helper functions
def symbolsToPrices(symbols,startDate,endDate):
    """
    From a list of stock symbols and a range of dates, returns a prices by symbols numpy array
    listing the opening prices for the stocks in the given date range.
    """
    #add check to account for missing values in data
    quotes = [finance.quotes_historical_yahoo_ochl(symbol, startDate, endDate,asobject = True).open for symbol in symbols]
    print "prices shape:"
    print np.array(quotes).T.shape
    return np.array(quotes).T

def pricesToReturns(prices):
    """Takes a numpy array of prices by assets and returns a numpy array of 
    percent returns for each period 
    """
    #convert array to array of floats to avoid rounding errors if int
    prices = prices.astype(float)
    return (prices[1:,:] - prices[:-1,:])/prices[:-1,:]

def pricesToReturnsForIndex(prices):
    """Takes a nxm numpy array (n time periods by m assets) of prices
    and returns a numpy array of returns for the index containing equal shares
    (not dollar value) of each asset
    """
    indexPrices = prices.sum(axis=1)
    indexPrices = indexPrices.astype(float)
    return (indexPrices[1:]-indexPrices[:-1])/indexPrices[:-1]


def findSolutionCardinality(w,threshold):
    '''
    counts the number of entries in w larger than the threshold
    '''
    return np.array([np.array(w)> threshold]).astype(int).sum()

#test this
# def errorSquared(R,x,y):
#     '''
#     Given a nxm matrix R, nx1 vector x and mx1 vector
#     '''
#     return np.linalg.norm(np.dot(R,x) - y)**2


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
    returns = pricesToReturns(prices)
    indexReturns = pricesToReturnsForIndex(prices)
    TrackingErrorSquared = cvx.sum_squares(returns*x -indexReturns)
    obj = cvx.Maximize(x[0])
    constraints = [x>=0,sum(x)==1,TrackingErrorSquared <=error**2]
    prob = cvx.Problem(obj,constraints)

    optimalVal = 0
    optimalSol = np.array([])
    for i in range(numAssets):
        prob.obj = cvx.Maximize(x[i])
        prob.solve(warm_start= True)
        value = prob.value
        #should this be value < optimalVal
        if value > optimalVal:
            optimalVal = value
            optimalSol = x.value
    return optimalVal, np.array(optimalSol).reshape(-1,)

if __name__ == "__main__":
    #symbols = ['AAPL','SNY']
    symbols = ['COP', 'AXP', 'RTN', 'BA', 'AAPL', 'PEP', 'NAV', 'GSK', 'MSFT',
       'KMB', 'R', 'SAP', 'GS', 'CL', 'WMT', 'GE', 'SNE', 'PFE', 'AMZN',
       'MAR', 'NVS', 'KO', 'MMM', 'CMCSA', 'SNY', 'IBM', 'CVX', 'WFC',
       'DD', 'CVS', 'TOT', 'CAT', 'CAJ', 'BAC', 'WBA', 'AIG', 'TWX', 'HD',
       'TXN', 'VLO', 'F', 'CVC', 'TM', 'PG', 'LMT', 'K', 'HMC', 'GD',
       'HPQ', 'MTU', 'XRX', 'YHOO', 'XOM', 'JPM', 'MCD', 'CSCO',
       'NOC', 'MDLZ', 'UN']
    startDate = datetime.datetime(2012, 1, 1)
    endDate = datetime.datetime(2013, 1, 1)
    prices = symbolsToPrices(symbols,startDate,endDate)
    print("data loaded")
    returns = pricesToReturns(prices)
    indexReturns = pricesToReturnsForIndex(prices)
    regularizer = .02
    problem = SparseIndexProblem(returns,indexReturns,regularizer)
    problem.solveProblem()
    #n_clusters = 4
    #problem.solveProblemWithClustering(n_clusters)
    problem.printSummary()
    print np.array(symbols)[problem.OptimalWeights > problem.threshold]
    cardSol = solveCardinalityObjective(prices,sqrt(problem.TrackingErrorSquared))
    print "For the cardinality objective function:"
    print("The optimal value is given by %s:" % str(cardSol[0]))
    print("The optimal solution is given by: ")
    print  str(cardSol[1])
