import numpy as np
import cvxpy as cvx
import pandas as pd
import datetime
from sklearn.cluster import KMeans
from matplotlib import finance
from math import sqrt

# a better idea might be to inherit from the cvxpy.Problem class
class SparseIndexProblem(object):

    
    def __init__(self,returns,indexReturns,regularizer=1):
        self.returns = returns
        self.indexReturns = indexReturns
        self.numAssets = returns.shape[1]
        self.regularizer = regularizer
        #use none instead for these uninitialized values?
        self.OptimalValue = None
        self.TrackingErrorSquared = None
        self.Optimalweights = None
        self.t = None
        #self.OptimalValue = np.float('nan')
        #self.TrackingErrorSquared = np.float('nan')
        #self.Optimalweights = np.array([])
        #self.t = np.float('nan')
    


    def SolveProblem(self):
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
            if prob.value < optimalVal:
                #optimalIndex = i
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


    def PrintSummary(self):
        print "with regularizer %f: " % self.regularizer
        print "The optimal weights are given by: \n %s" % str(self.OptimalWeights)
        print "The tracking error is given by %s" % str(sqrt(self.TrackingErrorSquared)) 
        print "The optimal Solution (lower bound) is given by %s" % str(self.OptimalValue)
        #upperBound = self.TrackingErrorSquared + self.regularizer*np.sum([self.Optimalweights < .00000001])
        #print "The upper bound for optimal solution is given by %s" % str(upperBound)

#helper functions

def SymbolsToPrices(symbols,startDate,endDate):
    """
    From a list of stock symbols and a range of dates, returns a prices by symbols numpy array
    listing the opening prices for the stocks in the given date range.
    """
    #add check to account for missing values in data
    quotes = [finance.quotes_historical_yahoo_ochl(symbol, startDate, endDate,asobject = True).open for symbol in symbols]
    return np.array(quotes).T

def PricesToReturns(prices):
    """Takes a numpy array of prices by assets and returns a numpy array of 
    percent returns for each period 
    """
    return (prices[1:,:] - prices[:-1,:])/prices[:-1,:]

def PricesToReturnsForIndex(prices):
    """Takes a nxm numpy array (n time periods by m assets) of prices
    and returns a numpy array of returns for the index containing equal shares
    (not dollar value) of each asset
    """
    indexPrices = prices.sum(axis=1)
    return (indexPrices[1:]-indexPrices[:-1])/indexPrices[:-1]


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
    returns = PricesToReturns(prices)
    indexReturns = PricesToReturnsForIndex(prices)
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
    #df =pd.read_csv('a1.csv')
    #prices = df.as_matrix()
    prices = SymbolsToPrices(symbols,startDate,endDate)
    print("data loaded")
    returns = PricesToReturns(prices)
    indexReturns = PricesToReturnsForIndex(prices)
    problem = SparseIndexProblem(returns,indexReturns,.005)
    problem.SolveProblem()
    #n_clusters = 4
    #problem.solveProblemWithClustering(n_clusters)
    problem.PrintSummary()
    cardSol = solveCardinalityObjective(prices,sqrt(problem.TrackingErrorSquared))
    print "For the cardinality objective function:"
    print("The optimal value is given by %s:" % str(cardSol[0]))
    print("The optimal solution is given by %s: " % str(cardSol[1]))
