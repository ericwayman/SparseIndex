import numpy as np
import cvxpy as cvx
import pandas as pd

# a better idea might be to inherit from the cvxpy.Problem class
class SparseIndexProblem(object):

    
    def __init__(self,returns,indexReturns,regularizer=1):
        self.returns = returns
        self.indexReturns = indexReturns
        self.numAssets = returns.shape[1]
        self.regularizer = regularizer
        self.OptimalValue = np.float('nan')
        self.TrackingErrorSquared = np.float('nan')
        self.Optimalweights = np.array([])
        self.t = np.float('nan')
    
    #override solve
    def SolveProblem(self):
        x = cvx.Variable(self.numAssets)
        t = cvx.Variable()
        constraints = [x>=0,sum(x)==1]
        
        #optimalIndex = -1
        optimalVal = np.float('inf')
        optimalT = np.float('inf')
        optimalSol = np.array([])
        for i in range(self.numAssets):
            trackingError = cvx.sum_squares(self.returns*x -self.indexReturns )
            obj = cvx.Minimize( trackingError+t)
            constraints = [x>=0,t>=0,sum(x)==1,cvx.log(x[i])+cvx.log(t)>= cvx.log(self.regularizer)]
            prob = cvx.Problem(obj,constraints)
            prob.solve(warm_start =True)
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
        print "The tracking error squared is given by %s" % str(self.TrackingErrorSquared) 
        print "The optimal Solution (lower bound) is given by %s" % str(self.OptimalValue)
        #upperBound = self.TrackingErrorSquared + self.regularizer*np.sum([self.Optimalweights < .00000001])
        #print "The upper bound for optimal solution is given by %s" % str(upperBound)

#helper functions
def PricesToReturns(prices):
    """Takes a numpy array of prices and returns a numpy array of 
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

if __name__ == "__main__":
    df =pd.read_csv('a1.csv')
    prices = df.as_matrix()
    returns = PricesToReturns(prices)
    indexReturns = PricesToReturnsForIndex(prices)
    problem = SparseIndexProblem(returns,indexReturns,1)
    problem.SolveProblem()
    problem.PrintSummary()
