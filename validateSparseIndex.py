from sparseIndex import *
import cvxpy as cvx
import pandas as pd
import multiprocessing as mp
import numpy as np
import datetime
from sklearn.cluster import KMeans
from matplotlib import finance
from math import sqrt


#helper functions
#most of these should probably be moved into the sparseIndex class
def trackingError(R,x,y):
    '''
    given a mxn matrix R a nx1 vector x and a mx1 vector returns the 
    tracking error ||Rx - y||_2
    '''
    return np.linalg.norm(np.dot(R,x)-y)

def relativeTrackingError(R,x,y):
    '''
    given a mxn matrix R a nx1 vector x and a mx1 vector returns the 
    relative tracking error ||Rx - y||_2/||y||_2
    '''
    return trackingError(R,x,y)/np.linalg.norm(y)

#should load data into a sql database first and have these functions pull from the
#data base rather than online
def generateReturns(symbols,startDate,endDate):
    '''
    Given symbols- a list of stock ticker symbols and 
    date time objects start date and end date,
    returns a matrix of returns with rows indexed by time and columns indexed by assets
    '''
    #start date must precede end date
    if endDate < startDate:
        raise ValueError("Start Date must precede end Date")
    prices = symbolsToPrices(symbols,startDate,endDate)
    return pricesToReturns(prices)

def generateIndexReturns(symbols, startDate,endDate):
    if endDate < startDate:
        raise ValueError("Start Date must precede end Date")
    prices = symbolsToPrices(symbols,startDate,endDate)
    return pricesToReturnsForIndex(prices)

def trainProblem(returns,indexReturns,regularizer):
    '''
    Given an nxm numpy array (n time periods by m assets) of returns,
    an nx1 vector of index returns, and a positive real number regularizer
    returns a solved SparseIndexProblem object
    '''
    problem = SparseIndexProblem(returns,indexReturns,regularizer)
    #update to solve with multiProcesssing instead
    problem.solveProblem()
    return problem



def validateDataOnRange(symbols,regularizer,startDate, endDate,trainingLength,validationLength):
    '''
    symbols is a list of string names for stock in the index
    regularizer is the float for the penalty parameter
    startDate and endDate are datetime.date objects 
    trainingLength and validationLength are datetime.timedelta objects
    representing the length of time for the training and validation periods
    The problem is trained on the interval [startDate,startDate+trainingLength)
    then validated on [startDate+trainingLength,startDate+trainingLength+validationLength)
    the training period is then shifted by validationLength and the model is retrained on
    [startDate+validationLength,startDate+trainingLength+validationLength) and validated on 
    the next period and so on until valided on [endDate-validationLength, endDate]
    '''
    x = startDate
    relErrorlist = []
    while x + trainingLength + validationLength <= endDate:
        #initialize stard date and end date for the training period
        sd = x
        ed = x +trainingLength
        #generate return data
        returns = generateReturns(symbols,sd,ed)
        #generate returns for the index.  This should be modified later
        indexReturns = generateIndexReturns(symbols,sd,ed)
        #train problem on data
        solvedProblem = trainProblem(returns,indexReturns,regularizer):
        #validate problem 
        testReturns = generateReturns(symbols,sd,ed)
        testIndexReturns = generateIndexReturns(symbols,ed,ed+validationLength)
        #record relative tracking error and append to error list
        relativeError = relativeTrackingError(testReturns,solvedProblem.Optimalweights,testIndexReturns)
        relErrorlist.append(relativeError)
        #increment date
        x = x + validationLength
    return relErrorlist


if __name__ == "__main__":
    '''
    Train on 12 months test on next month. 
    Use a rolling horizon to shift training data by one month each iteration
    We're interested in the relative tracking error:
    ||Rx -y||_2 / ||y||_2
    '''
    #?nice way to iterate through date time object by month?

