import multiprocessing
import numpy as np
import cvxpy as cvx
import pandas as pd
import datetime
from matplotlib import finance
from math import sqrt
import sparseIndex as si
import pdb
#local imports


class Consumer(multiprocessing.Process):
    #each Consumer represents a process associated with task_queues and result_queues
    #takes a task from the task_queue and send the answer to the result queue_

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            
            #otherwise solve the next task.  This is where the problem is solved.
            #print '%s: %s' % (proc_name, next_task)

            answer = next_task()
            #print answer
            self.task_queue.task_done()
            #print answer
            self.result_queue.put(answer)
        return


class Task(cvx.Problem):
    #is it possible to make each task a copy of a solved problem to make use of warm starts
    """
    A task object represents a problem of the form:
    min ||Rx -y||_2^2 + t
    subject to: x_i \geq \lambda/t, x \geq 0, 1^tx = 1, t \geq 0
    R is the matrix of asset returns (time periods by assets)
    y is the time series of index returns
    lambda is the regularition factor
    min over all i for these problems is our final solution
    So we solve each problem independently in parallel
    """

    def __init__(self, returns, indexReturns,regularizer,i):
    #initiate a problem with data
        numAssets = returns.shape[1]   
        self.x = cvx.Variable(numAssets)
        self.index = i
        self.t=cvx.Variable()
        baseConstraints = [self.x>=0,self.t>=0,sum(self.x)==1]
        #should have a check to see if i is in the range of num of assets
        constraints = baseConstraints+ [cvx.log(self.x[i])+cvx.log(self.t)>= cvx.log(regularizer)]
        TrackingErrorSquared = cvx.sum_squares(returns*self.x -indexReturns )
        obj = cvx.Minimize( TrackingErrorSquared+self.t)
        cvx.Problem.__init__(self,obj,constraints)


    def __call__(self):
    #call solver on problem   
        self.solve()
        #answer = solutionQueue.Solution(self.index,self.x,self.value)
        #or return (self.value, *solution object associated with self*)
        return (self.value, self.x.value, self.t.value)
    def __str__(self):
    #call summary giving a string reprsentation of the solution
        return 'Add string summary'


if __name__ == '__main__':
    #make data to feed into problems
    symbols = ['COP', 'AXP', 'RTN', 'BA', 'AAPL', 'PEP', 'NAV', 'GSK', 'MSFT',
       'KMB', 'R', 'SAP', 'GS', 'CL', 'WMT', 'GE', 'SNE', 'PFE', 'AMZN',
       'MAR', 'NVS', 'KO', 'MMM', 'CMCSA', 'SNY', 'IBM', 'CVX', 'WFC',
       'DD', 'CVS', 'TOT', 'CAT', 'CAJ', 'BAC', 'WBA', 'AIG', 'TWX', 'HD',
       'TXN', 'VLO', 'F', 'CVC', 'TM', 'PG', 'LMT', 'K', 'HMC', 'GD',
       'HPQ', 'MTU', 'XRX', 'YHOO', 'XOM', 'JPM', 'MCD', 'CSCO',
       'NOC', 'MDLZ', 'UN']
    startDate = datetime.datetime(2012, 1, 1)
    endDate = datetime.datetime(2013, 1, 1)

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    #results = solutionQueue.SolutionQueue()
    results = multiprocessing.Queue()

    prices = si.symbolsToPrices(symbols,startDate,endDate)
    returns = si.pricesToReturns(prices)
    indexReturns = si.pricesToReturnsForIndex(prices)
    numAssets = returns.shape[1]
    #threshold to determine of weights are nonzero
    #threshold = 1.0/(100.0*float(numAssets))
    threshold = .02
    print "threshold: %f" % threshold
    # Start consumers
    numConsumers = multiprocessing.cpu_count() * 2
    print 'Creating %d consumers' % numConsumers
    #create consumer processes
    consumers = [ Consumer(tasks, results)
                  for i in xrange(numConsumers) ]

    
    # Enqueue jobs
    numJobs = numAssets
    #number of jobs = number of assets

    regularizer = .005
    for i in range(numJobs):
        #add Tasks to task queue 
        #each 'task is a cvx.Minimize object'
        #here is where we put problems in queue
        tasks.put(Task(returns, indexReturns,regularizer,i))

    
    # Add a poison pill for each consumer
    for i in range(numConsumers):
        tasks.put(None)
    
    
     #start each process
    for w in consumers:
        w.start()

    # Wait for all of the tasks to finish
    tasks.join()
    
    print "start printing results"
    # Start printing results
    currentMin = np.float('inf')
    solution = None
    currentUpperBound = np.float('inf')
    while numJobs:
        result = results.get()
        if result[0] < currentMin:
            currentMin = result[0]
            solution = result
        errorSquared = result[0]-result[2]
        uB= errorSquared + regularizer*si.findSolutionCardinality(result[1],threshold)
        if uB < currentUpperBound:
            currentUpperBound = uB
        print result[0]
        numJobs -= 1

    print "the optimal value: %f" %currentMin
    print "the upper bound: %f" %currentUpperBound
    print "the cardinatliy of the solution: %f" %si.findSolutionCardinality(solution[1],threshold)
    print "tracking error: %f" %sqrt(solution[0] - solution[2])
    optimalWeights = np.array(solution[1]).reshape(-1,)
    nonzeroWeights = optimalWeights[optimalWeights>threshold]
    print "The (nonzero) optimal weights: \n %s" %nonzeroWeights
    #pdb.set_trace()
    print "The corresponding symbols \n: %s" %np.array(symbols)[optimalWeights>threshold]

