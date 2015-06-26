#script to pull return data from yahoo and store in a SQl database
import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import create_engine
from matplotlib import finance
import pdb

#helper functions
def symbolsToPriceDict(symbols,startDate,endDate):
    """
    From a list of stock symbols and a range of dates, returns a prices by symbols numpy array
    listing the opening prices for the stocks in the given date range.
    """
    #add check to account for missing values in data
    quotes = [list(finance.quotes_historical_yahoo_ochl(symbol, startDate, endDate,asobject = True).open) for symbol in symbols]
    return dict(zip(symbols,quotes))


if __name__ == "__main__":
    disk_engine = create_engine('sqlite:///returnData.db')
    symbols = ['COP', 'AXP', 'RTN', 'BA', 'AAPL', 'PEP', 'NAV', 'GSK', 'MSFT',
       'KMB', 'R', 'SAP', 'GS', 'CL', 'WMT', 'GE', 'SNE', 'PFE', 'AMZN',
       'MAR', 'NVS', 'KO', 'MMM', 'CMCSA', 'SNY', 'IBM', 'CVX', 'WFC',
       'DD', 'CVS', 'TOT', 'CAT', 'CAJ', 'BAC', 'WBA', 'AIG', 'TWX', 'HD',
       'TXN', 'VLO', 'F', 'CVC', 'TM', 'PG', 'LMT', 'HMC', 'GD',
       'HPQ', 'MTU', 'XRX', 'YHOO', 'XOM', 'MCD', 'CSCO',
       'NOC', 'MDLZ','ORCL','INTC','BP','EBAY']
    startDate = dt.datetime(2004, 1, 1)
    endDate = dt.datetime(2014, 1, 1)
    #create price matrix
    prices = symbolsToPriceDict(symbols,startDate,endDate)

    #pdb.set_trace()
    #find list of dates for trading days.  initially we'll assume all assets trade on all days
    #so we pull dates frome the first symbol
    #in the future we need to find all trading dates and fill in missing data for symbols with misssing days
    dates = list(finance.quotes_historical_yahoo_ochl(symbols[0], startDate, endDate,asobject = True).date)
    df = pd.DataFrame(data = prices, columns = symbols, index = dates)
    df.to_sql('returns', disk_engine)

