import pandas as pd 
import numpy as np
import yfinance as yf
from datetime import datetime as dt
from plotnine import *
from mizani.formatters import percent_format

# define list of tickers
tickerlist = ['UNH', 'MSFT', 'GS', 'HD', 'CAT', 'CRM', 'MCD',
               'V', 'AMGN', 'TRV', 'AXP', 'BA', 'HON', 'JPM',
                 'IBM', 'AAPL', 'AMZN', 'JNJ', 'PG', 'CVX',
                   'MRK', 'DIS', 'NKE', 'MMM', 'KO', 'WMT', 
                   'DOW', 'CSCO', 'INTC', 'VZ']

# inputdf = yf.download(tickers=tickerlist,period='25y')  
# inputdf.to_pickle(r'/Users/asbjornfyhn/Desktop/Emp Fin/AEF/MA1/DJ_comp.pkl')
# load data
inputdf = pd.read_pickle(r'MA1/DJ_comp.pkl')
emptyCols = inputdf.columns[inputdf.isna().any()].values
inputdf = inputdf.drop(columns=emptyCols)
df = (inputdf.stack()
      .stack()
      .reset_index()
      .rename(columns={'Date':'date','level_1':'ticker','level_2':'variable',0:'value'}))