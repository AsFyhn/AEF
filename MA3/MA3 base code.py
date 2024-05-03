import pandas as pd
import numpy as np
import sqlite3

from plotnine import *
from mizani.formatters import percent_format
from itertools import product
from scipy.stats import expon
from scipy.optimize import minimize


#Connecting to the database
tidy_finance = sqlite3.connect(
    database=f"/Users/emilkolko/Downloads/tidy_finance_python.sqlite"
)

#Reading in crsp_monthly dataset
crsp_monthly = (pd.read_sql_query(
    sql="SELECT permno, month, ret_excess FROM crsp_monthly",
    con=tidy_finance,
    parse_dates={"month"})
)

#Dropping all stocks before 1962
crsp_monthly = crsp_monthly.query("month >= '1962-01-01'")

#Dropping all stocks after 2020
crsp_monthly = crsp_monthly.query("month < '2021-01-01'")

#Dropping all stocks with missing values
crsp_monthly = crsp_monthly.groupby("permno").filter(lambda x: x.shape[0] == 708)

#Summarizing the table and showing how many stocks the investment universe consists of

print(crsp_monthly.groupby("permno").size().shape[0])

#Summarizing the table with mean return
summary_stats = crsp_monthly.groupby('permno').agg(
    mean_return=('ret_excess', 'mean'),      # Mean of 'return' per 'permno'
)

# Calculate the average of excess return across all stocks
average_mean_return = summary_stats['mean_return'].mean()








 


