import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

from plotnine import *
from mizani.formatters import percent_format
from regtabletotext import prettify_result

#Connecting to the database
tidy_finance = sqlite3.connect(database="/Users/emilkolko/Downloads/tidy_finance_python.sqlite")

#Reading in crsp_monthly and factors_ff3_monthly
crsp_monthly = (pd.read_sql_query(
    sql="SELECT permno, month, ret_excess, mktcap FROM crsp_monthly",
    con=tidy_finance,
    parse_dates={"month"})
)

factors_ff3_monthly = pd.read_sql_query(
  sql="SELECT month, mkt_excess FROM factors_ff3_monthly",
  con=tidy_finance,
  parse_dates={"month"}
)

#Transforming the crsp_monthly dataset to contain mktcap_lag_12
mktcap_lag_12 = (crsp_monthly
  .assign(month=lambda x: x["month"]+pd.DateOffset(months=12))
  .get(["permno", "month", "mktcap"])
  .rename(columns={"mktcap": "mktcap_lag_12"})
  .dropna()
)

#Merging the datasets so they contain both mktcap and mktcap_lag_12
data_for_sorts = (crsp_monthly
  .merge(mktcap_lag_12, how="inner", on=["permno", "month"])
)
#Printing the first 24 rows of the dataset to check if the data transformation was successful
print(data_for_sorts.head(24))