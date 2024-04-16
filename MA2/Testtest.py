import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

from plotnine import *
from mizani.formatters import percent_format
from regtabletotext import prettify_result

tidy_finance = sqlite3.connect(database="/Users/emilkolko/Downloads/tidy_finance_python.sqlite")

crsp_monthly = (pd.read_sql_query(
    sql="SELECT permno, month, ret_excess, mktcap_lag FROM crsp_monthly",
    con=tidy_finance,
    parse_dates={"month"})
)

factors_ff3_monthly = pd.read_sql_query(
  sql="SELECT month, mkt_excess FROM factors_ff3_monthly",
  con=tidy_finance,
  parse_dates={"month"}
)

beta = (pd.read_sql_query(
    sql="SELECT permno, month, beta_monthly FROM beta",
    con=tidy_finance,
    parse_dates={"month"})
)
beta_lag = (beta
  .assign(month=lambda x: x["month"]+pd.DateOffset(months=1))
  .get(["permno", "month", "beta_monthly"])
  .rename(columns={"beta_monthly": "beta_lag"})
  .dropna()
)

data_for_sorts = (crsp_monthly
  .merge(beta_lag, how="inner", on=["permno", "month"])
)

beta_portfolios = (data_for_sorts
  .groupby("month")
  .apply(lambda x: (x.assign(
      portfolio=pd.qcut(
        x["beta_lag"], q=[0, 0.5, 1], labels=["low", "high"]))
    )
  )
  .reset_index(drop=True)
  .groupby(["portfolio","month"])
  .apply(lambda x: np.average(x["ret_excess"], weights=x["mktcap_lag"]))
  .reset_index(name="ret")
)

print(beta_portfolios.head(24))

print(data_for_sorts.head(24))
