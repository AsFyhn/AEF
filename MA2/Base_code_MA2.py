import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
import tabulate

from plotnine import *
from mizani.formatters import percent_format
from regtabletotext import prettify_result

# Connecting to the database
tidy_finance = sqlite3.connect(
    database="/Users/emilkolko/Downloads/tidy_finance_python.sqlite"
)

# Reading in crsp_monthly and factors_ff3_monthly
crsp_monthly = pd.read_sql_query(
    sql="SELECT permno, month, ret_excess, mktcap, mktcap_lag FROM crsp_monthly",
    con=tidy_finance,
    parse_dates={"month"},
)

factors_ff3_monthly = pd.read_sql_query(
    sql="SELECT month, mkt_excess FROM factors_ff3_monthly",
    con=tidy_finance,
    parse_dates={"month"},
)

# Exercise 2
# Transforming the crsp_monthly dataset to contain mktcap_lag_12
mktcap_lag_12 = (
    crsp_monthly.assign(month=lambda x: x["month"] + pd.DateOffset(months=12))
    .get(["permno", "month", "mktcap"])
    .rename(columns={"mktcap": "mktcap_lag_12"})
    .dropna()
)

# Merging the datasets
data_for_sorts = crsp_monthly.merge(mktcap_lag_12, how="inner", on=["permno", "month"])

# Computing the momentum of stock i for 12-month momentum
data_for_sorts["Mom_12"] = (
    100
    * (data_for_sorts["mktcap_lag"] - data_for_sorts["mktcap_lag_12"])
    / data_for_sorts["mktcap_lag_12"]
)

# Printing the first 24 rows of the dataset to check if the data transformation was successful
print(data_for_sorts.head(24))

# Exercise 3
# Bullet 1
# Creating the decile groups for the 12-month momentum

data_for_sorts["Mom_12_decile"] = pd.qcut(data_for_sorts["Mom_12"], q=10, labels=False)

# Sorting the data by decile and month
sorted_data = data_for_sorts.sort_values(by=["Mom_12_decile", "month"])
# Bullet 2
# Calculating the equal-weighted average values of Mom_12 and mc for each of the ten portfolios
average_return = (
    sorted_data.groupby(["Mom_12_decile"])
    .apply(
        lambda x: pd.Series(
            {
                "Average_Mom_12": np.mean(x["Mom_12"]),
                "Average_mktcap": np.mean(x["mktcap"]),
            }
        )
    )
    .reset_index()
)

# rounding the values to 2 decimals
average_return = average_return.round(2)

# Specifiying the headers for the table
headers = [
    "Momentum decile",
    "Average momentum",
    "Average market cap",
]
# Printing the table
print(tabulate.tabulate(average_return, headers, tablefmt="simple", showindex=False))

# Bullet 3
