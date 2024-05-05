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

# Pivot the data to have stocks as columns and months as rows
returns_pivot = crsp_monthly.pivot(index='month', columns='permno', values='ret_excess')

# Calculate the expected returns (mean excess returns per stock) as a vector
mu = returns_pivot.mean(axis=0).values

# Calculate the covariance matrix of excess returns
sigma = returns_pivot.cov().values

# Calculate the minimum variance portfolio
# First we create a vector of ones with the same length as the number of stocks
n_industries = crsp_monthly.groupby("permno").size().shape[0]
print(n_industries)
# Next we calculate the inverse of the covariance matrix and multiply it with the vector of ones
w_mvp = np.linalg.inv(sigma) @ np.ones(n_industries)
# Finally we normalize the weights so they sum to one
w_mvp = w_mvp/w_mvp.sum()

print(returns_pivot)

print(mu)

print(sigma)

print(w_mvp)


# n_industries = crsp_monthly.shape[1]

# mu = np.array(crsp_monthly.mean()).T
# sigma = np.array(crsp_monthly.cov())
# w_mvp = np.linalg.inv(sigma) @ np.ones(n_industries)
# w_mvp = w_mvp/w_mvp.sum()

# weights_mvp = pd.DataFrame({
#   "Industry": crsp_monthly.columns.tolist(),
#   "Minimum variance": w_mvp
# })
# print(weights_mvp.round(3))

# print(mu)
# print(sigma)



# Writing a function that computes the optmial weights
def compute_efficient_weight(sigma, 
                             mu, 
                             gamma=4, 
                             beta=0,
                             w_prev=np.ones(sigma.shape[1])/sigma.shape[1]):
    """Compute efficient portfolio weights."""
    
    n = sigma.shape[1]
    iota = np.ones(n)
    sigma_processed = sigma+(beta/gamma)*np.eye(n)
    mu_processed = mu+beta*w_prev

    sigma_inverse = np.linalg.inv(sigma_processed)

    w_mvp = sigma_inverse @ iota
    w_mvp = w_mvp/np.sum(w_mvp)
    w_opt = w_mvp+(1/gamma)*\
        (sigma_inverse-np.outer(w_mvp, iota) @ sigma_inverse) @ mu_processed
        
    return w_opt

w_efficient = compute_efficient_weight(sigma, mu)

print(w_efficient)


gammas = [2, 4, 8, 20]
betas = 20*expon.ppf(np.arange(1, 100)/100, scale=1)

transaction_costs = (pd.DataFrame(
    list(product(gammas, betas)), 
    columns=["gamma", "beta"]
  )
  .assign(
    weights=lambda x: x.apply(lambda y:
      compute_efficient_weight(
        sigma, mu, gamma=y["gamma"], beta=y["beta"]/10000, w_prev=w_mvp), 
      axis=1
    ),
    concentration=lambda x: x["weights"].apply(
      lambda x: np.sum(np.abs(x-w_mvp))
    )
  )
)

rebalancing_plot = (
    ggplot(transaction_costs, 
           aes(x="beta", y="concentration",
               color="factor(gamma)", linetype="factor(gamma)")) +
    geom_line() +
    guides(linetype=None) +
    labs(x="Transaction cost parameter", y="Distance from MVP",
         color="Risk aversion",
         title=("Portfolio weights for different risk aversion and "
                "transaction cost"))
)
rebalancing_plot.draw()






 


