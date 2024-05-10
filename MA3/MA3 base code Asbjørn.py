import pandas as pd
import numpy as np
import sqlite3
import sys 
sys.path.append(r'/Users/asbjornfyhn/Desktop/Python')
from functions import plotgenerator as pg
from scipy.optimize import minimize

#Connecting to the database
tidy_finance = sqlite3.connect(
    database=f"/Users/asbjornfyhn/Desktop/Emp Fin/data/tidy_finance_python.sqlite"
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
crsp_monthly = crsp_monthly.groupby("permno").filter(lambda x: x.shape[0] == crsp_monthly.month.unique().shape[0])

# Estimate the expected returns 
def expected_returns(crsp_monthly):
    """
    This function estimates the expected returns for each stock in the crsp_monthly dataset
    """
    return crsp_monthly.groupby("permno").ret_excess.mean().values

# Estimate the covariance matrix
def covariance_matrix(crsp_monthly):
    """
    This function estimates the covariance matrix for the stocks in the crsp_monthly dataset
    """
    return crsp_monthly.pivot(index="month", columns="permno", values="ret_excess").cov().values

mu = expected_returns(crsp_monthly)
sigma = covariance_matrix(crsp_monthly)

#
def compute_naive_portfolio(N):
    """
    This function creates a naive portfolio with N stocks. The portfolio is equally weighted.
    """
    return np.repeat(1/N,N)

# optimization constraints and parameters
def equality_constraint(w):
    return np.sum(w)-1
def jacobian_equality(w):
    return np.ones_like(w)
options = {
  "tol":1e-20,
  "maxiter": 10000,
  "method":"SLSQP"
}

def compute_efficient_weight_L1_TC(mu, sigma, gamma, beta, initial_weights):
    """Compute efficient portfolio weights with L1 constraint."""       
    
    def objective(w):
      return (gamma*0.5*w.T @ sigma @ w-(1+mu) @ w
               +(beta/10000)/2*np.sum(np.abs(w-initial_weights)))
    def gradient(w):
      return (-mu+gamma*sigma @ w 
              +(beta/10000)*0.5*np.sign(w-initial_weights)) 
    constraints = (
      {"type": "eq", "fun": equality_constraint, "jac": jacobian_equality}
    )
    result = minimize(
        x0=initial_weights,
        fun=objective,
        jac=gradient,
        constraints=constraints,
        tol=options["tol"],
        options={"maxiter": options["maxiter"]},
        method=options["method"]
    )
    return result.x


############################################
#  Testing out-of-sample performance
############################################
window_length = 120
periods = len(crsp_monthly['month'].unique())-window_length

n_assets = len(crsp_monthly['permno'].unique())
w_prev_1 = w_prev_2 = w_prev_3 = np.ones(n_assets)/n_assets

beta = 200
gamma = 4


performance_values = np.empty((periods, 3)) + np.nan
performance_values = {
  "MV (TC)": performance_values.copy(), 
  "Naive": performance_values.copy(), 
  "MV": performance_values.copy()
}


def adjust_weights(w, next_return):
    w_prev = 1+w*next_return
    return np.array(w_prev/np.sum(np.array(w_prev)))

def evaluate_performance(w, w_previous, next_return, beta=50):
    """Calculate portfolio evaluation measures."""  
    
    raw_return = np.dot(next_return, w)
    turnover = np.sum(np.abs(w-w_previous))
    net_return = raw_return-beta/10000*turnover
    
    return np.array([raw_return, turnover, net_return])

# sort the values so we can iterate over the data
crsp_monthly.sort_values(["month","permno"], inplace=True)

for p in range(periods):
  p_ = p*n_assets
  wi_ = window_length*n_assets

  returns_window = crsp_monthly.iloc[p_:(p_+wi_-n_assets), :]
  next_return = crsp_monthly.iloc[p_+wi_, :]

  sigma_window = covariance_matrix(returns_window)
  mu_window = expected_returns(returns_window)

  # Transaction-cost adjusted portfolio
  w_1 = compute_efficient_weight_L1_TC(
    mu=mu_window, 
    sigma=sigma_window, 
    beta=beta, 
    gamma=gamma, 
    initial_weights=w_prev_1
  )

  performance_values["MV (TC)"][p_, :] = evaluate_performance(w_1, w_prev_1, next_return, beta=beta)
  w_prev_1 = adjust_weights(w_1, next_return)

    # Naive portfolio
  w_2 = compute_naive_portfolio(n_assets)
  performance_values["Naive"][p_, :] = evaluate_performance(w_2, w_prev_2, next_return)
  w_prev_2 = adjust_weights(w_2, next_return)

  # # Mean-variance efficient portfolio (w/o transaction costs)
  # w_3 = compute_efficient_weight(sigma=sigma_window, mu=mu, gamma=gamma)
  # performance_values["MV"][p, :] = evaluate_performance(
  #   w_3, w_prev_3, next_return
  # )
  # w_prev_3 = adjust_weights(w_3, next_return)



length_year = 12

performance_table = (performance
  .groupby("strategy")
  .aggregate(
    mean=("net_return", lambda x: length_year*100*x.mean()),
    sd=("net_return", lambda x: np.sqrt(length_year)*100*x.std()),
    sharpe_ratio=("net_return", lambda x: (
      (length_year*100*x.mean())/(np.sqrt(length_year)*100*x.std()) 
        if x.mean() > 0 else np.nan)
    ),
    turnover=("turnover", lambda x: 100*x.mean())
  )
  .reset_index()
)
performance_table.round(3)