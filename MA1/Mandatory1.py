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
inputdf.get('Adj Close').stack().reset_index()
# transform the dataset
df = (inputdf
      .get('Adj Close')
      .stack()
      .reset_index()
      .rename(columns={'Date':'date','level_1':'ticker',0:'adj_close'})
      .set_index('date')
      .groupby('ticker')['adj_close']
      .resample('m')
      .last()
      .reset_index()
      .assign(date=lambda x: x['date'].dt.strftime('%Y-%m-%d'))  # Convert date to string
      .query("date >= '2000-01-01'")  # Filter for dates >= 1999-04-01
      .query("date <= '2023-12-31'")  # Filter for dates >= 1999-04-01
      )

# remove all tickers where there is no data for each month 
df = (df
  .groupby(["ticker"])
  .apply(lambda x: x.assign(counts=x["adj_close"].dropna().count()))
  .reset_index(drop=True)
  .query("counts == counts.max()")
  .drop(columns='counts')
  .assign(ret=lambda x: x["adj_close"].pct_change())
  .assign(log_return = lambda x: np.log(x['adj_close']/x['adj_close'].shift(1)) )
)

# check that we have 27 tickers
no_of_tickers = int(df['ticker'].unique().size)
print(f'The number of tickers in the dataset: { no_of_tickers}')

#
return_col = 'log_return'
mu = np.array(df.groupby('ticker')[return_col].mean()).T
sigma = np.array(df.pivot(index='date',columns='ticker',values=return_col).cov())

# Calculate Sharpe ratio for each stock
sharpe_ratio = (mu - 0) / np.diag(sigma)

# Print the Sharpe ratio
sr_high = df.groupby('ticker')['ticker'].first().iloc[sharpe_ratio.argmax()]
print(f'Highest Sharpe ratio of {sharpe_ratio.max():.2f} is {sr_high}')

def compute_efficient_frontier(mu_est: np.array, sigma_est: np.array, yearly_factor: int=1) -> pd.DataFrame:
    """
    The function performs each of the following steps:
    1. Compute the minimum variance portfolio weight mvp_weights for the input Sigma_est. 
        The function can handle positive definite variance-covariance matrices of arbitrary dimensions N x N.
    2. Compute the eﬀicient portfolio weights efp_weights for the inputs Sigma_est and mu_est 
      that delivers two times the expected return of the minimum variance portfolio weight.
    3. Make use of the two-mutual fund theorem to characterize a range of portfolio weights on the eﬀicient frontier: 
      Specifically, compute the weights of a sequence of portfolios which are combinations of the minimum variance portfolio weight
        and the eﬀicient portfolio, c_weights = c * mvp_weights  + (1 - c) * efp_weights where c ∈ {-0.1, ... , 1.2}
    
      Args: 
        mu_est (np.array): N x 1 return matrix
        sigma_est (np.array): N x N variance-covariance matrix
      Returns: 
        df: with column c which is the weight of the minimum variance portfolio and N corresponding columns with the weights 
          of each asset
    """
    N = mu_est.shape[0]
    if sigma_est.shape[0] != N: 
        raise ValueError('The size length of vector mu_est should be the same as for sigma_est')
    iota = np.ones(N)
    sigma_inv = np.linalg.inv(sigma_est) 

    #----- minimum variance portfolio
    mvp_weights = sigma_inv @ iota
    mvp_weights = mvp_weights/mvp_weights.sum()
    print(mvp_weights)
    mvp_return = mu.T @ mvp_weights * yearly_factor
    mvp_volatility = np.sqrt(mvp_weights.T @ sigma_est @ mvp_weights) * np.sqrt(yearly_factor)
    print(f'Return of the minimum variance portfolio is: {mvp_return:.2f} and its volatility is {mvp_volatility:.2f}')

    #----- efficient frontier portfolio
    mu_bar = mvp_return*2
    C = iota.T @ sigma_inv @ iota
    D = iota.T @ sigma_inv @ mu_est
    E = mu_est.T @ sigma_inv @ mu_est
    lambda_tilde = 2 * (mu_bar - D/C) / (E-D**2/C)
    efp_weights = mvp_weights + lambda_tilde/2 * (sigma_inv@mu_est - D* mvp_weights ) 
    print(efp_weights)
    efp_return = mu_est.T @ efp_weights 
    efp_vol = np.sqrt(efp_weights.T @ sigma_est @ efp_weights)*np.sqrt(12)
    print(f'Return of the efficient frontier portfolio is: {efp_return:.2f} and its volatility is {efp_vol:.2f}')

    #----- mutual fund theorem
    a = np.linspace(-0.2, 1.2, 121)
    res = pd.DataFrame(columns=["mu", "sd"], index=a).astype(float)
    for i in a:
        w = i*mvp_weights+(1-i)*efp_weights
        for j in range(len(w)):
          res.loc[i, f"w_{j+1}"] = w[j]  # Assign each element of w to a named column
        res.loc[i, "mu"] = (w.T @ mu)*yearly_factor
        res.loc[i, "sd"] = np.sqrt(w.T @ sigma @ w)*np.sqrt(yearly_factor)

    return res

results = compute_efficient_frontier(mu_est=mu,sigma_est=sigma,yearly_factor=12)
res_figure = (
  ggplot(results, aes(x="sd", y="mu")) +
  geom_point() + 
  labs(x="Annualized standard deviation",
       y="Annualized expected return",
       title="Efficient frontier for DOW index constituents") +
  scale_x_continuous(labels=percent_format()) +
  scale_y_continuous(labels=percent_format())
)
res_figure