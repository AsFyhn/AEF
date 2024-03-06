import pandas as pd 
import numpy as np
import yfinance as yf
from datetime import datetime as dt
from plotnine import *
from mizani.formatters import percent_format
from copy import deepcopy

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
sharpe_ratio = (mu - 0) / np.sqrt(np.diag(sigma))

# Print the Sharpe ratio
sr_high = df.groupby('ticker')['ticker'].first().iloc[sharpe_ratio.argmax()]
print(f'Highest Sharpe ratio of {sharpe_ratio.max():.2f} is {sr_high}')

def calc_return_std(weights, mu, sigma_matrix, factor):
  """
  Calculate the expected return and standard deviation of a portfolio.

  Parameters:
  weights (numpy.ndarray): The weights of the assets in the portfolio.
  mu (numpy.ndarray): The expected returns of the assets.
  sigma_matrix (numpy.ndarray): The covariance matrix of the assets.
  factor (float): A scaling factor.

  Returns:
  tuple: A tuple containing the expected return and standard deviation of the portfolio.
  """
  return_vec = mu.T @ weights * factor 
  vol = np.sqrt(weights.T @ sigma_matrix @ weights) * np.sqrt(factor)
  return return_vec, vol

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
    class results: pass
    results.inputs = {}
    results.inputs['mu'] = mu_est
    results.inputs['sigma'] = sigma_est
    results.inputs['yearly_factor'] = yearly_factor

    N = mu_est.shape[0]
    if sigma_est.shape[0] != N: 
        raise ValueError('The size length of vector mu_est should be the same as for sigma_est')
    iota = np.ones(N)
    sigma_inv = np.linalg.inv(sigma_est) 

    #----- minimum variance portfolio
    mvp_weights = sigma_inv @ iota
    mvp_weights = mvp_weights/mvp_weights.sum()
    mvp_return,mvp_volatility =  calc_return_std(mvp_weights,mu,sigma_est,yearly_factor)
    # print(f'Return of the minimum variance portfolio is: {mvp_return:.2f} and its volatility is {mvp_volatility:.2f}')
    results.mvp_weights = mvp_weights  # store minimum variance portfolio weights in object

    #----- efficient frontier portfolio
    mu_bar = mvp_return/12*2
    C = iota.T @ sigma_inv @ iota
    D = iota.T @ sigma_inv @ mu_est
    E = mu_est.T @ sigma_inv @ mu_est
    lambda_tilde = 2 * (mu_bar - D/C) / (E-D**2/C)
    efp_weights = mvp_weights + lambda_tilde/2 * (sigma_inv@mu_est - D* mvp_weights ) 
    # efp_return,efp_vol =  calc_return_std(efp_weights,mu,sigma_est,yearly_factor)
    # print(f'Return of the efficient frontier portfolio is: {efp_return:.2f} and its volatility is {efp_vol:.2f}')
    results.efp_weights = efp_weights # store efficient frontier portfolio weights in object

    #----- mutual fund theorem
    a = np.linspace(-0.2, 10, 200)
    res = pd.DataFrame(columns=["mu", "sd"], index=a).astype(float)
    for i in a:
        w = (1-i)*mvp_weights+i*efp_weights
        for j in range(len(w)):
          res.loc[i, f"w_{j+1}"] = w[j]  # Assign each element of w to a named column
        res.loc[i, "mu"] = (w.T @ mu)*yearly_factor
        res.loc[i, "sd"] = np.sqrt(w.T @ sigma @ w)*np.sqrt(yearly_factor)
    
    results.res = res   # store dataframe in object 

    return results

cef = compute_efficient_frontier(mu_est=mu,sigma_est=sigma,yearly_factor=12)
mvpRet, mvpVol = calc_return_std(weights=cef.mvp_weights,mu=cef.inputs['mu'],sigma_matrix=cef.inputs['sigma'],factor=cef.inputs['yearly_factor'])
efpRet, efpVol = calc_return_std(weights=cef.efp_weights,mu=cef.inputs['mu'],sigma_matrix=cef.inputs['sigma'],factor=cef.inputs['yearly_factor'])


#---- tangency portfolio
# ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿
#       connsider if there is a better way
# ????????????????????????????????????????????????
d = np.vstack((cef.inputs['sigma'],np.ones(27)))
c = np.append(cef.inputs['mu'],0)
e = np.hstack((d,c.reshape(c.shape[0],1)))
q = np.append(np.zeros(cef.inputs['mu'].shape[0]),1)
tan_weights = np.linalg.inv(e) @ q
tan_weights = tan_weights[:-1]
tanRet, tanVol = calc_return_std(weights=tan_weights,mu=cef.inputs['mu'],sigma_matrix=cef.inputs['sigma'],factor=cef.inputs['yearly_factor'])
# sharpe-ratio
sharpeRatio = tanRet/tanVol

vol = np.linspace(0,tanVol,101)
x = np.linspace(0,1,101)
ret = np.nan + np.zeros(vol.shape[0])
for i, v in enumerate(x):
   ret[i] = tanRet * v
tanLine = pd.DataFrame((vol,ret),index=['vol','ret']).T

res_figure = (
  ggplot(cef.res, aes(x="sd", y="mu")) +
  geom_point() + 
  geom_line(tanLine, aes(x='vol',y='ret'))+
  geom_point(
    pd.DataFrame({"mu": [mvpRet, efpRet,tanRet],"sd":[mvpVol, efpVol,tanVol]}),
    size=4, 
    color='darkblue',
    ) +
  geom_point(
    pd.DataFrame({"mu": cef.inputs['mu']*12,
                  "sd": np.sqrt(np.diag(cef.inputs['sigma'])) * np.sqrt(12)
                  })
  ) +
  labs(x="Annualized standard deviation",
       y="Annualized expected return",
       title="Efficient frontier for DOW index constituents") +
  scale_x_continuous(labels=percent_format()) +
  scale_y_continuous(labels=percent_format())
)
res_figure


# simulating returns 
def simulate_returns(periods=200,
                     expected_returns=mu,
                     covariance_matrix=sigma):
    """
        periods (int): Number of periods
        expected_returns (array-like): Expected returns for each asset
        covariance_matrix (array-like): Covariance matrix of returns
    """
    return np.random.multivariate_normal(
       mean=expected_returns, 
       cov=covariance_matrix, 
       size=periods
    )

def calc_eff_tang_port(x):
  d = np.vstack((x.inputs['sigma'],np.ones(27)))
  c = np.append(x.inputs['mu'],0)
  e = np.hstack((d,c.reshape(c.shape[0],1)))
  q = np.append(np.zeros(x.inputs['mu'].shape[0]),1)
  tan_weights = np.linalg.inv(e) @ q
  tan_weights = tan_weights[:-1]
  return tan_weights 


simLength = 100
simdraws = {i:{} for i in range(simLength)}
np.random.seed(100)
for i in range(simLength):
  if i % 10==0: print(f'{i+1} out of {simLength}')
  draw1 = pd.DataFrame(simulate_returns(periods=200,expected_returns=cef.inputs['mu'],covariance_matrix=cef.inputs['sigma']))
  mu_sim_est = np.array(draw1.mean()).T
  sigma_sim_est = np.array(draw1.cov())
  eff = compute_efficient_frontier(mu_est=mu_sim_est,sigma_est=sigma_sim_est,yearly_factor=12)
  sim_df = deepcopy(eff.res)
  sim_df['sim_draw_no'] = i
  tan_weights = calc_eff_tang_port(x=eff)
  tanRet, tanVol = calc_return_std(weights=tan_weights,mu=eff.inputs['mu'],sigma_matrix=eff.inputs['sigma'],factor=eff.inputs['yearly_factor'])
  simdraws[i] = {'Frontier':sim_df,
                          'Tangency Weights': tan_weights,
                          'Sharpe Ratio': tanRet/tanVol,
                          'input':{'mu':mu_sim_est,
                                   'sigma':sigma_sim_est,
                                   'data':draw1}}

sim_df = pd.concat([simdraws[i]['Frontier'] for i in range(simLength)])

# plot of first simulatioj
res_figure2 = (
  ggplot(sim_df.loc[sim_df['sim_draw_no']==0], aes(x="sd", y="mu")) +
  geom_point() + 
  geom_point(cef.res, colour='red') + 
  labs(x="Annualized standard deviation",
       y="Annualized expected return",
       title="Efficient frontier for DOW index constituents") +
  scale_x_continuous(labels=percent_format()) +
  scale_y_continuous(labels=percent_format())
)
res_figure2


# plot of all 100 simulations
res_figure3 = (
    ggplot(sim_df, aes(x="sd", y="mu",group='sim_draw_no')) +
    geom_point(alpha=0.05)  # Plot first element with label
    +geom_point(cef.res.assign(sim_draw_no = -1),alpha=1,colour='black')
)
res_figure3

# plot histogram of sharpe ratios
(
    ggplot(pd.DataFrame([simdraws[i]['Sharpe Ratio'] for i in range(simLength)],columns=['Sharpe Ratio'])) + 
    aes(x = 'Sharpe Ratio') +
    geom_histogram(binwidth=0.1)
)

# check for the outliers 
for i in range(simLength):
  s, d = calc_return_std(weights=simdraws[i]['Tangency Weights'], sigma_matrix=simdraws[i]['input']['sigma'],mu=simdraws[i]['input']['mu'],factor=12)
  if s/d <= 0 : 
     print(f'Negative value of {s/d} when i={i}')

no_min = np.array([simdraws[i]['Sharpe Ratio'] for i in range(simLength)]).argmin()
simdraws[no_min]['input']['data'].mean()/np.array(simdraws[no_min]['input']['data'].cov())