import pandas as pd 
import numpy as np
import yfinance as yf
from datetime import datetime as dt
from plotnine import *
from mizani.formatters import percent_format

tickerlist = ['UNH', 'MSFT', 'GS', 'HD', 'CAT', 'CRM', 'MCD',
               'V', 'AMGN', 'TRV', 'AXP', 'BA', 'HON', 'JPM',
                 'IBM', 'AAPL', 'AMZN', 'JNJ', 'PG', 'CVX',
                   'MRK', 'DIS', 'NKE', 'MMM', 'KO', 'WMT', 
                   'DOW', 'CSCO', 'INTC', 'VZ']

# inputDf = yf.download(tickers=tickerlist,period='5y',interval='1mo') # Data has been stored in a pickle file and can be read using below line
#  notice that following code could be used instead of specifying interval:  .resample("m").last() 
inputDf = pd.read_pickle(r'/Users/asbjornfyhn/Desktop/Emp Fin/AEF/Exercises/DowJonesDecomp.pkl') 

df = (inputDf[['Adj Close']].stack()
      .reset_index()
      .rename(columns={'level_1':'ticker',
                       'Date':'date',
                       'Adj Close':'adj_close'})
      .sort_values(['ticker', 'date'])
      .set_index('date')
) 
# 
df['log_return'] = df.groupby('ticker')['adj_close'].apply(lambda x: np.log(x / x.shift(1))).values

#

retVec = df.groupby('ticker')['log_return'].mean().values
covMatrix = df.reset_index().dropna().pivot(index='date',columns='ticker',values='log_return').cov().values

no_obs = len(retVec)
mu = retVec.T
iota = np.ones(no_obs).T
sigma_inv = np.linalg.inv(covMatrix)
#
mvp_weights = sigma_inv @ iota #sigma_inv % iota # Alternatively:  
mvp_weights = mvp_weights / sum(mvp_weights)

#
mvp_return = mu @ mvp_weights 
mvp_var = mvp_weights.T @ covMatrix @ mvp_weights
mvp_vol = np.sqrt(mvp_var)
mvp_vol_ann = mvp_vol*np.sqrt(12)

# efficient portfolio
# w = w_mvp + tilde{lambda}/2 (sigma_inv * mu - D/C * sigma_inv * iota)
# where C = iota_prime * sigma_inv * iota 
# D = iota_prime * sigma_inv * mu, E = mu_prime * sigma_inv * mu 
# lastly \tilde{lambda} = 2 * ( \bar{mu} * - D/C ) / (E - D^2/C)
# notice that \bar{mu} is the investor required return

mu_bar = mvp_return*2
C = iota.T @ sigma_inv @ iota
D = iota.T @ sigma_inv @ retVec
E = retVec.T @ sigma_inv @ retVec
lambda_tilde = 2 * (mu_bar - D/C) / (E-D**2/C)
efp_weights = mvp_weights + lambda_tilde/2 * (sigma_inv@retVec - D* mvp_weights ) #/C * sigma_inv @ iota)

# C = iota.T @ sigma_inv @ iota
# D = iota.T @ sigma_inv @ mu
# E = mu.T @ sigma_inv @ mu
# lambda_tilde = 2*(mu_bar-D/C)/(E-D**2/C)
# efp_weights = mvp_weights+lambda_tilde/2*(sigma_inv @ mu-D*mvp_weights)

#
efp_return = retVec.T @ efp_weights 
efp_var = efp_weights.T @ covMatrix @ efp_weights
efp_vol = np.sqrt(efp_var)
efp_vol_ann = efp_vol*np.sqrt(12)

c_weights = np.linspace(-4,1.5,100)
eff_front_ret = np.nan + np.zeros(len(c_weights))
eff_front_vol = np.nan + np.zeros(len(c_weights))
eff_front_vol_ann = np.nan + np.zeros(len(c_weights))

for i, c in enumerate(c_weights):
  eff_front_weights = c * mvp_weights + (1-c) * efp_weights
  eff_front_ret[i] = retVec.T @ eff_front_weights 
  eff_front_vol[i] = np.sqrt(eff_front_weights.T @ covMatrix @ eff_front_weights)#
  eff_front_vol_ann[i] = eff_front_vol[i] *np.sqrt(12)

results = pd.DataFrame([eff_front_vol_ann,eff_front_ret],index=['vol','ret']).T
results['ret'] = results['ret'] * 12


res_figure = (
  ggplot(results, aes(x="vol", y="ret")) +
  geom_point() +
  geom_point(
    pd.DataFrame({"ret": [mvp_return*12, efp_return*12],"vol":[mvp_vol_ann, efp_vol_ann]}),
    size=4)
   +
  geom_point(
    pd.DataFrame({"ret": mu*12,
                  "vol": np.sqrt(np.diag(covMatrix)) * np.sqrt(12)
                  })
  ) +
  labs(x="Annualized standard deviation",
       y="Annualized expected return",
       title="Efficient frontier for DOW index constituents") +
  scale_x_continuous(labels=percent_format()) +
  scale_y_continuous(labels=percent_format())
)
print(res_figure)
