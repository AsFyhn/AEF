import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
import os
from plotnine import *
from mizani.formatters import percent_format

################################################################
from plotnine import *
################################################################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
  train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
)
################################################################

# Connecting to the database
tidy_finance = sqlite3.connect(
    database=f"/Users/asbjornfyhn/Desktop/Emp Fin/data/tidy_finance_python.sqlite"
)

# Reading in crsp_monthly and factors_ff3_monthly
crsp_monthly_inp = pd.read_sql_query(
    sql="SELECT permno, month, ret_excess, mktcap, mktcap_lag FROM crsp_monthly",
    con=tidy_finance,
    parse_dates={"month"},
)
crsp_monthly = crsp_monthly_inp.copy(deep=True)

# remove all companies that has a mktcap that is below the 20th percentile
for month in crsp_monthly["month"].unique():
    mktcap_20 = crsp_monthly[crsp_monthly["month"] == month]["mktcap"].quantile(0.05)
    crsp_monthly = crsp_monthly.loc[~((crsp_monthly['month'] == month) & (crsp_monthly['mktcap'] < mktcap_20))]


# create 60 columns with the lagged ret_excess
crsp_monthly_lags = crsp_monthly.copy(deep=True)
crsp_monthly_lags.head(5)
for i in range(2, 62):
    crsp_monthly_lags[f'ret_excess_lag_{i}'] = crsp_monthly_lags.groupby('permno')['ret_excess'].shift(i)
    crsp_monthly_lags[f'ret_excess_lag_{i}_sq'] = crsp_monthly_lags[f'ret_excess_lag_{i}']**2

crsp_monthly_lags.dropna(subset=[f'ret_excess_lag_61'],inplace=True)
crsp_monthly_lags.set_index(['permno', 'month'], inplace=True)
exog = crsp_monthly_lags[[f'ret_excess_lag_{i}' for i in range(2,62)]+[f'ret_excess_lag_{i}_sq' for i in range(2,62)]]
endog = crsp_monthly_lags.loc[:,'ret_excess']

# Standardize the data - dependent variable has a mean of 0 and the predictors a standard deviation of 1
endog = endog - endog.groupby('month').mean()
exog = exog / exog.groupby('month').std()

# Training data
train_endog = endog.loc[endog.index.get_level_values('month') < '2030-01-01']
train_exog = exog.loc[exog.index.get_level_values('month') < '2030-01-01']
len([col for col in train_exog.columns if 'sq' in col])
########################################
# CONSIDER WHETHER THIS IS RELEVANT 
########################################
# alphas = np.logspace(0, 5, 100)
# coefficients_ridge = []
# for a in alphas:
#     ridge = Ridge(alpha=a, fit_intercept=False)
#     coefficients_ridge.append(ridge.fit(train_exog, train_endog).coef_)

# coefficients_ridge = (pd.DataFrame(coefficients_ridge)
#   .assign(alpha=alphas, model="Ridge")
#   .melt(id_vars=["alpha", "model"])
# )
# # run linear regression
# coefficients_plot = (
#   ggplot(coefficients_ridge, 
#          aes(x="alpha", y="value", color="variable")) + 
#   geom_line()  +
#   labs(x="Penalty factor (lambda)", y="",
#        title="Estimated coefficient paths for different penalty factors") +
#   scale_x_log10() +
#   theme(legend_position="none")
# )

############################################################
##### TUNING: Hyperparameter tuning with cross-validation
############################################################

initial_years = 5
assessment_months = 48*4
n_splits = 100#int(len(train_endog)/assessment_months) - 1
length_of_year = 12
alphas = np.logspace(-4, 4, 40)

data_folds = TimeSeriesSplit(
  n_splits=n_splits, 
  test_size=assessment_months, 
  max_train_size=initial_years * length_of_year
)

params = {
  "alpha": alphas,
}
ridge_pipeline = Ridge(fit_intercept=False,max_iter=5000)

finder = GridSearchCV(
  estimator=ridge_pipeline,
  param_grid=params,
  scoring="neg_root_mean_squared_error", # RMSE -- https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
  cv=data_folds, # cross validation
  verbose=1, # Print messages choose 1 through 3 to get more information
  n_jobs=-1, # Use all cores
)

finder.fit(
  train_exog, train_exog
)

best_est = finder.best_estimator_
best_est_fit = best_est.fit(train_exog, train_endog)
coef = best_est_fit.coef_


heatmap = pd.DataFrame({'lag':range(2,len(coef)//2+2),'r':coef[len(coef)//2:],'r^2':coef[:-len(coef)//2]})
heatmap1 = heatmap.set_index('lag').stack().reset_index().rename(columns={'level_1':'variable',0:'value'})

(
  ggplot(heatmap) +
  geom_col(aes(x='lag', y='r'), position='dodge', fill="blue", alpha=0.5) +  # Bar chart for 'r'
  geom_point(aes(x='lag', y='r^2'), color="red", size=3) +  # Scatter plot for 'r^2'
  labs(x="lag", y="Coefficient", title="") +
  scale_color_discrete(guide=False) +
  scale_y_continuous(labels=percent_format()) +
  theme(legend_position="none")
)

plot_momentum_longshort_year = (
  ggplot(heatmap1, 
         aes(x='lag', y='value', fill="variable")) +
  geom_col(position='dodge') +
  labs(x="lag", y="Coefficient", title="") +
  scale_color_discrete(guide=False) +
  scale_y_continuous(labels=percent_format()) +
  theme(legend_position="none")
)
plot_momentum_longshort_year







import dataframe_image as dfi
import pandas as pd

heatmap = pd.read_pickle('/Users/asbjornfyhn/Desktop/heatmap.pkl')
heatmap = pd.concat([heatmap.iloc[(i-1)*len(heatmap)//3:i*len(heatmap)//3].reset_index(drop=True) for i in range(1,4)],axis=1)
heatmap.columns = ['lag','r','r^2','lag ','r ','r^2 ','lag  ','r  ','r^2  ']

heatmapTable = heatmap.style.background_gradient(cmap='RdYlBu',subset=['lag','r','r^2','lag ','r ','r^2 ','lag  ','r  ','r^2  '])
heatmapTable.hide()
dfi.export(heatmapTable, '/Users/asbjornfyhn/Desktop/df_styled.png')



