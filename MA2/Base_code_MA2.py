import pandas as pd
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
#Bullet 1
#Creating the decile groups for the 12-month momentum

data_for_sorts['Mom_12_decile'] = pd.qcut(data_for_sorts['Mom_12'], q=10, labels=False) + 1

#Sorting the data by decile and month
sorted_data = data_for_sorts.sort_values(
    by=['Mom_12_decile', 'month'])

#Bullet 2
#Calculating the equal-weighted average values of Mom_12 and mc for each of the ten portfolios
average_return = (
    sorted_data
    .groupby(['Mom_12_decile'])
    .apply(lambda x: pd.Series({
        'Average_Mom_12': np.mean(x['Mom_12']),
        'Average_mktcap': np.mean(x['mktcap'])
    }))
    .reset_index()
)

# Specifiying the headers for the table
headers = {'Mom_12_decile':"Momentum decile",
    'Average_Mom_12':"Average momentum",
    'Average_mktcap':"Average market cap",
}
average_return.rename(columns=headers,inplace=True)
# # Printing the table
# print(tabulate.tabulate(average_return, headers, tablefmt="simple", showindex=False))

for col in list(headers.values()):
  average_return[col] = average_return[col].astype(float)

# Format the DataFrame style
formatted_dfAgg = average_return.style.format({
    'Momentum decile': '{:.0f}',  # No decimals
    'Average momentum': '{:.2f}',  # Two decimal places
    'Average market cap': '{:.2f}',  # Two decimal places
    'na_rep': "" 
})
formatted_dfAgg.hide()
formatted_dfAgg

# Bullet 3
# Bullet 3
# Computing the value weighted excess returns for each of the ten portfolios
Vw_excess_returns = (
    sorted_data.groupby(["Mom_12_decile"])
    .apply(
        lambda x: pd.Series(
            {
                "Decile_portfolio_excess_return": np.average(x["ret_excess"], weights=x["mktcap"])
            }
        )
    )
    .reset_index()
)

# Calculating the CAPM alpha for each of the ten portfolios
# Merging the data with the factors_ff3_monthly dataset
data_for_sorts = data_for_sorts.merge(
    factors_ff3_monthly, how="inner", on="month"
)
# Defining the linear regression function
def lin_reg(x,y):
    reg = sm.OLS(y, sm.add_constant(x)).fit()
    return reg.params["const"], reg.tvalues['const'], reg.pvalues['const']

# Running the CAPM regression for each of the ten portfolios
CAPM_alphas = (
    data_for_sorts.groupby(["Mom_12_decile"])
    .apply(
        lambda x: lin_reg(x["mkt_excess"], x["ret_excess"])
    )
    .reset_index()
)
CAPM_alphas['Alpha'] = [x[0] for x in CAPM_alphas[0]]
CAPM_alphas['t-statistic'] = [x[1] for x in CAPM_alphas[0]]
CAPM_alphas['p-value'] = [x[2] for x in CAPM_alphas[0]]
CAPM_alphas.drop(columns=[0], inplace=True)


# Merging the calculations to one dataset to present it in a table
Alphas_excessret = Vw_excess_returns.merge(
    CAPM_alphas, how="left", on="Mom_12_decile"
)

# Specifiying the headers for the table
headers_v2 = {'Mom_12_decile':"Momentum decile",
    'Decile_portfolio_excess_return':"Excess return",
}
# Renaming the headers
Alphas_excessret.rename(columns=headers_v2,inplace=True)

# Putting headers in place
for col in list(headers_v2.values()):
  Alphas_excessret[col] = Alphas_excessret[col].astype(float)

# Format the DataFrame style
formatted_dfAgg_v2 = Alphas_excessret.style.format({
    'Momentum decile': '{:.0f}',  # No decimals
    'Excess return': '{:.4f}', # 4 decimal places
    'Alpha': '{:.4f}',
    't-statistic': '{:.2f}',
    'p-value': '{:.2f}',
    'na_rep': "" 
})
formatted_dfAgg_v2.hide()
print(Alphas_excessret)

plot_momentum_portfolios_summary = (
  ggplot(Alphas_excessret, 
         aes(x="Momentum decile", y="Alpha", fill="Momentum decile")) +
  geom_bar(stat="identity") +
  labs(x="Momentum decile", y="CAPM alpha", fill="Momentum decile",
       title="CAPM alphas of momentum-sorted portfolios") +
  scale_x_continuous(breaks=range(1, 11)) +
  scale_y_continuous(labels=percent_format()) +
  theme(legend_position="none")
)
plot_momentum_portfolios_summary.draw()
np.logspace(-4,4,40)
# Analyzing the momentum strategy
data_for_sorts['portfolio'] = np.where(data_for_sorts["Mom_12_decile"]==1,'low',np.where(data_for_sorts["Mom_12_decile"]==10,'high','neutral'))

mom_ls = (data_for_sorts
          .query("portfolio in ['low', 'high']")
          .pivot_table(index="month", columns="portfolio", values="ret_excess")
          .assign(long_short=lambda x: x["high"]-x["low"])
          .merge(factors_ff3_monthly, how="left", on="month")
  )

model_fit = (sm.OLS.from_formula(
    formula="long_short ~ 1", 
    data=mom_ls
  )
  .fit(cov_type="HAC", cov_kwds={"maxlags": 60})
)
prettify_result(model_fit)


