import pandas as pd
import sqlite3
from plotnine import *


tidy_finance_python = sqlite3.connect( database="/Users/asbjornfyhn/Desktop/Emp Fin/data/tidy_finance_python.sqlite")

# pd.read_sql_query('''SELECT name FROM sqlite_schema WHERE type ='table' AND  name NOT LIKE 'sqlite_%';''', con =tidy_finance_python)

cpi_monthly = (pd.read_sql_query('select * from cpi_monthly;',con=tidy_finance_python, parse_dates={"month"}))

crsp_monthly = (
    pd.read_sql_query(sql=("SELECT * FROM crsp_monthly"),
                        con=tidy_finance_python, parse_dates={"month"})
                        .dropna()
)

df = pd.merge(crsp_monthly,cpi_monthly,how='left',left_on='month',right_on='month')
df = df.sort_values(by='date')

df['mktcap_inflation'] = df['mktcap']/df['cpi']

industry = (df.groupby(['month','industry'])['mktcap_inflation']
            .sum()
            .reset_index()
            )

plot1 = (
    ggplot(industry)  # What data to use
    + aes(x="month", y="mktcap_inflation",color='industry')  # What variable to use
    + geom_line()  # Geometric object to use for drawing
)

# groupby exchange 


exchangeCount = (crsp_monthly.groupby(['month','exchange'])['permno']
                 .count()
                 .reset_index()
                 .rename(columns={'permno':'count'})
                 )