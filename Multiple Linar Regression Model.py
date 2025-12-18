import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import yfinance as yf

# Importing different stock market data into different dataframes

indices = ["^AXJO", "^N225", "^HSI", "^GDAXI", "^FCHI",  "^GSPC", "^DJI", "^IXIC", "SPY"]

# Define desired dates:
start_date = "2008-01-02"
end_date = "2018-08-21"

import warnings
warnings.filterwarnings("ignore")

data = yf.download(indices, start = start_date, end = end_date )

aord = data.xs("^AXJO", level = 1, axis = 1).copy()
nikkei = data.xs("^N225",level = 1, axis = 1).copy()
hsi = data.xs( "^HSI",level = 1, axis = 1).copy()
daxi = data.xs("^GDAXI", level =1, axis = 1).copy()
cac40 = data.xs("^FCHI",level =1, axis = 1).copy()
sp500 = data.xs("^GSPC",level =1, axis = 1).copy()
dji = data.xs( "^DJI",level =1, axis = 1).copy()
nasdaq = data.xs("^IXIC",level =1, axis = 1).copy()
spy =data.xs("SPY",level =1, axis = 1).copy()


# Due to the timezone issues, we extract and calculate appropriate stock market data for analysis
# Indicepanel is the DataFrame of our trading model
indicepanel=pd.DataFrame(index=spy.index)

indicepanel['spy']=spy['Open'].shift(-1)-spy['Open']
indicepanel['spy_lag1']=indicepanel['spy'].shift(1)
indicepanel['sp500']=sp500["Open"]-sp500['Open'].shift(1)
indicepanel['nasdaq']=nasdaq['Open']-nasdaq['Open'].shift(1)
indicepanel['dji']=dji['Open']-dji['Open'].shift(1)

indicepanel['cac40']=cac40['Open']-cac40['Open'].shift(1)
indicepanel['daxi']=daxi['Open']-daxi['Open'].shift(1)

indicepanel['aord']=aord['Close']-aord['Open']
indicepanel['hsi']=hsi['Close']-hsi['Open']
indicepanel['nikkei']=nikkei['Close']-nikkei['Open']
indicepanel['Price']=spy['Open']

print(indicepanel.head())

# Lets check wheterh do we have NaN Values in indicepanel
print(indicepanel.isnull().sum())

# We can use method 'fillna()' from dataframe to forward filling the Nan values
# Then we can drop the reminding Nan values
indicepanel = indicepanel.fillna(method = 'ffill')
indicepanel = indicepanel.dropna()

# Lets check wheterh do we have NaN Values in indicepanel once more
print(indicepanel.isnull().sum())

# Save to CSV
indicepanel.to_csv('indicepanel')

print(indicepanel.shape)

# Split the data into (1) train set and (2) test set

Train = indicepanel.iloc[-2000:-1000,:]
Test = indicepanel.iloc[-1000:, :]
print(Train.shape, Test.shape)

# Generate scatter matrix among all stock markets (and the price of SPY) to observe the association
from pandas.plotting import scatter_matrix
sm = scatter_matrix(Train, figsize = (10,10))

# Find the indice with largest correlation
corr_array = Train.iloc[:, :-1].corr()['spy']
print(corr_array)

formula = 'spy~spy_lag1 + sp500 + nasdaq + dji + cac40 + aord + daxi + nikkei + hsi'
lm = smf.ols(formula = formula, data = Train).fit()
print(lm.summary())

# Make prediction
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)
plt.scatter(Train['spy'], Train['PredictedY'])
plt.show()

                    
