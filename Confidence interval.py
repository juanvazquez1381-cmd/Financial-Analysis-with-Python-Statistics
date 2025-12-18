import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf


# Gets stock info and creates data frame
ms = yf.Ticker('MSFT')
ms = ms.history(period = "max")
ms = pd.DataFrame(ms)

# We will use log return for average stock return in Microsoft

ms['logReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])
ms = ms.dropna() # drop nan values

# Lets build 90% confidence interval for log return
sample_size = ms['logReturn'].shape[0]
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1) / sample_size**0.5

# left and right quantile
z_left = norm.ppf(0.05)
z_right =norm.ppf(1-0.05)
print(z_left)

# upper and lower bound
 
interval_left =  sample_mean + (z_left * sample_std)
interval_right = sample_mean + (z_right *sample_std)

# 90% confidence interval tells you that there will be 90% chance that the average stock return lies between "interval_left"
# and "interval_right".

print('90% confidence interval is ', (interval_left, interval_right))
