import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf

# Gets stock info and creates data frame
ms = yf.Ticker('MSFT')
ms = ms.history(period = "max")
ms = pd.DataFrame(ms)
ms['logReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])
ms.dropna()
# Log return goes up and down during the period
ms['logReturn'].plot(figsize = (20,8))
plt.axhline(0, color = 'red')
plt.show()

# Calculate Test Statistic
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof = 1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead
# of t-distribution
# mu = 0 under the null hypothesis
zhat = (sample_mean - 0)/ (sample_std/n**0.5)
print(zhat)

# Set Desicion Criteria
alpha = 0.05 # Confidence level

zleft = norm.ppf(alpha/2, 0, 1)
zright = -zleft # z-distribution is symmetric
print(zleft, zright)

#  Make decision - shall we reject H0?
print('Given the zscore is {} then at significant level of {}. we shall reject: {}'.format(zhat, alpha, zhat>zright or zhat<zleft))

# step 2
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead of t-distribtuion
# mu = 0 under the null hypothesis
zhat = (sample_mean - 0)/ (sample_std/n**0.5)
print(zhat)

# step 3
alpha = 0.05

zright = norm.ppf(1-alpha, 0, 1)
print(zright)

# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, zhat>zright))

# step 3 (p-value)
p = 1 - norm.cdf(zhat, 0, 1)
print(p)

# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, p < alpha))
