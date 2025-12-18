import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
msft = yf.Ticker("MSFT")
msft = msft.history(period = 'max')

msft = pd.DataFrame(data = msft)


msft['LogReturn'] = np.log(msft['Close']).shift(-1) - np.log(msft['Close'])

# Plot a histogram to show the distribution of log return of Microsoft's stock. 
# You can see it is very close to a normal distribution

from scipy.stats import norm
mu = msft['LogReturn'].mean()
sigma = msft['LogReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(msft['LogReturn'].min()-0.02,msft['LogReturn'].max()+0.02,0.0005)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

msft['LogReturn'].hist(bins = 50, density = True, color = 'skyblue', edgecolor = 'black',
                       alpha = 0.6, label = 'Historical Returns',figsize = (12,8))

plt.plot(density['x'], density['pdf'], color = 'red', linewidth=2, 
         label = rf'Normal PDF ($\mu={mu:.4f}, \sigma={sigma:.4f}$)')

plt.title('Historical Log Returns vs. Fitted Normal Distribution', fontsize=16)
plt.xlabel('Log Return', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()

# probability that the stock price of microsoft will drop over 5% in a day
prob_return1 = norm.cdf(-0.05, mu, sigma)
a = prob_return1
print(f"The probability that the stock price will drop over 5% is {a}")

# Now is your turn, calculate the probability that the stock price of microsoft will drop over 10% in a day
prob_return1 = norm.cdf(-.10,mu,sigma)
b = prob_return1
print(f'The Probability it will drop over 10% in a day is {b}')

# Drop over 20% in 220 days
mu220 = 220 * mu
sigma220 = (220**0.5) * sigma
drop20 = norm.cdf(-.20, mu220, sigma220)
print(f"The probability of dropping over 20% in 220 days is {drop20}")

# drop over 40% in 220 days
mu220 = 220*mu
sigma220 = (220**0.5) * sigma
print('The probability of dropping over 40% in 220 days is ', norm.cdf(-0.4, mu220, sigma220))

# Value at risk(VaR)
VaR = norm.ppf(0.05, mu, sigma)
print("Single day value at risk", VaR)

# Quatile 
# 5% quantile
print('5% quantile ', norm.ppf(0.05, mu, sigma))
# 95% quantile
print('95% quantile ', norm.ppf(0.95, mu, sigma))

# This is your turn to calcuate the 25% and 75% Quantile of the return
# 25% quantile
q25 = norm.ppf(.25, mu, sigma)
print('25% quantile ', q25)
# 75% quantile
q75 = norm.ppf(.75, mu, sigma) 
print('75% quantile ', q75)

