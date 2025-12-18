import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Sample mean and SD keep changing but always within a certain range
Fstsample = pd.DataFrame(np.random.normal(10, 5, size = 30))
print('sample mean is ', Fstsample[0].mean())
print("sample sta dev is", Fstsample[0].std(ddof =1))

print("-----------------------------------------------------")

# Empirical Distribution of mean
# Empirical Distribution of mean
meanlist = []
for i in range(10000):
    sample = pd.DataFrame(np.random.normal(10, 5, size = 30))
    meanlist.append(sample[0].mean())
collection = pd.DataFrame()
collection['meanlist'] = meanlist

# Use 'density=True' instead of 'normed=1'
collection['meanlist'].hist(bins=100, density=True, figsize=(15,8))
plt.show()


# See what central limit theorem tells you...the sample size is larger enough, 
# the distribution of sample mean is approximately normal
# apop is not normal, but try to change the sample size from 100 to a larger number. The distribution of sample mean of apop 
# becomes normal.
sample_size = 100
samplemeanlist = []
apop =  pd.DataFrame([1, 0, 1, 0, 1])
for t in range(10000):
    sample = apop[0].sample(sample_size, replace=True)  # small sample size
    samplemeanlist.append(sample.mean())

acollec = pd.DataFrame()
acollec['meanlist'] = samplemeanlist
acollec.hist(bins=100, density = True,figsize=(15,8))
plt.show()
