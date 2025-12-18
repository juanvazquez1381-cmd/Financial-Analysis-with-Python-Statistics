import pandas as pd
import numpy as np

data = pd.DataFrame()
data["Population"] = [47, 48, 85, 20, 19, 13, 72, 16, 50, 60]

a_sample_with_replacement = data['Population'].sample(5, replace = True)
print(a_sample_with_replacement)
print("---------------------------------------------------------")
a_sample_without_replacement = data['Population'].sample(5, replace = False)
print(a_sample_without_replacement)

print("---------------------------------------------------------")
# Calculate mean and variance
population_mean = data.mean()
population_var = np.var(data) # Population variance by default
print('Population mean is ', population_mean)
print('Population variance is', population_var)
print("---------------------------------------------------------")

# Calculate sample mean and sample standard deviation, size =10
# You will get different mean and varince every time when you excecute the below code

a_sample = data["Population"].sample(10, replace = True)
sample_mean = a_sample.mean()
sample_var = a_sample.var()
print("Sample mean is ", sample_mean)
print("Sample variance is", sample_var)
print("---------------------------------------------------------")

# Average of an unbiased estimator
sample_length = 500
sample_variance_collection = [data['Population'].sample(10, replace = True).var(ddof=1) for i in range(sample_length)]


for i in range(sample_length):
    data['Population'].sample(10, replace = True).var(ddof =1)

    
