import pandas as pd
import matplotlib.pyplot as plt


# To recall, this is the code to mimic the roll dice game for 50 times

die = pd.DataFrame([1, 2, 3, 4, 5, 6])
trial = 50
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]

# This is the code for summarizing the results of the sum of the faces by frequency
freq = pd.DataFrame(results)[0].value_counts()
sort_freq = freq.sort_index()
print(sort_freq)

# plot the bar chart base of the result
sort_freq.plot(kind = 'bar', color = 'deeppink', figsize =(15,8))


# Using relative frequency, we can rescale the frequency so that we can compare results from different number of trials
relative_freq = sort_freq/trial
relative_freq.plot(kind='bar', color='deeppink', figsize=(15, 8))


# Let us try to increase the number of trials to 10000, and see what will happen...
trial = 10000
results = [die.sample(2, replace = True).sum().loc[0] for i in range(trial)]
freq = pd.DataFrame(results)[0].value_counts()
sort_freq = freq.sort_index()
relative_freqs = sort_freq/trial
relative_freqs.plot(kind = 'bar', color = 'deeppink', figsize = (15,8))

# assume that we have fair dice, which means all faces will be shown with equal probability
# then we can say we know the 'Distribtuion' of the random variable - sum_of_dice

X_distri = pd.DataFrame(index = [2,3,4,5,6,7,8,9,10,11,12])
X_distri['Prob'] = [1,2,3,4,5,6,5,4,3,2,1]
X_distri['Prob'] = X_distri['Prob']/36
print(X_distri)

mean = pd.Series(X_distri.index * X_distri['Prob']).sum()
var = pd.Series(((X_distri.index - mean)**2)*X_distri['Prob']).sum()

print(f"Mean: {mean}, Variance: {var}")

# if we calculate mean and variance of outcomes (with high enough number of trials, eg 20000)...
trial = 20000
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]

# Print the mean variance of the 20000 trials
results = pd.Series(results)
print("Mean:", results.mean(),"Variance:", results.var())
