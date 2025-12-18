# Import numpy and pandas as package
import numpy as np
import pandas as pd

# Mimics the roll of two dices

die = pd.DataFrame([1, 2, 3, 4, 5, 6])
sum_of_die = die.sample(2, replace = True).sum().loc[0]

print("The sume of the two dies is", sum_of_die)

# Mimics the rolle of three dices
die = pd.DataFrame([1, 2, 3, 4, 5, 6])
sum_of_three = die.sample(3, replace = True).sum().loc[0]
print(f"The sum of three dices is {sum_of_three}")

# The following code mimics the roll dice game for 50 times. And the results are all stored into "Result"
# Lets try and get the results of 50 sum of faces.

trial = 50
result = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]

# print the first 20 results
print(result)

trial = 100
result = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]
numbers = []
for i in result:
    numbers.append(i)
print(numbers)
