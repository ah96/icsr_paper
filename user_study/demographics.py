import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import math

# Load the data
with_explanations = pd.read_excel("With_Explanations.xlsx")
without_explanations = pd.read_excel("Without_Explanations.xlsx")

print(with_explanations)

######################################################################################################

# Descriptive statistics for age and sex
def participant_stats(df):
    avg_age = df['Age'].mean()
    sex_distribution = df['Sex'].value_counts(normalize=True)
    return avg_age, sex_distribution

avg_age_with, sex_dist_with = participant_stats(with_explanations)
avg_age_without, sex_dist_without = participant_stats(without_explanations)

print(f"Average age with explanations: {avg_age_with}")
print(f"Sex distribution with explanations: \n{sex_dist_with}")
print(f"Average age without explanations: {avg_age_without}")
print(f"Sex distribution without explanations: \n{sex_dist_without}")

# Combine both datasets
combined_data = pd.concat([with_explanations, without_explanations])

# Calculate mean and variance for age
age_mean = combined_data['Age'].mean()
age_variance = combined_data['Age'].var()
age_min = combined_data['Age'].min()
age_max = combined_data['Age'].max()

print(f"Mean age of all participants: {age_mean}")
print(f"Variance of age among all participants: {age_variance}")
print(f"SD of age among all participants: {math.sqrt(age_variance)}")
print(f"Minimal age of all participants: {age_min}")
print(f"Maximal age of all participants: {age_max}")

# Calculate sex distribution
sex_distribution = combined_data['Sex'].value_counts(normalize=True) * 100

print("Sex distribution for all participants:")
print(sex_distribution)

# Female: 60.83%, Male 39.17%
# Age: mean: 21.08, variance: 11.05