import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Read data
data = pd.read_csv("C:/Users/Donghan/Desktop/551A1/adult.data", header = None)

# Statistical summary
def summary(data, label_col, groupby = True):
    '''
    data: pd.DataFrame
    label_col: string, label's column name
    groupby: boolean, if groupy by labels
    '''
    if groupby = True:
        return data.groupby(data[label_col]).describe()
    return data.describe()

# Hist plot (for distribution discovery)
# For continous variable
sns.distplot(data[...])

# For normal distribution fit line, if not roughly fit, then not in normal distribution
from scipy.stats import norm
sns.distplot(data[...], fit = norm)

# Box plot
def boxplot(data, label_col, feature_col):
    '''
    use this function iif one of axis is categorical
    '''
    var = label_col
    data = pd.concat([data[feature_col], data[label_col]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y=0, data=data)
    fig.axis()

# Check missing data percentage
def missing_percent(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Remove missing data for all variables
def data_cleaning(data):
    missing_data = missing_percent(data)
    data_cleaned = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
    for i in data_cleaned.columns:
        data_cleaned = data_cleaned.drop(data_cleaned.loc[data_cleaned[i].isnull()].index)
    return data_cleaned.isnull().sum().max(), data_cleaned


# Pair scatter plot for all variables
sns.pairplot(data)

# OR scatter plot for two specifed columns
data.plot.scatter(x=..., y=..., ylim=(0,...))

# Correlation plot
sns.heatmap(data.corr())

# Correlation matrix
data.corr()


# One-hot encoding
pd.get_dummies(data)
