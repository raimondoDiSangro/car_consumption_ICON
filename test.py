import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/auto-mpg.csv')

pd.set_option('display.max_columns', None) # mostra tutte le colonne

print(df.head())
print(df.dtypes)

# Exploratory data Analysis visualization and analysis
plt.figure(figsize=(10, 8))
sns.histplot(df.mpg)
plt.show()

# searching for null values
print(df.isnull().sum())


# Correlation
#f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df.corr(), annot=True, fmt=".2f")
#ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

sns.pairplot(df, diag_kind='kde')
plt.show()

#plt.figure(figsize=[14, 6])
#sns.barplot(x=df['model year'] + 1900, y=df['mpg'])
#plt.title('Consumption Gallon by Years')
#plt.show()

print(df.corr('spearman'))

# train test split
x = df.drop(["mpg"], axis=1)
y = df.mpg
print(x.head())

test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)

