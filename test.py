import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv('data/auto_mpg.csv')

pd.set_option('display.max_columns', None)  # show all the columns
# print(df.describe())
# print(df.head())
# print(df.dtypes)

list(df.columns)
df.drop(labels=['car name'], axis=1, inplace=True)

# Exploratory data Analysis visualization and analysis
plt.figure(figsize=(10, 8))
sns.histplot(df.mpg)
# plt.show()

# Correlation
# f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df.corr(), annot=True, fmt=".2f")
# ax.set_title("Correlation Matrix", fontsize=20)
# plt.show()

sns.pairplot(df, diag_kind='kde')
# plt.show()

plt.figure(figsize=[14, 6])
sns.barplot(x=df['model year'] + 1900, y=df['mpg'])
plt.title('Consumption Gallon by Years')
# plt.show()

# print(df.corr('spearman'))


# df['acceleration_power_ratio'] = df['acceleration'] / df['horsepower']

y = df['mpg']
df.drop('mpg', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=3)
model = Pipeline(steps=[('scaler', StandardScaler(),), ('lasso', LassoCV(),)])
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# cylinders,displacement,horsepower,weight,acceleration,model year,origin
# input_data = (4,97.0,88.0,2130,14.5,70,3) #27
# input_data = (8,318.0,140.0,3735,13.2,78,1) #19.4
input_data = (4, 79.0, 58.0, 1755, 16.9, 81, 3)  # 39.1
# input_data=(6,168.0,116.0,2900,12.6,81,3) #25.4
# input_data=(8,351.0,266.0,2860,6,73,3) #de tomaso pantera 13.1-15.8 / 12.6


# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = model.predict(input_data_reshaped)
print(prediction)
