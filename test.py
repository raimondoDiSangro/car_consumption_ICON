import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# test test
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import svm
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR

# test test

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv('data/auto_mpg.csv')

pd.set_option('display.max_columns', None)  # show all the columns
# print(df.describe())
# print(df.head())
# print(df.dtypes)

# list(df.columns)
df.drop(labels=['car name'], axis=1, inplace=True)
# print(df.head())

# Exploratory data Analysis visualization and analysis
# plt.figure(figsize=(10, 8))
# sns.histplot(df.mpg)
# plt.show()

# Correlation
# f, ax = plt.subplots(figsize=[14, 8])
# sns.heatmap(df.corr(), annot=True, fmt=".2f")
# ax.set_title("Correlation Matrix", fontsize=20)
# plt.show()

# sns.pairplot(df, diag_kind='kde')
# plt.show()

plt.figure(figsize=[14, 6])
sns.barplot(x=df['model year'], y=df['mpg'])
plt.title('Consumption Gallon by Years')
# plt.show()

# print(df.corr('spearman'))


# df['acceleration_power_ratio'] = df['acceleration'] / df['horsepower']

y = df['mpg']
df.drop('mpg', axis=1, inplace=True)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=3)
model_pipe = Pipeline(steps=[('scaler', StandardScaler(),), ('lasso', LassoCV(),)])
# model = Pipeline('scaler', StandardScaler())
model_pipe.fit(X_train, y_train)
prediction = model_pipe.predict(X_test)

# print(X_test)
# print(prediction)
print("lassoCV train accuracy", model_pipe.score(X_train, y_train))
print("lassoCV test accuracy", model_pipe.score(X_test, y_test))

print("lassoCV mean absolute error:", mean_absolute_error(y_test, prediction))
print("lassoCV r2 score:",r2_score(y_test, prediction))
print("lassoCV mean squared error:", mean_squared_error(y_test, prediction))
print("lassoCV mean absolute error percentage",
      mean_absolute_percentage_error(y_test, prediction))

#todo
n_features = 7
rng = np.random.RandomState(0)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svr', SVR(epsilon=0.2))])

print("svr train accuracy",regr.score(X_train, y_train))
print("svr train accuracy",regr.score(X_test, y_test))

prediction = regr.predict(X_test)

print("SVR mean absolute error:", mean_absolute_error(y_test, prediction))
print("SVR r2 score:",r2_score(y_test, prediction))
print("SVR mean squared error:", mean_squared_error(y_test, prediction))
print("SVRM mean absolute error percentage",
      mean_absolute_percentage_error(y_test, prediction))


# cylinders,displacement,horsepower,weight,acceleration,model year,origin
input_data = (4,97.0,88.0,2130,14.5,1970,3) #27
# input_data = (8,318.0,140.0,3735,13.2,1978,1) #19.4
# input_data = (4, 79.0, 58.0, 1755, 16.9, 1981, 3)  # 39.1
# input_data=(6,168.0,116.0,2900,12.6,1981,3) #25.4
# input_data=(8,351.0,266.0,2860,6,1973,3) #de tomaso pantera 15.1
# input_data = (8,351.0,142.0,4054,14.3,1979,1) #15.5
# input_data = (8, 302.0, 130.0, 4295, 14.9, 1977, 1)  # 15
# input_data = (4,114.0,91.0,2582,14.0,1973,2) #20


# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model_pipe.predict(input_data_reshaped)
print(prediction)
prediction = regr.predict(input_data_reshaped)
print(prediction)
