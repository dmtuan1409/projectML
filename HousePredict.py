import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.set_index("Id", inplace=True)
test.set_index("Id", inplace=True)
y_train = train["SalePrice"]
y_test = test["SalePrice"]

train.drop("SalePrice", axis=1, inplace=True)
test.drop("SalePrice", axis=1, inplace=True)

train_test = train.append(test)

array = list(['MSSubClass','LotFrontage','LotArea','LowQualFinSF','GrLivArea','TotRmsAbvGrd','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','WoodDeckSF','OpenPorchSF','EnclosedPorch','MoSold','YrSold','MiscVal','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','1stFlrSF','2ndFlrSF','OverallQual','OverallCond'])
X=train_test[array]

X.fillna(value=0.0, inplace=True)

X_train = X[0:1460]
X_test = X[1460:]

scale = MinMaxScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=42, test_size=0.3)
model = linear_model.LinearRegression()
model.fit(X_train_sub, y_train_sub)
print('Train R^2 score: {}'.format(model.score(X_train_sub, y_train_sub)))
ypred = model.predict(X_test_scale)
print('Test R^2 score: {}'.format(model.score(X_test_scale, y_test)))
