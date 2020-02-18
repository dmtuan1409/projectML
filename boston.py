# dự đoán giá nhà ở boston sử dụng gradientboosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

boston = load_boston()
x, y = boston.data, boston.target

xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=12, test_size=0.15)

gbr = GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.01,  min_samples_split=3)

gbr.fit(xtrain, ytrain)
print("Accuracy score (training): {0:.3f}".format(gbr.score(xtrain, ytrain)))
print("Accuracy score (test): {0:.3f}".format(gbr.score(xtest, ytest)))
