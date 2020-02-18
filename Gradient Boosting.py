import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

#load data
train = pd.read_csv("train1.csv")
test = pd.read_csv("test1.csv")
#set "passenger id" variable as index
train.set_index("PassengerId", inplace=True)
test.set_index("PassengerId", inplace=True)

y_train = train["Survived"]
train.drop("Survived", axis=1, inplace=True)
train_test=train.append(test)
columns_to_drop = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
train_test.drop(columns_to_drop, axis=1, inplace=True)

train_test_dummies = pd.get_dummies(train_test, columns=["Sex"])
train_test_dummies.fillna(value=0.0, inplace=True)
X_train = train_test_dummies[0:891]
X_test = train_test_dummies[891:]

scale = MinMaxScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0, test_size=0.3)

learning_rate = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for i in learning_rate:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate=i, max_features=2, max_depth=2, random_state=0)
    gb.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", i)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
    print()


