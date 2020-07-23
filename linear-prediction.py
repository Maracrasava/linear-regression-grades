import tensorflow
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv('student-mat.csv', sep=";")

# Turning internet column into integers with 0 and 1
data["internet"] = data["internet"].map({"yes": 1, "no": 0})

data = data[["G1", "G2", "G3", "internet", "failures", "studytime", "absences", "freetime", "health", "age"]]


forecast = "G3"

# Deleting G3 column, where 1 is the axis number (0 for rows and 1 for columns.)
X = np.array(data.drop([forecast], 1))
# Output G3 column for testing and training
y = np.array(data[forecast])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)
acc = linear.score(X_test, y_test)
print(acc)

# print(f'{linear.coef_} are coefficients' + f' and {linear.intercept_} is intercept of our hypotheses')

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])
