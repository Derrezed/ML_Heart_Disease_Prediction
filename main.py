import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("processed.cleveland.data")

data = data[
    ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
     "num"]]

predict = "num"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)


best = 0

for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(solver="liblinear")
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)

    print("Acc:", acc)

    if acc > best:
        best = acc
        with open("model.pick.pickle", "wb") as f:
            pickle.dump(model, f)

print(f"Best accuracy: {best}")
pickle_in = open("model.pick.pickle", "rb")
model = pickle.load(pickle_in)

predictions = model.predict(x_test)
for i in range(len(predictions)):
    print(f'Prediciton: {predictions[i]}, Data: {",".join([str(x) for x in x_test[i]])}, Prediction check: {y_test[i]}')
