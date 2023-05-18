from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X,y = data.data, data.target

print('X',X.shape)
print('y', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

print('X_train',X_train.shape)
print('y_train', y_train.shape)
print('X_test',X_test.shape)
print('y_test', y_test.shape)

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, y_pred)

print(acc)

