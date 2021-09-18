import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

data = pd.read_csv('./out.csv').dropna()

data_values = data.iloc[:, :6].values
data_complexity = data.iloc[:, 6:].values

data_values = [list(map(int, x)) for x in data_values]
data_complexity = [int(x) for x in data_complexity]

X_train, X_test, Y_train, Y_test = train_test_split(data_values, data_complexity, train_size=.7, random_state=10)

p = MaxAbsScaler()

p.fit(X_train)
X_train = p.transform(X_train)

X_test = p.transform(X_test)

kmeans = KMeans(n_clusters=5, random_state=0, max_iter=1000).fit(X_train)
X_predicted = kmeans.predict(X_test)

print(accuracy_score(X_predicted, Y_test) * 100)
print(X_predicted)
print(Y_test)
