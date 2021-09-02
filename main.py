import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

complexity_dictionary = {'n': 0, 'n_square': 1, '1': 2, 'nlogn': 3, 'logn': 4}

dataframe_training = pd.read_csv('base_dados/finalFeatureData.csv').dropna()
dataframe_training = dataframe_training.drop(columns=['file_name'])
x_array_training, x_array_testing = dataframe_training.iloc[:, :13].values, dataframe_training.iloc[:, 14:].values

x_array_training = [list(map(int, x)) for x in x_array_training]
x_array_testing = np.array([int(complexity_dictionary[row[0]]) for row in x_array_testing])

dataframe_testing = pd.read_csv('base_dados/testFeatureData.csv').dropna()
dataframe_testing = dataframe_testing.drop(columns=['file_name'])
y_array_training, y_array_testing = dataframe_testing.iloc[:, :13].values, dataframe_testing.iloc[:, 14:].values

y_array_training = [list(map(int, x)) for x in y_array_training]
y_array_testing = np.array([int(complexity_dictionary[row[0]]) for row in y_array_testing])

# wcss = []
# for i in range(1, 50):
#    kmeans = KMeans(n_clusters=i, init='random').fit(x_array_training)
#    print(i, kmeans.inertia_)
#    wcss.append(kmeans.inertia_)
# plt.plot(range(1, 50), wcss)
# plt.title('O Metodo Elbow')
# plt.xlabel('Numero de Clusters')
# plt.ylabel('WSS')
# plt.show()

kmeans = KMeans(n_clusters=5, init='random', random_state=534, max_iter=534)
kmeans.fit(x_array_training)
y_predicted = kmeans.predict(y_array_training)

print(accuracy_score(y_predicted, y_array_testing) * 100)
print(y_predicted)
print(y_array_testing)
