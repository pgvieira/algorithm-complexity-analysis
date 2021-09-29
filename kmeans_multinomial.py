from collections import Counter

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']
data_frame = pd.DataFrame(columns=array_suport)

data = pd.read_csv('base_dados/out_multinomial.csv').dropna()

x_train, x_test, y_train, y_test = train_test_split(data.drop('complexity', axis=1), data['complexity'], test_size=.3, random_state=0)

kmeans = KMeans(n_clusters=5, max_iter=1000, random_state=5)
kmeans.fit(x_train)
predictions = kmeans.predict(x_test)

print(accuracy_score(y_test, predictions))

print('Predictions', Counter(predictions))
print('True values', Counter(y_test))

print('Matrix de Confusão')
print(confusion_matrix(y_test, predictions))

print('Precisão Computacional')
print(classification_report(y_test, predictions))

for i in range(len(y_test)):
    data_frame.loc[i] = x_test.values[i]

array_true = [predictions[i] == y_test.values[i] for i in range(len(y_test))]

data_frame = data_frame.assign(complexity=y_test.values)
data_frame = data_frame.assign(complexity_predicted=predictions)
data_frame = data_frame.assign(acertos=array_true)

print(data_frame)
