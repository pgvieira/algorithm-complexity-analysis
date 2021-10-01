from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']
data_frame = pd.DataFrame(columns=array_suport)

data = pd.read_csv('base_dados/out_binomial.csv').dropna()

data_values = data.iloc[:, :13].values

data_complexity = data.iloc[:, 13:].values
data_complexity = [int(x) for x in data_complexity]

kmeans = KMeans(n_clusters=2, max_iter=1000, random_state=5)
predictions = kmeans.fit_predict(data_values, data_complexity)

print(accuracy_score(data_complexity, predictions))

print('Predictions', Counter(predictions))
print('True values', Counter(data_complexity))

print('Matrix de Confusão')
print(confusion_matrix(data_complexity, predictions))

print('Precisão Computacional')
print(classification_report(data_complexity, predictions))

for i in range(len(data_complexity)):
    data_frame.loc[i] = data_values[i]

array_true = [predictions[i] == data_complexity[i] for i in range(len(data_complexity))]

data_frame = data_frame.assign(complexity=data_complexity)
data_frame = data_frame.assign(complexity_predicted=predictions)
data_frame = data_frame.assign(acertos=array_true)

print(data_frame)

plt.figure(figsize=(20, 12))

labels = ['0 { n - 1 - logn }', '1 { n_square - nlogn }']
predictions = [Counter(predictions).get(0), Counter(predictions).get(1)]
data_complexity = [Counter(data_complexity).get(0), Counter(data_complexity).get(1)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, predictions, width, label='Predicted')
rects2 = ax.bar(x + width/2, data_complexity, width, label='Real Values')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('graficos/kmeans_binomial.png', format='png')
plt.show()
