import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']
data_frame = pd.DataFrame(columns=array_suport)

data = pd.read_csv('base_dados/out_multinomial.csv').dropna()

data_values = data.iloc[:, :13].values

data_complexity = data.iloc[:, 13:].values
data_complexity = [int(x) for x in data_complexity]

logmodel = LogisticRegression(n_jobs=5, max_iter=1000).fit(data_values, data_complexity)
predictions = logmodel.predict(data_values)

print(logmodel.score(data_values, data_complexity))

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

with open('models_regression_store/logistic_regression_multinomial.pkl', 'wb') as file:
    pickle.dump(logmodel, file)


plt.figure(figsize=(20, 12))

labels = ['n', 'n_square', '1', 'nlogn', 'logn']
predictions = [Counter(predictions).get(0), Counter(predictions).get(1), Counter(predictions).get(2), Counter(predictions).get(3), Counter(predictions).get(4)]
data_complexity = [Counter(data_complexity).get(0), Counter(data_complexity).get(1), Counter(data_complexity).get(2), Counter(data_complexity).get(3), Counter(data_complexity).get(4)]

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
plt.savefig('graficos/logistic_regression_multinomial.png', format='png')
plt.show()
