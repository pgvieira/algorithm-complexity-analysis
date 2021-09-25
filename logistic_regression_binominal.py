import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']
data_frame = pd.DataFrame(columns=array_suport)

data = pd.read_csv('base_dados/out_best_performance.csv').dropna()

x_train, x_test, y_train, y_test = train_test_split(data.drop('complexity', axis=1), data['complexity'], test_size=.3, random_state=0)

logmodel = LogisticRegression(n_jobs=5, max_iter=500)
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)

print(logmodel.score(x_test, y_test))

print('Predictions', Counter(predictions))
print('True values', Counter(y_test))

print('Matrix de Confusão')
print(confusion_matrix(y_test, predictions))

print('Precisão Computacional')
print(classification_report(y_test, predictions))


for i in range(len(y_test)):
    data_frame.loc[i] = x_test.values[i]

array_true = []

for i in range(len(y_test)):
    array_true.append(predictions[i] == y_test.values[i])

data_frame = data_frame.assign(truefalse=array_true)

print(data_frame)


