import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('base_dados/out_type_classes.csv').dropna()

x_train, x_test, y_train, y_test = train_test_split(data.drop('complexity', axis=1), data['complexity'], test_size=.3, random_state=0)

logmodel = LogisticRegression(n_jobs=5, max_iter=1000)
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)

print(logmodel.score(x_test, y_test))

print('Predictions', Counter(predictions))
print('True values', Counter(y_test))

print('Matrix de Confusão')
print(confusion_matrix(y_test, predictions))

print('Precisão Computacional')
print(classification_report(y_test, predictions))
