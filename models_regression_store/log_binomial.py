import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../base_dados/out_binomial.csv').dropna()

x_train = data.drop('complexity', axis=1).values
y_train = data['complexity'].values

logmodel = LogisticRegression(n_jobs=5, max_iter=500)
logmodel.fit(x_train, y_train)

with open('logistic_regression_binomial.pkl', 'wb') as file:
    pickle.dump(logmodel, file)
