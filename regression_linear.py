import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols, logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

columns = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map',
           'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State', 'complexity']

data_csv = pd.read_csv('./out.csv')


# corr = data_csv.corr()
# fig, ax = plt.subplots(figsize=(20, 16))
# colormap = sns.diverging_palette(220, 10, as_cmap=True)
# dropvals = np.zeros_like(corr)
# dropvals[np.triu_indices_from(dropvals)] = True
# sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropvals)
# plt.show()


def calculate_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif



columns_drop = ['num_Switch', 'num_Break', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Nasted_Loop', 'num_Priority', 'num_State', 'complexity']
X = data_csv.drop(columns_drop, 1)
print(calculate_vif(X))

formula = 'complexity ~ num_If * (num_Method + num_Variables + num_Recursive + num_Loof) + ' \
          'num_Loof * (num_Variables + num_Method) + ' \
          'num_Recursive * (num_Method + num_Variables) + ' \
          'num_Variables * (num_Method) + ' \
          'num_Method'
mlr = ols(formula, data=data_csv)
estimates = mlr.fit()
print(estimates.summary())
