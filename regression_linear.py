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


X = data_csv.iloc[:, :-1]
print(calculate_vif(X))

formula = "complexity ~ num_If + num_Switch + num_Loof + num_Break + num_Priority + num_Sort + num_Hash_Map + num_Hash_Set + num_Recursive + num_Nasted_Loop + num_Variables + num_Method + num_State"
mlr = ols(formula, data=data_csv)
estimates = mlr.fit()
print(estimates.summary())


formula = "complexity ~ num_If + num_Switch + num_Loof + num_Break + num_Priority + num_Sort + num_Hash_Map + num_Hash_Set + num_Recursive + num_Nasted_Loop + num_Variables + num_Method + num_State"
logit_model = logit(formula, data=data_csv)
logit_estimates = logit_model.fit()
print(logit_estimates.summary())
