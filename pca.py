import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

data_csv = pd.read_csv('./out.csv')

X = data_csv.drop('complexity', 1)
Y = data_csv['complexity']

pca = PCA(n_components=13)
pca.fit(X)

print('Auto-valores:')
print(pca.explained_variance_)
print()

print('Auto-vetores:')
print(pca.components_)
print()

print('Variância explicada:')
print(pca.explained_variance_ratio_)
print()

X = pca.transform(X)

columns = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map',
           'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']

new_data = pd.DataFrame(X, columns=columns)
new_data['complexity'] = data_csv['complexity']

sns.pairplot(new_data, vars=columns, hue='complexity', diag_kind='hist')
plt.show()

