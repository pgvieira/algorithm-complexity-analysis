import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

data = pd.read_csv('./out.csv').dropna()
data = data.drop('complexity', 1)

correlation = data.corr()

plot = sn.heatmap(correlation, annot=True, fmt='.1f', linewidths=.6)
plt.show()
