# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import ImportData
ImportData = ImportData.ImportData()
from sklearn import decomposition
from sklearn import datasets

random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))
#random_state = np.random.RandomState(11235813)
#np.random.seed(112358)

# Choose the data you want to run it on

# SemmedDB
ImportData.TP_files = ['../data/c_drug_treats_disease.csv']
ImportData.TN_files = ['../data/c_tn.csv']

# SemmedDB plus NDF, do nothing

# Import the data
X, y, id_list = ImportData.import_data()
print(len(X))
print(len(y))
print(np.unique(y, return_counts=True))

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)


for name, label in [('Does Not Treat', 0), ('Treats', 1)]:
	ax.text3D(X[y == label, 0].mean(),
			  X[y == label, 1].mean(),
			  X[y == label, 2].mean(), name,
			  horizontalalignment='center',
			  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
y2 = []
for i in range(len(y)):
	if y[i] == 1.0:
		y2.append('y')
	if y[i] == 0.0:
		y2.append('m')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y2, cmap=plt.cm.viridis, edgecolor='k')
plt.title("PCA. Treats=yellow, Doesn't_treat=maroon")

"""
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
"""

plt.show()