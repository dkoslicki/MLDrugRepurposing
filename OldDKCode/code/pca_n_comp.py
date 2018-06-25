

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd


from sklearn import decomposition
from sklearn import datasets

random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))
#random_state = np.random.RandomState(11235813)
#np.random.seed(112358)

try:
    node_vec = pd.read_csv('/home/dkoslicki/Dropbox/Repositories/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/rel_large_2_1.emb', sep = ' ', skiprows=1, header = None, index_col=None)
    map_df = pd.read_csv('/home/dkoslicki/Dropbox/Repositories/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/map.csv', index_col=None)
except FileNotFoundError:
    node_vec = pd.read_csv('rel_large_2_1.emb', sep = ' ', skiprows=1, header = None, index_col=None)
    map_df = pd.read_csv('map.csv', index_col=None)


node_vec = node_vec.sort_values(0).reset_index(drop=True)

#print(node_vec)
#print(map_df)

map_dict = {}

for row in range(len(map_df)):
    map_dict[map_df['curie'][row]] = map_df['id'][row]

try:
    TP = pd.read_csv('/home/dkoslicki/Dropbox/Repositories/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_drug_treats_disease.csv',index_col=None)
    TN = pd.read_csv('/home/dkoslicki/Dropbox/Repositories/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_tn.csv',index_col=None)
except FileNotFoundError:
    TP = pd.read_csv('c_drug_treats_disease.csv',index_col=None)
    TN = pd.read_csv('c_tn.csv',index_col=None)


y = []
X = []

y1 = []
X1 = []

y2 = []
X2 = []

c = 0

id_list = []

for row in range(len(TP)):
    if 'count' in list(TP):
        if int(TP['count'][row]) < 2:
            continue
    try:
        if TP['disease'][row].split(':')[0] == 'OMIM':
            continue
        source_id = map_dict[TP['drug'][row]]
        target_id = map_dict[TP['disease'][row]]
        id_list += [[TP['drug'][row], TP['disease'][row]]]
    except KeyError:
        c += 1
        continue

    X1 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]
    y1 += [1]

for row in range(len(TN)):
    if 'count' in list(TN):
        if int(TN['count'][row]) < 2:
            continue
    try:
        if TN['target'][row].split(':')[0] == 'OMIM':
            continue
        source_id = map_dict[TN['source'][row]]
        target_id = map_dict[TN['target'][row]]
        id_list += [[TN['source'][row], TN['target'][row]]]
    except KeyError:
        c += 1
        continue

    X2 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]
    y2 += [0]

#u_subset = np.random.choice(np.arange(len(y1)), len(y2), replace = False)
#o_subset = np.random.choice(np.arange(len(y2)), len(y1), replace = True)

X1 = np.array(X1)
y1 = np.array(y1)
X2 = np.array(X2)
y2 = np.array(y2)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,y2))

logistic = linear_model.LogisticRegression(class_weight='balanced', C=15.5)

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# Plot the PCA spectrum
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X, y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()