# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
		#if TP['disease'][row].split(':')[0] == 'DOID':
		#    continue
		source_id = map_dict[TP['source'][row]]
		target_id = map_dict[TP['target'][row]]
		id_list += [[TP['source'][row], TP['target'][row]]]
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
		#if TN['target'][row].split(':')[0] == 'DOID':
		#    continue
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

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)


X2 = X[X[:,0] < -0.5]
print(X2)
id_list = np.array(id_list)
blob = id_list[X[:,0] < -0.5]
blob2 = id_list[X[:,0] >= -0.5]
yb = y[X[:,0] < -0.5]
yb2 = y[X[:,0] >= -0.5]

doids = 0
omims = 0
doid_list = []
omim_list = []
for row in blob:
	if row[1].split(':')[0] == 'DOID':
		doids += 1
		doid_list += [row]
	elif row[1].split(':')[0] == 'OMIM':
		omims += 1
		omim_list += [row]
	else:
		print('something is wrong...')
		print(row)

print('----------------')
if doids < omims:
	print(doid_list)
else:
	print(omim_list)


doids = 0
omims = 0
doid_list = []
omim_list = []
for row in blob2:
	if row[1].split(':')[0] == 'DOID':
		doids += 1
		doid_list += [row]
	elif row[1].split(':')[0] == 'OMIM':
		omims += 1
		omim_list += [row]
	else:
		print('something is wrong...')
		print(row)

print('----------------')
if doids < omims:
	print(doid_list)
else:
	print(omim_list)


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
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y2, cmap=plt.cm.viridis,
		   edgecolor='k')

"""
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
"""

plt.show()