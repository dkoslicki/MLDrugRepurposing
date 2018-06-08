import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.linear_model as lm
import sklearn.externals as ex
import sklearn.metrics as met
import sklearn.model_selection as ms
import time

random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))

class ImportData:
	def __init__(self):
		self.TP_files = ['/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_drug_treats_disease.csv', '/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/NDF_TP_curie.csv']
		self.TN_files = ['/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_tn.csv', '/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/NDF_TN_curie.csv']
		self.node_vec_file = '/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/rel_large_2_1.emb'
		self.map_df_file = '/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/map.csv'

	def import_data(self):
		"""
		Import the true positives and true negatives into a form usable by sklearn
		:param TP_files: list of csv file names
		:param TN_files: list of csv file name
		:return: (X, y) vectors (features, classes)
		"""
		TP_list = []
		for file in self.TP_files:
			TP_list.append(pd.read_csv(file, index_col=None))

		TN_list = []
		for file in self.TN_files:
			TN_list.append(pd.read_csv(file, index_col=None))

		node_vec = pd.read_csv(self.node_vec_file, sep=' ', skiprows=1, header = None, index_col=None)
		map_df = pd.read_csv(self.map_df_file, index_col=None)

		node_vec = node_vec.sort_values(0).reset_index(drop=True)

		map_dict = {}

		for row in range(len(map_df)):
			map_dict[map_df['curie'][row]] = map_df['id'][row]

		X1 = []
		X2 = []

		c = 0

		id_list = []

		for TP in TP_list:
			for row in range(len(TP)):
				if 'count' in list(TP):
					if int(TP['count'][row]) < 2:
						continue
				try:
					source_id = map_dict[TP['source'][row]]
					target_id = map_dict[TP['target'][row]]
				except KeyError:
					c += 1
					continue

				if [source_id, target_id] not in id_list:
					id_list += [[source_id, target_id]]
					X1 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]

		y1 = [1]*len(X1)

		for TN in TN_list:
			for row in range(len(TN)):
				if 'count' in list(TN):
					if int(TN['count'][row]) < 2:
						continue
				try:
					source_id = map_dict[TN['source'][row]]
					target_id = map_dict[TN['target'][row]]
				except KeyError:
					c += 1
					continue

				if [source_id, target_id] not in id_list:
					id_list += [[source_id, target_id]]
					X2 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]

		y2 = [0]*len(X2)

		X1 = np.array(X1)
		y1 = np.array(y1)
		X2 = np.array(X2)
		y2 = np.array(y2)
		X = np.concatenate((X1,X2))
		y = np.concatenate((y1,y2))

		return X,y