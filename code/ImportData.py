import pandas as pd
import numpy as np
import time

random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))

class ImportData:
	def __init__(self):
		self.TP_files = ['../data/c_drug_treats_disease.csv', '../data/NDF_TP_curie.csv']
		self.TN_files = ['../data/c_tn.csv', '../data/NDF_TN_curie.csv']
		self.node_vec_file = '../data/rel_large_2_1.emb.tar.gz'
		self.map_df_file = '../data/map.csv'
		self.cutoff = 2  # include semmeddb stuff if it has at least this many associated publications

	def import_data(self):
		"""
		Import the true positives and true negatives into a form usable by sklearn
		:param TP_files: list of csv file names
		:param TN_files: list of csv file name
		:return: (X, y, id_list) vectors (features, classes, ids)
		"""
		TP_list = []
		for file in self.TP_files:
			TP_list.append(pd.read_csv(file, index_col=None))

		TN_list = []
		for file in self.TN_files:
			TN_list.append(pd.read_csv(file, index_col=None))

		node_vec = pd.read_csv(self.node_vec_file, compression='gzip', sep=' ', skiprows=1, header=None, index_col=None)
		map_df = pd.read_csv(self.map_df_file, index_col=None)

		node_vec = node_vec.sort_values(0).reset_index(drop=True)

		map_dict = {}

		for row in range(len(map_df)):
			map_dict[map_df['curie'][row]] = map_df['id'][row]

		X_TP = []
		X_TN = []

		c = 0

		id_list = []
		id_list_dict = dict()
		curie_list = []

		for TP in TP_list:
			for row in range(len(TP)):
				if 'count' in list(TP):
					if int(TP['count'][row]) < self.cutoff:
						continue
				try:
					source_id = map_dict[TP['source'][row]]
					target_id = map_dict[TP['target'][row]]
					source_curie = TP['source'][row]
					target_curie = TP['target'][row]
				except KeyError:
					c += 1
					continue

				if (source_id, target_id) not in id_list_dict:
					id_list += [[source_id, target_id]]
					id_list_dict[source_id, target_id] = 1
					curie_list += [[source_curie, target_curie]]
					X_TP += [list(node_vec.iloc[source_id, 1:]) + list(node_vec.iloc[target_id, 1:])]

		y_TP = [1]*len(X_TP)

		for TN in TN_list:
			for row in range(len(TN)):
				if 'count' in list(TN):
					if int(TN['count'][row]) < self.cutoff:
						continue
				try:
					source_id = map_dict[TN['source'][row]]
					target_id = map_dict[TN['target'][row]]
					source_curie = TN['source'][row]
					target_curie = TN['target'][row]
				except KeyError:
					c += 1
					continue

				if (source_id, target_id) not in id_list_dict:
					id_list += [[source_id, target_id]]
					id_list_dict[source_id, target_id] = 1
					curie_list += [[source_curie, target_curie]]
					X_TN += [list(node_vec.iloc[source_id, 1:]) + list(node_vec.iloc[target_id, 1:])]

		y_TN = [0]*len(X_TN)

		X_TP = np.array(X_TP)
		y_TP = np.array(y_TP)
		X_TN = np.array(X_TN)
		y_TN = np.array(y_TN)
		X = np.concatenate((X_TP,X_TN))
		y = np.concatenate((y_TP,y_TN))
		return X, y, id_list, curie_list