import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ImportData
from sklearn.model_selection import GridSearchCV

ImportData = ImportData.ImportData()

# Initialize a random state
random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))

# Choose the data you want to run it on

# SemmedDB
#ImportData.TP_files = ['../data/c_drug_treats_disease.csv']
#ImportData.TN_files = ['../data/c_tn.csv']

# SemmedDB plus NDF, do nothing

# Just NDF
#ImportData.TP_files = ['../data/NDF_TP_curie.csv']
#ImportData.TN_files = ['../data/NDF_TN_curie.csv']

#ImportData.cutoff = 3

# Import the data
X, y, id_list = ImportData.import_data()

model = RandomForestClassifier(class_weight='balanced')
param_grid = {'n_estimators': [10, 50, 100, 1000],
			'max_features': ['auto', 10, 25, 50, 100],
			'max_depth': [10, 20, 50],
			'min_samples_split': [2, 10, 50, 100],
			'min_samples_leaf': [1, 10, 100, 1000],
			}

clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, verbose=100)

res = clf.fit(X, y)

print(clf.best_score_)

print(clf.best_estimator_)