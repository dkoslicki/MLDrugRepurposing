import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ImportData
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
			'max_depth': [None, 10, 20, 50],
			'min_samples_split': [2, 10, 50, 100],
			'min_samples_leaf': [1, 10, 100, 1000],
			'max_leaf_nodes': [None, 10, 100, 1000]
			}


cv = ms.StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)

clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, verbose=100)

res = clf.fit(X, y)

print(clf.best_score_)

print(clf.best_estimator_)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r"
		  % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

"""
[CV]  max_depth=None, max_leaf_nodes=None, n_estimators=200, min_samples_leaf=1, min_samples_split=2, max_features=100, score=0.9243212016175621

"""