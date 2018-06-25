import scipy as sci
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.linear_model as lm
import sklearn.externals as ex
import sklearn.metrics as met
import sklearn.model_selection as ms
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from PlotLearningCurve import plot_learning_curve
import numpy as np
import ImportData
ImportData = ImportData.ImportData()

# Initialize a random state
random_state = np.random.RandomState(int(time.time()))
np.random.seed(int(time.time()/100))

# SemmedDB
ImportData.TP_files = ['../data/c_drug_treats_disease.csv']
ImportData.TN_files = ['../data/c_tn.csv']
ImportData.cutoff = 2
X_sem, y_sem, id_list, curie_list = ImportData.import_data()

# NDF
import ImportData
ImportData = ImportData.ImportData()
ImportData.TP_files = ['../data/NDF_TP_curie.csv']
ImportData.TN_files = ['../data/NDF_TN_curie.csv']
X_NDF, y_NDF, id_list_NDF, curie_list = ImportData.import_data()

# Make the choice of who is train and who is test
X_train = X_NDF
y_train = y_NDF

X_test = X_sem
y_test = y_sem

# Fit
#model = lm.LogisticRegression(class_weight='balanced', C=.5)
model = RandomForestClassifier(class_weight='balanced', max_depth=15, max_leaf_nodes=None, n_estimators=500, min_samples_leaf=1, min_samples_split=2, max_features=50, n_jobs=-1, oob_score=True)
model.fit(X_train, y_train)



# Test
probs = model.predict_proba(X_test)
pred = model.predict(X_test)
f1 = met.f1_score(y_test, pred, average='binary')
fpr, tpr, thresholds = met.roc_curve(y_test, probs[:, 1])
roc_auc = met.auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.4f, F1 = %0.4f)' % (roc_auc, f1))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Coin Flip', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


