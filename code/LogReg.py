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
#random_state = np.random.RandomState(11235813)
#np.random.seed(112358)

try:
    node_vec = pd.read_csv('/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/rel_large_2_1.emb', sep = ' ', skiprows=1, header = None, index_col=None)
    map_df = pd.read_csv('/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/map.csv', index_col=None)
except FileNotFoundError:
    node_vec = pd.read_csv('rel_large_1_4.emb', sep = ' ', skiprows=1, header = None, index_col=None)
    map_df = pd.read_csv('map.csv', index_col=None)


node_vec = node_vec.sort_values(0).reset_index(drop=True)

#print(node_vec)
#print(map_df)

map_dict = {}

for row in range(len(map_df)):
    map_dict[map_df['curie'][row]] = map_df['id'][row]

try:
    TP = pd.read_csv('/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_drug_treats_disease.csv',index_col=None)
    TN = pd.read_csv('/home/dkoslicki/Desktop/RTX/FinnStuff/snap/snap-master/examples/node2vec/LogReg/c_tn.csv',index_col=None)
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

TP_flag = 'count' in list(TP)
TN_flag = 'count' in list(TN)

for row in range(len(TP)):
    if TP_flag:
        if int(TP['count'][row]) < 3:
            continue
    try:
        source_id = map_dict[TP['source'][row]]
        target_id = map_dict[TP['target'][row]]
        id_list += [source_id, target_id]
    except KeyError:
        c += 1
        continue

    X1 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]
    y1 += [1]

print(len(y1))

for row in range(len(TN)):
    if TN_flag:
        if int(TN['count'][row]) < 3:
            continue
    try:
        source_id = map_dict[TN['source'][row]]
        target_id = map_dict[TN['target'][row]]
        id_list += [source_id, target_id]
    except KeyError:
        c += 1
        continue

    X2 += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]
    y2 += [0]

print(len(y2))

#u_subset = np.random.choice(np.arange(len(y1)), len(y2), replace = False)
#o_subset = np.random.choice(np.arange(len(y2)), len(y1), replace = True)

X1 = np.array(X1)
y1 = np.array(y1)
X2 = np.array(X2)
y2 = np.array(y2)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,y2))

model = lm.LogisticRegression(class_weight='balanced', C=.5)

from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(class_weight='balanced', n_estimators=10, max_depth=5)  # Doesn't over-fit

from sklearn.ensemble import VotingClassifier
#model = VotingClassifier(estimators=[('lr', model1), ('gnb', model2), ('rf', model3)], voting='soft')

#model = skl.linear_model.LogisticRegressionCV(class_weight='balanced', n_jobs=-1)

from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=100)

cv = ms.StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)

from PlotLearningCurve import plot_learning_curve
import numpy as np
plot_learning_curve(model, "Learning curve", X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1, shuffle=True)

tprs = []
aucs = []
f1s = []
mean_fpr = np.linspace(0, 1, 100)

i = 0

shuffled_idx = np.arange(len(y))
np.random.shuffle(shuffled_idx)

test_f1_mean = np.mean(ms.cross_val_score(model, X[shuffled_idx], y[shuffled_idx], cv=10, n_jobs=-1, scoring='f1'))
print('using cross val score F1 = %0.4f' % (test_f1_mean))

for train, test in cv.split(X, y):
    model_i = model.fit(X[train], y[train])
    probas_ = model_i.predict_proba(X[test])
    pred = model_i.predict(X[test])
    f1 = met.f1_score(y[test], pred, average='binary')
    f1s.append(f1)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = met.roc_curve(y[test], probas_[:, 1])
    tprs.append(sci.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = met.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.4f, F1 = %0.4f)' % (i, roc_auc, f1))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Coin Flip', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_f1 = np.mean(f1s)
mean_tpr[-1] = 1.0
mean_auc = met.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=u'Mean ROC (AUC = %0.4f \u00B1 %0.4f, \n        \
            Mean F1 = %0.4f)' % (mean_auc, std_auc, mean_f1),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#model_all = model.fit(X,y)

#ex.joblib.dump(model, 'LReg_under.pkl')


"""
comb = '/home/womackf/all_drug_dis.csv'


print('loading ', comb)

all_df = pd.read_csv(comb).sample(frac=1/500).reset_index(drop=True)

x_i = []
for row in range(20):
    try:
        source_id = map_dict[all_df['drug'][row]]
        target_id = map_dict[all_df['disease'][row]]

        print([all_df['drug'][row],all_df['disease'][row]])
    except KeyError:
        continue

    x_i += [list(node_vec.iloc[source_id,1:]) + list(node_vec.iloc[target_id,1:])]


ex.joblib.dump(model, 'Lreg1.pkl')
"""

