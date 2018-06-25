# Train the model
model = RandomForestClassifier(class_weight='balanced', max_depth=15, max_leaf_nodes=None, n_estimators=500, min_samples_leaf=1, min_samples_split=2, max_features=50, n_jobs=-1, oob_score=True)
model.fit(X, y)


import sys
import pandas as pd
sys.path.append('/home/dkoslicki/Dropbox/Repositories/RTX/code/reasoningtool/QuestionAnswering')
import ReasoningUtilities as RU

# Need to properly match this up with X since some rows have been removed from Sem to get X

# Just look at a few
probs = []
for itr in range(10000):
	i = np.random.randint(len(X))
	j = np.random.randint(len(X))
	prob_treats = model.predict_proba([np.concatenate([X[i, 0:128], X[j, 128:]])])[0,0]
	if prob_treats > .99:
		probs.append((i, j, prob_treats))

probs_sorted = sorted(probs, key=lambda x: x[2], reverse=True)

for i, j, prob in probs_sorted[0:10]:
	source_curie = curie_list[i][0]
	target_curie = curie_list[j][1]
	source_name = RU.get_node_property(source_curie, "name", node_label="chemical_substance", name_type="id")
	target_name = RU.get_node_property(target_curie, "name", node_label="disease", name_type="id")
	print("%s (%s)-> %s (%s), %f\n" % (source_name, source_curie, target_name, target_curie, prob))
