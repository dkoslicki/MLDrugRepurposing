import sklearn.linear_model as lm
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

class predictor():

    def __init__(self, model_file = 'LogModel.pkl'):
        self.model = joblib.load(model_file)

    def prob(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def import_file(self, file, graph_file = 'rel_max.emb.gz', map_file = 'map.csv'):
        graph = pd.read_csv(graph_file, sep = ' ', skiprows=1, header = None, index_col=None)
        self.graph = graph.sort_values(0).reset_index(drop=True)
        self.map_df = pd.read_csv(map_file, index_col=None)

        data = pd.read_csv(file, index_col=None)

        map_dict = {}
        X_list = []
        drop_list = []
        for row in range(len(data)):
            source_id = self.map_df.loc[self.map_df['curie'] == data['source'][row], 'id']
            target_id = self.map_df.loc[self.map_df['curie'] == data['target'][row], 'id']
            if len(source_id) >0 and len(target_id)>0:
                source_id = source_id.iloc[0]
                target_id = target_id.iloc[0]
                X_list += [list(self.graph.iloc[source_id,1:]) + list(self.graph.iloc[target_id,1:])]
            else:
                drop_list += [row]

        print(drop_list)

        self.X = np.array(X_list)
        self.data = data.drop(data.index[drop_list]).reset_index(drop=True)
        self.dropped_data = data.iloc[drop_list].reset_index(drop=True)

    def prob_file(self):
        return self.prob(self.X)

    def predict_file(self):
        return self.predict(self.X)

    def build_pred_df(self):
        probs = self.prob_file()
        preds = self.predict_file()
        df = self.data.copy()
        df['treat_prob'] = [prob[1] for prob in probs]
        df['prediction'] = preds
        df = df.sort_values('treat_prob', ascending = False).reset_index(drop=True)
        return df

    def build_pred_df_all(self):
        probs = self.prob_file()
        preds = self.predict_file()
        df = pd.concat([self.data,self.dropped_data])
        df['treat_prob'] = [prob[1] for prob in probs] + [np.nan]*len(self.dropped_data)
        df['prediction'] = list(preds) + [np.nan]*len(self.dropped_data)
        df = df.sort_values('treat_prob', ascending = False).reset_index(drop=True)
        return df

    def test(self):
        self.import_file('test_set.csv')
        print('df w/o nodes not in largest connected component:')
        print('------------------------------------------------')
        df = self.build_pred_df()
        print(df)
        print('\n\n')
        print('df with nodes not in largest connected component:')
        print('-------------------------------------------------')
        df_all = self.build_pred_df_all()
        print(df_all)

#if __name__ == '__main__':
#	pr = predictor()
#	pr.test()



