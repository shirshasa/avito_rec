
from tools import recall_at

import polars as pl
import pickle


class CandidatesGenerator:
    """
    A wrapper to ALS, KNN and other RecTools models to generate candidates.
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.state = {}

    def fit(self):
        self.model.fit(self.dataset)
        self.state['model'] = self.model
        self.state['dataset'] = self.dataset

    def refit(self, dataset):
        self.dataset = dataset
        self.model.fit(self.dataset)
        self.state['model'] = self.model
        self.state['dataset'] = self.dataset
        print("Model refitted with new dataset.")

    def evaluate(self, df_eval, k = 40):
        eval_users = df_eval['cookie'].unique().to_list()

        recos = self.model.recommend(
            users=eval_users,
            dataset=self.dataset,
            k=k,
            filter_viewed=True,
        )
        
        df_pred = recos[['user_id', 'item_id']]
        df_pred.columns = ['cookie', 'node']

        df_pred_pl = pl.from_pandas(df_pred).with_columns(
            pl.col("cookie").cast(pl.Int64),
            pl.col("node").cast(pl.Int64)
        )
        
        recall = recall_at(df_eval, df_pred_pl, k=k)
        print(f"recall@{k}", recall)

        self.state[f"recall@{k}"] = recall
        # self.state['eval_preds'] = df_pred
        # self.state['eval_preds@k'] = k
        return df_pred_pl


    def predict(self, test_users: list, k=40) -> pl.DataFrame:
        recos = self.model.recommend(
            users=test_users,
            dataset=self.dataset,
            k=k,
            filter_viewed=True,
        )
        
        df_pred = recos[['user_id', 'item_id']]
        df_pred.columns = ['cookie', 'node']
        df_pred_pl = pl.from_pandas(df_pred).with_columns(
            pl.col("cookie").cast(pl.Int64),
            pl.col("node").cast(pl.Int64)
        )
        # self.state['test_preds'] = df_pred_pl
        # self.state['test_preds@k'] = k
        return df_pred_pl
    
    def save_submit(self, path):
        df_pred = self.state['test_preds'][['cookie', 'node']]
        df_pred.write_csv(path, index=False)
        print(f"Submission saved to {path}")
        return df_pred
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
        print(f"Model state saved to {path}")
        return self.state
    

    def load(self, path):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)

        self.model = self.state['model']
        self.dataset = self.state['dataset']
        
        print(f"Model state loaded from {path}")
        return self.state
