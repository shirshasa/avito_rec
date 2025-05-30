
import polars as pl
import random
import catboost as cb
from catboost import Pool
import pickle
import os
from datetime import datetime

random.seed(2025)


class SecondStageRanker:

    def __init__(self, df):

        # prepare training data
        self.df = df
        self.features = []
        self.cat_features = []

        params = {
            "boosting_type": "Plain",
            "early_stopping_rounds": 20,
            "eval_metric": "AUC",
            "learning_rate": 0.1,
            "max_ctr_complexity": 1,
            "nan_mode": "Min",
            "num_trees": 150,
            "objective": "PairLogitPairwise:max_pairs=50",
            "random_state": 42,
            "task_type": "CPU",
        }
        self.model = cb.CatBoost(params = params)

    
    
    def split_data(self, features=(), cat_features=(), emb_features=(), eval_ratio=0.1):
        df = self.df
        self.features = features
        self.cat_features = cat_features if cat_features is not None else []
    

        if features is None:
            features = df.columns

        assert 'cookie' in df.columns, "DataFrame must contain 'cookie' column"
        assert 'is_target' not in features, "is_target should not be in features"

        all_users = df['cookie'].unique().to_list()
        random.shuffle(all_users)
        num_eval = int(len(all_users) * eval_ratio)
        eval_users, test_users, train_users = all_users[:num_eval], all_users[num_eval: 2 * num_eval], all_users[2 * num_eval:]

        eval_pool = self._get_pool(df.filter(pl.col('cookie').is_in(eval_users)), features, cat_features, emb_features)
        test_pool = self._get_pool(df.filter(pl.col('cookie').is_in(test_users)), features, cat_features, emb_features)
        train_pool = self._get_pool(df.filter(pl.col('cookie').is_in(train_users)), features, cat_features, emb_features)

        print(f"train: {len(train_users)}, eval: {len(eval_users)}, test: {len(test_users)}")
        print(f"train: {train_pool.shape}, eval: {eval_pool.shape}, test: {test_pool.shape}")
        
        self.train_pool, self.eval_pool, self.test_pool =  train_pool, eval_pool, test_pool
    
    def _get_pool(self, df, features, cat_features = None, emb_features = None):
        df = df.sort(by = ['cookie', 'node'])
        group_id = df['cookie'].to_list()
        y = df['is_target'].to_list()
        assert 'is_target' not in features, "is_target should not be in features"
        
        pool = Pool(
            data=df[features].to_pandas(),
            group_id=group_id, 
            label = y, 
            cat_features=cat_features,
            embedding_features=emb_features
        )
        return pool


    def fit(self):
        self.model.fit(X= self.train_pool, verbose=10, eval_set=self.eval_pool, use_best_model=True)


    def evaluate(self):
        stat = self.model.eval_metrics(
            self.test_pool, 
            metrics=["AUC", "Accuracy", "PrecisionAt:top=40", "RecallAt:top=40"], 
            ntree_start=self.model.tree_count_-1 # type: ignore
        )
        for k in stat:
            print(f"{k}: {stat[k]}")

    
    def predict_test(self, df: pl.DataFrame, k=40) -> pl.DataFrame:
        df = df.sort(by=['cookie'])
        pool = Pool(
            data=df[self.features].to_pandas(),
            group_id= df['cookie'].to_list(), 
            cat_features=self.cat_features,
        )
        
        df = df.with_columns(
            catboost_score = self.model.predict(pool)
        )
        print(f"Predictions done..")
        print(df.head())
        df_submit_pred = df.sort(by=['cookie', 'catboost_score'], descending=True).group_by('cookie').head(k)[['cookie', 'node', 'catboost_score']]

        return df_submit_pred
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"Path {path} already exists.")
            cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{path}_{cur_time}"
            if not os.path.exists(path):
                os.makedirs(path)
        
        model_path = f"{path}/catboost_ranker_model.cbm"
        state_path = f"{path}/catboost_ranker_state.pkl"

        self.model.save_model(model_path, format="cbm")
        print("Model saved as catboost_ranker_model.cbm")

        state = {
            'features': self.features,
            'cat_features': self.cat_features,
        }

        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        print(f"Ranker saved to {path}.")


    @classmethod
    def load(cls, path):
        model_path = f"{path}/catboost_ranker_model.cbm"
        state_path = f"{path}/catboost_ranker_state.pkl"

        params = {
            "boosting_type": "Plain",
            "early_stopping_rounds": 20,
            "eval_metric": "AUC",
            "learning_rate": 0.1,
            "max_ctr_complexity": 1,
            "nan_mode": "Min",
            "num_trees": 100,
            "objective": "PairLogitPairwise:max_pairs=50",
            "random_state": 42,
            "task_type": "CPU"
        }
        
        model = cb.CatBoost(params = params)
        model.load_model(model_path, format="cbm")

        new_obj = cls(None)
        new_obj.model = model
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            new_obj.features = state['features']
            new_obj.cat_features = state['cat_features']

        print(f"Ranker loaded from {model_path}")

        return new_obj