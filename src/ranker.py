
import polars as pl
import random
import catboost as cb
from catboost import Pool
import pickle

random.seed(2025)


class SecondStageRanker:

    def _set_if_none(self, df, deafault_func, df_historical=None):
        if df is None:
            assert df_historical is not None, "Historical features must be provided if df is None"
            df = deafault_func(df_historical)

        else:
            assert 'cookie' in df, "Historical features must contain 'category' and 'cookie' columns"

        return df


    def __init__(
            self, df_ranker, df_hist, df_node, model, df_hist_cat_feats=None, df_hist_loc_feats=None
        ):
        assert 'node' in df_node and 'category' in df_node and 'node_most_freq_location' in df_node, \
            "df_node must contain 'node', 'category' and 'node_most_freq_location' columns"

        self.df_node = df_node
        self.df_hist_cat_feats = self._set_if_none(df_hist_cat_feats, self._get_historical_cat_features, df_hist)
        self.df_hist_loc_feats = self._set_if_none(df_hist_loc_feats, self._get_historical_location_features, df_hist)

        print(self.df_hist_loc_feats.head())
        print(self.df_hist_cat_feats.head())


        # prepare training data
        self.df_ranker = df_ranker
        if self.df_ranker is not None:
            self.df_ranker = self._join_features(self.df_ranker)

        self.model = model
        self.features = []
        self.cat_features = []
    

    def _join_features(self, df) -> pl.DataFrame:
        df = df.join(
            self.df_node, 
            on = ['node'],
            how='left'
        ) # add category, node_most_freq_location and other feats of the current node

        df = df.join(
            self.df_hist_cat_feats, on=['category', 'cookie'], how='left'
        ) # add historical features of that category

        df = df.join(
            self.df_hist_loc_feats,
            left_on=['node_most_freq_location', 'cookie'],
            right_on=['location', 'cookie'],
            how='left'
        ) # add historical features of that location

        return df


    def _get_historical_cat_features(self, df) -> pl.DataFrame:
        assert 'category' in df and 'cookie' in df

        df_features = df.group_by(['cookie', 'category']).agg(
            [
                pl.col('is_target').sum().alias('num_contacts'),
                pl.col('is_target').count().alias('num_events'),
                pl.col('is_target').mean().alias('pr_contact'),
                pl.col('surface').unique().alias('surface_unique_counts'),
                pl.col('location').unique().alias('location_unique_counts'),

                pl.col('surface').mode().first().alias('most_freq_node'),
                pl.col('event').mode().first().alias('most_freq_event'),

                pl.col('node').last().alias('node_last'),
                pl.col('node').filter(pl.col('is_target') == 1).last().alias('node_last_contact'),
                pl.col('location_top').filter(pl.col('is_target') == 1).last().alias('location_top_last_contact'),

                pl.col('node').filter(pl.col('is_target') == 1).mode().first().alias('most_freq_node_contact'),
                pl.col('location_top').filter(pl.col('is_target') == 1).mode().first().alias('most_freq_top_location_contact'),
                pl.col('event').filter(pl.col('is_target') == 1).mode().first().alias('most_freq_event_contact'),



            ]
        ).with_columns(
            [
                pl.col('surface_unique_counts').list.len(),
                pl.col('location_unique_counts').list.len()
            ]
        )

        return df_features
    
    def _get_historical_location_features(self, df):
        assert 'location' in df and 'cookie' in df

        df_features = df.group_by(['cookie', 'location']).agg(
            [
                pl.col('is_target').sum().alias('num_contacts_location'),
                pl.col('is_target').count().alias('num_events_location'),
                pl.col('is_target').mean().alias('pr_contact_location'),
            ]
        )

        return df_features

    
    def split_data(self, df, features=(), cat_features=(), eval_ratio=0.1):
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

        eval_pool = self._get_pool(df.filter(pl.col('cookie').is_in(eval_users)), features, cat_features)
        test_pool = self._get_pool(df.filter(pl.col('cookie').is_in(test_users)), features, cat_features)
        train_pool = self._get_pool(df.filter(pl.col('cookie').is_in(train_users)), features, cat_features)

        print(f"train: {len(train_users)}, eval: {len(eval_users)}, test: {len(test_users)}")
        print(f"train: {train_pool.shape}, eval: {eval_pool.shape}, test: {test_pool.shape}")
        
        self.train_pool, self.eval_pool, self.test_pool =  train_pool, eval_pool, test_pool
    
    def _get_pool(self, df, features, cat_features = None):
        df = df.sort(by = ['cookie', 'node'])
        group_id = df['cookie'].to_list()
        y = df['is_target'].to_list()
        assert 'is_target' not in features, "is_target should not be in features"
        
        pool = Pool(
            data=df[features].to_pandas(),
            group_id=group_id, 
            label = y, 
            cat_features=cat_features,
        )
        return pool


    def fit(self):
        self.model.fit(X= self.train_pool, metric_period=100, eval_set=self.eval_pool)


    def evaluate(self):
        stat = self.model.eval_metrics(self.test_pool, metrics=["RecallAt:top=40"], ntree_start=self.model.tree_count_-1 )
        for k in stat:
            print(f"{k}: {stat[k]}")

    
    def predict_test(self, df_test_pred: pl.DataFrame, k=40) -> pl.DataFrame:

        df = self._join_features(df_test_pred)

        pool = Pool(
            data=df[self.features].to_pandas(),
            group_id= df['cookie'].to_list(), 
            cat_features=self.cat_features,
        )

        df_test_pred = df_test_pred.with_columns(
            catboost_score = self.model.predict(pool)
        )
        df_submit_pred = df_test_pred.sort(by=['cookie', 'catboost_score'], descending=True).group_by('cookie').head(k)[['cookie', 'node']]

        # df_submit_pred.write_csv(file)

        return df_submit_pred
    
    def save(self):
        self.model.save_model("catboost_ranker_model.cbm", format="cbm")
        print("Model saved as catboost_ranker_model.cbm")

        state = {
            'features': self.features,
            'cat_features': self.cat_features,
        }

        with open("catboost_ranker_state.pkl", "wb") as f:
            pickle.dump(state, f)


        self.df_node.write_parquet("df_node.pq")
        self.df_hist_cat_feats.write_parquet("ranker_historical_cat_features.pq")
        self.df_hist_loc_feats.write_parquet("ranker_historical_loc_features.pq")


    @classmethod
    def load(cls, path):
        model_path = f"{path}/catboost_ranker_model.cbm"
        state_path = f"{path}/catboost_ranker_state.pkl"


        df_cat_features = pl.read_parquet(f"{path}/ranker_cat_features.pq")
        df_historical_cat_feats = pl.read_parquet(f"{path}/ranker_historical_cat_features.pq")

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

        new_obj = cls(None, None, df_cat_features, model, df_historical_cat_feats)
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            new_obj.features = state['features']
            new_obj.cat_features = state['cat_features']

        print(f"Ranker loaded from {model_path}")

        return new_obj