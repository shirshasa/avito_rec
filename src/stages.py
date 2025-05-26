import pandas as pd
import polars as pl
import numpy as np

import os
import threadpoolctl
import warnings

warnings.filterwarnings('ignore')

from datetime import timedelta

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender
from rectools.dataset import Dataset
from rectools.models import (
    ImplicitALSWrapperModel,
    LightFMWrapperModel,
    ImplicitItemKNNWrapperModel,
)
from lightfm import LightFM
import catboost as cb

# For vector models optimized ranking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas");

###

from data import get_train_val
from features import get_user_features, get_node_features, get_text_embeddings_user, \
    get_text_embeddings_node, load_text_embeddings, load_cat_df
from feature_pool import FeaturePool
from candidates_generator import CandidatesGenerator
from ranker import SecondStageRanker
from tools import recall_at, reduce_memory_usage_pl



RANDOM_STATE = 42

def get_interactions_df(df: pl.DataFrame, event2weight):
    df = df.with_columns(
        pl.col("event").replace_strict(event2weight, default=1).alias("weight")
    )
    interactions = df.select(["cookie", 'event_date', 'node', 'weight'])
    interactions = interactions.rename(
        {
            "cookie": "user_id",
            "node": "item_id",
            "event_date": "datetime"
        }
    )
    return interactions.to_pandas()

def create_dataset(df_train: pl.DataFrame, df_text, interactions_only=True) -> Dataset:

    event2weight = {
        17 : 1, #    88.851425
        11: 3, #    6.276978
        12 : 3, #   1.223713
        3: 2,    # 0.463616
        8: 2,   # 0.218001,
        16: 2,   #  0.149694
        10 : 20, #   1.216446
        15: 10,
        5: 10,
        19: 10,
        4: 10,
        13:10,
        14: 10,
        2:10,
        0:10,
        9: 10
    }
    interactions = get_interactions_df(df_train, event2weight)

    if not interactions_only:
        df_user = get_user_features(df_train)
        df_node = get_node_features(df_train, df_text)

        user_pool = FeaturePool(key='cookie')
        user_pool.add_features(df_user, feature_set='simple')

        item_pool = FeaturePool(key='node')
        item_pool.add_features(df_node, feature_set='simple')

        user_params = dict(
            add_categorical=0, add_numerical=0, add_ratios=1, add_embeddings=0, 
            filter_by_name=('surface', 'event', 'location', 'node', 'category', 'is_contact_last'),
            add_by_name = ('category_last_contact', ) # most_freq_category
        )
        item_params = dict(
            add_categorical=1, add_numerical=0, add_ratios=1, add_embeddings=0, 
            filter_by_name=('surface', 'location', 'event')
        )

        df_user_melted, user_cols = user_pool.get_melted_dataframe(**user_params) # type: ignore
        df_item_melted, item_cols = item_pool.get_melted_dataframe(**item_params) # type: ignore


        dataset = Dataset.construct(
            interactions_df=interactions,
            user_features_df=df_user_melted,
            item_features_df=df_item_melted,
            cat_user_features=('category_last_contact', ),
            cat_item_features=('category', ),
        )
    else:
        dataset = Dataset.construct(interactions_df=interactions,)
    return dataset


def get_candidates():

    data_dir = '/kaggle/input/avito-cup-2025-recsys/'
    df_train, df_eval, df_test_users = get_train_val(data_dir = data_dir)
    df_text = load_text_embeddings(data_dir)

    df_user = get_user_features(df_train)
    df_node = get_node_features(df_train, df_text)

    df_user_embeddings = get_text_embeddings_user(df_train, df_text)
    df_node_embeddings = get_text_embeddings_node(df_train, df_text)

    user_pool = FeaturePool(key='cookie')
    user_pool.add_features(df_user, feature_set='simple')
    user_pool.add_embedding_features(df_user_embeddings, feature_set='simple_emb')
    user_pool.transform_numerical_to_min_max()

    item_pool = FeaturePool(key='node')
    item_pool.add_features(df_node, feature_set='simple')
    item_pool.add_embedding_features(df_node_embeddings, feature_set='simple_emb')
    item_pool.transform_embeddings_to_min_max()

    subset_user = ['simple',]
    subset_item = ['simple',]

    df_user_melted, user_cols = user_pool.get_melted_dataframe(subset=subset_user)
    df_item_melted, item_cols = item_pool.get_melted_dataframe(subset=subset_item)

    event2weight = {
        17 : 1, #    88.851425
        11: 3, #    6.276978
        12 : 3, #   1.223713
        3: 2,    # 0.463616
        8: 2,   # 0.218001,
        16: 2,   #  0.149694
            
        10 : 20, #   1.216446
        15: 10,
        5: 10,
        19: 10,
        4: 10,
        13:10,
        14: 10,
        2:10,
        0:10,
        9: 10
    }


    interactions = get_interactions_df(df_train, event2weight)

    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=df_user_melted,
        cat_user_features=user_cols,
        item_features_df=df_item_melted,
        cat_item_features=item_cols,
    )
    
    # ------------------------------------------------------------------------------------
    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=60,  # latent embeddings size
            # regularization=0.1,
            iterations=10,
            # alpha=50,  # confidence multiplier for non-zero entries in interactions
            random_state=RANDOM_STATE,
        ),
        fit_features_together=False,  # way to fit paired features
    )

    gen1 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    gen1.fit()
    gen1.evaluate(df_eval, k=40)
    gen1.predict(df_test_users['cookie'].unique().to_list(), k=40)
    gen1.save('als_model.pkl')

    # ------------------------------------------------------------------------------------

    model = ImplicitItemKNNWrapperModel(
        CosineRecommender(
            K=40,
            num_threads=3,
        ),
    )
    gen2 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    gen2.fit()
    gen2.evaluate(df_eval, k=40)
    gen2.predict(df_test_users['cookie'].unique().to_list(), k=40)
    gen2.save('knn_model.pkl')

    # ------------------------------------------------------------------------------------

    model = ImplicitItemKNNWrapperModel(
        TFIDFRecommender(
        K=40,
        num_threads=3,
    ))
    gen3 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    gen3.fit()
    gen3.evaluate(df_eval, k=40)
    gen3.predict(df_test_users['cookie'].unique().to_list(), k=40)
    gen3.save('tfidf_model.pkl')

    # ------------------------------------------------------------------------------------

    model = ImplicitItemKNNWrapperModel(
        BM25Recommender(
        K=40,
        num_threads=3,
    ))
    gen4 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    gen4.fit()
    gen4.evaluate(df_eval, k=40)
    gen4.predict(df_test_users['cookie'].unique().to_list(), k=40)
    gen4.save('bm25_model.pkl')


    # ------------------------------------------------------------------------------------
    event2weight = {
        17 : 1, #    88.851425
        11: 1, #    6.276978
        12 : 1, #   1.223713
        3: 1,    # 0.463616
        8: 1,   # 0.218001,
        16: 1,   #  0.149694
            
        10 : 5, #   1.216446
        15: 2,
        5: 2,
        19: 2,
        4: 2,
        13:2,
        14: 2,
        2:2,
        0:2,
        9: 2
    }


    interactions = get_interactions_df(df_train, event2weight)
    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=df_user_melted,
        cat_user_features=user_cols,
        item_features_df=df_item_melted,
        cat_item_features=item_cols,
    )
    

    model = LightFMWrapperModel(
        LightFM(
            no_components=100, loss="warp", random_state=RANDOM_STATE, max_sampled=20,
        ),
        epochs=10,
        recommend_n_threads=3,
        num_threads=3,
    )
    gen5 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    gen5.fit()
    gen5.evaluate(df_eval, k=40)
    gen5.predict(df_test_users['cookie'].unique().to_list(), k=40)
    gen5.save('lightfm_model.pkl')
    # ------------------------------------------------------------------------------------


def run_2_stages():
    RANDOM_STATE=60
    CAND_DAYS_TRESHOLD = 7
    data_dir = '/kaggle/input/avito-cup-2025-recsys/'
    df_train, df_eval, _ = get_train_val(data_dir = data_dir)
    df_cat = load_cat_df(data_dir)
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    df_user = get_user_features(df_train)
    df_node = get_node_features(df_train, df_cat)

    df_user_ranker = get_user_features(df_cand)
    df_node_ranker = get_node_features(df_cand, df_cat)

    # first stage ------------------------------------------------------------------

    pred_users = df_ranker['cookie'].unique().to_list()
    ranker_preds = pl.read_parquet('ranker_preds.pq')
    print("Ranking predictions shape", ranker_preds.shape)
    print("users in ranker preds vs loaded", pred_users.n_unique(),ranker_preds['cookie'].n_unique())
    ranker_preds = ranker_preds.join(
        df_user_ranker, on='cookie', how='left'
    ).join(
        df_node_ranker, on='node', how='left'
    )

    eval_users = df_eval['cookie'].unique().to_list()
    eval_preds = pl.read_parquet('ranker_preds_all.pq')
    print("Eval predictions shape", eval_preds.shape)
    print("users in eval preds vs loaded", eval_users.n_unique(), eval_preds['cookie'].n_unique())

    # second stage ------------------------------------------------------------------

    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq').select(['location', 'category', 'node'])
    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')
    

    df_ranker  = df_ranker.with_columns(pl.col("is_contact").alias("is_target"))
    df_cand  = df_cand.with_columns(pl.col("is_contact").alias("is_target"))

    df_targets = df_ranker[['cookie', 'node', 'is_target']].sort(by = ['is_target'], descending=True).unique(['cookie', 'node' ])
    ranker_preds = ranker_preds.join(df_targets, on = ['cookie', 'node'], how = 'left').with_columns(pl.col('is_target').cast(int))
    ranker_preds = ranker_preds.with_columns(pl.col('is_target').fill_null(0))
    print(df_targets['is_target'].mean(), "is_target mean in ranker preds")


    params = {
        "boosting_type": "Plain",
        "early_stopping_rounds": 10,
        "eval_metric": "RecallAt:top=40",
        # "learning_rate": 0.1,
        "max_ctr_complexity": 1,
        "nan_mode": "Min",
        "num_trees": 200,
        "objective": "PairLogitPairwise",
        "random_state": 42,
        "thread_count": 3,
    }
    model = cb.CatBoost(params = params)

    ranker = SecondStageRanker(
        df_ranker=ranker_preds, 
        df_hist = df_cand, 
        df_node=df_cat_features, 
        model=model
    )
    
    features = [
        'cookie',
        'node',
        'category',
        'num_contacts',
        'num_events',
        'surface_unique_counts',
        'location_unique_counts'
    ]
    cat = [
        # 'category',
    ]
    ranker.split_data(ranker.df_ranker, features=features, cat_features=cat, eval_ratio=0.1)
    ranker.model.fit(X= ranker.train_pool, verbose=10, eval_set=ranker.eval_pool)

    print(ranker.model.eval_metrics(
        ranker.test_pool,
        metrics=["AUC", "Accuracy", "PrecisionAt:top=40", "RecallAt:top=40"],ntree_start=ranker.model.tree_count_ -1 # type: ignore
    ))

    ranker.save()



def run_1_stage():
    RANDOM_STATE=60

    data_dir = '/kaggle/input/avito-cup-2025-recsys/'
    df_train, df_eval, _ = get_train_val(data_dir = data_dir)
    df_cat = load_cat_df(data_dir)

    CAND_DAYS_TRESHOLD = 7
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold) # 14 days
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    dataset = create_dataset(df_cand, df_cat, interactions_only=False)
    dataset_all = create_dataset(df_train, df_cat, interactions_only=False)
    # first stage ------------------------------------------------------------------

    model = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=60,  # latent embeddings size
            iterations=10,
            random_state=RANDOM_STATE,
        ),
        fit_features_together=True,  # way to fit paired features
    )

    gen1 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    print("Fitting...")
    gen1.fit()
    print("Evaluating...")
    gen1.evaluate(df_ranker, k=40)

    print("Predicting...")
    pred_users = df_ranker['cookie'].unique().to_list()
    ranker_preds = gen1.predict(pred_users, k=200)
    print("Ranking predictions shape", ranker_preds.shape)

    # first stage on all train data -----------------------------------------------------
    model_all = ImplicitALSWrapperModel(
        AlternatingLeastSquares(
            factors=60,  # latent embeddings size
            iterations=10,
            random_state=RANDOM_STATE,
        ),
        fit_features_together=True,  # way to fit paired features
    )
    gen1_all = CandidatesGenerator(
        model=model_all,
        dataset=dataset_all
    )
    print("Fitting on all train data...")
    gen1_all.fit()
    print("Evaluating on all train data...")
    gen1_all.evaluate(df_eval, k=40)
    print("Predicting on all eval data...")
    pred_users_all = df_eval['cookie'].unique().to_list()
    ranker_preds_all = gen1_all.predict(pred_users_all, k=200)
    print("Ranking predictions shape on all train data", ranker_preds_all.shape)

    ranker_preds_all.write_parquet('ranker_preds_all.pq')
    ranker_preds.write_parquet('ranker_preds.pq')
    # second stage ------------------------------------------------------------------













