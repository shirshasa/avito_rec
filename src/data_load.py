from tools import reduce_memory_usage_pl

import numpy as np
import polars as pl
import pandas as pd
from datetime import timedelta

from rectools.dataset import Dataset

from features import get_user_features, get_node_features, get_text_embeddings_user, \
    load_text_embeddings_node, load_text_embeddings, load_cat_df
from feature_pool import FeaturePool
from candidates_generator import CandidatesGenerator
from ranker import SecondStageRanker
from tools import recall_at, reduce_memory_usage_pl

EVAL_DAYS_TRESHOLD = 14


def get_train_val(data_dir = '/kaggle/input/avito-cup-2025-recsys/', eval_days=EVAL_DAYS_TRESHOLD):
    # load
    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{data_dir}/clickstream.pq')
    # df_text_features = pl.scan_parquet(f'{DATA_DIR}/text_features.pq')
    df_event = pl.read_parquet(f'{data_dir}/events.pq')

    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = reduce_memory_usage_pl(df_clickstream, name='df_clickstream')

    # split
    treshhold = df_clickstream['event_date'].max() - timedelta(days=eval_days)
    df_train = df_clickstream.filter(df_clickstream['event_date']<= treshhold)

    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq')
    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')

    df_train = df_train.join(df_event, on='event', how='left')
    df_train = df_train.join(df_cat_features.select(["item", "location", "category"]), on=['item'], how='left')
    df_train = reduce_memory_usage_pl(df_train, name='df_train')

    top25_locations = df_train.group_by('location').count().sort('count', descending=True)[:25]['location'].to_list()
    df_train = df_train.with_columns(
        pl.when(pl.col('location').is_in(top25_locations))
        .then(pl.col('location'))
        .otherwise(-1) # keep original value
        .alias("location_top")
    )

    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact')==1)['event'].unique()
        )
    )
    df_eval.shape

    df_eval = df_eval.filter(
            pl.col('cookie').is_in(df_train['cookie'].unique())
        ).filter(
            pl.col('node').is_in(df_train['node'].unique())
        )

    df_eval = reduce_memory_usage_pl(df_eval, name='df_eval')

    return df_train, df_eval, df_test_users


def get_cand_ranker(df_train, cand_days=7):
    cand_treshhold = df_train['event_date'].max() - timedelta(days=cand_days)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    df_ranker  = df_ranker.with_columns(pl.col("is_contact").alias("is_target"))
    df_cand  = df_cand.with_columns(pl.col("is_contact").alias("is_target"))

    return df_ranker, df_cand


def get_clickstream(data_dir) -> pl.DataFrame:

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{data_dir}/clickstream.pq')
    # df_text_features = pl.scan_parquet(f'{DATA_DIR}/text_features.pq')
    df_event = pl.read_parquet(f'{data_dir}/events.pq')

    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = reduce_memory_usage_pl(df_clickstream, name='df_clickstream')

    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq')
    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')

    df_clickstream = df_clickstream.join(df_event, on='event', how='left')
    df_clickstream = df_clickstream.join(df_cat_features.select(["item", "location", "category"]), on=['item'], how='left')
    df_clickstream = reduce_memory_usage_pl(df_clickstream, name='df_clickstream')

    top25_locations = df_clickstream.group_by('location').count().sort('count', descending=True)[:25]['location'].to_list()
    df_clickstream = df_clickstream.with_columns(
        pl.when(pl.col('location').is_in(top25_locations))
        .then(pl.col('location'))
        .otherwise(-1) # keep original value
        .alias("location_top")
    )

    return df_clickstream

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

def create_dataset_als(df_train: pl.DataFrame, df_text, interactions_only=True) -> Dataset:

    event2weight = {
        17 : 1, #    88.851425
        11: 2, #    6.276978
        12 : 2, #   1.223713
        3: 2,    # 0.463616
        8: 2,   # 0.218001,
        16: 2,   #  0.149694
        10 : 50, #   1.216446
        15: 20,
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
            filter_by_name=('surface', 'location', 'node', 'category', 'is_contact_last'),
            add_by_name = ('category_last_contact', 'most_freq_category') # most_freq_category
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
            cat_user_features=('category_last_contact', 'most_freq_category'),
            cat_item_features=('category', ),
        )
    else:
        dataset = Dataset.construct(interactions_df=interactions,)
    return dataset


def create_dataset_other(df_train: pl.DataFrame) -> Dataset:
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
    dataset = Dataset.construct(interactions_df=interactions,)
    return dataset