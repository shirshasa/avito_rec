from tools import reduce_memory_usage_pl

import numpy as np
import polars as pl
import pandas as pd
from datetime import timedelta


def get_train_val(data_dir = '/kaggle/input/avito-cup-2025-recsys/'):

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{data_dir}/clickstream.pq')
    # df_text_features = pl.scan_parquet(f'{DATA_DIR}/text_features.pq')
    df_event = pl.read_parquet(f'{data_dir}/events.pq')

    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = reduce_memory_usage_pl(df_clickstream, name='df_clickstream')

    EVAL_DAYS_TRESHOLD = 14

    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
    df_train = df_clickstream.filter(df_clickstream['event_date']<= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq')
    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')

    df_train = df_train.join(df_event, on='event', how='left')
    df_train = df_train.join(df_cat_features.select(["item", "location", "category"]), on=['item'], how='left')
    df_train = reduce_memory_usage_pl(df_train, name='df_train')

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