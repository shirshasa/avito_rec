import pandas as pd
import polars as pl
import numpy as np

import os
import threadpoolctl
import warnings

warnings.filterwarnings('ignore')

from datetime import timedelta
from gc import collect

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

from data_load import get_train_val, create_dataset_als, create_dataset_other, get_clickstream
from features import load_cat_df
from feature_pool import FeaturePool
from candidates_generator import CandidatesGenerator
from tools import recall_at, reduce_memory_usage_pl



RANDOM_STATE = 42



def get_candidates(df_test, dataset, df_evaluate=None, k=200, k_eval=40, random_state=60, model_name="als"):
    if model_name == "als":
        print("Using ALS model for candidates generation")
        model = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=256,  # latent embeddings size
                iterations=25,
                random_state=random_state,
            ),
            fit_features_together=False,  # way to fit paired features
        )
    elif model_name == "lightfm":
        print("Using LightFM model for candidates generation")
        model = LightFMWrapperModel(
            LightFM(
                no_components=128, loss="warp", random_state=RANDOM_STATE, max_sampled=20,
            ),
            epochs=20,
            recommend_n_threads=24,
            num_threads=24,
        )
    elif model_name == "knn":
        model = ImplicitItemKNNWrapperModel(
            TFIDFRecommender(
                K=k,
                num_threads=24,
            )
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    gen1 = CandidatesGenerator(
        model=model,
        dataset=dataset
    )
    print("Fitting...")
    gen1.fit()

    if df_evaluate is not None:
        print("Evaluating on provided dataset...")
        gen1.evaluate(df_evaluate, k=k_eval)

    print("Predicting...")
    pred_users = df_test['cookie'].unique().to_list()
    ranker_preds = gen1.predict(pred_users, k=k)
    print("Ranking predictions shape", ranker_preds.shape)
    return ranker_preds



def run_1_stage_als():
    RANDOM_STATE=60

    data_dir = 'data/'
    ranker_folder = f'{data_dir}/candidates/cand/'
    train_folder = f'{data_dir}/candidates/train/'
    all_folder = f'{data_dir}/candidates/all/'

    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    if not os.path.exists(ranker_folder):
        os.makedirs(ranker_folder, exist_ok=True)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)
    if not os.path.exists(all_folder):
        os.makedirs(all_folder, exist_ok=True)


    df_train, df_eval, _ = get_train_val(data_dir = data_dir)
    df_cat = load_cat_df(data_dir)

    CAND_DAYS_TRESHOLD = 7
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold) # 14 days
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    dataset = create_dataset_als(df_cand, df_cat, interactions_only=False)
    dataset_all = create_dataset_als(df_train, df_cat, interactions_only=False)
    # first stage ------------------------------------------------------------------

    ranker_preds = get_candidates(df_ranker, dataset, df_evaluate=df_ranker, k=200, k_eval=40, random_state=RANDOM_STATE, model_name="als")
    ranker_preds.write_parquet(f'{ranker_folder}/preds_als_features_200.pq')

    # first stage on all train data -----------------------------------------------------
    ranker_preds_all = get_candidates(df_eval, dataset_all, df_evaluate=df_eval, k=200, k_eval=40, random_state=RANDOM_STATE, model_name="als")
    ranker_preds_all.write_parquet(f'{train_folder}/preds_als_features_200.pq')

    # fisrt stage on all data -----------------------------------------------------------
    del df_train, df_eval, df_cand, df_ranker, dataset, dataset_all
    collect()

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = get_clickstream(data_dir)

    dataset = create_dataset_als(df_clickstream, df_cat, interactions_only=False)

    ranker_preds_all = get_candidates(df_test_users, dataset, k=200, random_state=RANDOM_STATE, model_name="als")
    ranker_preds_all.write_parquet(f'{all_folder}/preds_als_features_200.pq')


def run_1_stage_lightfm():
    RANDOM_STATE=60

    data_dir = 'data/'
    ranker_folder = f'{data_dir}/candidates/cand/'
    train_folder = f'{data_dir}/candidates/train/'
    all_folder = f'{data_dir}/candidates/all/'

    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    if not os.path.exists(ranker_folder):
        os.makedirs(ranker_folder, exist_ok=True)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)
    if not os.path.exists(all_folder):
        os.makedirs(all_folder, exist_ok=True)

    df_train, df_eval, _ = get_train_val(data_dir = data_dir)

    CAND_DAYS_TRESHOLD = 7
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold)
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    dataset = create_dataset_other(df_cand)
    dataset_all = create_dataset_other(df_train)

    # first stage ------------------------------------------------------------------

    ranker_preds = get_candidates(df_ranker, dataset, df_evaluate=df_ranker, k=300, k_eval=40, random_state=RANDOM_STATE, model_name="lightfm")
    ranker_preds.write_parquet(f'{ranker_folder}/preds_lightfm_300.pq')

    # first stage on all train data -----------------------------------------------------
    ranker_preds_all = get_candidates(df_eval, dataset_all, df_evaluate=df_eval, k=300, k_eval=40, random_state=RANDOM_STATE, model_name="lightfm")
    ranker_preds_all.write_parquet(f'{train_folder}/preds_lightfm_300.pq')
    
    # fisrt stage on all data -----------------------------------------------------------
    del df_train, df_eval, df_cand, df_ranker, dataset, dataset_all
    collect()

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = get_clickstream(data_dir)

    dataset = create_dataset_other(df_clickstream)
    ranker_preds_all = get_candidates(df_test_users, dataset, k=300, random_state=RANDOM_STATE, model_name="lightfm")
    ranker_preds_all.write_parquet(f'{all_folder}/preds_lightfm_300.pq')


def run_1_stage_knn():
    RANDOM_STATE=60

    data_dir = 'data/'
    ranker_folder = f'{data_dir}/candidates/cand/'
    train_folder = f'{data_dir}/candidates/train/'
    all_folder = f'{data_dir}/candidates/all/'

    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    if not os.path.exists(ranker_folder):
        os.makedirs(ranker_folder, exist_ok=True)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)
    if not os.path.exists(all_folder):
        os.makedirs(all_folder, exist_ok=True)

    df_train, df_eval, _ = get_train_val(data_dir = data_dir)

    CAND_DAYS_TRESHOLD = 7
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold)
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    dataset = create_dataset_other(df_cand)
    dataset_all = create_dataset_other(df_train)

    # first stage ------------------------------------------------------------------

    ranker_preds = get_candidates(df_ranker, dataset, df_evaluate=df_ranker, k=100, k_eval=40, random_state=RANDOM_STATE, model_name="knn")
    ranker_preds.write_parquet(f'{ranker_folder}/preds_knn_100.pq')

    # first stage on all train data -----------------------------------------------------
    ranker_preds_all = get_candidates(df_eval, dataset_all, df_evaluate=df_eval, k=100, k_eval=40, random_state=RANDOM_STATE, model_name="knn")
    ranker_preds_all.write_parquet(f'{train_folder}/preds_knn_100.pq')
    
    # fisrt stage on all data -----------------------------------------------------------
    del df_train, df_eval, df_cand, df_ranker, dataset, dataset_all
    collect()

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')
    df_clickstream = get_clickstream(data_dir)

    dataset = create_dataset_other(df_clickstream)
    ranker_preds_all = get_candidates(df_test_users, dataset, k=100, random_state=RANDOM_STATE, model_name="knn")
    ranker_preds_all.write_parquet(f'{all_folder}/preds_knn_100.pq')


def run_1_stage_als_no_feats():
    RANDOM_STATE=60

    data_dir = 'data/'
    ranker_folder = f'{data_dir}/candidates/cand/'
    train_folder = f'{data_dir}/candidates/train/'
    all_folder = f'{data_dir}/candidates/all/'

    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    if not os.path.exists(ranker_folder):
        os.makedirs(ranker_folder, exist_ok=True)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)
    if not os.path.exists(all_folder):
        os.makedirs(all_folder, exist_ok=True)

    df_train, df_eval, _ = get_train_val(data_dir = data_dir)

    CAND_DAYS_TRESHOLD = 7
    cand_treshhold = df_train['event_date'].max() - timedelta(days=CAND_DAYS_TRESHOLD)
    df_cand = df_train.filter(df_train['event_date'] <= cand_treshhold)

    df_ranker = df_train.filter(df_train['event_date'] > cand_treshhold)
    df_ranker = df_ranker.filter(pl.col('cookie').is_in(df_cand['cookie'].unique()))

    dataset = create_dataset_als(df_cand, None, interactions_only=True)
    dataset_all = create_dataset_als(df_train, None, interactions_only=True)

    # first stage ------------------------------------------------------------------

    ranker_preds = get_candidates(df_ranker, dataset, df_evaluate=df_ranker, k=200, k_eval=40, random_state=RANDOM_STATE, model_name="als")
    ranker_preds.write_parquet(f'{ranker_folder}/preds_als_no_feats_200.pq')

    # first stage on all train data -----------------------------------------------------
    ranker_preds_all = get_candidates(df_eval, dataset_all, df_evaluate=df_eval, k=200, k_eval=40, random_state=RANDOM_STATE, model_name="als")
    ranker_preds_all.write_parquet(f'{train_folder}/preds_als_no_feats_200.pq')
    
    # fisrt stage on all data -----------------------------------------------------------
    del df_train, df_eval, df_cand, df_ranker, dataset, dataset_all
    collect()

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_test_users = reduce_memory_usage_pl(df_test_users, name='df_test_users')

    df_clickstream = get_clickstream(data_dir)

    dataset = create_dataset_als(df_clickstream, None, interactions_only=True)
    ranker_preds_all = get_candidates(df_test_users, dataset, k=200, random_state=RANDOM_STATE, model_name="als")
    ranker_preds_all.write_parquet(f'{all_folder}/preds_als_no_feats_200.pq')


if __name__ == "__main__":
    run_1_stage_lightfm()

    # Results
    # 1 stage candidates generation with ALS model on all data: preds_als_no_feats_200_exp.pq
    # cand: 0.078
    # train: 0.152
    # 1 stage candidates generation with LightFM model on all data: preds_lightfm_200.pq
    # cand: 0.08
    # train: 0.13
    # 1 stage candidates generation with KNN model on all data: preds_knn_100.pq
    # cand: 0.06
    # train: 0.119
    # 1 stage candidates generation with ALS model on ranker data: preds_als_features_200.pq
    # cand: 0.08
    # train: 0.16 -> 0.154

   






