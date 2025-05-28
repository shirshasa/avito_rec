
import pandas as pd
import polars as pl
import numpy as np
import os
import threadpoolctl
import warnings

warnings.filterwarnings('ignore')

from datetime import timedelta
from gc import collect
import catboost as cb

# For vector models optimized ranking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas");

###

from data_load import get_train_val, get_cand_ranker, get_clickstream
from features import load_text_embeddings_node, load_text_embeddings, \
    load_cat_df, get_user_cat_features, get_user_loc_features, \
    get_node_loc_cat_features, join_features, add_dist_similarity, get_user_features, \
    get_node_features
from feature_pool import FeaturePool
from ranker import SecondStageRanker
from tools import recall_at, reduce_memory_usage_pl


CAND_DAYS_TRESHOLD = 7
RANDOM_STATE = 42

def prepare_data_for_ranking(
        predictions, df_node, df_user_cat, df_user_loc, df_node_emb, cat, nodes_cols,
        df_cand, df_cat,
        df_ranker=None
    ):
    # prepare data for catboost
    predictions = join_features(predictions, df_node, df_user_cat, df_user_loc)
    predictions = predictions.with_columns([pl.col(c).fill_null(-1).cast(int) for c in cat])

    # add targets
    if df_ranker is not None:
        df_targets = (
            df_ranker[['cookie', 'node', 'is_target']]
            .sort(by = ['is_target'], descending=True)
            .unique(['cookie', 'node' ], keep='first', maintain_order=True)
        )
        predictions = predictions.join(df_targets, on = ['cookie', 'node'], how = 'left').with_columns(pl.col('is_target').cast(int))
        predictions = predictions.with_columns(pl.col('is_target').fill_null(0))
        print("% positive targets in training data:", predictions['is_target'].mean())
        # add column, if date 

    # add nodes distances from embeddings
    predictions  = add_dist_similarity(predictions, df_node_emb, nodes_cols)

    # add user history features
    df_user_hist, df_node_hist = get_hist_features(df_cand, df_cat)
    print("Adding user and node history features...")

    predictions = predictions.join(df_user_hist, on='cookie', how='left')
    predictions = predictions.join(df_node_hist, on='node', how='left')

    return predictions

def get_hist_features(df_cand, df_cat):
    df_user_ranker = get_user_features(df_cand)
    df_node_ranker = get_node_features(df_cand, df_cat)

    user_pool = FeaturePool(key='cookie')
    user_pool.add_features(df_user_ranker, feature_set='simple')

    item_pool = FeaturePool(key='node')
    item_pool.add_features(df_node_ranker, feature_set='simple')

    user_params = dict(
        add_categorical=0, add_numerical=0, add_ratios=1, add_embeddings=0, 
        filter_by_name=('node',),
        # add_by_name = ('category_last_contact', 'most_freq_category') # most_freq_category
    )
    item_params = dict(
        add_categorical=0, add_numerical=0, add_ratios=1, add_embeddings=0, 
        filter_by_name=('surface', 'location', 'event')
    )
    user_df = user_pool.get_features(**user_params) # type: ignore
    node_df = item_pool.get_features(**item_params) # type: ignore
    return user_df, node_df


def load_ranker_preds(data_dir, data_path):
    if os.path.exists(data_path):
        print("Ranker predictions already exists, loading from", data_path)
        return pl.read_parquet(data_path)
    
    from consts import nodes, cat

    df_train, _, _ = get_train_val(data_dir = data_dir)
    df_ranker, df_cand = get_cand_ranker(df_train, cand_days=CAND_DAYS_TRESHOLD)
    del df_train
    collect()

    df_cat = load_cat_df(data_dir)
    df_user_loc = get_user_loc_features(df_cand)
    df_user_cat = get_user_cat_features(df_cand)
    df_node = get_node_loc_cat_features(df_cat)

    # nodes embeddings
    df_node_emb = load_text_embeddings_node(data_dir)

    # load prediction from generator

    # ranker_preds1 = pl.read_parquet('./data/candidates/cand/preds_als_features_200.pq')
    ranker_preds2 = pl.read_parquet('./data/candidates/cand/preds_als_no_feats_200.pq')
    # ranker_preds3 = pl.read_parquet('./data/candidates/cand/preds_lightfm_300.pq')
    # ranker_preds4 = pl.read_parquet('./data/candidates/cand/preds_knn_100.pq')

    all_preds = ranker_preds2  # pl.concat([ranker_preds1, ranker_preds2, ranker_preds3, ranker_preds4], how='vertical')
    all_preds = all_preds.unique(['cookie', 'node'])
    all_preds = all_preds.sort(by=['cookie'])
    ranker_preds = all_preds

    ranker_preds = prepare_data_for_ranking(
        predictions=ranker_preds, 
        df_node=df_node, 
        df_user_cat=df_user_cat, 
        df_user_loc=df_user_loc, 
        df_node_emb=df_node_emb, 
        cat=cat, 
        nodes_cols=nodes,
        df_ranker=df_ranker,
        df_cand=df_cand,
        df_cat=df_cat
    )

    ranker_preds.write_parquet(data_path)

    return ranker_preds

def train_ranker():
    from consts import feats, cat

    preprcoessed_data_path = './data/processed/ranker_preds_als_no_feat.pq'

    ranker_preds = load_ranker_preds(data_dir = './data/', data_path=preprcoessed_data_path)

    print("Ranker predictions shape", ranker_preds.shape)
    print("Start splitting data..")


    ranker = SecondStageRanker(df=ranker_preds)
    params = {
        "boosting_type": "Plain",
        "early_stopping_rounds": 10,
        "eval_metric": "RecallAt:top=40",
        # "learning_rate": 0.1,
        "max_ctr_complexity": 2,
        "nan_mode": "Min",
        "num_trees": 250,
        "objective": "PairLogitPairwise:max_pairs=50",
        "random_state": 42,
        "task_type": "CPU",
        "thread_count": -1,
    }

    model = cb.CatBoost(params = params)
    ranker.model = model
    ranker.split_data(features=feats, cat_features=cat, eval_ratio=0.2)
    ranker.fit()
    ranker.evaluate()
    ranker.save('./models/catboost_als_no_feats')
    return ranker


def evaluate_ranker():
    from consts import nodes
    # load ranker
    ranker = SecondStageRanker.load('./models/catboost_lightfm')
    cat = ranker.cat_features

    data_dir = './data/'
    df_train, df_eval, _ = get_train_val(data_dir = data_dir)
    df_ranker, df_cand = get_cand_ranker(df_train, cand_days=CAND_DAYS_TRESHOLD)

    print("Ranker users num", len(df_ranker['cookie'].unique()))
    print("Cand users num", len(df_cand['cookie'].unique()))
    print("Eval users num", len(df_eval['cookie'].unique()))

    # to test on train data
    # df_train = df_cand
    # df_eval = df_ranker

    df_cat = load_cat_df(data_dir)
    df_user_loc = get_user_loc_features(df_train)
    df_user_cat = get_user_cat_features(df_train)
    df_node = get_node_loc_cat_features(df_cat)

    # nodes embeddings
    df_node_emb = load_text_embeddings_node(data_dir)

    # load predictions for eval sel
    def evaluate(eval_preds):
        
        eval_preds = prepare_data_for_ranking(
            predictions=eval_preds, 
            df_node=df_node, 
            df_user_cat=df_user_cat, 
            df_user_loc=df_user_loc, 
            df_node_emb=df_node_emb, 
            cat=cat, 
            nodes_cols=nodes,
            df_cand=df_train,
            df_cat=df_cat,
        )
        
        eval_preds = ranker.predict_test(df=eval_preds, k=40)
        
        preds_users = eval_preds['cookie'].unique().to_list()
        eval_users = df_eval['cookie'].unique().to_list()

        print("Eval users:", len(eval_users), "Predictions users:", len(preds_users))
        print(eval_preds.filter(eval_preds['cookie'] == 0))
        recall = recall_at(df_eval, eval_preds, k=40)
        print(f"Recall at 40: {recall:.4f}")

    als_preds = './data/candidates/train/preds_als_no_feats_200.pq'
    eval_preds1 = pl.read_parquet(als_preds)

    # lightfm_preds = './data/candidates/train/preds_lightfm_300.pq'
    # eval_preds2 = pl.read_parquet(lightfm_preds)

    # knn_preds = './data/candidates/train/preds_knn_100.pq'
    # eval_preds3 = pl.read_parquet(knn_preds)

    # all_preds = pl.concat([eval_preds1, eval_preds2, eval_preds3], how='vertical')
    # all_preds = all_preds.unique(['cookie', 'node'])
    # all_preds = all_preds.sort(by=['cookie'])

    # print("All predictions shape", all_preds.shape)
    # print(all_preds.head())

    evaluate(eval_preds1)


def save_submission():
    from consts import nodes
    # load ranker
    ranker = SecondStageRanker.load('./models/catboost_0_91_emb')
    cat = ranker.cat_features

    data_dir = './data/'
    _, _, df_test = get_train_val(data_dir = data_dir)
    # df_ranker, df_cand = get_cand_ranker(df_train, cand_days=CAND_DAYS_TRESHOLD)
    df_clickstream = get_clickstream(data_dir)

    eval_users = df_test['cookie'].unique().to_list()

    df_cat = load_cat_df(data_dir)
    df_user_loc = get_user_loc_features(df_clickstream)
    df_user_cat = get_user_cat_features(df_clickstream)
    df_node = get_node_loc_cat_features(df_cat)

    # nodes embeddings
    df_node_emb = load_text_embeddings_node(data_dir)
    als_preds = './data/candidates/all/preds_als_features_200.pq'
    eval_preds = pl.read_parquet(als_preds)

    print("Submission predictions shape", eval_preds.shape)
    print("Users in submission predictions:", len(set(eval_users) - set(eval_preds['cookie'].unique())) == 0)


    eval_preds = prepare_data_for_ranking(
            predictions=eval_preds, 
            df_node=df_node, 
            df_user_cat=df_user_cat, 
            df_user_loc=df_user_loc, 
            df_node_emb=df_node_emb, 
            cat=cat, 
            nodes_cols=nodes,
            df_cand=df_clickstream,
            df_cat=df_cat,
    )
        
    eval_preds = ranker.predict_test(
            df=eval_preds,
            k=40
    )
    eval_preds[['cookie', 'node']].write_csv('./subs/submission1.csv')

    

if __name__ == "__main__":
    train_ranker()
    # evaluate_ranker()
    
    # Example of how to use the ranker
    # ranker = SecondStageRanker.load('./models/catboost_0_91_emb')
    # df_test = pl.read_parquet('./data/test_users.pq')
    # df_test = ranker.predict_test(df_test, k=40)
    # df_test.write_csv('./data/submission.csv')
