import pandas as pd
import polars as pl

import os
import threadpoolctl
import warnings

warnings.filterwarnings('ignore')

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import (
    ImplicitALSWrapperModel,
    ImplicitBPRWrapperModel,
    LightFMWrapperModel,
    PureSVDModel,
    ImplicitItemKNNWrapperModel,
    EASEModel
)

# For vector models optimized ranking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas");

from tools import recall_at

RANDOM_STATE=2025

def get_als_weight_preds(
    interactions_df,
    user_features_df=None,
    item_features_df=None,
    cat_user_features=None,
    cat_item_features=None,
    eval_users = None,
    test_users=None,
    df_eval=None
):
    # create dataset with both user and item features
    dataset = Dataset.construct(
        interactions_df=interactions_df,
        user_features_df=user_features_df,
        cat_user_features=cat_user_features,
        item_features_df=item_features_df,
        cat_item_features=cat_item_features,
    )
    

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
    model.fit(dataset)

    ans = {
        "model": model,
        "dataset": dataset
    }

    # validation 
    if eval_users and df_eval:
        recos = model.recommend(
            users=eval_users,
            dataset=dataset,
            k=40,
            filter_viewed=True,
        )
        
        df_pred = recos[['item_id', 'user_id']]
        df_pred.columns = ['node', 'cookie']
        
        recall = recall_at(df_eval, pl.from_pandas(df_pred), k=40)
        print(recall)

        ans['recall@40'] = recall
        ans['eval_preds'] = df_pred


    if test_users:
        recos = model.recommend(
            users=test_users,
            dataset=dataset,
            k=40,
            filter_viewed=True,
        )
        
        df_pred = recos[['user_id', 'item_id']]
        df_pred.columns = ['cookie', 'node']
        ans['test_preds'] = df_pred

    return ans