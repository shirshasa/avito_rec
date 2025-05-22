import pandas as pd
import polars as pl
from data import get_train_val
from features import get_user_features, get_node_features, get_text_embeddings_user, \
    get_text_embeddings_node, load_text_embeddings
from feature_pool import FeaturePool
from als import get_als_weight_preds
import pickle
from datetime import datetime

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

def get_als_candidates():

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
    user_pool.transform_categorical_to_ohe()
    user_pool.transform_numerical_to_min_max()

    item_pool = FeaturePool(key='node')
    item_pool.add_features(df_node, feature_set='simple')
    item_pool.add_embedding_features(df_node_embeddings, feature_set='simple_emb')
    item_pool.transform_categorical_to_ohe()
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

    als_results = get_als_weight_preds(
        interactions,
        user_features_df=df_user_melted,
        item_features_df=user_cols,
        cat_user_features=df_item_melted,
        cat_item_features=item_cols,
        eval_users=df_eval['cookie'].unique().to_list(),
    )

    # save results

    date_checkpoint = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('als_results_{date_checkpoint}.pkl', 'wb') as f:
        pickle.dump(als_results, f)
    print(f"ALS results saved to als_results_{date_checkpoint}.pkl")
    return als_results

