from datetime import timedelta
import polars as pl
import numpy as np
import implicit
from lightfm import LightFM
import pandas as pd

from gc import collect


def reduce_memory_usage_pl(df, name):
    """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
        Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 """
    print(f"Memory usage of dataframe {name} is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    Numeric_Float_types = [pl.Float32,pl.Float64]    
    for col in df.columns:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))
            else:
                pass
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    print(f"Memory usage of dataframe {name} became {round(df.estimated_size('mb'), 2)} MB")
    return df



def recall_at(df_solution: pl.DataFrame, df_pred: pl.DataFrame, k=40, ):
    assert df_pred.group_by(['cookie']).agg(pl.col('node').count())['node'].max() < 41 , 'send more then 40 nodes per cookie' # type: ignore
    assert 'node' in df_pred.columns, 'node columns does not exist'
    assert 'cookie' in df_pred.columns, 'cookie columns does not exist'
    assert df_pred.with_columns(v = 1).group_by(['cookie','node']).agg(pl.col('v').count())['v'].max() == 1 , 'more then 1 cookie-node pair'
    assert df_pred['cookie'].dtype == pl.Int64, 'cookie must be int64'
    assert df_pred['node'].dtype == pl.Int64, 'node must be int64'
    
    return  df_solution[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on = ['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum()/pl.col('value').count()
        ]
    )['value'].mean()
