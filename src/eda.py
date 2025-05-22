import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from consts import  *


def check_clicks_stream():
    df_clickstream = pl.scan_parquet(
        f'{DATA_DIR}/clickstream.pq',
    )

    df_clickstream.cast(
        {
            "event": pl.UInt8,
            "platform": pl.UInt8,
            "surface": pl.UInt8,
            "node": pl.UInt32,
            "item": pl.UInt32,
            "cookie": pl.UInt32,
        }
    )

    print(df_clickstream.head(3).collect())

    # print unique number
    print(df_clickstream.select(pl.col("cookie").n_unique()).collect())
    print(df_clickstream.select(pl.col("item").n_unique()).collect())
    print(df_clickstream.select(pl.col("event").n_unique()).collect())
    print(df_clickstream.select(pl.col("platform").n_unique()).collect())
    print(df_clickstream.select(pl.col("surface").n_unique()).collect())
    print(df_clickstream.select(pl.col("node").n_unique()).collect())

    print("number of rows", df_clickstream.count().collect())

    print("event_date: ")
    print("min", df_clickstream.select(pl.col("event_date").min()).collect())
    print("max", df_clickstream.select(pl.col("event_date").max()).collect())


def plot_stuff_on_clicks_stream():
    df_clickstream = pl.scan_parquet(
        f'{DATA_DIR}/clickstream.pq',
    )

    df_clickstream.cast(
        {
            "event": pl.UInt8,
            "platform": pl.UInt8,
            "surface": pl.UInt8,
            "node": pl.UInt32,
            "item": pl.UInt32,
            "cookie": pl.UInt32,
        }
    )

    df_sample = df_clickstream.sample(10_000).collect().to_pandas()