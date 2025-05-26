import polars as pl
import numpy as np

from tools import reduce_memory_usage_pl
from datetime import timedelta


def get_activity_features(df_train: pl.DataFrame) -> pl.DataFrame:
    N = df_train['cookie'].n_unique()
    MAX_DATE = df_train['event_date'].max()
    MIN_DATE = df_train['event_date'].min()
    TOTAL_DAYS = (df_train['event_date'].max() - df_train['event_date'].min()).days # type: ignore

    df_train = df_train.sort(["cookie", "event_date"])

    # last time activity
    cols = ['cookie', 'event', 'event_date', 'platform', 'surface', 'node', 'is_contact', 'location', 'category']
    last_df = df_train.group_by('cookie', maintain_order=True).last().select(cols)
    last_df = last_df.with_columns(
        since=(MAX_DATE - pl.col("event_date")).dt.total_days(),
        pr_since=((MAX_DATE - pl.col("event_date")).dt.total_days() / TOTAL_DAYS).round(2)
    ).drop("event_date").rename(lambda c: c + "_last" if c != "cookie" else c)

    cols = ['cookie', 'event', 'event_date', 'platform', 'surface', 'node', 'location', 'category']
    last_contact_df = df_train.filter(df_train['is_contact'] == 1) \
        .group_by('cookie', maintain_order=True).last().select(cols)
    last_contact_df = last_contact_df.with_columns(
        since=(MAX_DATE - pl.col("event_date")).dt.total_days(),
        pr_since=((MAX_DATE - pl.col("event_date")).dt.total_days() / TOTAL_DAYS).round(2)
    ).drop("event_date").rename(lambda c: c + "_last_contact" if c != "cookie" else c)

    # activity features
    # by days
    df_train = df_train.with_columns(
        pl.col("event_date").dt.date().alias("date")
    )

    df1 = df_train.group_by(['cookie']).agg(pl.col('date').n_unique().alias("active_days_count"))
    df2 = df_train.filter(df_train['is_contact'] == 1) \
        .group_by(['cookie']).agg(pl.col('date').n_unique().alias("contact_days_count"))

    df_days = df1.join(df2, on='cookie', how='left').with_columns(
        (pl.col("contact_days_count") / pl.col("active_days_count")).round(3).alias("contact_active_days_ratio"),
        (pl.col("active_days_count") / TOTAL_DAYS).round(3).alias("active_days_pr")
    )

    # by events
    df1 = df_train.group_by(['cookie']).len().rename({"len": "total_events"})
    df2 = df_train.filter(df_train['is_contact'] == 1).group_by(['cookie']).len().rename({"len": "total_contacts"})

    df_events = df1.join(df2, on='cookie', how='left').with_columns(
        (pl.col("total_contacts") / pl.col("total_events")).round(3).alias("contact_active_events_ratio")
    )

    # 105 users with only contact events - strange peak in distribution, maybe this users are kind of outliers

    # by last week activity
    treshhold = MAX_DATE - timedelta(days=7) # type: ignore

    df1 = df_train.filter(df_train['event_date'] >= treshhold).group_by(['cookie']).len().rename(
        {"len": "last_week_events"})
    df2 = df_train.filter((df_train['event_date'] >= treshhold) & (df_train['is_contact'] == 1)) \
        .group_by(['cookie']).len().rename({"len": "last_week_contacts"})

    df_last_week_events = df1.join(df2, on='cookie', how='left').with_columns(
        (pl.col("last_week_contacts") / pl.col("last_week_events")).round(3).alias(
            "contact_active_last_week_events_ratio")
    )

    # merge all features
    dfs = [df_days, df_events, df_last_week_events, last_contact_df, last_df]
    df_activity = dfs[0]
    for df in dfs[1:]:
        df_activity = df_activity.join(df, on='cookie', how='left').fill_null(0)

    print(df_activity.shape[0] == N)

    df_activity = reduce_memory_usage_pl(df_activity, name='df_activity')

    return df_activity

def get_category_if_it_more_then(arr, threshold=0.5):
    n = len(arr)
    values, counts = np.unique(arr, return_counts=True)
    top_indices = np.argsort(counts)[-1:]
    if counts[top_indices[0]] / n >= threshold:
        x = values[top_indices[0]]
        return int(x)

def get_category_vector_features(df_train: pl.DataFrame, col="category", normalize=False):
    counts_df = (
        df_train
        .fill_null(0)
        .group_by(["cookie", col])
        .agg(pl.len())  # count events per (user_id, col)
        .pivot(
            values="len",
            index="cookie",
            on=col,
            aggregate_function=None
        )
        .fill_null(0)
    )
    counts_df = counts_df.rename({x: f"{col}_{x}_count" for x in counts_df.columns if x != "cookie"})


    if normalize:
        category_cols = [x for x in counts_df.columns if x != "cookie"]
        category_cols.sort()
        
        counts_df = counts_df.with_columns(
            total_count=pl.sum_horizontal(pl.col(category_cols))
        )
    
        counts_df = counts_df.select(
            [
                pl.col("cookie"),
                *[(pl.col(col) / pl.col("total_count")).alias(col.replace("_count", "_pr")).round(3) for col in category_cols]
            ]
        )

    return counts_df

def get_category_features(df_train: pl.DataFrame):

    def _add_mean_std_per_vectoriez_d_features(df: pl.DataFrame, agg_coll="cookie"):
        category_cols = [col for col in df.columns if col != agg_coll]
        df = df.with_columns(
            mean_cat_count=pl.mean_horizontal(pl.col(category_cols)).round().cast(pl.UInt32),
            std_cat_count= pl.concat_list(category_cols).list.std().round().cast(pl.UInt32)
        )
        return df

    df_simple_features = df_train.group_by("cookie").agg(
        pl.col("category").mode().first().alias("most_freq_category"),
        pl.col("event").mode().first().alias("most_freq_event"),
        pl.col("surface").mode().first().alias("most_freq_surface"),
        pl.col("category").map_elements(get_category_if_it_more_then, return_dtype=pl.Int8).alias("dominant_category").fill_null(-1),
        pl.col("category").map_elements(lambda x: get_category_if_it_more_then(x, 0.3), return_dtype=pl.Int8).alias("almost_dominant_category").fill_null(-1),
    )
    
    df_count_category = get_category_vector_features(df_train, col="category", normalize=False)
    df_count_category = _add_mean_std_per_vectoriez_d_features(df_count_category, agg_coll="cookie")

    df_pr_category = get_category_vector_features(df_train, normalize=True)

    df_count_contact_category = get_category_vector_features(df_train.filter(df_train['is_contact'] == 1), normalize=False)
    df_count_contact_category = _add_mean_std_per_vectoriez_d_features(df_count_contact_category, agg_coll="cookie")
    df_count_contact_category.columns = [col if col == "cookie" else f"{col}_contact" for col in df_count_contact_category.columns]
    
    df_pr_event = get_category_vector_features(df_train, col="event", normalize=True)

    df_pr_surface = get_category_vector_features(df_train, col="surface", normalize=True)

    dfs = [
        df_simple_features, df_count_category, df_pr_category, df_count_contact_category, df_pr_event, df_pr_surface
    ]
    categories_df = dfs[0]

    for df in dfs[1:]:
        categories_df = categories_df.join(df, on="cookie", how="left")

    print(categories_df.shape[0] == df_train['cookie'].n_unique())
    categories_df = reduce_memory_usage_pl(categories_df, name='categories_df')

    return categories_df

def load_text_embeddings(data_dir):
    cols = ["item", "category", "node"]
    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq', low_memory=True).select(cols)
    df_text_features = pl.read_parquet(f'{data_dir}/text_features.pq')

    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')
    df_text_features = df_text_features.join(df_cat_features, on='item', how='right')

    dim = 64
    for i in range(dim):
        df_text_features = df_text_features.with_columns(
            pl.col("title_projection").arr.get(i).alias(f"emb_{i}")
        )
    emb_columns = [c for c in df_text_features.columns if 'emb' in c]

    return df_text_features.select(emb_columns + ['item', 'node', 'category'])

def load_cat_df(data_dir):
    cols = ["item", "category", "node"]
    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq', low_memory=True).select(cols)

    df_cat_features = reduce_memory_usage_pl(df_cat_features, name='df_cat_features')
    return df_cat_features

def get_text_embeddings_user(df_train: pl.DataFrame, df_text: pl.DataFrame) -> pl.DataFrame:
    """
    >> df_text = load_text_embeddings(data_dir)
    >> df_text_embeddings = get_text_embeddings_user(df_train, df_text)
    """
    emb_columns = [c for c in df_text.columns if 'emb' in c]


    df_user_embeddings = (
        df_train
        .select(['cookie', 'item', 'event_date'])
        .join(df_text.select(emb_columns + ['item']),
            on='item',
            how='left'
        )
        .group_by('cookie').tail(5).
        group_by('cookie').agg(
            [pl.col(c).mean().round().cast(pl.Int8) for c in emb_columns]
        )
    )
    return df_user_embeddings

def get_user_features(df_train: pl.DataFrame) -> pl.DataFrame:
    df_user = df_train.group_by("cookie").agg(
        pl.col("node").mode().first().alias("most_freq_node"),
        pl.col("location").mode().first().alias("most_freq_location"),
        pl.col("is_contact").sum().alias("total_contacts"),
        pl.col("is_contact").mean().alias("contact_ratio"),
    )

    print(df_user.shape[0] == df_train['cookie'].n_unique())

    df_activity = get_activity_features(df_train)
    print("Added activity features.")
    df_categories = get_category_features(df_train)
    print("Added category features.")
    df_user = df_user.join(df_activity, on="cookie", how="left")
    df_user = df_user.join(df_categories, on="cookie", how="left")

    df_user = reduce_memory_usage_pl(df_user, name='df_user')

    return df_user

def get_text_embeddings_node(df_train: pl.DataFrame, df_text: pl.DataFrame) -> pl.DataFrame:        
    emb_columns = [c for c in df_text.columns if 'emb' in c]
    df_text_embeddings = df_text.group_by('node').agg(
        [pl.col(c).mean().round().cast(pl.Int8) for c in emb_columns]
    )
    return df_text_embeddings

def get_node_features(df_train: pl.DataFrame, df_cat_text: pl.DataFrame) -> pl.DataFrame:
    assert 'node' in df_cat_text.columns, 'node column does not exist in df_text'
    assert 'node' in df_train.columns, 'node column does not exist in df_train'
    assert 'category' in df_cat_text.columns, 'category column does not exist in df_text'

    df_node = df_train.with_columns(
            pl.col("category").fill_null(40)
        ).group_by("node").agg(
            pl.col("is_contact").sum().alias("total_contacts"),
            pl.col("is_contact").mean().alias("ctr"),
            pl.col("event").mode().first().alias("most_freq_event"),
            pl.col("surface").mode().first().alias("most_freq_surface"),
            pl.col("location").mode().first().alias("most_freq_location"),
            pl.col("event").len().alias("total_events"),
            pl.col("event").n_unique().alias("unique_events"),
        )
        

    df_category = (
        df_cat_text
        .group_by(["node", "category"])
        .len()
        .rename({"len":"item_counts"})
        .unique(['node']) # only one node had 2 categories other had 1
    )

    print("Added node features.")

    df_node = df_node.join(df_category, on="node", how="left")

    df_node = reduce_memory_usage_pl(df_node, name='df_node')

    print(df_node.shape[0] == df_train['node'].n_unique())

    return df_node