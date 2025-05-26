
import polars as pl
import pandas as pd


BIG_CATEGORIES = [
    'most_freq_node',
    'most_freq_location',
    'node_last_contact',
    'location_last_contact',
    'node_last',
    'location_last',
]

class FeaturePool:
    """
    A class to manage a pool of features.
    1. It has a dictionary to store features by sets.
    2. It provides methods to add and retrieve features.
    3. It has diffrent types of features:
        - Category
        - Numerical
        - Ratios
        - Embeddings
    4. It derives features types (like category or ratio or numerical) from the data types of the features.
    5. It adds embedding features to a separete pool and adds a name to each embedding feature.
    6. It has a method to get the feature types of the features in the pool.


    """
    def __init__(self, key:str, catboost_mode=False):
        self.key = key
        self.catboost_mode = catboost_mode
        self.features = {}
        self.embeddings = {}
        self.feature_types = {}

    def add_features(self, features: pl.DataFrame, feature_set: str):
        """
        Add features to the pool.
        :param features: A DataFrame containing the features to be added.
        :param feature_set: The name of the feature set.
        """
        if feature_set not in self.features:
            self.features[feature_set] = features
        else:
            print(f"Feature set {feature_set} already exists. Please use a different name.")

        # add info about category and numerical features
        ratio_cols = self._get_ratio_columns(features)
        cat_cols = list(set(self._detect_categorical_columns(features)) - set(ratio_cols))

        if self.catboost_mode:
            cat_cols = cat_cols + [col for col in BIG_CATEGORIES if col in features.columns]
        
        num_cols = list((set(features.columns) - set(cat_cols)) - set(ratio_cols))
        num_cols.remove(self.key)

        self.feature_types[feature_set] = {
            'categorical': cat_cols,
            'numerical': num_cols,
            'ratios': ratio_cols
        }

        print(
            f"""Feature set {feature_set} added.\n\
            Columns:\n \
            - {len(cat_cols)} categorical features,\n \
            - {len(num_cols)} numerical features\n \
            - {len(ratio_cols)} ratio features."""
        )

        print("Shape match:", len(ratio_cols) + len(cat_cols) + len(num_cols) + 1 == len(features.columns))

    def get_categorical_features_names(self):
        cat_features = []
        for name, features in self.features.items():
            cat_features.extend(self.feature_types[name]['categorical'])
        return list(set(cat_features))
    
    def get_numerical_features_names(self):
        num_features = []
        for feature_set, features in self.features.items():
            num_features.extend(self.feature_types[feature_set]['numerical'])
        return list(set(num_features))
    
    def get_ratio_features_names(self):
        ratio_features = []
        for feature_set, features in self.features.items():
            ratio_features.extend(self.feature_types[feature_set]['ratios'])
        return list(set(ratio_features))

    
    def _get_ratio_columns(self, df, tol=0.01):
        cols_to_scale = []
        Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
        Numeric_Float_types = [pl.Float32,pl.Float64] 
        
        for col in df.columns:
            col_type = df[col].dtype

            if col_type not in Numeric_Int_types and col_type not in Numeric_Float_types:
                continue

            col_min = df[col].min()
            col_max = df[col].max()
            if (col_min >= -tol and col_max <= 1 + tol) or ('pr' in col or 'ratio' in col):
                cols_to_scale.append(col)
        return cols_to_scale

    def _detect_categorical_columns(self, df, max_unique=53):
        cat_cols = set()
        for col in df.columns:
            semantic_numeric = 'count' in col or 'sum' in col or 'total' in col or 'mean' in col or 'pr' in col or 'since' in col or 'unique' in col
            if df[col].n_unique() < max_unique and not semantic_numeric:
                cat_cols.add(col)
        
        return list(cat_cols)
    

    def transform_categorical_to_ohe(self):
        """
        Transform categorical features to one-hot encoding.
        """
        for feature_set, features in self.features.items():
            cat_features = self.feature_types[feature_set]['categorical']
            if len(cat_features) > 0:
                ohe_features = features.select(cat_features).to_dummies(drop_first=True)
                self.feature_types[feature_set]['categorical'] = list(ohe_features.columns)
                self.features[feature_set] = pl.concat([features.drop(cat_features), ohe_features], how='horizontal')


    def transform_numerical_to_min_max(self):
        """
        Transform numerical features to standard scale.
        """
        for feature_set, features in self.features.items():
            num_features = self.feature_types[feature_set]['numerical']
            if len(num_features) == 0:
                continue
            for col in num_features:
                min = features[col].min()
                max = features[col].max()

                features = features.with_columns(
                    (features[col] - min) / (max - min)
                )
            self.features[feature_set] = features

    def add_embedding_features(self, features: pl.DataFrame, feature_set: str):
        """
        Add embedding features to the pool.
        :param features: A DataFrame containing the features to be added.
        :param feature_set: The name of the feature set.
        """
        if feature_set not in self.embeddings:
            self.embeddings[feature_set] = features
        else:
            print(f"Feature set {feature_set} already exists. Please use a different name.")


    def get_embedding_features_names(self):
        """
        Get the embedding features from the pool.
        :return: A DataFrame containing the embedding features.
        """
        return self.embeddings.keys()
    

    def transform_embeddings_to_min_max(self):
        """
        Transform embedding features to mean and standard deviation.
        """
        for feature_set, features in self.embeddings.items():
            for col in features.columns:
                if col == self.key:
                    continue

                min = features[col].min()
                max = features[col].max()
                features = features.with_columns(
                    (features[col] - min) / (max - min)
                )
            self.embeddings[feature_set] = features
    
    def get_features(
            self, subset=None, add_categorical=True, add_numerical=True, add_ratios=True, add_embeddings=True,
            filter_by_name=(), add_by_name=()
        ):
        """
        Get the features from the pool.
        :param add_categorical: Whether to add categorical features.
        :param add_numerical: Whether to add numerical features.
        :param add_embeddings: Whether to add embedding features.
        :return: A DataFrame containing the features.
        """
        def _check_column(col):
            check = [pattern not in col for pattern in filter_by_name]
            return all(check) or not filter_by_name
        

        cols = [self.key]

        if add_ratios:
            for name, type2cols in self.feature_types.items():
                cur_cols = [col for col in type2cols['ratios'] if _check_column(col)]
                cols.extend(cur_cols)

        if add_categorical:
            for name, type2cols in self.feature_types.items():
                cur_cols = [col for col in type2cols['categorical'] if _check_column(col)]
                cols.extend(cur_cols)

        if add_numerical:
            for name, type2cols in self.feature_types.items():
                cur_cols = [col for col in type2cols['numerical'] if _check_column(col)]
                cols.extend(cur_cols)

        dfs = []
        for name, features in self.features.items():
            if subset is None or name in subset:
                cols_cur = [col for col in features.columns if col in cols]
                if add_by_name:
                    extra_cols = [col for col in add_by_name if col in features.columns]
                    print(f"Warning: {extra_cols} in {name} features")
                    cols_cur.extend(extra_cols)

                dfs.append(features.select(cols_cur))

        if add_embeddings:
            for name, features in self.embeddings.items():
                if subset is None or name in subset:
                    dfs.append(features)

        df = dfs[0]
        for df2 in dfs[1:]:
            df = df.join(df2, how='left', on=self.key)

        return df

    def get_melted_dataframe(
            self, subset=None, add_categorical=True, add_numerical=True, add_ratios=True, add_embeddings=True,
            filter_by_name=(), add_by_name=()
        ):

        df = self.get_features(
            subset, add_categorical=add_categorical, add_numerical=add_numerical, add_embeddings=add_embeddings,
            add_ratios=add_ratios, filter_by_name=filter_by_name, add_by_name=add_by_name
        )
        feat_columns = df.columns
        feat_columns.remove(self.key)

        df = df.to_pandas()
        df = df.rename(columns={self.key: 'id'})
        df = pd.melt(df, id_vars=['id'], var_name='feature', value_vars=feat_columns)

        return df, feat_columns

