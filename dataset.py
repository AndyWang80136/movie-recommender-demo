from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import mmh3
import pandas as pd
from feature_analysis.data import ML100K, DatasetLoader
from sklearn.base import BaseEstimator, TransformerMixin


class HashBucketEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features into buckets by hash
    """

    def __init__(self, num_buckets: int, random_seed: int = 42):
        super().__init__()
        self.num_buckets = num_buckets
        self.random_seed = random_seed
        self._data = defaultdict(int)
        self._group = defaultdict(list)

    @property
    def collision_rate(self):
        """collision rate for different users into same bucket
        """
        num_collision_samples = sum([
            len(value) - 1 for value in self._group.values() if len(value) > 1
        ])
        total_samples = len(self._data)
        return round(num_collision_samples /
                     total_samples, 2) if total_samples != 0 else 0.

    @property
    def classes_(self):
        return list(range(self.num_buckets))

    def fit(self, data: pd.DataFrame):
        """fit each element in data and create groups

        Args:
            data: dataframe

        Returns:
            self: encoder
        """
        for d in data:
            self.search(d)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform each element in dataframe 

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: transformed dataframe
        """
        return df.apply(lambda a: self._data[a])

    def search(self, key: Union[int, str]) -> int:
        """search bucket group by key

        Args:
            key: key

        Returns:
            int: bucket group number
        """
        if key not in self._data:
            self._data[key] = mmh3.hash(str(key),
                                        self.random_seed) % self.num_buckets
            self._group[self._data[key]].append(key)
        return self._data[key]


class ML1M(ML100K):

    def __init__(self,
                 hash_buckets: Optional[Dict[str, int]] = None,
                 **kwargs):
        if hash_buckets is not None:
            categorical_encoders = kwargs.get('categorical_encoders', {})
            categorical_encoders.update({
                feature:
                HashBucketEncoder(num_buckets=bucket_size)
                for feature, bucket_size in hash_buckets.items()
            })
            kwargs.update(categorical_encoders=categorical_encoders)
        super().__init__(**kwargs)
        self.custom_features = ['year', 'freshness']

    def load_df(self, data_dir: Union[Path, str]) -> pd.DataFrame:
        """load dataframe

        Args:
            data_dir: dataframe directory

        Returns:
            pd.DataFrame: loaded dataframe
        """
        user_df = pd.read_csv(
            Path(data_dir).joinpath('users.dat'),
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python')
        movie_df = pd.read_csv(
            Path(data_dir).joinpath('movies.dat'),
            sep='::',
            names=['item_id', 'movie_title', 'movie_genres'],
            encoding='latin-1',
            engine='python')
        rating_df = pd.read_csv(
            Path(data_dir).joinpath('ratings.dat'),
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python')
        df = rating_df.merge(user_df, how='left',
                             on='user_id').merge(movie_df,
                                                 how='left',
                                                 on='item_id')

        df = self.create_new_features(df,
                                      features=('rating_datetime', 'phase'))
        self.phase = df['phase']
        self.user_df, self.movie_df = user_df, movie_df
        return df

    def categorical_encoder_fit(self, df: pd.DataFrame):
        categorical = self.categorical.copy()
        if 'user_id' in self.categorical:
            self.categorical.remove('user_id')
            self.categorical_encoders['user_id'].fit(
                self.user_df.user_id.values)
        if 'item_id' in self.categorical:
            self.categorical.remove('item_id')
            self.categorical_encoders['item_id'].fit(
                self.movie_df.item_id.values)
        super().categorical_encoder_fit(df)
        self.categorical = categorical

    @staticmethod
    def create_rating_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """transform timestamp to datetime

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: dataframe with rating timestamp
        """
        df['rating_datetime'] = df['timestamp'].apply(datetime.fromtimestamp)
        return df

    @staticmethod
    def create_phase(df: pd.DataFrame) -> pd.DataFrame:
        """create experiment phase by rating datetime, roughly train:val:test => 0.8:0.1:0.1

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: dataframe with phase
        """
        train_cutoff_datetime = datetime(2000, 12, 2)
        val_cutoff_datetime = datetime(2000, 12, 28)
        df['phase'] = df['rating_datetime'].apply(
            lambda a: 'test' if a >= val_cutoff_datetime else 'val'
            if a >= train_cutoff_datetime else 'train')
        return df

    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame]:
        """get train, val, and test dataframe by phase

        Args:
            df: dataframe

        Returns:
            Tuple[pd.DataFrame]: tuple of dataframe
        """
        train_df = df[self.phase == 'train']
        val_df = df[self.phase == 'val']
        test_df = df[self.phase == 'test']
        return train_df, val_df, test_df


DatasetLoader.add('ML1M', ML1M)
