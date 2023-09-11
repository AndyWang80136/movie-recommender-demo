from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from feature_analysis.data import ML100K, DatasetLoader


class ML1M(ML100K):

    def __init__(self, **kwargs):
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
        return df

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
