from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .mixins import CosineSimilaritySearchingMixin


class MovieCF(CosineSimilaritySearchingMixin, metaclass=ABCMeta):
    """Abstract base class of Item-CF and User-CF

    Args:
        rating_df : rating dataframe with user_id, item_id, rating
        user_df : user dataframe with user_id
        item_df: item dataframe with item_id
    """

    def __init__(self, rating_df: pd.DataFrame, user_df: pd.DataFrame,
                 item_df: pd.DataFrame):
        self.user_df: pd.DataFrame = user_df
        self.item_df: pd.DataFrame = item_df
        self.rating_df: pd.DataFrame = rating_df

        self.user_interacted_items: pd.DataFrame = rating_df.groupby(
            'user_id').agg(
                item_ids=pd.NamedAgg(column='item_id', aggfunc=list),
                ratings=pd.NamedAgg(column='rating', aggfunc=list),
            )
        self.user_encoder: LabelEncoder = self.initialize_user_encoder()
        self.item_encoder: LabelEncoder = self.initialize_item_encoder()
        self.matrix: np.ndarray = self.initialize_matrix()
        self.searcher: CosineSimilaritySearchingMixin = self.initialize_searcher(
        )

    def initialize_user_encoder(self):
        return LabelEncoder().fit(self.rating_df['user_id'])

    def initialize_item_encoder(self):
        return LabelEncoder().fit(self.rating_df['item_id'])

    @abstractmethod
    def initialize_matrix(self):
        ...

    @abstractmethod
    def infer(self):
        ...
