import copy
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from movie_recommender.utils import is_valid_sequence

from ....utils.typing import ItemIds, UserIds
from .cf import MovieCF

__all__ = ['RatingItemCF', 'GenreItemCF']

SIMILARITY_TOP_K: int = 10
OUTPUT_TOP_K: int = 100


class ItemCF(MovieCF, metaclass=ABCMeta):
    """Abstract Base class of Item-Based CF
    """

    @abstractmethod
    def create_query_matrix(self):
        ...

    def check_user_status(
            self, user_ids: Union[UserIds, int]) -> Union[List[bool], bool]:
        """check the user id in history data or not 

        Args:
            user_ids: single user id or user id list

        Returns:
            Union[List[bool], bool]: user id in history data or not
        """
        if not is_valid_sequence(user_ids):
            user_ids = [user_ids]
        state = np.isin(user_ids, self.user_interacted_items.index)
        return state if len(state) != 1 else state[0]

    def infer(self,
              item_id: Optional[int] = None,
              item_ids: Optional[ItemIds] = None,
              user_id: Optional[int] = None,
              similarity_top_k: int = SIMILARITY_TOP_K,
              output_top_k: int = OUTPUT_TOP_K) -> pd.DataFrame:
        """ItemCF infer function. Provide one of [`item_id`, `item_ids`, `user_id`]

        Args:
            item_id: item id
            item_ids: Sequence of item id
            user_id: user id
            similarity_top_k: Top-k item in similarity search
            output_top_k: Top-k item in output dataframe

        Returns:
            pd.DataFrame: inference dataframe
        """
        if (item_id is not None) ^ (item_ids is not None):
            if item_id is not None:
                item_ids = [item_id]
        elif (item_id is None) and (item_ids is None):
            assert user_id is not None
        else:
            raise NotImplementedError()

        return self.infer_items_by_item_ids(
            item_ids=item_ids,
            user_id=user_id,
            similarity_top_k=similarity_top_k,
            output_top_k=output_top_k,
        )

    def search_similar_items(
            self,
            item_ids: ItemIds,
            top_k: int = SIMILARITY_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        """Search Top-k similar items for each item id in `item_ids`. 
        
        Outputs: 
            Top-k item ids: shape: `(len(item_ids), top_k)`
            Top-k item index: shape: `(len(item_ids), top_k)`
        Args:
            item_ids: Sequence of item id
            top_k: Top-k similar items

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Top-k item ids, Top-k index)
        
        """
        query_matrix = self.create_query_matrix(item_id=item_ids)
        assert isinstance(query_matrix, np.ndarray) and query_matrix.ndim == 2
        # top_k + 1 is for excluding self item ids
        query_top_k = min(top_k + 1, self.matrix.shape[0])
        top_k_pairs = self.searcher.search(query_matrix, query_top_k)

        top_k_item_similarities, top_k_item_index = top_k_pairs[
            'similarities'], top_k_pairs['index']

        top_k_item_ids = self.item_encoder.inverse_transform(
            top_k_item_index.ravel()).reshape(top_k_item_index.shape)

        # exclude self item id
        np_item_ids = np.asarray([item_ids]).reshape(query_matrix.shape[0], 1)
        row_mask, col_mask = np.where(top_k_item_ids == np_item_ids)
        mask = np.ones(shape=(query_matrix.shape[0], query_top_k), dtype=bool)
        mask[row_mask, col_mask] = False

        # if no self item id, just take top-k items
        top_k_mask = np.setdiff1d(np.arange(query_matrix.shape[0]), row_mask)
        mask[top_k_mask, -1] = False

        top_k_item_similarities = top_k_item_similarities[mask].reshape(
            query_matrix.shape[0], -1)
        top_k_item_ids = top_k_item_ids[mask].reshape(query_matrix.shape[0],
                                                      -1)

        return top_k_item_ids, top_k_item_similarities

    def infer_items_by_item_ids(
            self,
            item_ids: Optional[ItemIds] = None,
            user_id: Optional[int] = None,
            similarity_top_k: int = SIMILARITY_TOP_K,
            output_top_k: int = OUTPUT_TOP_K) -> pd.DataFrame:
        """infer function by given one of [`item_ids`, `user_id`].
        If `user_id` is given, `item_ids` will be the interacted items from the specific `user_id`.
        Meanwhile, the ratings of items will be accessed as the scores of the items.
        Otherwise, just use the similarity as the scores  

        Args:
            item_ids: Sequence of itme id
            user_id: user id
            similarity_top_k: Top-k items in similarity search
            output_top_k: Top-k items output items

        Returns:
            pd.DataFrame: infer dataframe with `item_id` and `score`
        """
        assert (item_ids is not None) ^ (user_id is not None)
        if user_id is not None:
            if not self.check_user_status(user_ids=user_id):
                return pd.DataFrame([], columns=['item_id', 'score'])
            
            item_ids = self.user_interacted_items.loc[user_id].item_ids
            ratings = self.user_interacted_items.loc[user_id].ratings

        top_k_similar_item_ids, top_k_item_similarity = self.search_similar_items(
            item_ids=item_ids, top_k=similarity_top_k)

        if user_id is not None:
            top_k_item_scoring = np.multiply(top_k_item_similarity.T,
                                             np.asarray(ratings)).T
        else:
            top_k_item_scoring = top_k_item_similarity

        pred_df = pd.DataFrame(
            zip(*[top_k_similar_item_ids.ravel(),
                  top_k_item_scoring.ravel()]),
            columns=['item_id', 'score'])

        sort_df = pred_df.groupby('item_id').agg({
            'score': sum
        }).sort_values(by='score', ascending=False).reset_index(drop=False)
        return sort_df[~sort_df.item_id.isin(item_ids)].reset_index(
            drop=True).head(output_top_k)


class RatingItemCF(ItemCF):
    """Item-Based CF by using rating as similarity matrix
    """

    def initialize_matrix(self) -> np.ndarray:
        """Initialize rating similarity matrix

        Returns:
            np.ndarray: similarity matrix
        """
        matrix = np.zeros(shape=(len(self.item_encoder.classes_),
                                 len(self.user_encoder.classes_)),
                          dtype=np.float32)
        user_index = self.user_encoder.transform(self.rating_df.user_id)
        item_index = self.item_encoder.transform(self.rating_df.item_id)
        matrix[item_index, user_index] = self.rating_df.rating
        return matrix

    def create_query_matrix(self, item_id: Union[ItemIds, int]) -> np.ndarray:
        """create query matrix for similarity searching by `item_id`

        Args:
            item_id: item id or sequence of item id

        Returns:
            np.ndarray: query matrix
        """
        if not is_valid_sequence(item_id):
            item_id = [item_id]
        index = self.item_encoder.transform(item_id)
        query = copy.deepcopy(self.matrix[index, :])
        return query


class GenreItemCF(ItemCF):
    """Item-Based CF by using genre tags as similarity matrix
    """

    def __init__(self,
                 rating_df: pd.DataFrame,
                 user_df: pd.DataFrame,
                 item_df: pd.DataFrame,
                 genre_col: str = 'movie_genres'):
        self.genre_col = genre_col
        item_df[self.genre_col] = item_df[self.genre_col].apply(
            lambda a: a.split('|'))
        super().__init__(rating_df=rating_df, user_df=user_df, item_df=item_df)

    def initialize_user_encoder(self) -> LabelEncoder:
        """initial user encoder

        Returns:
            LabelEncoder: user encoder
        """
        return LabelEncoder().fit(self.user_df['user_id'])

    def initialize_item_encoder(self) -> LabelEncoder:
        """initial item encoder

        Returns:
            LabelEncoder: item encoder
        """
        return LabelEncoder().fit(self.item_df['item_id'])

    @property
    def genres(self):
        if getattr(self, '_genres', None) is None:
            self._genres = reduce(lambda a, b: set(a).union(b),
                                  self.item_df['movie_genres'])
            self._genres = sorted(self._genres)
        return self._genres

    def initialize_matrix(self) -> np.ndarray:
        """Initialize genre similarity matrix

        Returns:
            np.ndarray: similarity matrix
        """
        self.feature_encoder = MultiLabelBinarizer().fit(
            self.item_df[self.genre_col])
        matrix = np.zeros(shape=(len(self.item_encoder.classes_),
                                 len(self.feature_encoder.classes_)),
                          dtype=np.float32)
        matrix[self.item_encoder.transform(
            self.item_df.item_id)] = self.feature_encoder.transform(
                self.item_df[self.genre_col])
        return matrix

    def create_query_matrix(self, item_id: Union[ItemIds, int]) -> np.ndarray:
        """create query matrix for similarity searching

        Args:
            item_id: item id or sequence of item id

        Returns:
            np.ndarray: query matrix
        """
        if not is_valid_sequence(item_id):
            item_id = [item_id]
        index = self.item_encoder.transform(item_id)
        query = copy.deepcopy(self.matrix[index, :])
        return query
