import copy
from abc import ABCMeta, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from movie_recommender.utils import is_valid_sequence

from ....utils.typing import UserIds
from .cf import MovieCF

__all__ = ['RatingUserCF', 'ContentUserCF']

SIMILARITY_TOP_K: int = 10
OUTPUT_TOP_K: int = 100


class UserCF(MovieCF, metaclass=ABCMeta):
    """Abstract Base class of User-Based CF
    """

    @abstractmethod
    def create_query_matrix(self):
        ...

    @abstractmethod
    def check_user_status(self):
        ...

    def infer(self,
              user_id: Optional[int] = None,
              user_ids: Optional[UserIds] = None,
              similarity_top_k: int = SIMILARITY_TOP_K,
              output_top_k: int = OUTPUT_TOP_K):
        """UserCF infer function. Provide one of [`user_id`, `user_ids`]

        Args:
            user_id: user id
            user_ids: Sequence of user id
            similarity_top_k: Top-k users in similarity search
            output_top_k: Top-k users in output dataframe

        Returns:
            pd.DataFrame: inference dataframe
        """
        assert (user_id is not None) ^ (user_ids is not None)
        return self.infer_item_by_user_id(
            user_id=user_id,
            similarity_top_k=similarity_top_k,
            output_top_k=output_top_k
        ) if user_id is not None else self.infer_item_by_user_ids(
            user_ids=user_ids,
            similarity_top_k=similarity_top_k,
            output_top_k=output_top_k)

    def search_similar_users(
            self,
            user_id: Optional[Union[UserIds, int]] = None,
            query_matrix: Optional[np.ndarray] = None,
            top_k: int = SIMILARITY_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        """Search Top-k similar users for each user id in `user_id` or `query_matrix`.
        Provide one of [`user_id`, `query_matrix`]
        
        Outputs: 
            Top-k user ids: shape: `(len(user_id), top_k)`
            Top-k user index: shape: `(len(user_id), top_k)`

        Args:
            user_id: user id or Sequence of user id
            query_matrix: query matrix
            top_k: Top-k users

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Top-k user ids, Top-k user index)
        """
        assert (user_id is not None) ^ (query_matrix is not None)
        if query_matrix is None:
            query_matrix = self.create_query_matrix(user_id=user_id)
        assert isinstance(query_matrix, np.ndarray) and query_matrix.ndim == 2
        # top_k + 1 is for excluding self user ids
        query_top_k = min(top_k + 1, self.matrix.shape[0])
        top_k_pairs = self.searcher.search(query_matrix, top_k=query_top_k)
        top_k_user_similarities, top_k_user_index = top_k_pairs[
            'similarities'], top_k_pairs['index']
        top_k_user_ids = self.user_encoder.inverse_transform(
            top_k_user_index.ravel()).reshape(top_k_user_index.shape)

        if user_id is not None:
            # remove self user id if `user_id` is given
            user_id = user_id if is_valid_sequence(user_id) else [user_id]
            assert query_matrix.shape[0] == len(user_id)
            np_user_ids = np.asarray(user_id).reshape(query_matrix.shape[0],
                                                      -1)
            mask = np.ones(shape=(query_matrix.shape[0], top_k + 1),
                           dtype=bool)
            row_mask, col_mask = np.where(top_k_user_ids == np_user_ids)
            top_k_mask = np.setdiff1d(np.arange(query_matrix.shape[0]),
                                      row_mask)
            mask[row_mask, col_mask] = False
            mask[top_k_mask, -1] = False

            top_k_user_similarities = top_k_user_similarities[mask].reshape(
                query_matrix.shape[0], top_k)
            top_k_user_ids = top_k_user_ids[mask].reshape(
                query_matrix.shape[0], top_k)
        return top_k_user_ids, top_k_user_similarities

    def infer_item_by_user_id(
            self,
            user_id: int,
            similarity_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            similarity_top_k: int = SIMILARITY_TOP_K,
            output_top_k: int = OUTPUT_TOP_K) -> pd.DataFrame:
        """infer function by given `user_id`.
        If `user_id` not in history data, empty dataframe is returned

        Args:
            user_id: user id
            similarity_pairs: Similarity pairs of similar users ids and index
            similarity_top_k: Top-k users in similarity search
            output_top_k: Top-k output items

        Returns:
            pd.DataFrame: infer dataframe with `item_id` and `score`
        """

        if not self.check_user_status(user_ids=user_id):
            return pd.DataFrame([], columns=['item_id', 'score'])

        if similarity_pairs is None:
            similar_user_ids, cos_sim = self.search_similar_users(
                user_id=user_id, top_k=similarity_top_k)
            # only one user input, so take first element of similar_user_ids and cos_sim
            similar_user_ids, cos_sim = similar_user_ids[0], cos_sim[0]
        else:
            similar_user_ids, cos_sim = np.asarray(
                similarity_pairs[0]), np.asarray(similarity_pairs[1])
        assert similar_user_ids.ndim == 1 and cos_sim.ndim == 1

        similar_users_info = self.user_interacted_items.loc[
            similar_user_ids].copy()
        similar_users_info['weight'] = cos_sim
        infer_items_ratings = similar_users_info.apply(lambda a: dict(
            zip(a['item_ids'],
                np.asarray(a['ratings']) * a['weight'])),
                                                       axis=1)
        infer_items_ratings = sum(
            infer_items_ratings.apply(lambda a: Counter(a)), Counter())

        # filter out current user_id interacted items
        if user_id in self.user_interacted_items.index:
            for item_id in self.user_interacted_items.loc[user_id, 'item_ids']:
                if item_id in infer_items_ratings:
                    infer_items_ratings.pop(item_id)

        pred_df = pd.DataFrame(infer_items_ratings.most_common(output_top_k),
                               columns=['item_id', 'score'])
        return pred_df

    def infer_item_by_user_ids(
            self,
            user_ids: UserIds,
            similarity_top_k: int = SIMILARITY_TOP_K,
            output_top_k: int = OUTPUT_TOP_K) -> Dict[int, pd.DataFrame]:
        """apply multithread on each user id

        Args:
            user_ids: Sequence of user id
            similarity_top_k: Top-k similar users 
            output_top_k: Top-k output items

        Returns:
            Dict[int, pd.DataFrame]: dict with user_id as key and dataframe as value
        """
        user_id_infer_state = self.check_user_status(user_ids=user_ids)
        cos_sim_index = np.cumsum(user_id_infer_state) - 1
        similar_user_ids, cos_sim = self.search_similar_users(
            user_id=np.asarray(user_ids)[user_id_infer_state],
            top_k=similarity_top_k)
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = {
                user_id:
                executor.submit(
                    self.infer_item_by_user_id,
                    **dict(
                        user_id=user_id,
                        similarity_pairs=(similar_user_ids[idx],
                                          cos_sim[idx]) if is_infer else None,
                        similarity_top_k=similarity_top_k,
                        output_top_k=output_top_k))
                for user_id, is_infer, idx in zip(
                    user_ids, user_id_infer_state, cos_sim_index)
            }

        return {k: v.result() for k, v in futures.items()}


class RatingUserCF(UserCF):
    """User-Based CF by using rating as similarity matrix
    """

    def check_user_status(
            self, user_ids: Union[UserIds, int]) -> Union[List[bool], bool]:
        """check the user id in history data or not 

        Args:
            user_ids: single user id or user id list

        Returns:
            Union[List[bool], bool]: user id in history data or not
        """
        user_id_list = [user_ids] if not is_valid_sequence(user_ids) else user_ids
        state = np.isin(user_id_list,
                        self.user_interacted_items.index).tolist()
        return state if is_valid_sequence(user_ids) else state[0]

    def create_query_matrix(self, user_id: Union[UserIds, int]) -> np.ndarray:
        """create query matrix for similarity searching by `user_id`

        Args:
            user_id: user id or sequence of user id

        Returns:
            np.ndarray: query matrix
        """
        if not is_valid_sequence(user_id):
            user_id = [user_id]
        index = self.user_encoder.transform(user_id)
        query = copy.deepcopy(self.matrix[index, :])
        return query

    def initialize_matrix(self) -> np.ndarray:
        """Initialize rating similarity matrix

        Returns:
            np.ndarray: similarity matrix
        """
        matrix = np.zeros(shape=(len(self.user_encoder.classes_),
                                 len(self.item_encoder.classes_)),
                          dtype=np.float32)
        user_index = self.user_encoder.transform(self.rating_df.user_id)
        item_index = self.item_encoder.transform(self.rating_df.item_id)
        matrix[user_index, item_index] = self.rating_df.rating
        return matrix


class ContentUserCF(UserCF):
    """User-Based CF by using user content tags as similarity matrix
    """

    def __init__(self,
                 rating_df: pd.DataFrame,
                 user_df: pd.DataFrame,
                 item_df: pd.DataFrame,
                 columns: List[str] = ['gender', 'age_interval',
                                       'occupation']):
        self.columns = columns
        super().__init__(rating_df=rating_df, user_df=user_df, item_df=item_df)

    def check_user_status(
            self, user_ids: Union[UserIds, int]) -> Union[List[bool], bool]:
        """check the user id status

        Args:
            user_ids: single user id or user id list

        Returns:
            Union[List[bool], bool]: user id in history data or not
        """
        user_id_list = [user_ids] if not is_valid_sequence(user_ids) else user_ids
        return [True] * len(user_id_list) if is_valid_sequence(user_ids) else True

    def initialize_matrix(self) -> np.ndarray:
        """Initialize user content similarity matrix

        Returns:
            np.ndarray: similarity matrix
        """
        self.feature_encoder = MultiLabelBinarizer().fit(
            self.user_df[self.columns].values)
        matrix = np.ones(shape=(len(self.user_encoder.classes_),
                                len(self.feature_encoder.classes_)),
                         dtype=np.float32)
        matrix[self.user_encoder.transform(
            self.user_encoder.classes_)] = self.feature_encoder.transform(
                self.user_df.set_index('user_id').loc[
                    self.user_encoder.classes_][self.columns].values)
        return matrix.astype(np.float32)

    def create_query_matrix(self,
                            user_id: Optional[int] = None,
                            user_features: Optional[List[str]] = None):
        """create query matrix for similarity searching

        Args:
            user_id: user id or sequence of user id
            user_features: user features in list

        Returns:
            np.ndarray: query matrix
        """
        assert (user_id is not None) ^ (user_features is not None)
        if user_id is not None:
            if not is_valid_sequence(user_id):
                user_id = [user_id]
            query = self.feature_encoder.transform(
                self.user_df.set_index('user_id').loc[user_id][
                    self.columns].values).astype(np.float32)
        else:
            query = self.feature_encoder.transform([user_features])
        return query
