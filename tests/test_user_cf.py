import numpy as np
import pandas as pd
import pytest

from movie_recommender.algorithms import ContentUserCF, RatingUserCF


class TestRatingUserCF:

    @pytest.fixture(scope='class')
    def rating_user_cf(self, rating_df, user_df, item_df):
        return RatingUserCF(rating_df=rating_df,
                            user_df=user_df,
                            item_df=item_df)

    def test_init(self, rating_user_cf):
        assert hasattr(rating_user_cf, 'matrix')
        assert hasattr(rating_user_cf, 'user_interacted_items')
        assert hasattr(rating_user_cf, 'searcher')
        assert hasattr(rating_user_cf, 'user_encoder')
        assert hasattr(rating_user_cf, 'item_encoder')

    def test_initialize_matrix(self, rating_user_cf):
        assert isinstance(rating_user_cf.matrix, np.ndarray)

    def test_infer_user(self, rating_user_cf):
        df = rating_user_cf.infer(user_id=1,
                                  similarity_top_k=2,
                                  output_top_k=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_infer_users(self, rating_user_cf):
        user_ids = [1, 2, -100]
        user_infer_dict = rating_user_cf.infer(user_ids=user_ids,
                                               similarity_top_k=2,
                                               output_top_k=2)
        assert isinstance(user_infer_dict, dict)
        assert set(user_infer_dict.keys()) == set(user_ids)
        assert all(
            isinstance(i, pd.DataFrame) for i in user_infer_dict.values())
        assert all(
            len(infer_df) == 2
            for user_id, infer_df in user_infer_dict.items()
            if user_id in [1, 2])
        assert user_infer_dict[-100].empty

    def test_check_user_status(self, rating_user_cf):
        user_ids = [1, 2, 3, -100]
        assert np.array_equal(
            rating_user_cf.check_user_status(user_ids=user_ids),
            [True, True, True, False])

    def test_create_query_matrix(self, rating_user_cf):
        query_matrix = rating_user_cf.create_query_matrix(user_id=1)
        assert isinstance(
            query_matrix, np.ndarray
        ) and query_matrix.ndim == 2 and query_matrix.shape[0] == 1

        query_matrix = rating_user_cf.create_query_matrix(user_id=[1, 2])
        assert isinstance(
            query_matrix, np.ndarray
        ) and query_matrix.ndim == 2 and query_matrix.shape[0] == 2

    @pytest.mark.parametrize('user_ids, expected_similar_user_ids',
                             [(1, [[3, 2]]), ([1, 2], [[3, 2], [3, 1]])])
    def test_search_similar_users(self, rating_user_cf, user_ids,
                                  expected_similar_user_ids):
        similar_user_ids, similarities = rating_user_cf.search_similar_users(
            user_id=user_ids, top_k=2)
        num_users = len(user_ids) if isinstance(user_ids, list) else 1
        assert isinstance(similar_user_ids, np.ndarray)
        assert isinstance(similarities, np.ndarray)
        assert similar_user_ids.ndim == 2 and similarities.ndim == 2
        assert similar_user_ids.shape[0] == num_users and similarities.shape[
            0] == num_users
        assert np.array_equal(similar_user_ids, expected_similar_user_ids)

    @pytest.mark.parametrize('params', [
        dict(user_id=1, similarity_top_k=2, output_top_k=2),
        dict(user_id=-100, similarity_top_k=2, output_top_k=2),
        dict(user_id=1,
             similarity_pairs=([3, 2], [0.5, 0.1]),
             similarity_top_k=2,
             output_top_k=2)
    ])
    def test_infer_item_by_user_id(self, rating_user_cf, params):
        df = rating_user_cf.infer_item_by_user_id(**params)
        assert isinstance(df, pd.DataFrame)

    def test_infer_item_by_user_ids(self, rating_user_cf):
        user_ids = [1, 2, -100]
        infer_dict = rating_user_cf.infer_item_by_user_ids(user_ids=user_ids,
                                                           similarity_top_k=2,
                                                           output_top_k=2)
        assert isinstance(infer_dict, dict)
        assert all(
            isinstance(infer_df, pd.DataFrame)
            for infer_df in infer_dict.values())
        assert set(infer_dict.keys()) == set(user_ids)
        assert infer_dict[-100].empty


class TestContentUserCF:

    @pytest.fixture(scope='class')
    def content_user_cf(self, rating_df, user_df, item_df):
        return ContentUserCF(rating_df=rating_df,
                             user_df=user_df,
                             item_df=item_df,
                             columns=['gender', 'age_interval'])

    def test_initialize_matrix(self, content_user_cf):
        assert isinstance(content_user_cf.matrix, np.ndarray)
        assert content_user_cf.matrix.ndim == 2 and content_user_cf.matrix.shape[
            1] == 4

    @pytest.mark.parametrize(
        'params',
        [dict(user_id=1), dict(user_features=['M', '25'])])
    def test_create_query_matrix(self, content_user_cf, params):
        query_matrix = content_user_cf.create_query_matrix(**params)
        assert isinstance(query_matrix,
                          np.ndarray) and query_matrix.shape == (1, 4)

    def test_check_user_status(self, content_user_cf):
        user_ids = [1, 2, 3, -100]
        assert all(content_user_cf.check_user_status(user_ids=user_ids))
