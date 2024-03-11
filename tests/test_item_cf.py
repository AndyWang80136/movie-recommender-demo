import numpy as np
import pandas as pd
import pytest

from movie_recommender.algorithms import GenreItemCF, RatingItemCF


class TestRatingItemCF:

    @pytest.fixture(scope='class')
    def rating_item_cf(self, rating_df, user_df, item_df):
        return RatingItemCF(rating_df=rating_df,
                            user_df=user_df,
                            item_df=item_df)

    def test_init(self, rating_item_cf):
        assert hasattr(rating_item_cf, 'matrix')
        assert hasattr(rating_item_cf, 'user_interacted_items')
        assert hasattr(rating_item_cf, 'searcher')
        assert hasattr(rating_item_cf, 'user_encoder')
        assert hasattr(rating_item_cf, 'item_encoder')

    def test_initialize_matrix(self, rating_item_cf):
        assert isinstance(rating_item_cf.matrix, np.ndarray)

    @pytest.mark.parametrize('top_k', [3, 10])
    def test_search_similar_items(self, top_k, rating_item_cf):
        top_k_item_ids, top_k_item_similarities = rating_item_cf.search_similar_items(
            item_ids=[1, 2], top_k=top_k)
        # output dim will be `top-k` if top-k < actual item size - 1
        test_top_k = min(top_k, len(rating_item_cf.item_encoder.classes_) - 1)
        assert isinstance(
            top_k_item_ids,
            np.ndarray) and top_k_item_ids.shape == (2, test_top_k)
        assert isinstance(
            top_k_item_similarities,
            np.ndarray) and top_k_item_similarities.shape == (2, test_top_k)

    def test_infer_items_by_item_ids(self, rating_item_cf):
        infer_df_by_user = rating_item_cf.infer_items_by_item_ids(
            user_id=1, similarity_top_k=3, output_top_k=2)
        infer_df_by_items = rating_item_cf.infer_items_by_item_ids(
            item_ids=[1, 2, 5], similarity_top_k=3, output_top_k=2)

        assert isinstance(infer_df_by_user, pd.DataFrame) and isinstance(
            infer_df_by_items, pd.DataFrame)
        assert infer_df_by_user.item_id.values.tolist(
        ) == infer_df_by_items.item_id.values.tolist()

    def test_infer_items_by_item_ids_exclude_self(self, rating_item_cf):
        infer_df_by_items = rating_item_cf.infer_items_by_item_ids(
            item_ids=[1, 2, 5], similarity_top_k=3, output_top_k=4)
        assert not any(np.isin(infer_df_by_items.item_id.values, [1, 2, 5]))

    def test_infer(self, rating_item_cf):
        df = rating_item_cf.infer(user_id=1,
                                  similarity_top_k=3,
                                  output_top_k=2)
        assert isinstance(df, pd.DataFrame) and len(df) == 2


class TestGenreItemCF:

    @pytest.fixture(scope='class')
    def genre_item_cf(self, rating_df, user_df, item_df):
        return GenreItemCF(rating_df=rating_df,
                           user_df=user_df,
                           item_df=item_df,
                           genre_col='movie_genres')

    def test_init(self, genre_item_cf):
        assert hasattr(genre_item_cf, 'genre_col')

    def test_initialize_encoders(self, genre_item_cf, user_df, item_df):
        assert np.array_equal(genre_item_cf.user_encoder.classes_,
                              user_df['user_id'].values)
        assert np.array_equal(genre_item_cf.item_encoder.classes_,
                              item_df['item_id'].values)

    def test_property_genres(self, genre_item_cf):
        assert genre_item_cf.genres == ['genre1', 'genre2', 'genre3', 'genre4']

    def test_initialize_matrix(self, genre_item_cf):
        assert isinstance(genre_item_cf.matrix, np.ndarray)
        assert genre_item_cf.matrix.shape[1] == len(genre_item_cf.genres)

    def test_create_query_matrix(self, genre_item_cf):
        query_matrix = genre_item_cf.create_query_matrix(item_id=1)
        assert isinstance(query_matrix, np.ndarray) and query_matrix.shape == (
            1, len(genre_item_cf.genres))

    def test_infer(self, genre_item_cf):
        df = genre_item_cf.infer(item_id=1, similarity_top_k=3, output_top_k=2)
        assert isinstance(df, pd.DataFrame) and len(df) == 2
