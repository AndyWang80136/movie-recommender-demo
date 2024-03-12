import numpy as np
import pandas as pd
import pytest

from movie_recommender.algorithms.rerank import (MMRReranker,
                                                 UCBImpressionReranker)


@pytest.fixture(scope='module')
def df():
    return pd.DataFrame(
        [
            (1, 0.9, ['genre1', 'genre2']),
            (2, 0.8, ['genre1', 'genre2']),
            (3, 0.7, ['genre2', 'genre3']),
            (4, 0.6, ['genre1', 'genre3']),
        ],
        columns=['item_id', 'score', 'movie_genres'],
    )


@pytest.fixture(scope='module')
def impression_df():
    return pd.DataFrame(
        [
            (2, 0.8, ['genre1', 'genre2']),
            (3, 0.7, ['genre2', 'genre3']),
        ],
        columns=['item_id', 'score', 'movie_genres'],
    )


@pytest.fixture(scope='module')
def clicked_df():
    return pd.DataFrame(
        [
            (3, 0.7, ['genre2', 'genre3']),
        ],
        columns=['item_id', 'score', 'movie_genres'],
    )


class TestMMRReranker:

    @pytest.fixture(scope='class')
    def mmr_ranker(self):
        return MMRReranker(similarity_column='movie_genres',
                           param_lamba=0.5,
                           output_column_name='mr_score')

    def test_rank(self, mmr_ranker, df):
        rank_df = mmr_ranker.rank(df=df, top_k=1)
        assert len(rank_df) == 1 and np.array_equal(
            rank_df.columns, [*df.columns, mmr_ranker.output_column_name
                              ]) and rank_df.iloc[0].item_id == 1
        rank_df = mmr_ranker.rank(df=df, top_k=3)
        assert np.array_equal(rank_df['item_id'].values, [1, 3, 2])
        rank_df = mmr_ranker.rank(df=df, top_k=10)
        assert len(rank_df) == 4


class TestUCBImpressionReranker:

    @pytest.fixture(scope='function')
    def ranker(self):
        return UCBImpressionReranker(column='item_id',
                                     output_column_name='ucb_score',
                                     exploitation_column='score')

    def test_rank(self, ranker, df, clicked_df, impression_df):
        out_df = ranker.rank(df=df,
                             clicked_df=clicked_df,
                             impression_df=impression_df)
        assert isinstance(out_df, pd.DataFrame)
        assert ranker.output_column_name in out_df.columns

    def test_calculate_exploitation_score(self, ranker, df):
        scores = ranker.calculate_exploitation_score(df=df)
        assert isinstance(scores, np.ndarray)
        assert np.array_equal(scores, df[ranker.exploitation_column].values)

    def test_calculate_exploration_score(self, ranker, df):
        ranker._n_times = 2
        ranker.params = {
            1: dict(clicks=0, impressions=1),
            2: dict(clicks=0, impressions=1),
            3: dict(clicks=0, impressions=2),
            4: dict(clicks=0, impressions=2),
        }
        scores = ranker.calculate_exploration_score(df=df)
        assert isinstance(scores, np.ndarray)
        assert scores[0] == scores[1]
        assert scores[2] == scores[3]

    def test_calculate_score(self, ranker, df):
        scores = ranker.calculate_score(df=df)
        assert isinstance(scores, np.ndarray)

    def test_reset(self, ranker, df, clicked_df, impression_df):

        out_df1 = ranker.rank(df=df,
                              clicked_df=clicked_df,
                              impression_df=impression_df,
                              reset=True)
        ranker_params = ranker.params
        ranker_n_times = ranker._n_times
        out_df2 = ranker.rank(df=df,
                              clicked_df=clicked_df,
                              impression_df=impression_df,
                              reset=True)
        assert out_df1.equals(out_df2)
        assert ranker_params == ranker.params
        assert ranker_n_times == ranker._n_times
