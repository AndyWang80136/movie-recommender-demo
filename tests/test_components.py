from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from movie_recommender.data import Item, Recommendations, User


class TestUser:

    @pytest.mark.parametrize('user_dict', [
        dict(user_id=1, gender='M', age_interval=25, occupation='occupation'),
        dict(user_id=2,
             gender='F',
             age_interval=40,
             occupation='occupation',
             statistics={'A': 1}),
        dict(user_id=3)
    ])
    def test_user(self, user_dict: dict):
        user = User(**user_dict)
        assert all(asdict(user)[k] == v for k, v in user_dict.items())

    def test_user_get(self):
        user = User(user_id=1,
                    gender='F',
                    statistics=dict(A=dict(B=1)),
                    algorithms=dict(B=dict(A=2)))
        assert user.get('user_id') == 1
        assert user.get('statistics/A') == dict(B=1)
        assert user.get('algorithms/B/A') == 2
        with pytest.raises(KeyError):
            user.get('algorithms/A/A')


class TestItem:

    @pytest.mark.parametrize('item_dict', [
        dict(item_id=1),
        dict(item_id=1, movie_title='title'),
        dict(item_id=2,
             movie_title='title',
             algorithms={'model': {
                 'name': 'model_1',
                 'score': 0.5
             }}),
        dict(item_id=2, movie_title='title', statistics={'A': 1})
    ])
    def test_item(self, item_dict: dict):
        item = Item(**item_dict)
        assert all(asdict(item)[k] == v for k, v in item_dict.items())

    def test_user_get(self):
        item = Item(item_id=1,
                    statistics=dict(A=dict(B=1)),
                    algorithms=dict(B=dict(A=2)))
        assert item.get('item_id') == 1
        assert item.get('statistics/A') == dict(B=1)
        assert item.get('algorithms/B/A') == 2
        with pytest.raises(KeyError):
            item.get('algorithms/B/B')


@pytest.mark.parametrize('item_dict', [
    dict(item_id=1, movie_genres='genre1|genre2'),
    dict(item_id=2, movie_genres='genre')
])
def test_item_movie_genres(item_dict):
    item = Item(**item_dict)
    assert isinstance(asdict(item)['movie_genres'], list)
    assert asdict(item)['movie_genres'] == item_dict['movie_genres'].split('|')


class TestRecommendations:

    @pytest.fixture(scope='class')
    def recommendations(self):
        return Recommendations(recommendations=[
            Item(item_id=1, movie_title=''),
            Item(item_id=2, movie_title='')
        ],
                               user_id=1,
                               status='success')

    def test_len(self, recommendations: Recommendations):
        assert len(recommendations) == 2

    def test_add(self):
        item1 = Item(item_id=1)
        recommendations = Recommendations(user_id=1)
        recommendations.add(item1)
        assert len(recommendations) == 1
        assert recommendations.recommendations == [item1]

    def test_to_df(self, recommendations: Recommendations):
        df = recommendations.to_df(columns=['item_id', 'movie_title'])
        assert isinstance(df, pd.DataFrame) and np.array_equal(
            df.columns.values, ['item_id', 'movie_title'])
        assert len(df) == 2

    def test_item_ids(self, recommendations: Recommendations):
        assert recommendations.item_ids == [1, 2]

    def test_get(self, recommendations: Recommendations):
        item = recommendations.get(item_id=1)
        assert isinstance(item, Item)
        assert item.item_id == 1

    def test_iter(self, recommendations: Recommendations):
        items = [item for item in recommendations]
        assert items == recommendations.recommendations

    def test_getitem(self, recommendations: Recommendations):
        assert isinstance(recommendations[0], Item)
        assert isinstance(recommendations[1], Item)
