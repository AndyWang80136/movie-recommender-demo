import numpy as np
import pandas as pd
import pytest

from movie_recommender.data.components import Item, User
from movie_recommender.data.engines import ItemEngine, UserEngine


class TestUserEngine:

    @pytest.fixture(scope='class')
    def user_engine(self, user_df):
        return UserEngine(df=user_df)

    def test_create(self, user_engine):
        user = user_engine.create(user_id=1)
        assert isinstance(user, UserEngine.INSTANCE_CLASS)

    @pytest.mark.parametrize('query_id', [1, [1], [1, 2, 3]])
    def test_search(self, user_engine, query_id):
        user = user_engine.search(id=query_id)
        assert isinstance(user, UserEngine.INSTANCE_CLASS) if not isinstance(
            query_id, (list, tuple, np.ndarray)) else all(
                isinstance(i, UserEngine.INSTANCE_CLASS) for i in user)
        user_df = user_engine.search(id=query_id, return_instance=False)
        assert isinstance(user_df, pd.DataFrame)

    @pytest.mark.parametrize('query_id', [-1, '1'])
    def test_search_exception(self, user_engine, query_id):
        with pytest.raises(ValueError):
            user_engine.search(id=query_id)


class TestItemEngine:

    @pytest.fixture(scope='class')
    def item_engine(self, item_df):
        return ItemEngine(df=item_df)

    def test_create(self, item_engine):
        item = item_engine.create(item_id=1)
        assert isinstance(item, ItemEngine.INSTANCE_CLASS)

    @pytest.mark.parametrize('query_id', [1, [1], [1, 2, 3]])
    def test_search(self, item_engine, query_id):
        item = item_engine.search(id=query_id)
        assert isinstance(item, ItemEngine.INSTANCE_CLASS) if not isinstance(
            query_id, (list, tuple, np.ndarray)) else all(
                isinstance(i, ItemEngine.INSTANCE_CLASS) for i in item)
        item_df = item_engine.search(id=query_id, return_instance=False)
        assert isinstance(item_df, pd.DataFrame)

    @pytest.mark.parametrize('query_id', [-1, '1'])
    def test_search_exception(self, item_engine, query_id):
        with pytest.raises(ValueError):
            item_engine.search(id=query_id)
