import os

import pandas as pd
import pytest


@pytest.fixture(scope='session', name='user_df')
def test_user_df():
    return pd.DataFrame([(1, 'M', '25'), (2, 'F', '40'), (3, 'M', '25'),
                         (4, 'F', '40')],
                        columns=['user_id', 'gender', 'age_interval'])


@pytest.fixture(scope='session', name='item_df')
def test_item_df():
    return pd.DataFrame([
        (1, 'genre1|genre2'),
        (2, 'genre2|genre3'),
        (3, 'genre1'),
        (4, 'genre2'),
        (5, 'genre2|genre4'),
    ],
                        columns=['item_id', 'movie_genres'])


@pytest.fixture(scope='session', name='rating_df')
def test_rating_df():
    return pd.DataFrame([
        (1, 1, 4),
        (1, 2, 3),
        (1, 5, 5),
        (2, 1, 1),
        (2, 3, 5),
        (2, 4, 1),
        (3, 1, 4),
        (3, 2, 3),
        (3, 3, 5),
    ],
                        columns=['user_id', 'item_id', 'rating'])
