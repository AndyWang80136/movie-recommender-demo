from movie_recommender.data import MovieData


def test_movie_data(user_df, item_df, rating_df):
    movie_data = MovieData(users=user_df, items=item_df, ratings=rating_df)
    assert hasattr(movie_data, 'users')
    assert hasattr(movie_data, 'items')
    assert hasattr(movie_data, 'ratings')