from typing import List

import uvicorn
from fastapi import FastAPI, Query
from typing_extensions import Annotated

from dataset import *
from movie_recommender.algorithms import (ContentUserCF, GenreItemCF,
                                          RatingItemCF, RatingUserCF)
from movie_recommender.data import MovieData
from movie_recommender.utils import PSQLLoader

app = FastAPI()

loader = PSQLLoader()
data = MovieData(users=loader.load('sql/user_df.sql'),
                 items=loader.load('sql/item_df.sql'),
                 ratings=loader.load('sql/rating_df.sql',
                                     query_params=dict(phase='train')))

content_user_cf = ContentUserCF(user_df=data.users,
                                item_df=data.items,
                                rating_df=data.ratings)
rating_user_cf = RatingUserCF(user_df=data.users,
                              item_df=data.items,
                              rating_df=data.ratings)
rating_item_cf = RatingItemCF(rating_df=data.ratings,
                              user_df=data.users,
                              item_df=data.items)
genre_item_cf = GenreItemCF(rating_df=data.ratings,
                            user_df=data.users,
                            item_df=data.items)


@app.get("/recall/itemcf-ratings/users/{user_id}")
async def itemcf_ratings_user(user_id: int,
                              similarity_top_k: int = 50,
                              output_top_k: int = 100):
    pred = rating_item_cf.infer(user_id=user_id,
                                similarity_top_k=similarity_top_k,
                                output_top_k=output_top_k)
    return pred.to_json(orient='split')


@app.get("/recall/usercf-ratings/users/{user_id}")
async def usercf_ratings_user(user_id: int,
                              similarity_top_k: int = 50,
                              output_top_k: int = 100):
    pred = rating_user_cf.infer(user_id=user_id,
                                similarity_top_k=similarity_top_k,
                                output_top_k=output_top_k)
    return pred.to_json(orient='split')


@app.get("/recall/usercf-content/users/{user_id}")
async def usercf_content_user(user_id: int,
                              similarity_top_k: int = 50,
                              output_top_k: int = 100):
    pred = content_user_cf.infer(user_id=user_id,
                                 similarity_top_k=similarity_top_k,
                                 output_top_k=output_top_k)
    return pred.to_json(orient='split')


@app.get("/recall/itemcf-genres/users/{user_id}")
async def itemcf_genres_user(user_id: int,
                             similarity_top_k: int = 50,
                             output_top_k: int = 100):
    pred = genre_item_cf.infer(user_id=user_id,
                               similarity_top_k=similarity_top_k,
                               output_top_k=output_top_k)
    return pred.to_json(orient='split')


@app.get("/recall/item-genres/items/")
async def item_genres(item_ids: Annotated[List[int], Query()] = None,
                      similarity_top_k: int = 50,
                      output_top_k: int = 100):
    pred = genre_item_cf.infer(item_ids=item_ids,
                               similarity_top_k=similarity_top_k,
                               output_top_k=output_top_k)
    return pred.to_json(orient='split')
