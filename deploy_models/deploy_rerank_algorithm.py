import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from movie_recommender.algorithms import MMRReranker, UCBImpressionReranker

app = FastAPI()

ucb_impression_ranker = UCBImpressionReranker(column='item_id',
                                              exploitation_column='score',
                                              output_column_name='ucb_score',
                                              param_lambda=0.1)
mmr_ranker = MMRReranker(similarity_column='movie_genres',
                         param_lamba=0.95,
                         score_column='ucb_score',
                         output_column_name='score')


class Data(BaseModel):
    rank_df: str
    clicked_df: str
    impression_df: str


@app.post("/rerank/ucb-mmr/")
async def rerank_service(data: Data,
                         top_k: int = 10,
                         last_n: int = 1,
                         reset: bool = False):
    rank_df = pd.read_json(data.rank_df, orient='split')
    clicked_df = pd.read_json(data.clicked_df, orient='split')
    impression_df = pd.read_json(data.impression_df, orient='split')

    ucb_rank_df = ucb_impression_ranker.rank(df=rank_df,
                                             clicked_df=clicked_df,
                                             impression_df=impression_df,
                                             reset=reset,
                                             top_k=top_k)
    rerank_df = mmr_ranker.rank(df=ucb_rank_df, top_k=top_k, last_n=last_n)

    return rerank_df.to_json(orient='split')
