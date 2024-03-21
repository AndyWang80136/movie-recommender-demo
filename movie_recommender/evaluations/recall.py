import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from movie_recommender.utils import PSQLLoader

from ..algorithms import RECALL_METHOD_INFO
from ..inference import infer_users_async
from ..inference.utils import RecallAlgo, get_recall_algo_params
from .metrics import evaluate_recall

__all__ = [
    'evaluate_params', 'evaluate_params_async', 'evaluate_recall_hyperparams',
    'evaluate_movielens_dataset'
]


def evaluate_params(model_params: List[RecallAlgo],
                    rating_df: pd.DataFrame) -> List[dict]:
    return asyncio.run(evaluate_params_async(model_params, rating_df))


async def evaluate_params_async(model_params: List[RecallAlgo],
                                rating_df: pd.DataFrame) -> List[dict]:
    """async function for evlauating with params

    Args:
        model_params: list of model parameters
        rating_df: groundtruth dataframe

    Returns:
        List[dict]: metric dict for each model params
    """
    eval_users = rating_df.user_id.unique().tolist()
    eval_ratings = rating_df.groupby('user_id').agg(
        items=pd.NamedAgg(column='item_id', aggfunc=list))

    infer_result_list = await infer_users_async(model_params=model_params,
                                                user_ids=eval_users)
    metrics = [None] * len(infer_result_list)
    for index, infer_result in enumerate(infer_result_list):
        user_id = infer_result['user_id']
        pred_items = infer_result['prediction_df']['item_id'].values.tolist()
        gt_items = eval_ratings.loc[user_id]['items']
        recall = evaluate_recall(pred_items=pred_items, gt_items=gt_items)
        user_type = 'new-user' if pred_items else 'history-user'
        metrics[index] = dict(user_type=user_type,
                              user_id=user_id,
                              **infer_result['model_param'],
                              **recall)
    return metrics


async def evaluate_recall_hyperparams(
        rating_df: pd.DataFrame, algo_url: str,
        hyperparams: Dict[str, Union[int,
                                     List[int]]]) -> Dict[str, List[dict]]:
    """apply multiprocess async evaluation function

    Args:
        rating_df: groundtruth dataframe
        algo_url: algorithm url
        hyperparams: model hyperparameters

    Returns:
        Dict[str, List[dict]]: metric list for algorithms
    """
    assert set(hyperparams.keys()).issubset(
        RECALL_METHOD_INFO
    ), f'Please assign hyperparams with dict using algorithm name as key {RECALL_METHOD_INFO} '
    hyperparam_dict = get_recall_algo_params(algo_url=algo_url,
                                             algo_params=hyperparams)

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, evaluate_params, hyperpram_list,
                                 rating_df)
            for hyperpram_list in hyperparam_dict.values()
        ]
    algo_metrics = dict(
        zip(hyperparam_dict.keys(), await asyncio.gather(*futures)))
    return algo_metrics


def evaluate_movielens_dataset(
        algo_url: str,
        hyperparams: Dict[str, Union[int, List[int]]],
        phase: Optional[str] = None,
        rating_df: Optional[pd.DataFrame] = None) -> Dict[str, List[dict]]:
    """evaluate on movielens dataset rating df

    Args:
        algo_url: algorithm url
        hyperparams: model hyperparameters
        phase: custom phase of movielens dataset (val)
        rating_df: groudtruth rating dataframe

    Returns:
        Dict[str, List[dict]]: metric list for algorithms
    """
    assert (phase is not None) ^ (rating_df is not None)
    if phase is not None:
        assert phase in ('train', 'val', 'test')
    eval_rating_df = PSQLLoader().load(
        'sql/rating_df.sql', query_params=dict(
            phase=phase)) if phase is not None else rating_df
    logger.info(
        f'Evaluating recall algorithms on data with hyperparameters: {hyperparams} \n\n {eval_rating_df}'
    )
    algo_metric_dict = asyncio.run(
        evaluate_recall_hyperparams(algo_url=algo_url,
                                    rating_df=eval_rating_df,
                                    hyperparams=hyperparams),
        debug=True,
    )
    return algo_metric_dict
