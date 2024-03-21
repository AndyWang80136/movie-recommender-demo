import asyncio
from typing import List

import aiohttp
import pandas as pd

from .utils import RecallAlgo

__all__ = ['infer_users_async']


async def infer_users_async(model_params: List[RecallAlgo],
                            user_ids: List[int]) -> List[dict]:
    """async function for inference on user id on every parameters 

    Args:
        model_params: list of model parameters
        user_ids: list of users

    Returns:
        List[dict]: inference results
    """

    async def _infer_user_with_session(session: aiohttp.ClientSession,
                                       model_param: RecallAlgo, user_id: int):
        url = f'{model_param.url}/users/{user_id}'
        async with session.get(url,
                               params=model_param.hyperparams) as response:
            pred_df = pd.read_json(await response.json(), orient='split')
        return dict(user_id=user_id,
                    prediction_df=pred_df,
                    model_param=model_param._asdict())

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=timeout) as session:
        coroutines = [
            _infer_user_with_session(session, model_param, user_id)
            for model_param in model_params for user_id in user_ids
        ]
        infer_list = await asyncio.gather(*coroutines)
        return infer_list
