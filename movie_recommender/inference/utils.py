from collections import namedtuple
from itertools import product
from typing import Dict, List, Union

from movie_recommender.utils import is_valid_sequence

__all__ = ['RecallAlgo', 'get_recall_algo_params']

RecallAlgo = namedtuple('RecallAlgo', ['algo', 'url', 'hyperparams'])


def get_recall_algo_params(
        algo_url: str,
        algo_params: Dict[str, Union[int, List[int]]]) -> List[RecallAlgo]:
    params = {
        algo_name: {
            k: [v] if not is_valid_sequence(v) else v
            for k, v in hyperparams.items()
        }
        for algo_name, hyperparams in algo_params.items()
    }
    
    algo_params = {}
    for algo_name, hyperparams in params.items():
        param_list = [
            dict(zip(hyperparams.keys(), param))
            for param in product(*hyperparams.values())
        ]
        algo_params[algo_name] = [
            RecallAlgo(algo=algo_name,
                       url=f'{algo_url}/{algo_name}',
                       hyperparams=param_dict) for param_dict in param_list
        ]
    return algo_params
