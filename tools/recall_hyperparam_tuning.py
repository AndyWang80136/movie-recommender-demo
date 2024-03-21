from pathlib import Path
from typing import Dict, List, Union

import fire
import pandas as pd
from loguru import logger

from movie_recommender.algorithms.recall import RECALL_METHOD_INFO
from movie_recommender.evaluations import evaluate_movielens_dataset
from movie_recommender.utils.io import dump_json

BASE_URL = 'http://localhost:8000/recall'


def generate_metric_info(algo_metrics: Dict[str, List[dict]]):
    metric_df = pd.concat(
        [pd.DataFrame(metric) for metric in algo_metrics.values()])

    param_df = metric_df.pop('hyperparams').apply(pd.Series)
    metric_df = pd.concat((metric_df, param_df), axis=1)
    user_metric_df = metric_df.groupby(['algo',
                                        *param_df.columns.tolist()]).agg({
                                            'metric_value':
                                            'mean'
                                        }).reset_index(drop=False)
    best_df = user_metric_df.sort_values(by='metric_value',
                                         ascending=False).groupby(['algo'
                                                                   ]).head(1)
    best_metrics = {
        row.pop('algo'): row.to_dict()
        for _, row in best_df.iterrows()
    }
    best_params = {
        algo: {
            k: v
            for k, v in metric.items() if k != 'metric_value'
        }
        for algo, metric in best_metrics.items()
    }

    return dict(raw_metrics=algo_metrics,
                best_metrics=best_metrics,
                best_params=best_params)


def tune_parameters(save_dir: Path = Path('./param_tuning_results'),
                    similarity_top_k: Union[int, List[int]] = 10,
                    output_top_k: Union[int, List[int]] = 100,
                    phase: str = 'val'):
    hyperparams = {
        algo: dict(similarity_top_k=similarity_top_k,
                   output_top_k=output_top_k)
        for algo in RECALL_METHOD_INFO
    }
    algo_metric_dict = evaluate_movielens_dataset(
        algo_url=BASE_URL,
        phase=phase,
        hyperparams=hyperparams,
    )

    metric_info = generate_metric_info(algo_metric_dict)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for k, v in metric_info.items():
        dump_json(v, Path(save_dir).joinpath(f'{k}.json'))
        logger.info(f'Save {k} at {Path(save_dir).joinpath(f"{k}.json")}')


if __name__ == '__main__':
    fire.Fire(tune_parameters)
