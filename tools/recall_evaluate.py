from pathlib import Path

import fire
import pandas as pd
from loguru import logger

from movie_recommender.evaluations import evaluate_movielens_dataset
from movie_recommender.utils import dump_json, load_json

BASE_URL = 'http://localhost:8000/recall'


def generate_metrics(algo_metrics: dict) -> dict:
    metrics = {}
    for algo_name, metric in algo_metrics.items():
        metric_df = pd.DataFrame(metric).groupby('metric_type').agg(
            {'metric_value': 'mean'})
        metrics[algo_name] = dict(
            zip(metric_df.index, metric_df['metric_value']))

    return dict(raw_metrics=algo_metrics, metrics=metrics)


def evaluate(
        config: Path,
        phase: str,
        save_dir: Path = Path('./eval_results'),
):
    infer_params = load_json(config)
    algo_metric_dict = evaluate_movielens_dataset(algo_url=BASE_URL,
                                                  phase=phase,
                                                  hyperparams=infer_params)
    algo_metrics = generate_metrics(algo_metrics=algo_metric_dict)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for k, v in algo_metrics.items():
        dump_json(v, Path(save_dir).joinpath(f'{k}.json'))
        logger.info(f'Save {k} at {Path(save_dir).joinpath(f"{k}.json")}')


if __name__ == '__main__':
    fire.Fire(evaluate)
