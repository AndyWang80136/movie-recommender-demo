from pathlib import Path
from typing import Union

import fire
from feature_analysis.process import train_process
from feature_analysis.utils import load_yaml
from loguru import logger

from dataset import *


def train(config: Union[Path, str]):
    config_dict = load_yaml(config)
    result = train_process(**config_dict)
    logger.info(result)


if __name__ == '__main__':
    fire.Fire(train)