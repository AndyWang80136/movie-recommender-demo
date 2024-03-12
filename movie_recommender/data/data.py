from dataclasses import dataclass

import pandas as pd

__all__ = ['MovieData']


@dataclass
class MovieData:
    users: pd.DataFrame
    items: pd.DataFrame
    ratings: pd.DataFrame
