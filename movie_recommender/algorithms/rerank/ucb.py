from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import _ImpressionReranker

__all__ = ['UCBImpressionReranker']


@dataclass
class UCBImpressionReranker(_ImpressionReranker):
    """Reranker based on Upper Confidence Bound on impressions
    """
    column: str = 'item_id'
    exploitation_column: str = 'score'
    output_column_name: str = 'ucb_score'
    param_lambda: float = 0.5

    def calculate_exploitation_score(self, df: pd.DataFrame) -> np.ndarray:
        """calculate exploitation score

        Args:
            df: dataframe

        Returns:
            np.ndarray: score array
        """
        return df[self.exploitation_column].values

    def calculate_exploration_score(self, df: pd.DataFrame) -> np.ndarray:
        """calculate exploration score, calculate on impressions and number of times

        Args:
            df: dataframe

        Returns:
            np.ndarray: score array
        """
        num_impressions_per_item = np.asarray(
            [self.params[item]['impressions'] for item in df[self.column]])
        assert np.all(num_impressions_per_item <= self._n_times)
        return (np.log(self._n_times + 1) /
                (num_impressions_per_item + 1))**0.5

    def calculate_score(self, df: pd.DataFrame) -> np.ndarray:
        """calculate ucb score

        Args:
            df: dataframe

        Returns:
            np.ndarray: score array
        """
        exploitation_score = self.calculate_exploitation_score(df=df)
        exploration_score = self.calculate_exploration_score(df=df)
        return exploitation_score + self.param_lambda * exploration_score
