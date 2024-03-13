from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class _ImpressionReranker(metaclass=ABCMeta):
    """Base class for item impressions and clicks
    """
    column: str
    output_column_name: str = 'score'
    _n_times: int = 0

    def __post_init__(self):
        self.reset_params()

    def reset_params(self):
        """reset default parameters
        """
        self.params = defaultdict(lambda: {'clicks': 0, 'impressions': 0})
        self._n_times = 0

    def update_impressions_param(self, value: List[Union[str, int]]):
        """update impressions items

        Args:
            value: item value
        """
        for v in value:
            self.params[v]['impressions'] += 1

    def update_clicks_param(self, value: List[Union[str, int]]):
        """update clicks items

        Args:
            value: item value
        """
        for v in value:
            self.params[v]['clicks'] += 1

    @staticmethod
    def get_column_type(df: pd.DataFrame, column_name: str) -> type:
        """get the type of `df[column_name]`

        Args:
            df: dataframe
            column_name: column name

        Returns:
            type: instance type
        """
        column_type = set(df[column_name].apply(type).values)
        assert len(column_type) == 1
        return list(column_type)[0]

    def get_value_from_df(self, df: pd.DataFrame) -> list:
        """get the list of value on `df[self.column]`

        Args:
            df: dataframe

        Returns:
            list: list of value
        """
        if self.get_column_type(df,
                                self.column) not in [list, tuple, np.ndarray]:
            return list(set(df[self.column]))
        else:
            return list(reduce(lambda a, b: set(a).union(b), df[self.column]))

    @abstractmethod
    def calculate_score(self, df: pd.DataFrame):
        ...

    def rank(self,
             df: pd.DataFrame,
             clicked_df: pd.DataFrame,
             impression_df: pd.DataFrame,
             reset: bool = False,
             top_k: Optional[int] = None) -> pd.DataFrame:
        """rank the dataframe with the generated score

        Args:
            df: dataframe
            clicked_df: clicked dataframe
            impression_df: impression dataframe
            reset: reset parameters or not 
            top_k: Top-k output dataframe

        Returns:
            pd.DataFrame: ranked dataframe
        """
        if reset:
            self.reset_params()
        
        out_df = df.copy()
        out_df[self.output_column_name] = self.calculate_score(out_df)
        out_df = out_df.sort_values(by=self.output_column_name,
                                    ascending=False).reset_index(drop=True)
        if top_k:
            out_df = out_df.head(top_k)
        
        self._n_times += 1
        if not impression_df.empty:
            impression_value_list = self.get_value_from_df(impression_df)
            self.update_impressions_param(impression_value_list)
        if not clicked_df.empty:
            clicked_value_list = self.get_value_from_df(clicked_df)
            self.update_clicks_param(clicked_value_list)

        return out_df
