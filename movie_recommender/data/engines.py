from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.distutils.misc_util import is_sequence
from sklearn.preprocessing import LabelEncoder

from .components import Item, User
from .mixins import InstanceFactoryMixin

__all__ = ['UserEngine', 'ItemEngine']


@dataclass
class _DataFrameEngine:
    df: pd.DataFrame
    id_column: str = 'id'

    def __post_init__(self):
        id_val = self.df[self.id_column].values
        self.encoder = LabelEncoder().fit(id_val)
        self.df = self.df.set_index(self.encoder.transform(id_val))

        for column in self.columns:
            setattr(self, column, self.df[column].values)

    def search(self, id: int, return_instance: bool = True) -> pd.DataFrame:
        try:
            id_list = np.array(
                [id]) if not is_sequence(id) else np.asarray(id).ravel()
            index = self.encoder.transform(id_list)
            df = self.df.loc[index]
            if return_instance:
                output = self._to_instance(df=df)
                if not is_sequence(id):
                    output = output[0]
            else:
                output = df
            return output
        except ValueError:
            raise

    def _to_instance(self, df: pd.DataFrame):
        return [self.create(**info) for _, info in df.iterrows()]

    @property
    def columns(self):
        return sorted(c for c in self.df.columns)

    def __len__(self):
        return len(self.df)


@dataclass
class ItemEngine(_DataFrameEngine, InstanceFactoryMixin):
    df: pd.DataFrame
    id_column: str = 'item_id'
    INSTANCE_CLASS: ClassVar[object] = Item


@dataclass
class UserEngine(_DataFrameEngine, InstanceFactoryMixin):
    df: pd.DataFrame
    id_column: str = 'user_id'
    INSTANCE_CLASS: ClassVar[object] = User
