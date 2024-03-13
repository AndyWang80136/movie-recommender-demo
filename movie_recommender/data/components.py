from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import pandas as pd

from .mixins import InstanceMixin

__all__ = ['User', 'Item', 'Recommendations']


@dataclass
class User(InstanceMixin):
    user_id: int
    gender: Optional[str] = None
    age_interval: Optional[int] = None
    occupation: Optional[str] = None
    statistics: dict = field(default_factory=dict)
    algorithms: dict = field(default_factory=dict)


@dataclass
class Item(InstanceMixin):
    item_id: int
    movie_genres: Optional[Union[str, List[str]]] = None
    movie_title: Optional[str] = None
    statistics: dict = field(default_factory=dict)
    algorithms: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.movie_genres is not None:
            self.movie_genres = self.movie_genres.split('|')


@dataclass
class Recommendations:
    """stores item information of recommendations
    """
    recommendations: List[Item] = field(default_factory=list)
    user_id: Optional[int] = None
    status: str = 'not started'
    statistics: dict = field(default_factory=dict)
    algorithms: list = field(default_factory=list)
    _index: dict = field(default_factory=dict)

    def __post_init__(self):
        for item in self.recommendations:
            self._index[item.item_id] = item

    def __getitem__(self, index: int):
        if index < len(self.recommendations):
            return self.recommendations[index]
        else:
            raise IndexError

    def __iter__(self):
        index = 0
        while index < len(self.recommendations):
            yield self.recommendations[index]
            index += 1

    def __len__(self):
        return len(self.recommendations)

    def add(self, item: Item):
        self.recommendations.append(item)
        self._index[item.item_id] = item

    def get(self, item_id: int):
        try:
            return self._index[item_id]
        except KeyError:
            raise

    def to_df(self, columns: Tuple[str] = ('item_id', )):
        output_columns = [c.split('/')[-1] for c in columns]
        return pd.DataFrame([[item.get(c) for c in columns]
                             for item in self.recommendations],
                            columns=output_columns)

    @property
    def item_ids(self):
        return [item.item_id for item in self.recommendations]
