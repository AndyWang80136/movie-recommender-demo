from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ['MMRReranker']


@dataclass
class MMRReranker:
    """Maximum Marginal Revelance on ranked items 
    """
    similarity_column: str = 'movie_genres'
    score_column: str = 'score'
    param_lamba: float = 0.5
    output_column_name: str = 'mr_score'

    def __post_init__(self):
        self._encoder = MultiLabelBinarizer()

    def rank(self,
             df: pd.DataFrame,
             top_k: int = 10,
             last_n: int = 1) -> pd.DataFrame:
        """rank on dataframe based on MMR algorithm

        Args:
            df: sorted dataframe
            top_k: Top-k items
            last_n: only compute on last-n items in selected set

        Returns:
            pd.DataFrame: ranked dataframe
        """
        df = df.reset_index(drop=True).copy()
        columns = [
            'item_id', self.score_column, self.similarity_column,
            self.output_column_name
        ]
        ranked_df = pd.DataFrame([], columns=columns)
        similarity_matrix = cosine_similarity(
            self._encoder.fit_transform(df[self.similarity_column].values))

        while len(ranked_df) < top_k and not df.empty:
            if ranked_df.empty:
                df[self.output_column_name] = self.param_lamba * df[
                    self.score_column]
                selected_row = df.iloc[[0]]
            else:
                ranked_index = ranked_df.iloc[-last_n:].index.values.astype(
                    int)
                candidate_index = df.index.values.astype(int)
                df[self.output_column_name] = self.param_lamba * df[
                    self.score_column] - (1 - self.param_lamba) * np.mean(
                        similarity_matrix[candidate_index.reshape(-1, 1),
                                          ranked_index],
                        axis=1)
                sort_df = df.sort_values(by=self.output_column_name,
                                         ascending=False)
                selected_row = sort_df.iloc[[0]]

            ranked_df = pd.concat([ranked_df, selected_row[columns]])
            df = df.drop(selected_row.index)
        return ranked_df
