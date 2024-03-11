import copy

import faiss
import numpy as np


class CosineSimilaritySearchingMixin:
    """Mixin class for cosine similarity search in Faiss
    """

    def initialize_searcher(self):
        matrix = copy.deepcopy(self.matrix)
        self._searcher = faiss.IndexFlatIP(matrix.shape[1])
        faiss.normalize_L2(matrix)
        self._searcher.add(matrix)
        return self

    def search(self, query_matrix: np.ndarray, top_k: int):
        faiss.normalize_L2(query_matrix)
        top_k_similarities, top_k_index = self._searcher.search(
            query_matrix, top_k)
        return dict(similarities=top_k_similarities, index=top_k_index)
