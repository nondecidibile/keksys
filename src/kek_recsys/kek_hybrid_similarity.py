import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

import kek_utils
from kek_recsys.kek_recsys import RecSys


class HybridSimilarity(RecSys):

    def __init__(self, *models, knn=np.inf):
        super().__init__()
        self.models = list(models)
        self.knn = knn

    def get_similarity(self, dataset):

        if not self.models:
            raise RuntimeError("You already called rate")

        s = sparse.csr_matrix((dataset.shape[1], dataset.shape[1]), dtype=np.float32)
       
        while self.models:
            model, w = self.models.pop()
            model_similarity = model.get_similarity(dataset)
            model_similarity = model_similarity * w

            s += model_similarity
            del model_similarity
        
        s = normalize(s, norm='l2', axis=1)
        s = kek_utils.knn(s, self.knn)
        return s

    def get_scores(self, dataset, targets):
        s = self.get_similarity(dataset)
        scores = (dataset[targets, :] * s).tocsr()
        del s
        return scores
        