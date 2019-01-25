import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

import utils
from recsys.recsys import RecSys


class HybridSimilarity(RecSys):

    def __init__(self, model1, w1, model2, w2):
        super().__init__()
        
        self.model1 = model1
        self.w1 = w1
        self.model2 = model2
        self.w2 = w2

    def get_similarity(self, data):

        s = sparse.csr_matrix((data.shape[1], data.shape[1]), dtype=np.float32)

        model_sim = self.model1.get_similarity(data)
        model_sim = model_sim * self.w1
        s += model_sim
        del model_sim

        model_sim = self.model2.get_similarity(data)
        model_sim = model_sim * self.w2
        s += model_sim
        del model_sim
        
        s = normalize(s, norm='l2', axis=1)
        s = utils.knn(s, np.inf)
        return s

    def get_scores(self, data, targets):
        s = self.get_similarity(data)
        scores = (data[targets, :] * s).tocsr()
        del s
        return scores
        