import numpy as np
import scipy.sparse as sp
import time
from sklearn.preprocessing import normalize

from recsys.recsys import RecSys


class Hybrid(RecSys):
    def __init__(self, model1, w1, model2, w2):
        super().__init__()

        self.model1 = model1
        self.w1 = w1
        self.model2 = model2
        self.w2 = w2

    def get_scores(self, data, targets):

        scores = sp.csr_matrix((len(targets), data.shape[1]), dtype=np.float32)

        model_scores = self.model1.get_scores(data, targets).tocsr()
        model_scores = normalize(model_scores, norm='l2', axis=1)
        model_scores = model_scores * self.w1
        scores += model_scores
        del model_scores

        model_scores = self.model2.get_scores(data, targets).tocsr()
        model_scores = normalize(model_scores, norm='l2', axis=1)
        model_scores = model_scores * self.w2
        scores += model_scores
        del model_scores

        return scores
