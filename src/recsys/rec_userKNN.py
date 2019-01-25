import numpy as np
import kek_utils
from kek_recsys.kek_recsys import RecSys

class UserKNN(RecSys):
    
    def __init__(self, alpha=0.5, asym=True, knn=np.inf, h=0):
        super().__init__()
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.knn = knn

    def get_similarity(self, dataset=None):
        print("Computing UserKNN similarity...")
        s = kek_utils.cosine_similarity(dataset.T, alpha=self.alpha, asym=self.asym, h=self.h, dtype=np.float32)
        s = kek_utils.knn(s, self.knn)
        return s

    def get_scores(self, dataset, targets):
        s = self.get_similarity(dataset)
        print("Computing UserKNN scores...")
        scores = (dataset.T * s).tocsr()
        del s
        return scores.T[targets, :]
