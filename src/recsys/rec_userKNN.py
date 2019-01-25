import numpy as np
import utils
from recsys.recsys import RecSys

class UserKNN(RecSys):
    
    def __init__(self, alpha=0.5, asym=True, knn=np.inf, h=0):
        super().__init__()
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.knn = knn

    def get_similarity(self, data):
        print("Computing UserKNN similarity...")
        s = utils.cosine_similarity(data.T, alpha=self.alpha, asym=self.asym, h=self.h, dtype=np.float32)
        s = utils.knn(s, self.knn)
        return s

    def get_scores(self, data, targets):
        s = self.get_similarity(data)
        print("Computing UserKNN scores...")
        scores = (data.T * s).tocsr()
        del s
        return scores.T[targets, :]
