import numpy as np
import scipy.sparse as sp
import time
from sklearn.preprocessing import normalize

from kek_recsys.kek_recsys import RecSys


class Hybrid(RecSys):
    def __init__(self, *models, normalize=True):
        super().__init__()

        self.models = list(models)
        self.normalize = normalize

    def get_scores(self, dataset, targets):

        if not self.models:
            raise RuntimeError("You already called rate")

        scores = sp.csr_matrix((len(targets), dataset.shape[1]), dtype=np.float32)

        while self.models:

            model, w = self.models.pop()
            model_ratings = model.get_scores(dataset, targets).tocsr()

            if self.normalize:
                model_ratings = normalize(model_ratings, norm='l2', axis=1)
 

            model_ratings = model_ratings * w
            scores += model_ratings
            del model_ratings

        return scores
