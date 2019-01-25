import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

from recsys.recsys import RecSys

from implicit.als import AlternatingLeastSquares as als


class ALS(RecSys):

    def __init__(self, factors=10, iterations=10, reg=0.01):
        super().__init__()
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.model = als(factors=self.factors,
                         iterations=self.iterations,
                         regularization=self.reg,
                         use_gpu=False,
                         num_threads=mp.cpu_count())

    def get_scores(self, data, targets):

        self.model.fit(data.T)
        scores = np.empty((len(targets), data.shape[1]), dtype=np.float32)
        
        for i,target in enumerate(targets):
            r = self.model.recommend(userid=target, user_items=data, filter_already_liked_items=False, N=1000)

            items = []
            rates = []
            for item_id, rating in r:
                items.append(item_id)
                rates.append(rating)

            row = np.zeros((1, data.shape[1]), dtype=np.float32)
            row[0, items] = rates
            scores[i] = row

        return sp.csr_matrix(scores, dtype=np.float32)







