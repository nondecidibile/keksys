import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

from recsys.recsys import RecSys
from lightfm import LightFM


class Warp(RecSys):
    def __init__(self, NUM_TRACKS, no_components=10, learning_schedule='adagrad', learning_rate=0.05, epochs=1, num_threads=mp.cpu_count()):

        super().__init__()
        self.NUM_TRACKS = NUM_TRACKS
        self.no_components = no_components
        self.learning_schedule = learning_schedule
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_threads = num_threads

        self.model = LightFM(no_components=self.no_components,
                             learning_schedule=self.learning_schedule,
                             loss='warp',
                             learning_rate=self.learning_rate)

    
    def get_scores(self, dataset, targets):
        self.model.fit(interactions=dataset,
                       epochs=self.epochs,
                       num_threads=self.num_threads,
                       verbose=True)

        scores = np.empty((len(targets), dataset.shape[1]), dtype=np.float32)
        tracks = [i for i in range(self.NUM_TRACKS)]
        for i, target in enumerate(targets):
            new_row = self.model.predict(target, tracks)
            discard = np.argpartition(new_row, -1000)[:-1000]
            new_row[discard] = 0
            scores[i] = new_row

        return sp.csr_matrix(scores, dtype=np.float32)