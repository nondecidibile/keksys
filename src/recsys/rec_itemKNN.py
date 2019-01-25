import numpy as np
import scipy.sparse as sparse
import kek_utils
from kek_recsys.kek_recsys import RecSys


class ItemKNN(RecSys):
   
    def __init__(self, tracks_info, artist_w=0.075, album_w=0.075, alpha=0.5, asym=True, knn=np.inf, h=0):
        super().__init__()

        self.tracks_info = tracks_info
        self.artist_w = artist_w
        self.album_w = album_w
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.knn = knn

    def get_similarity(self, data=None):
        print("Computing ItemKNN similarity...")
        similarity = kek_utils.cosine_similarity(data, alpha=self.alpha, asym=self.asym, h=self.h, dtype=np.float32)

        # ARTIST
        artists = sparse.csr_matrix(self.tracks_info[:, 2])
        similarity += kek_utils.cosine_similarity(artists,alpha=0.5,asym=True,h=0,dtype=np.float32) * self.artist_w
        
        # ALBUM
        albums = sparse.csr_matrix(self.tracks_info[:, 1])
        similarity += kek_utils.cosine_similarity(albums,alpha=0.5,asym=True,h=0,dtype=np.float32) * self.album_w

        similarity = kek_utils.knn(similarity, self.knn)
        return similarity

    def get_scores(self, dataset, targets):
        similarity = self.get_similarity(dataset)
        print("Computing ItemKNN scores...")
        scores = (dataset[targets, :] * similarity).tocsr()
        del similarity
        return scores
