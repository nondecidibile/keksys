import numpy as np
import scipy.sparse as sparse
import utils
from recsys.recsys import RecSys


class ItemKNN(RecSys):
   
    def __init__(self, tracks_info, artist_w=0.075, album_w=0.075, alpha=0.5, asym=True, knn=np.inf, h=0):
        super().__init__()

        self.artist_w = artist_w
        self.album_w = album_w
        self.alpha = np.float32(alpha)
        self.asym = asym
        self.h = np.float32(h)
        self.knn = knn

        artists = tracks_info[:, 2]
        albums = tracks_info[:, 1]

        NUM_ARTISTS = max(artists)
        NUM_ALBUMS = max(albums)
        NUM_TRACKS = len(tracks_info)

        self.artists_mat = sparse.dok_matrix((NUM_ARTISTS, NUM_TRACKS), dtype=np.uint8)
        self.albums_mat = sparse.dok_matrix((NUM_ALBUMS, NUM_TRACKS), dtype=np.uint8)

        for i,row in enumerate(tracks_info):

            track = i
            artist = row[2]-1
            album = row[1]-1

            self.artists_mat[artist, track] = 1
            self.albums_mat[album, track] = 1

    def get_similarity(self, data):
        print("Computing ItemKNN similarity...")
        similarity = utils.cosine_similarity(data, alpha=self.alpha, asym=self.asym, h=self.h, dtype=np.float32)

        # ARTIST
        similarity += utils.cosine_similarity(self.artists_mat,alpha=0.5,asym=True,h=0,dtype=np.float32) * self.artist_w
        # ALBUM
        similarity += utils.cosine_similarity(self.albums_mat,alpha=0.5,asym=True,h=0,dtype=np.float32) * self.album_w

        similarity = utils.knn(similarity, self.knn)
        return similarity

    def get_scores(self, data, targets):
        similarity = self.get_similarity(data)
        print("Computing ItemKNN scores...")
        scores = (data[targets, :] * similarity).tocsr()
        del similarity
        return scores
