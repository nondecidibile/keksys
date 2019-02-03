import pandas as pd
import numpy as np
from scipy import sparse
import imp
import sklearn.preprocessing as pp


def load_targets(num_playlists):
    df_target_playlists = pd.read_csv('../input/target_playlists.csv')
    targets = np.array(list(filter(lambda x: x < num_playlists, np.array(df_target_playlists))))
    sorted_target_playlists = np.sort(targets.ravel())
    return sorted_target_playlists


def load_interactions(num_playlists, num_tracks, target_playlists, test_ratio=0):
    df_train = pd.read_csv('../input/train.csv')

    sparse_row_ind = []
    sparse_col_ind = []
    sparse_data = []

    for playlist_i in df_train.values:
        if np.int(playlist_i[0]) < num_playlists and np.int(playlist_i[1]) < num_tracks:
            sparse_row_ind.append(playlist_i[0])
            sparse_col_ind.append(playlist_i[1])
            sparse_data.append(1.0)

    row_ind = np.asarray(sparse_row_ind).astype(np.int)
    col_ind = np.asarray(sparse_col_ind).astype(np.int)
    data = np.asarray(sparse_data).astype(np.float)

    num_interactions = len(data)

    train_row = []
    train_col = []
    train_data = []
    test_row = []
    test_col = []
    test_data = []

    for i in range(len(row_ind)):
        row = row_ind[i]
        col = col_ind[i]

        if row in target_playlists and np.random.binomial(1, test_ratio):
            test_row.append(row)
            test_col.append(col)
            test_data.append(1)
        else:
            train_row.append(row)
            train_col.append(col)
            train_data.append(1)

    train = sparse.csr_matrix((train_data, (train_row, train_col)), shape=(num_playlists, num_tracks), dtype=np.uint8)
    test = sparse.csr_matrix((test_data, (test_row, test_col)), shape=(num_playlists, num_tracks), dtype=np.uint8)

    return [train, test, num_interactions]


def load_tracks_info(num_tracks):
    df_tracks_info = pd.read_csv('../input/tracks.csv')
    info = np.array(df_tracks_info)[0:np.int32(num_tracks)]
    #info = np.array(df_tracks_info)[:, 0:num_tracks]
    return info

def save_recommendations(predictions_list):
    with open('../submissions/ok.csv', 'w') as f:
        f.write("playlist_id,track_ids\n")
        for playlist_index, track_list in predictions_list.items():
            f.write(str(playlist_index) + ",")
            f.write(" ".join([str(el) for el in track_list]))
            f.write("\n")

def knn(s, knn=np.inf):
    
    s = sparse.csr_matrix(s)

    if knn != np.inf:
        for r in range(len(s.indptr) - 1):
            start = s.indptr[r]
            end = s.indptr[r + 1]

            row_data = s.data[start:end]

            if len(row_data) > knn:
                discard = np.argpartition(row_data, -knn)[:-knn] + start
                s.data[discard] = 0

        s = s.tocsr()

        d = s.data.nonzero()[0]
        data = s.data[d]

        indices = s.indices[d]

        indptr = [0]
        for i in range(s.shape[0]):
            start = s.indptr[i]
            end = s.indptr[i + 1]

            num_nonzero = np.count_nonzero(s.data[start:end])
            indptr.append(indptr[i] + num_nonzero)

        return sparse.csr_matrix((data,indices,indptr), shape=s.shape)
    return s

def cosine_similarity(data, alpha=0.5, asym=True, h=0, dtype=np.float32):

    ns = data.sum(axis=0).A.ravel().astype(dtype)

    if (not asym or alpha == 0.5) and h == 0:
        n_data = data * sparse.diags(np.divide(
            1, np.power(ns, alpha, out=ns, dtype=dtype),
            out=ns, where=ns != 0, dtype=dtype), format="csr", dtype=dtype)

        sim = (n_data.T * n_data)

    else:
        sim = (data.T * data).tocsr()

        if asym:
            n_factors = np.outer(
                np.power(ns, alpha, dtype=dtype),
                np.power(ns, 1 - alpha, dtype=dtype)
            ) + h

        else:
            ns = np.power(ns, alpha, dtype=dtype)
            n_factors = np.outer(ns, ns) + h

        n_factors = np.divide(1, n_factors, out=n_factors, where=n_factors != 0, dtype=dtype)
        sim = sim.multiply(n_factors).tocsr()
        del ns
        del n_factors

    return sim


def compute_map(recommendations_list, test_csr_matrix, target_indices):
	map_metric = 0
	n = 0
	for index in target_indices:
		test_playlist = test_csr_matrix.getrow(index)
		m = len(test_playlist.data)
		ap = 0
		if m > 0:
			n += 1
			recommendations = np.array(recommendations_list[index])

			k = 0
			tot = 0
			for j in recommendations:
				k += 1
				if j in test_playlist.indices:
					tot += 1
					ap += tot / k
			map_metric += ap / m
	map_metric = map_metric / n
	print("map = " + str(round(map_metric, 4)))