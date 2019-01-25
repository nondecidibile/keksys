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

def save_recommendations(target_indices, recommendations_list):
    out_recommendation_list = []
    for rec in recommendations_list:
        str_rec = [str(int(i)) for i in rec]
        str_rec = ' '.join(str_rec)
        out_recommendation_list.append(str_rec)
    data_out = {'playlist_id': target_indices, 'track_ids': out_recommendation_list}
    df_out = pd.DataFrame(data=data_out)
    df_out.to_csv("../submissions/out.csv", index=False)


def knn(s, knn=np.inf):
    if type(s) is not sparse.csr_matrix:
        s = sparse.csr_matrix(s)
    if knn != np.inf:
        for row in range(len(s.indptr) - 1):
            row_start = s.indptr[row]
            row_end = s.indptr[row + 1]

            row_data = s.data[row_start:row_end]

            if len(row_data) > knn:
                discard = np.argpartition(row_data, -knn)[:-knn] + row_start
                s.data[discard] = 0

        if not isinstance(s, sparse.csr_matrix):
            s = s.tocsr()

        data_i = s.data.nonzero()[0]
        data = s.data[data_i]

        indices = s.indices[data_i]

        indptr = [0]
        for row_i in range(s.shape[0]):
            row_start = s.indptr[row_i]
            row_end = s.indptr[row_i + 1]

            num_nonzero = np.count_nonzero(s.data[row_start:row_end])
            indptr.append(indptr[row_i] + num_nonzero)

        return sparse.csr_matrix((data,indices,indptr), shape=s.shape)
    return s

def cosine_similarity(input, alpha=0.5, asym=True, h=0., dtype=np.float32):

    norms = input.sum(axis=0).A.ravel().astype(dtype)

    if (not asym or alpha == 0.5) and h == 0:
        norm_input = input * sparse.diags(np.divide(
            1,
            np.power(norms, alpha, out=norms, dtype=dtype),
            out=norms,
            where=norms != 0,
            dtype=dtype
        ), format="csr", dtype=dtype)

        s = (norm_input.T * norm_input)

    else:
        s = (input.T * input).tocsr()

        if asym:
            assert 0. <= alpha <= 1., "alpha should be a number between 0 and 1"
            norm_factors = np.outer(
                np.power(norms, alpha, dtype=dtype),
                np.power(norms, 1 - alpha, dtype=dtype)
            ) + h

        else:
            norms = np.power(norms, alpha, dtype=dtype)
            norm_factors = np.outer(norms, norms) + h

        # Calculate inverse and normalize
        norm_factors = np.divide(1, norm_factors, out=norm_factors, where=norm_factors != 0, dtype=dtype)
        s = s.multiply(norm_factors).tocsr()
        del norms
        del norm_factors

    # Return computed similarity matrix
    return s


def compute_map(recommendations_list, test_csr_matrix, target_indices):
	map_metric = 0
	n = 0
	i = 0
	for index in target_indices:
		test_playlist = test_csr_matrix.getrow(index)
		m = len(test_playlist.data)
		ap = 0
		if m > 0:
			n += 1
			recommendations = np.array(recommendations_list[i])

			k = 0
			tot = 0
			for j in recommendations:
				k += 1
				if j in test_playlist.indices:
					tot += 1
					ap += tot / k
			map_metric += ap / m
		i += 1
	map_metric = map_metric / n
	print("map = " + str(round(map_metric, 4)))