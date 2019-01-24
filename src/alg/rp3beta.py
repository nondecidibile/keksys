import numpy as np
import time
from sklearn.preprocessing import normalize

import scipy.sparse as sp
from src.alg.recsys import RecSys


class RP3Beta(RecSys):

    def __init__(self, alpha=1, beta=0.6, knn=np.inf, normalize=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.knn = knn
        self.normalize = normalize

    def compute_similarity(self, dataset):

        urm = dataset
        # Pui is the row-normalized urm
        Pui = normalize(urm, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = urm.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(urm.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.knn]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        s = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize:
            s = normalize(s, norm='l1', axis=1)

        return s

    def rate(self, dataset, targets):
        s = self.compute_similarity(dataset)

        print("Computing rp3beta ratings...")
        start = time.time()
        ratings = (dataset[targets, :] * s).tocsr()
        print("elapsed: {:.3f}s\n".format(time.time() - start))
        del s
        return ratings