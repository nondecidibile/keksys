'''
import time

import numpy as np
import scipy.sparse as sp
from numpy.linalg import linalg as LA

from src.alg.bpr import BPR
from src.alg.recsys import RecSys
from src.alg.utils import knn
'''
import numpy as np
import scipy.sparse as sp
from kek_recsys.kek_recsys import RecSys
from kek_recsys.kek_bpr import BPR
import kek_utils


class Slim(RecSys):
	def __init__(self, lr=0.01, epochs=1, lambda_i=0, lambda_j=0, knn=np.inf):
		super().__init__()

		self.lambda_i = lambda_i
		self.lambda_j = lambda_j

		self.lr = lr
		self.epochs = epochs

		self.knn = knn

		self.num_interactions = None
		self.bpr_sampler = None

	def get_similarity(self, dataset):
		urm = dataset
		urm = urm.tocsr()

		self.num_interactions = urm.nnz
		
		urm = sp.csr_matrix(urm)
		self.bpr_sampler = BPR(urm)

		slim_dim = urm.shape[1]

		s = np.zeros((slim_dim, slim_dim), dtype=np.float32)

		self.train(self.lr, self.epochs, urm, s)
		s = kek_utils.knn(s.T, knn=self.knn)

		return s

	def get_scores(self, dataset, targets):

		s = self.get_similarity(dataset)

		scores = (dataset[targets, :] * s).tocsr()
		del s

		return scores

	def build_batches(self):
		batches = []
		full_batches = self.num_interactions

		for _ in range(full_batches):
			batches.append(self.bpr_sampler.sample_batch(1))

		return batches


	def stochastic_gd(self, lr, num_epochs, urm, slim_matrix):

		for i in range(num_epochs):

			batches = self.build_batches()

			for batch in batches:
				u = batch[0][0]
				i = batch[0][1]
				j = batch[0][2]

				user_indices = urm.indices[urm.indptr[u]:urm.indptr[u + 1]]

				x_ui = slim_matrix[i, user_indices]
				x_uj = slim_matrix[j, user_indices]

				x_uij = np.sum(x_ui - x_uj)
				
				gradient = 1 / (1 + np.exp(x_uij))

				
				slim_matrix[i, user_indices] -= lr * (- gradient + (self.lambda_i * slim_matrix[i, user_indices]))
				slim_matrix[i, i] = 0

				slim_matrix[j, user_indices] -= lr * (gradient + (self.lambda_j * slim_matrix[j, user_indices]))
				slim_matrix[j, j] = 0

	def train(self, lr, num_epochs, urm, slim_matrix):
		self.stochastic_gd(lr, num_epochs, urm, slim_matrix)

