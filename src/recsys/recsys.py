import numpy as np

class RecSys:

	def run(self, data, targets):
		
		# Compute scores
		print("Computing scores...")
		scores = self.get_scores(data, targets)

		# Compute predictions
		print("Computing predictions...")
		preds = self.get_predictions(scores, targets, data[targets, :])
		del scores

		return preds
	
	def get_predictions(self, scores, targets, mask):

		scores = scores.tocsr()
		mask = mask.tocsr()

		predictions = {}

		for playlist_index, playlist in enumerate(targets):

			scores_i_data = scores.data[scores.indptr[playlist_index]:scores.indptr[playlist_index+1]]
			scores_i_indices = scores.indices[scores.indptr[playlist_index]:scores.indptr[playlist_index+1]]

			mask_i_indices = mask.indices[mask.indptr[playlist_index]:mask.indptr[playlist_index + 1]]
			masked_i = np.in1d(scores_i_indices, mask_i_indices, assume_unique=True, invert=True)

			scores_i_data = scores_i_data[masked_i]
			scores_i_indices = scores_i_indices[masked_i]

			if len(scores_i_indices) > 10:
				best_indices = np.argpartition(scores_i_data, -10)[-10:]
				tracks_i = scores_i_indices[best_indices]
				sorted_best_indices = np.argsort(-scores_i_data[best_indices])
			else:
				#top_pop = np.array([2272, 18266, 13980, 2674, 17239, 10496, 15578, 5606, 10848, 8956])
				num_missing = 10 - len(scores_i_indices)
				missing_tracks_i = np.zeros(num_missing, dtype=np.float32)
				missing_scores_i = np.zeros(num_missing, dtype=np.float32)
				tracks_i = np.append(scores_i_indices, missing_tracks_i)
				scores_i = np.append(scores_i_data, missing_scores_i)
				sorted_best_indices = np.argsort(-scores_i)

			predictions[playlist] = list(np.resize(tracks_i[sorted_best_indices], 10))

		return predictions

	def get_similarity(self, data):
		print("get_similarity() not implemented on abstract class RecSys")
		
	def get_scores(self, data, targets):
		print("get_scores() not implemented on abstract class RecSys")
	
