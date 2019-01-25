import numpy as np

class RecSys:

	def run(self, data, targets):
		
		data = data.tocsr()

		# Compute ratings
		print("Computing ratings...")
		scores = self.get_scores(data, targets)

		# Compute predictions
		print("Computing predictions...")
		preds = self.get_predictions(scores, targets=targets)
		del scores

		return preds
	

	def get_predictions(self, scores, targets):

		predictions = []
	
		for i,target in enumerate(targets):
			scores_i = scores[i,:].A.ravel()

			top_idxs = np.argpartition(scores_i, -10)[-10:]
			sorted_idxs = np.argsort(-scores_i[top_idxs])
			pred = top_idxs[sorted_idxs]
			predictions.append(pred)
		
		return predictions


	def get_similarity(self, data):
		raise NotImplementedError        
		
	def get_scores(self, data, targets):
		raise NotImplementedError
	
