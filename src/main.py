import utils
from recsys.rec_slim import Slim
from recsys.rec_warp import Warp
from recsys.rec_itemKNN import ItemKNN
from recsys.rec_userKNN import UserKNN
from recsys.rec_als import ALS
from recsys.rec_hybrid_similarity import HybridSimilarity
from recsys.rec_hybrid import Hybrid

#
# Global variables
#

NUM_PLAYLISTS = 50446  # 50446
NUM_TRACKS = 20635  # 20635
TEST = False
TEST_RATIO = 0.2 if TEST else 0

#
# Load data
#

targets = utils.load_targets(num_playlists=NUM_PLAYLISTS)
train_data, test_data, _ = utils.load_interactions(num_playlists=NUM_PLAYLISTS, num_tracks=NUM_TRACKS,
													target_playlists=targets, test_ratio=TEST_RATIO)
tracks_info = utils.load_tracks_info(NUM_TRACKS)

#
# Build recommender system
#

item = ItemKNN(tracks_info,0.075,0.075)
slim = Slim(lambda_i=0.001, lambda_j=0.0001, epochs=3, lr=0.1)
h1 = HybridSimilarity(item, 0.7, slim, 0.3)

user = UserKNN(knn=64)
h2 = Hybrid(h1, 0.85, user, 0.15)

w = Warp(NUM_TRACKS=NUM_TRACKS, no_components=300, epochs=50)
a = ALS(factors=1024, iterations=5)
h3 = Hybrid(w, 0.7, a, 0.3)

recsys = Hybrid(h2, 0.85, h3, 0.15)

#
# Run recommender system
#

recs = recsys.run(train_data,targets)

#
# Compute MAP or export recommendations
#

if TEST:
	utils.compute_map(recs,test_data,targets)
else:
	utils.save_recommendations(recs)