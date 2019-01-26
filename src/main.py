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
TEST = True
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

slim = Slim(lambda_i=0.025, lambda_j=0.025, epochs=3, lr=0.1)
item = ItemKNN(tracks_info,0.075,0.075)
hybrid_1 = HybridSimilarity(item, 0.65, slim, 0.35)

user = UserKNN(knn=100)
hybrid_2 = Hybrid(hybrid_1, 0.8, user, 0.2)

warp = Warp(NUM_TRACKS=NUM_TRACKS, no_components=300, epochs=30)
als = ALS(factors=1024, iterations=2)
hybrid_3 = Hybrid(warp, 0.5, als, 0.5)

recsys = Hybrid(hybrid_2, 0.85, hybrid_3, 0.15)

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