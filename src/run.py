import time
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

NUM_PLAYLISTS = 10000  # 50446
NUM_TRACKS = 1000  # 20635
TEST = True
TEST_RATIO = 0.2 if TEST else 0

#
# Load data
#

targets = utils.load_targets(num_playlists=NUM_PLAYLISTS)
train_data, test_data, _ = utils.load_interactions(num_playlists=NUM_PLAYLISTS, num_tracks=NUM_TRACKS,
									target_playlists=targets, test_ratio=TEST_RATIO)

warp = Warp(NUM_TRACKS=NUM_TRACKS, no_components=300, epochs=30)
als = ALS(factors=1024, iterations=2)
hybrid_3 = Hybrid(warp, 0.5, als, 0.5)

recs = hybrid_3.run(train_data,targets)
utils.compute_map(recs,test_data,targets)
#utils.save_recommendations(recs)
