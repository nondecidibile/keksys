from src.alg.als import ALS
from src.alg.hybrid import Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.data import Cache
from src.alg.light import Light
from src.alg.svd import SVD
from src.writer import create_submission

cache = Cache()

slim = Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=30, loss='warp')
a = ALS(factors=1024, iterations=2)

h1 = HybridSimilarity((ItemKNN(("artist_set", 0.075, {}), ("album_set", 0.075, {})), 0.65),
                      (slim, 0.35))

forzajuve = Hybrid((h1, 0.8), (UserKNN(knn=100), 0.2))

f = Hybrid((forzajuve, 0.85), (Hybrid((l, 0.5), (a, 0.5)), 0.15))

preds = f.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("ok",preds)
