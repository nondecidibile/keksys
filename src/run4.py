from src.alg import Hybrid, UserKNN, ItemKNN
from src.alg.als import ALS
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.light import Light
from src.alg.slim import Slim
from src.data import Cache
from src.writer import create_submission

cache = Cache()

'''
# 0.09426
slim = Slim(lambda_i=0.001, lambda_j=0.001, all_dataset=False, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=50, loss='warp')
a = ALS(factors=1024, iterations=2)
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7), (slim, 0.3))
forzajuve = Hybrid((h1, 0.9), (UserKNN(knn=64), 0.1))
f = Hybrid((forzajuve, 0.85), (Hybrid((l, 0.7), (a, 0.3)), 0.15))
f.evaluate()
'''

'''
slim = Slim(lambda_i=0.002, lambda_j=0.0002, all_dataset=False, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=50, loss='warp')
a = ALS(factors=1024, iterations=2)
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.075, {}), ("album_set", 0.15, {})), 0.7), (slim, 0.3))
forzajuve = Hybrid((h1, 0.9), (UserKNN(knn=64), 0.1))
f = Hybrid((forzajuve, 0.875), (Hybrid((l, 0.675), (a, 0.325)), 0.125))
'''

slim = Slim(lambda_i=0.0022, lambda_j=0.00018, all_dataset=False, epochs=3, lr=0.1)
l = Light(no_components=300, epochs=50, loss='warp')
a = ALS(factors=1024, iterations=3)
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.12, {}), ("album_set", 0.18, {})), 0.65), (slim, 0.35))
forzajuve = Hybrid((h1, 0.9), (UserKNN(knn=64), 0.1))
f = Hybrid((forzajuve, 0.825), (Hybrid((l, 0.71), (a, 0.29)), 0.175))

preds = f.run(dataset=cache.fetch("interactions"), targets=cache.fetch("targets"))
create_submission("ok2",preds)
