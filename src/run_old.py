from src.alg.hybrid import Hybrid
from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.alg.als import ALS
from src.alg.light import Light
from src.writer import create_submission
from src.data import Cache

from src.alg.hybrid_similarity import HybridSimilarity
from src.alg.item_knn import ItemKNN
from src.alg.slim import Slim
from src.alg.user_knn import UserKNN
from src.alg.als import ALS
from src.alg.light import Light
from src.writer import create_submission
from src.data import Cache
from src.alg.svd import SVD

cache = Cache()

a1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.5), (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.03), 0.5))            
a = Hybrid((a1, 0.85), (UserKNN(knn=190), 0.15))

'''
i = ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {}))
u = UserKNN(knn=200)
s = SVD(factors=200, knn=50)

h = HybridSimilarity((i, 0.3), (s, 0.7))
b = Hybrid((h, 0.3), (u, 0.2))

f = Hybrid((a, 0.9), (b, 0.1))
'''

#a.evaluate()
recs = a.run()
create_submission("ok",recs)

'''
h1 = HybridSimilarity((ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {})), 0.7), (Slim(lambda_i=0.025, lambda_j=0.025, all_dataset=False, epochs=3, lr=0.1), 0.3))            
m1 = Hybrid((h1, 0.85), (UserKNN(knn=190), 0.15)) 

i = ItemKNN(("artist_set", 0.1, {}), ("album_set", 0.2, {}))
u = UserKNN(knn=200)
s = SVD(factors=200, knn=50)

h = HybridSimilarity((i, 0.3), (s, 0.7))
h1 = Hybrid((h, 0.3), (u, 0.2))

f = Hybrid((m1, 0.9), (h1, 0.1))
f.evaluate()
'''

