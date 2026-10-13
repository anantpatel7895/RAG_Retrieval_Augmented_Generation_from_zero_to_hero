from hnsw import HNSW
import numpy as np

data = [
 ("Python tutorial",   (1, 1)),
 ("Java tutorial",     (2, 1)),
 ("SQL basics",        (1, 3)),
 ("ML intro",          (3, 3)),
 ("Deep learning",     (4, 4)),
 ("Neural nets",       (3, 5)),
 ("Pasta recipe",      (8, 8)),
 ("Pizza recipe",      (9, 7)),
 ("Soup recipe",       (7, 9)),
 ("Cake recipe",       (9, 9)),
 ("Bread baking",      (8, 6)),
 ("Salad recipe",      (6, 8)),
]
names = [n for n, _ in data]
idx = HNSW(M=3, ef_construction=6, seed=14, names=names, trace=True)
for n, v in data:
    idx.add(v)

print("\n\n================ FINAL GRAPH ================")
idx.trace = False
idx.show()

print("\n\n================ QUERY: 'pizza dough' q=(8.8, 7.2), k=3 ================")
idx.dist_calls = 0
res = idx.search((8.8, 7.2), k=3, ef=4, trace=True)
for dd, n in res:
    print(f"  {names[n]:<14} dist={dd:.2f}")
print(f"  distance computations: {idx.dist_calls} (brute force would be {len(data)})")