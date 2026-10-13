import numpy as np
from hnsw import HNSW
from dataset import DIMS, NAMES, X, vec

print("DATASET: 20 chunks x 10 dims  (dims =", DIMS, ")\n")
for i, (n, v) in enumerate(zip(NAMES, X)):
    top = np.argsort(-v)[:3]
    print(f"{i:2d} {n:<28} " + "  ".join(f"{DIMS[j]}={v[j]:.2f}" for j in top if v[j] > 0))

idx = HNSW(M=3, ef_construction=8, seed=1, names=NAMES, trace=True)
for v in X:
    idx.add(v)
    idx.show()

idx.trace = False
print("\n\n================ FINAL GRAPH ================")
idx.show()
print("\nentry point:", NAMES[idx.entry], "| levels:", {NAMES[i]: l for i, l in enumerate(idx.levels) if l > 0})

queries = {
 "speed up my slow database queries":       {"programming": .3, "databases": .9},
 "best pasta dish to try in Rome":          {"cooking": .7, "travel": .6},
 "use AI to detect fraudulent payments":    {"ml_ai": .7, "finance": .6},
}
K, EF = 3, 4
for text, qd in queries.items():
    q = vec(qd)
    print(f"\n\n=== QUERY: '{text}'  (k={K}, ef={EF}) ===")
    idx.eval_log = []
    res = idx.search(q, k=K, ef=EF, trace=False)
    log = idx.eval_log; idx.eval_log = None
    for layer in sorted({l for l, _, _ in log}, reverse=True):
        seen = []
        for l, n, dd in log:
            if l == layer and n not in [s[0] for s in seen]:
                seen.append((n, dd))
        print(f"  layer {layer}: evaluated " + ", ".join(f"{NAMES[n]}({dd:.2f})" for n, dd in seen))
    print("  HNSW top-3 :", [(NAMES[n], round(dd, 2)) for dd, n in res])
    ex = np.argsort(np.linalg.norm(X - q, axis=1))[:K]
    print("  exact top-3:", [(NAMES[n], round(float(np.linalg.norm(X[n]-q)), 2)) for n in ex])
    uniq = len({n for _, n, _ in log})
    print(f"  unique nodes evaluated: {uniq} of {len(X)}   same result as brute force: {set(n for _, n in res) == set(ex.tolist())}")