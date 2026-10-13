import math, heapq, random
import numpy as np


class HNSW:
    def __init__(self, M=4, ef_construction=8, seed=42, names=None, trace=False):
        """
        Initialize the HNSW index.

        Parameters:
        - M: Maximum number of neighbors per node per layer (except layer 0, which can have up to 2*M).
        - ef_construction: Size of the dynamic list for the nearest neighbors during construction.
        - seed: Random seed for reproducibility.
        - names: Optional list of names for the nodes (for logging purposes).
        - trace: If True, enables detailed logging of the construction and search processes.    
        - entry: The entry point of the graph (the node with the highest level).
        - vecs: List of vectors corresponding to the nodes in the graph.
        - levels: List of maximum levels for each node.
        - graph: A list of dictionaries representing the graph structure. Each dictionary corresponds to a layer and maps node indices to their neighbors.
        - dist_calls: Counter for the number of distance calculations
        - eval_log: Optional list to log the evaluation of nodes during search (for debugging). 
        """
        self.M = M    # max neighbors per node per layer (except layer 0, which can have up to 2*M)
        self.Mmax0 = 2 * M
        self.efC = ef_construction
        self.mL = 1 / math.log(M)
        self.rng = random.Random(seed)
        self.vecs = []                 # node id -> vector
        self.levels = []               # node id -> max layer
        self.graph = []                # graph[layer][node] -> list of neighbors
        self.entry = None
        self.names = names
        self.trace = trace
        self.dist_calls = 0
        self.eval_log = None
        self.last_draw = None   # (U, raw) from the most recent random_level(), for the trace

    # ---------- helpers ----------
    def nm(self, i):
        return self.names[i] if self.names else str(i)

    def d(self, a, b):
        self.dist_calls += 1
        return float(np.linalg.norm(a - b))

    def log(self, *a):
        if self.trace:
            print(*a)

    def random_level(self):
        u = self.rng.random()
        raw = -math.log(1 - u) * self.mL
        self.last_draw = (u, raw)   # same RNG sequence as before; kept only for logging
        return int(raw)

    # ---------- core: search one layer ----------
    def search_layer(self, q, eps, ef, layer, tag=""):
        visited = set(eps)
        cand, best = [], []            # cand: min-heap, best: max-heap (neg dist)
        for e in eps:
            de = self.d(q, self.vecs[e])
            if self.eval_log is not None: self.eval_log.append((layer, e, de))
            heapq.heappush(cand, (de, e))
            heapq.heappush(best, (-de, e))
        while cand:
            dc, c = heapq.heappop(cand)
            if dc > -best[0][0] and len(best) >= ef:
                break
            for nb in self.graph[layer].get(c, []):
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self.d(q, self.vecs[nb])
                if self.eval_log is not None: self.eval_log.append((layer, nb, dn))
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, nb))
                    heapq.heappush(best, (-dn, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, n) for nd, n in best)   # [(dist, node)] ascending

    # ---------- neighbor selection heuristic ----------
    def select_neighbors(self, base_vec, cands, M, why=None):
        chosen = []
        for dist, c in sorted(cands):
            if len(chosen) >= M:
                break
            ok = True
            for _, s in chosen:
                if self.d(self.vecs[c], self.vecs[s]) < dist:
                    ok = False
                    if why is not None:
                        why.append(f"      reject {self.nm(c)} (d={dist:.2f}): closer to {self.nm(s)}")
                    break
            if ok:
                chosen.append((dist, c))
                if why is not None:
                    why.append(f"      keep   {self.nm(c)} (d={dist:.2f})")
        return chosen

    # ---------- insert ----------
    def add(self, vec):
        vec = np.asarray(vec, dtype=float)
        idx = len(self.vecs)
        L = self.random_level()   # level of the new node
        calls_at_start = self.dist_calls

        self.log(f"\n>>> INSERT {self.nm(idx)}  id={idx}  vec={vec.tolist()}")
        u, raw = self.last_draw
        self.log(f"    level draw: U={u:.5f}"
                 f"  ->  -ln(1-U)={-math.log(1 - u):.5f}"
                 f"  x  mL=1/ln({self.M})={self.mL:.5f}"
                 f"  =  {raw:.5f}  ->  floor  ->  L={L}")
        stack = "  ".join(f"L{x}" for x in range(L, -1, -1))
        note = ("layer 0 is the base layer - every node is in it" if L == 0
                else f"top layer {L}; one adjacency list per layer, all for the same vector")
        self.log(f"      lives on {L + 1} layer{'' if L == 0 else 's'}:  {stack}   ({note})")

        self.vecs.append(vec)
        self.levels.append(L)

        while len(self.graph) <= L: # what if the new node has a higher level than current max?
            # add new layer if needed
            self.log(f"    graph grew: empty layer {len(self.graph)} created"
                     f"  (index max layer now {len(self.graph)})")
            self.graph.append({})

        for l in range(L + 1): # initialize empty neighbor list for the new node at each layer
            self.log(f"    layer {l}: neighbor list initialized for {self.nm(idx)}"
                     f"  (empty, cap={self.Mmax0 if l == 0 else self.M})")
            self.graph[l][idx] = []

        if self.entry is None:
            self.entry = idx
            self.log(f"    index was empty -> {self.nm(idx)} becomes the entry point (no edges to build)")
            self.log(f"<<< DONE {self.nm(idx)}: {self.dist_calls - calls_at_start} distance calls")
            return idx

        top = self.levels[self.entry]
        ep = [self.entry]
        self.log(f"    entry point = {self.nm(self.entry)} (top layer {top});"
                 f" descend layers {top}..{L + 1}, then link layers {min(top, L)}..0")

        # greedy descent through layers above L
        for l in range(top, L, -1):
            before = self.dist_calls
            hit = self.search_layer(vec, ep, 1, l)[0]
            self.log(f"    layer {l}: greedy descent {self.nm(ep[0])} -> {self.nm(hit[1])}"
                     f"  (d={hit[0]:.3f}, {self.dist_calls - before} distance calls)")
            ep = [hit[1]]

        # connect on layers min(top, L) .. 0
        for l in range(min(top, L), -1, -1):
            layer_start = self.dist_calls
            Mmax = self.Mmax0 if l == 0 else self.M
            self.log(f"    layer {l}: LINKING  (Mmax={Mmax}, ef_construction={self.efC},"
                     f" entry={[self.nm(e) for e in ep]})")

            C = self.search_layer(vec, ep, self.efC, l)
            self.log(f"      candidates ({len(C)} found in {self.dist_calls - layer_start} calls): "
                     + (", ".join(f"{self.nm(n)}({dd:.3f})" for dd, n in C) or "(none)"))

            why = [] if self.trace else None
            chosen = self.select_neighbors(vec, C, self.M, why)
            if why:
                self.log(f"      heuristic (keep at most M={self.M}):")
                self.log("\n".join("  " + w for w in why))
            self.log(f"      chosen {len(chosen)}/{len(C)}: "
                     + (", ".join(f"{self.nm(n)}({dd:.3f})" for dd, n in chosen) or "(none)"))

            for dd, n in chosen:
                self.graph[l][idx].append(n)
                self.graph[l][n].append(idx)
                self.log(f"      edge {self.nm(idx)} <-> {self.nm(n)} (d={dd:.3f});"
                         f" {self.nm(n)} now deg={len(self.graph[l][n])}/{Mmax}")
                if len(self.graph[l][n]) > Mmax:       # prune overfull neighbor
                    nv = self.vecs[n]
                    before_deg = len(self.graph[l][n])
                    cl = [(self.d(nv, self.vecs[x]), x) for x in self.graph[l][n]]
                    pwhy = [] if self.trace else None
                    keep = self.select_neighbors(nv, cl, Mmax, pwhy)
                    dropped = set(self.graph[l][n]) - {x for _, x in keep}
                    self.graph[l][n] = [x for _, x in keep]
                    self.log(f"      prune {self.nm(n)}: deg {before_deg} > Mmax={Mmax}"
                             f" -> keep {[self.nm(x) for _, x in keep]},"
                             f" drop {[self.nm(x) for x in dropped]}")
                    if pwhy:
                        self.log("\n".join("  " + w for w in pwhy))
                    for x in dropped:                  # keep graph symmetric-ish
                        if n in self.graph[l][x]:
                            self.graph[l][x].remove(n)
                            self.log(f"        reverse edge {self.nm(x)} -> {self.nm(n)} removed"
                                     f" ({self.nm(x)} deg now {len(self.graph[l][x])})")

            self.log(f"    layer {l} done: {self.nm(idx)} -> {[self.nm(n) for n in self.graph[l][idx]]}"
                     f"  (deg={len(self.graph[l][idx])}/{Mmax},"
                     f" {self.dist_calls - layer_start} distance calls on this layer)")
            ep = [n for _, n in C]

        if L > top:
            self.entry = idx
            self.log(f"    L={L} > previous top={top} -> {self.nm(idx)} is the new entry point")
        else:
            self.log(f"    entry point unchanged: {self.nm(self.entry)} (top layer {top})")

        self.log(f"<<< DONE {self.nm(idx)}: {self.dist_calls - calls_at_start} distance calls"
                 f" for this insert, {self.dist_calls} total")
        return idx

    # ---------- query ----------
    def search(self, q, k=5, ef=50, trace=False):
        q = np.asarray(q, dtype=float)
        ep = [self.entry]
        for l in range(self.levels[self.entry], 0, -1):
            ep = [self.search_layer(q, ep, 1, l)[0][1]]
            if trace:
                print(f"  layer {l}: greedy -> {self.nm(ep[0])}")
        res = self.search_layer(q, ep, max(ef, k), 0)
        return res[:k]

    def show(self):
        for l in range(len(self.graph) - 1, -1, -1):
            print(f"Layer {l}  ({len(self.graph[l])} nodes)")
            for n in sorted(self.graph[l]):
                print(f"   {self.nm(n):>14} -> {[self.nm(x) for x in self.graph[l][n]]}")