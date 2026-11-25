"""
Label-free test of whether edge features align with graph structure.

Structure and features are fitted separately: a structure-only model (NOCD)
assigns every edge to its most likely community, and the edge features are
clustered on their own. If the features carry information about the
communities, the two edge labellings agree more than chance. The test
statistic is their mutual information; the null distribution comes from
permuting the feature-cluster labels across edges, which keeps both marginal
distributions and breaks only the link between an edge and its features.

MarkBP uses the feature-led start only if the alignment is far beyond chance,
z > Z_THRESHOLD (chosen on the proof-of-concept graphs, where the weakest real
signal had z = 7.6 and the strongest null case z = 3.6); otherwise it ignores
the features and falls back to structure alone.
"""

import numpy as np

Z_THRESHOLD = 5.0

try:
    from .kmeans_cache import cached_kmeans
except ImportError:  # Fallback for direct script execution
    from kmeans_cache import cached_kmeans


def mutual_information(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> float:
    """Mutual information (nats) between two label arrays."""
    joint = np.zeros((ka, kb))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def alignment_test(structure_labels: np.ndarray, feats: np.ndarray, k: int, seed: int,
                   num_clusters=None, permutations: int = 200,
                   max_fit: int = 50000) -> dict:
    """
    structure_labels: community of each edge under a structure-only model.
    feats: edge features (rows of zeros = no features; those edges are skipped).
    Returns the observed mutual information, the permutation null and p-value.
    """
    has = np.linalg.norm(feats, axis=1) > 0
    s = structure_labels[has]
    x = feats[has]
    m = num_clusters or k
    rng = np.random.default_rng(seed)
    if len(x) > max_fit:
        # Fit the clusters on a sample, then give every edge its nearest centre
        sample = np.sort(rng.choice(len(x), max_fit, replace=False))
        centers, _ = cached_kmeans(x[sample], m, seed)
        g = _nearest(x, centers)
    else:
        _, g = cached_kmeans(x, m, seed)
    observed = mutual_information(s, g, k, m)
    null = np.array([mutual_information(s, rng.permutation(g), k, m)
                     for _ in range(permutations)])
    return {"mi": observed,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "z": float((observed - null.mean()) / max(null.std(), 1e-12)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + permutations)),
            "edges_tested": int(has.sum()),
            "use_features": bool((observed - null.mean()) / max(null.std(), 1e-12)
                                 > Z_THRESHOLD)}


def feature_clusters(feats: np.ndarray, k: int, seed: int, max_fit: int = 50000):
    """Cluster labels of the edges with features (others get -1)."""
    has = np.linalg.norm(feats, axis=1) > 0
    x = feats[has]
    rng = np.random.default_rng(seed)
    if len(x) > max_fit:
        sample = np.sort(rng.choice(len(x), max_fit, replace=False))
        centers, _ = cached_kmeans(x[sample], k, seed)
        g = _nearest(x, centers)
    else:
        _, g = cached_kmeans(x, k, seed)
    labels = np.full(len(feats), -1, dtype=np.int64)
    labels[has] = g
    return labels


def homophily_test(edge_index: np.ndarray, feats: np.ndarray, k: int, seed: int,
                   max_pairs: int = 200000, permutations: int = 200) -> dict:
    """
    Do edges that share a node fall into the same feature cluster more often
    than chance? Uses no fitted model: if the features reflect communities,
    a node's edges from the same community share a cluster; features unrelated
    to the graph (e.g. shuffled) do not. Null: cluster labels permuted across
    edges. Pairs are sampled uniformly among incident-edge pairs.
    """
    g = feature_clusters(feats, k, seed)
    rng = np.random.default_rng(seed + 1)
    e = np.arange(edge_index.shape[1])
    ends = np.concatenate([edge_index[0], edge_index[1]])
    order = np.argsort(ends, kind="stable")
    node, inc = ends[order], np.concatenate([e, e])[order]
    starts = np.r_[0, np.flatnonzero(np.diff(node)) + 1]
    deg = np.diff(np.r_[starts, len(node)])
    # sample pairs of distinct incident edges, nodes weighted by pair count
    pairs_per = deg * (deg - 1) / 2
    ok = pairs_per > 0
    nodes = rng.choice(np.flatnonzero(ok), max_pairs, p=pairs_per[ok] / pairs_per[ok].sum())
    i = rng.integers(0, deg[nodes])
    j = (i + rng.integers(1, deg[nodes])) % deg[nodes]
    a, b = inc[starts[nodes] + i], inc[starts[nodes] + j]
    keep = (g[a] >= 0) & (g[b] >= 0)
    a, b = a[keep], b[keep]
    observed = float(np.mean(g[a] == g[b]))
    valid = g >= 0
    null = []
    for _ in range(permutations):
        perm = g.copy()
        perm[valid] = rng.permutation(g[valid])
        null.append(np.mean(perm[a] == perm[b]))
    null = np.array(null)
    z = float((observed - null.mean()) / max(null.std(), 1e-12))
    return {"agreement": observed, "agreement_null": float(null.mean()),
            "homophily_z": z, "pairs_tested": int(len(a)),
            "features_consistent": bool(z > Z_THRESHOLD)}


def _nearest(x: np.ndarray, centers: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Index of the nearest centre for every row of x, in chunks."""
    sq = (centers ** 2).sum(axis=1)
    out = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), chunk):
        block = x[i:i + chunk]
        out[i:i + chunk] = np.argmin(sq[None] - 2 * block @ centers.T, axis=1)
    return out
