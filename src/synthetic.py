"""
Synthetic Edge-Featured Graphs with Overlapping Communities

Nodes join 1-3 of K communities. Each community creates edges among its members
with expected internal degree `degree`; a fraction `noise` of all edges are
random background edges. Every edge records the community that created it
(-1 for background) and carries a feature vector

    x_uv = normalise(snr * mu_c + eps),  eps ~ N(0, I / dim)

where mu_c is a random unit vector for community c, so `snr` sets how much an
edge's features reveal its community (snr = 0: pure noise). Background edges
get pure noise.
"""

from typing import Dict

import numpy as np


def generate(num_nodes: int = 2000, num_communities: int = 10, degree: float = 8.0,
             noise: float = 0.3, snr: float = 1.0, dim: int = 32,
             membership_probs=(0.6, 0.3, 0.1), seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    k = num_communities

    # Overlapping memberships: 1, 2 or 3 communities per node
    counts = rng.choice(len(membership_probs), size=num_nodes, p=membership_probs) + 1
    members = [[] for _ in range(k)]
    for node, m in enumerate(counts):
        for c in rng.choice(k, size=m, replace=False):
            members[c].append(node)
    communities = [np.array(sorted(m), dtype=np.int64) for m in members]

    edges = {}
    for c, nodes in enumerate(communities):
        n = len(nodes)
        if n < 2:
            continue
        num_edges = int(round(degree * n / 2))
        a = rng.choice(nodes, num_edges)
        b = rng.choice(nodes, num_edges)
        for u, v in zip(a, b):
            if u != v:
                edges.setdefault((min(u, v), max(u, v)), c)

    num_noise = int(round(noise / (1 - noise) * len(edges)))
    added = 0
    while added < num_noise:
        u, v = rng.integers(0, num_nodes, 2)
        key = (min(u, v), max(u, v))
        if u != v and key not in edges:
            edges[key] = -1
            added += 1

    keys = sorted(edges)
    edge_index = np.array(keys, dtype=np.int64).T
    labels = np.array([edges[key] for key in keys], dtype=np.int64)

    mu = rng.normal(size=(k, dim))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    feats = rng.normal(scale=1.0 / np.sqrt(dim), size=(len(keys), dim))
    signal = labels >= 0
    feats[signal] += snr * mu[labels[signal]]
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    # Keep only nodes with at least one edge
    used = np.unique(edge_index)
    remap = np.full(num_nodes, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    communities = [remap[c][remap[c] >= 0] for c in communities]

    return {
        "num_nodes": int(len(used)),
        "edge_index": remap[edge_index],
        # Indices match edge_labels, so no community is dropped
        "communities": communities,
        "edge_labels": labels,
        "edge_features": feats.astype(np.float32),
    }
