"""
Link Communities Baseline

Ahn, Bagrow & Lehmann (Nature, 2010): cluster edges instead of nodes. Two edges
sharing a node k, e_ik and e_jk, have similarity
    S = |N+(i) & N+(j)| / |N+(i) | N+(j)|,  N+(x) = neighbours of x and x itself.
Edges are merged by single linkage in order of decreasing similarity, and the
dendrogram is cut where the partition density
    D = 2/M * sum_c m_c (m_c - (n_c - 1)) / ((n_c - 2)(n_c - 1))
is largest (m_c edges and n_c nodes in link community c; M edges in total).
Each node belongs to every link community containing one of its edges, which
gives overlapping node communities. Structure only.
"""

import numpy as np
import scipy.sparse as sp


def _edge_pairs(num_nodes: int, edge_index: np.ndarray):
    """All pairs of edges sharing a node, with their similarity."""
    u, v = edge_index
    m = len(u)
    a = sp.csr_matrix((np.ones(2 * m), (np.r_[u, v], np.r_[v, u])), shape=(num_nodes,) * 2)
    a.data[:] = 1.0
    a_plus = (a + sp.identity(num_nodes, format="csr")).tocsr()
    a_plus.data[:] = 1.0
    size = np.asarray(a_plus.sum(axis=1)).ravel()
    common = (a_plus @ a_plus).tocsr()

    # incident edges of each node
    ends = np.r_[u, v]
    other = np.r_[v, u]
    eid = np.r_[np.arange(m), np.arange(m)]
    order = np.argsort(ends, kind="stable")
    ends, other, eid = ends[order], other[order], eid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ends)) + 1, len(ends)]
    e1, e2, i_nodes, j_nodes = [], [], [], []
    for s, t in zip(starts[:-1], starts[1:]):
        k = t - s
        if k < 2:
            continue
        ii, jj = np.triu_indices(k, 1)
        e1.append(eid[s + ii]); e2.append(eid[s + jj])
        i_nodes.append(other[s + ii]); j_nodes.append(other[s + jj])
    if not e1:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    e1, e2 = np.concatenate(e1), np.concatenate(e2)
    i_nodes, j_nodes = np.concatenate(i_nodes), np.concatenate(j_nodes)
    inter = np.asarray(common[i_nodes, j_nodes]).ravel()
    sim = inter / (size[i_nodes] + size[j_nodes] - inter)
    return e1, e2, sim


def link_communities(num_nodes: int, edge_index: np.ndarray, min_size: int = 3):
    """Return (node communities as arrays, edge labels) at maximum partition density."""
    u, v = edge_index
    m = len(u)
    e1, e2, sim = _edge_pairs(num_nodes, edge_index)
    order = np.argsort(-sim, kind="stable")
    e1, e2, sim = e1[order], e2[order], sim[order]

    parent = np.arange(m)
    nodes = [{int(u[e]), int(v[e])} for e in range(m)]
    n_edges = np.ones(m, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def density_term(mc, nc):
        return mc * (mc - (nc - 1)) / ((nc - 2) * (nc - 1)) if nc > 2 else 0.0

    total = 0.0  # sum of density terms; singletons contribute 0
    best_total, best_cut = 0.0, 0
    i = 0
    while i < len(sim):
        j = i
        level = sim[i]
        while j < len(sim) and sim[j] == level:  # merge a whole similarity level
            a, b = find(e1[j]), find(e2[j])
            if a != b:
                if len(nodes[a]) < len(nodes[b]):
                    a, b = b, a
                total -= density_term(n_edges[a], len(nodes[a]))
                total -= density_term(n_edges[b], len(nodes[b]))
                nodes[a] |= nodes[b]
                nodes[b] = set()
                n_edges[a] += n_edges[b]
                parent[b] = a
                total += density_term(n_edges[a], len(nodes[a]))
            j += 1
        if total > best_total:
            best_total, best_cut = total, j
        i = j

    # replay merges up to the best cut
    parent = np.arange(m)
    for k in range(best_cut):
        a, b = find(e1[k]), find(e2[k])
        if a != b:
            parent[b] = a
    labels = np.array([find(e) for e in range(m)])
    _, labels = np.unique(labels, return_inverse=True)

    communities = []
    for c in range(labels.max() + 1):
        es = np.flatnonzero(labels == c)
        members = np.unique(np.r_[u[es], v[es]])
        if len(members) >= min_size:
            communities.append(members)
    return communities, labels, 2.0 / m * best_total
