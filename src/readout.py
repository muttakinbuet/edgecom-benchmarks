"""
Label-free Readout Selection

A fitted model gives two kinds of evidence about memberships: the affiliation
strengths F (threshold readout, F_uc > 0.5) and the community of every edge
(edge readout: node u joins c when at least a share tau of its edges are in c).
Which readout, and which tau, is right depends on the graph, so we choose among
candidates with the model's own held-out edges, without labels.

Each candidate membership matrix M is scored as a community-affiliation graph
model (AGM; Yang & Leskovec, 2012): community c links two of its members with
probability p_c, estimated from the training edges, and any pair with
background probability eps, so
    P(u ~ v) = 1 - (1 - eps) * prod_{c : M_uc = M_vc = 1} (1 - p_c).
The score is the held-out log-likelihood on the split used for early stopping:
the held-out edges plus the held-out share of all non-edges, estimated from as
many sampled non-edges as held-out edges. (A balanced score, weighting edges
and non-edges equally, rewards assigning every node to many communities.)
"""

from typing import Dict

import numpy as np

try:
    from .ocd_models import EdgeSampler
except ImportError:  # Fallback for direct script execution
    from ocd_models import EdgeSampler

TAUS = (0.1, 0.15, 0.2, 0.25, 0.33, 0.5)


def heldout_split(num_nodes: int, edge_index: np.ndarray, seed: int,
                  val_fraction: float = 0.1):
    """The held-out split of OverlappingCommunityModel.fit for this seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(edge_index.shape[1])
    num_val = max(1, int(val_fraction * edge_index.shape[1]))
    val_idx, train_idx = perm[:num_val], perm[num_val:]
    _, val_non_edges = EdgeSampler(num_nodes, edge_index[:, val_idx], num_val, rng,
                                   all_edge_index=edge_index).sample()
    return train_idx, edge_index[:, val_idx], val_non_edges


def edge_share_memberships(num_nodes, edge_index, edge_labels, k, tau):
    """Node u joins c when at least a share tau of its (assigned) edges are in c."""
    has = edge_labels >= 0
    u, v = edge_index[:, has]
    counts = np.zeros((num_nodes, k))
    np.add.at(counts, (u, edge_labels[has]), 1)
    np.add.at(counts, (v, edge_labels[has]), 1)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return share >= tau


def agm_score(member: np.ndarray, train_edges: np.ndarray, val_edges: np.ndarray,
              val_non_edges: np.ndarray) -> float:
    """Held-out log-likelihood (per held-out edge) of memberships under a fitted AGM."""
    member = member[:, member.any(axis=0)].astype(np.float64)
    n = member.shape[0]
    sizes = member.sum(axis=0)
    shared = member[train_edges[0]] * member[train_edges[1]]       # E x C
    inside = shared.sum(axis=0)
    pairs = sizes * (sizes - 1) / 2
    p = np.clip(inside / np.maximum(pairs, 1), 1e-6, 1 - 1e-6)
    background = (shared.sum(axis=1) == 0).sum()
    eps = np.clip(background / (n * (n - 1) / 2), 1e-12, 1 - 1e-6)
    log_q = np.log1p(-p)

    def log_no_edge(pairs_):
        s = member[pairs_[0]] * member[pairs_[1]]
        return np.log1p(-eps) + s @ log_q

    lq_e = log_no_edge(val_edges)
    lq_n = log_no_edge(val_non_edges)
    ll_edges = np.log(-np.expm1(np.minimum(lq_e, -1e-12))).sum()
    # The sampled non-edges stand for the held-out share of all non-edges
    num_edges = train_edges.shape[1] + val_edges.shape[1]
    share = val_edges.shape[1] / num_edges
    ll_non = share * (n * (n - 1) / 2 - num_edges) * lq_n.mean()
    return (ll_edges + ll_non) / val_edges.shape[1]


def candidates(F: np.ndarray, threshold: float, num_nodes: int, edge_index: np.ndarray,
               edge_labels: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    out = {"threshold": F > threshold}
    for tau in TAUS:
        out[f"edges_{tau:g}"] = edge_share_memberships(num_nodes, edge_index,
                                                       edge_labels, k, tau)
    return out


def select(F: np.ndarray, threshold: float, num_nodes: int, edge_index: np.ndarray,
           edge_labels: np.ndarray, k: int, seed: int):
    """Choose a readout by held-out AGM likelihood.

    Candidates are scored with edge shares counted on the training edges only,
    so held-out edges never shape the memberships they are scored against; the
    chosen readout is then applied to all edges. Returns (name, memberships,
    scores, all candidates on all edges).
    """
    train_idx, val_edges, val_non_edges = heldout_split(num_nodes, edge_index, seed)
    train_edges = edge_index[:, train_idx]
    scored = candidates(F, threshold, num_nodes, train_edges, edge_labels[train_idx], k)
    scores = {name: agm_score(m, train_edges, val_edges, val_non_edges)
              for name, m in scored.items() if m.any()}
    best = max(scores, key=scores.get)
    final = candidates(F, threshold, num_nodes, edge_index, edge_labels, k)
    return best, final[best], scores, final
