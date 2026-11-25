"""
Overlapping Community Detection Metrics

* Overlapping NMI (McDaid, Greene & Hurley, 2011; the "NMI_max" variant used by
  NOCD, CLARE and others).
* Bi-matching F1 and Jaccard (Yang & Leskovec 2013; CLARE, ProCom):
  the mean best-match score of predicted communities against the ground truth
  and of ground-truth communities against the prediction, averaged.

Communities are given as lists of node-id arrays; num_nodes is the size of
the graph they live in.
"""

from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp


def membership_matrix(communities: Sequence[Sequence[int]], num_nodes: int) -> sp.csc_matrix:
    """Binary node x community incidence matrix."""
    rows = np.concatenate([np.asarray(c, dtype=np.int64) for c in communities])
    cols = np.repeat(np.arange(len(communities)), [len(c) for c in communities])
    data = np.ones(len(rows), dtype=np.float64)
    m = sp.csc_matrix((data, (rows, cols)), shape=(num_nodes, len(communities)))
    m.data[:] = 1.0  # duplicate node ids within a community count once
    return m


def _overlaps(pred, truth, num_nodes):
    a = membership_matrix(pred, num_nodes)
    b = membership_matrix(truth, num_nodes)
    inter = (a.T @ b).toarray()
    size_a = np.asarray(a.sum(axis=0)).ravel()
    size_b = np.asarray(b.sum(axis=0)).ravel()
    return inter, size_a, size_b


def _h(w, n):
    w = np.asarray(w, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -w * np.log2(w / n)
    return np.where(w > 0, out, 0.0)


def _conditional_entropy(inter, size_x, size_y, n):
    """H(X|Y) summed over communities of X, per McDaid et al. (2011)."""
    d = inter
    b = size_x[:, None] - d
    c = size_y[None, :] - d
    a = n - d - b - c
    h_ab_cd = _h(a, n) + _h(b, n) + _h(c, n) + _h(d, n)
    h_xi = _h(size_x, n) + _h(n - size_x, n)
    h_yj = _h(size_y, n) + _h(n - size_y, n)
    cond = h_ab_cd - h_yj[None, :]
    # Only accept Y_j as informative about X_i under the LFK constraint
    valid = _h(a, n) + _h(d, n) >= _h(b, n) + _h(c, n)
    cond = np.where(valid, cond, h_xi[:, None])
    best = cond.min(axis=1) if cond.shape[1] else h_xi
    return np.minimum(best, h_xi).sum(), h_xi.sum()


def overlapping_nmi(pred: List, truth: List, num_nodes: int) -> float:
    if not pred or not truth:
        return 0.0
    inter, size_p, size_t = _overlaps(pred, truth, num_nodes)
    h_p_given_t, h_p = _conditional_entropy(inter, size_p, size_t, num_nodes)
    h_t_given_p, h_t = _conditional_entropy(inter.T, size_t, size_p, num_nodes)
    mutual = 0.5 * (h_p - h_p_given_t + h_t - h_t_given_p)
    denom = max(h_p, h_t)
    return float(mutual / denom) if denom > 0 else 0.0


def bimatching_scores(pred: List, truth: List, num_nodes: int) -> Dict[str, float]:
    if not pred or not truth:
        return {"f1": 0.0, "jaccard": 0.0}
    inter, size_p, size_t = _overlaps(pred, truth, num_nodes)
    union = size_p[:, None] + size_t[None, :] - inter
    jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    f1 = np.divide(2 * inter, size_p[:, None] + size_t[None, :],
                   out=np.zeros_like(inter), where=(size_p[:, None] + size_t[None, :]) > 0)
    return {
        "f1": float(0.5 * (f1.max(axis=1).mean() + f1.max(axis=0).mean())),
        "jaccard": float(0.5 * (jaccard.max(axis=1).mean() + jaccard.max(axis=0).mean())),
    }


def evaluate(pred: List, truth: List, num_nodes: int) -> Dict[str, float]:
    return {"onmi": overlapping_nmi(pred, truth, num_nodes),
            **bimatching_scores(pred, truth, num_nodes)}
