"""
CESNA Baseline

Communities from Edge Structure and Node Attributes (Yang, McAuley & Leskovec,
ICDM 2013). Non-negative affiliations F explain edges as in BigCLAM,
P(A_uv = 1) = 1 - exp(-F_u . F_v), and binary node attributes through a
logistic model, P(X_uk = 1) = sigmoid(W_k . F_u + b_k). The objective is
(1 - alpha) * log P(A) + alpha * log P(X) - lambda * |W|_1, maximised by full-batch
projected Adam from ego-net seeds (as in our BigCLAM). Node u joins community c
when F_uc > sqrt(-log(1 - eps)), eps = 2|E| / (N (N - 1)).

Our graphs have no node attributes, so binary attributes are derived from the
edge features: each node's mean incident edge feature, projected on its top
principal components and binarised by sign.
"""

import numpy as np
import scipy.sparse as sp

try:
    from .bigclam import BigCLAM
except ImportError:  # Fallback for direct script execution
    from bigclam import BigCLAM


def binary_attributes(num_nodes: int, edge_index: np.ndarray, edge_features: np.ndarray,
                      num_attributes: int = 32) -> np.ndarray:
    """Binary node attributes from the node-averaged edge features (PCA signs)."""
    u, v = edge_index
    e = np.arange(len(u))
    inc = sp.csr_matrix((np.ones(2 * len(u)), (np.concatenate([u, v]), np.concatenate([e, e]))),
                        shape=(num_nodes, len(u)))
    deg = np.asarray(inc.sum(axis=1)).ravel()
    node_feats = (inc @ edge_features) / np.maximum(deg, 1)[:, None]
    centred = node_feats - node_feats.mean(axis=0)
    _, _, vt = np.linalg.svd(centred[np.random.default_rng(0).permutation(num_nodes)[:20000]],
                             full_matrices=False)
    scores = centred @ vt[:num_attributes].T
    return (scores > 0).astype(np.float64)


class CESNA(BigCLAM):
    def __init__(self, num_communities: int, alpha: float = 0.5, l1: float = 1.0,
                 **kwargs):
        super().__init__(num_communities, **kwargs)
        self.alpha = alpha
        self.l1 = l1

    def fit(self, num_nodes: int, edge_index: np.ndarray, attributes: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        src = np.concatenate([edge_index[0], edge_index[1]])
        dst = np.concatenate([edge_index[1], edge_index[0]])
        num_edges = edge_index.shape[1]
        density = 2.0 * num_edges / (num_nodes * (num_nodes - 1))
        scale = np.sqrt(-np.log(1 - density) / self.k) * 3
        F = rng.uniform(0, 2 * scale, size=(num_nodes, self.k))
        if self.init == "egonet":
            seeds, adj = self._egonet_seeds(num_nodes, src, dst, rng)
            for c, s in enumerate(seeds):
                F[np.append(adj.indices[adj.indptr[s]:adj.indptr[s + 1]], s), c] = 1.0
        X = attributes
        W = rng.normal(scale=0.01, size=(X.shape[1], self.k))
        b = np.zeros(X.shape[1])

        params = [F, W, b]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        steps = [self.lr * scale, self.lr * 0.1, self.lr * 0.1]
        prev = None
        for epoch in range(1, self.epochs + 1):
            x = np.maximum(np.einsum("ij,ij->i", F[src], F[dst]), 1e-10)
            coef = 1.0 / np.expm1(x) + 1.0
            weights = sp.csr_matrix((coef, (src, dst)), shape=(num_nodes, num_nodes))
            grad_graph = weights @ F - (F.sum(axis=0, keepdims=True) - F)
            q = 1.0 / (1.0 + np.exp(-(F @ W.T + b)))
            resid = X - q
            grads = [(1 - self.alpha) * grad_graph + self.alpha * resid @ W,
                     self.alpha * resid.T @ F - self.l1 * np.sign(W),
                     self.alpha * resid.sum(axis=0)]
            for i, g in enumerate(grads):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
                update = steps[i] * (m[i] / (1 - beta1 ** epoch)) / (
                    np.sqrt(v[i] / (1 - beta2 ** epoch)) + eps)
                params[i] += update
            np.maximum(F, 0, out=F)
            if epoch % 100 == 0:
                ll = self.log_likelihood(F, edge_index)
                if prev is not None and abs(ll - prev) < self.tol * abs(prev):
                    break
                prev = ll
        self.F, self.W, self.b = F, W, b
        self.threshold = np.sqrt(-np.log(1 - density))
        return F
