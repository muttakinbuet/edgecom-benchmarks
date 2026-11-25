"""
BigCLAM Baseline

Structure-only overlapping community detection (Yang & Leskovec, WSDM 2013).
Each node u has non-negative affiliation strengths F_u in R^K, and an edge
(u, v) appears with probability 1 - exp(-F_u . F_v). F maximizes

    sum_{(u,v) in E} log(1 - exp(-F_u.F_v))  -  sum_{(u,v) not in E} F_u.F_v

by full-batch projected Adam, starting from ego-nets with locally minimal
conductance as in the paper (or at random). Node u joins community c when
F_uc > sqrt(-log(1 - eps)) with eps = 2|E| / (N (N - 1)), as in the paper.

Usage:
    python src/bigclam.py --subset data/subsets/cs_areas_le2006
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from .metrics import evaluate
    from .subset import load_subset
except ImportError:  # Fallback for direct script execution
    from metrics import evaluate
    from subset import load_subset

logger = logging.getLogger(__name__)


class BigCLAM:
    def __init__(self, num_communities: int, lr: float = 0.05, epochs: int = 3000,
                 seed: int = 0, init: str = "egonet", tol: float = 1e-4):
        self.k = num_communities
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.init = init
        self.tol = tol

    def _egonet_seeds(self, num_nodes: int, src: np.ndarray, dst: np.ndarray,
                      rng: np.random.Generator):
        """
        Pick K seed nodes whose ego-net (node plus neighbours) has locally
        minimal conductance, preferring the lowest conductance (BigCLAM paper).
        """
        adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes,) * 2)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        # Edges among a node's neighbours = triangles through the node
        triangles = np.asarray((adj @ adj).multiply(adj).sum(axis=1)).ravel() / 2
        vol = deg + adj @ deg
        internal = deg + triangles
        cut = vol - 2 * internal
        total = deg.sum()
        conductance = np.divide(cut, np.minimum(vol, total - vol),
                                out=np.ones(num_nodes), where=np.minimum(vol, total - vol) > 0)
        # Locally minimal: no neighbour has a lower ego-net conductance
        neighbour_min = np.full(num_nodes, np.inf)
        np.minimum.at(neighbour_min, src, conductance[dst])
        candidates = np.flatnonzero((conductance <= neighbour_min) & (deg > 0))
        # Random tie-breaking among equal conductance, then lowest first
        order = np.lexsort((rng.random(len(candidates)), conductance[candidates]))
        return candidates[order[:self.k]], adj

    def fit(self, num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        # Both directions, so each node's gradient sums over all its neighbours
        src = np.concatenate([edge_index[0], edge_index[1]])
        dst = np.concatenate([edge_index[1], edge_index[0]])
        num_edges = edge_index.shape[1]

        # Scale the initialization so F_u.F_v starts near the mean edge density
        density = 2.0 * num_edges / (num_nodes * (num_nodes - 1))
        scale = np.sqrt(-np.log(1 - density) / self.k) * 3
        F = rng.uniform(0, 2 * scale, size=(num_nodes, self.k))
        if self.init == "egonet":
            seeds, adj = self._egonet_seeds(num_nodes, src, dst, rng)
            for c, s in enumerate(seeds):
                ego = np.append(adj.indices[adj.indptr[s]:adj.indptr[s + 1]], s)
                F[ego, c] = 1.0

        m = np.zeros_like(F)
        v = np.zeros_like(F)
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        prev_ll = None
        for epoch in range(1, self.epochs + 1):
            x = np.einsum("ij,ij->i", F[src], F[dst])
            x = np.maximum(x, 1e-10)
            # d/dx log(1 - e^-x) = 1 / (e^x - 1); plus 1 restores the edge's
            # own term that the all-pairs penalty below subtracts
            coef = 1.0 / np.expm1(x) + 1.0
            weights = sp.csr_matrix((coef, (src, dst)), shape=(num_nodes, num_nodes))
            grad = weights @ F - (F.sum(axis=0, keepdims=True) - F)

            # Adam ascent step, projected onto F >= 0
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            F = np.maximum(F + self.lr * scale * m_hat / (np.sqrt(v_hat) + adam_eps), 0)

            if epoch % 100 == 0 or epoch == self.epochs:
                ll = self.log_likelihood(F, edge_index)
                logger.info(f"epoch {epoch}: log-likelihood {ll:.4e}")
                if prev_ll is not None and abs(ll - prev_ll) < self.tol * abs(prev_ll):
                    logger.info(f"converged at epoch {epoch}")
                    break
                prev_ll = ll

        self.F = F
        self.threshold = np.sqrt(-np.log(1 - density))
        return F

    @staticmethod
    def log_likelihood(F: np.ndarray, edge_index: np.ndarray) -> float:
        x = np.einsum("ij,ij->i", F[edge_index[0]], F[edge_index[1]])
        edge_term = np.sum(np.log(-np.expm1(-np.maximum(x, 1e-10))) + x)
        s = F.sum(axis=0)
        all_pairs = 0.5 * (s @ s - np.einsum("ij,ij->", F, F))
        return float(edge_term - all_pairs)

    def communities(self, min_size: int = 1):
        member = self.F > self.threshold
        comms = [np.flatnonzero(member[:, c]) for c in range(self.k)]
        return [c for c in comms if len(c) >= min_size]


def main():
    parser = argparse.ArgumentParser(description="Run BigCLAM on a subset benchmark.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--k", type=int, help="communities (default: ground-truth count)")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--init", choices=["egonet", "random"], default="egonet")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    num_nodes, edge_index, truth = load_subset(args.subset)
    k = args.k or len(truth)
    logger.info(f"{num_nodes:,} nodes, {edge_index.shape[1]:,} edges, "
                f"{len(truth)} ground-truth communities, K = {k}")

    results = []
    for seed in args.seeds:
        t0 = time.time()
        model = BigCLAM(k, lr=args.lr, epochs=args.epochs, seed=seed, init=args.init)
        model.fit(num_nodes, edge_index)
        pred = model.communities()
        scores = evaluate(pred, truth, num_nodes)
        scores.update(seed=seed, seconds=round(time.time() - t0, 1),
                      predicted=len(pred),
                      mean_pred_size=float(np.mean([len(c) for c in pred])) if pred else 0.0)
        logger.info(f"seed {seed}: {scores}")
        results.append(scores)

    summary = {m: {"mean": float(np.mean([r[m] for r in results])),
                   "std": float(np.std([r[m] for r in results]))}
               for m in ("onmi", "f1", "jaccard")}
    logger.info(f"summary: {json.dumps(summary)}")
    out = Path(args.subset) / "results_bigclam.json"
    with open(out, "w") as f:
        json.dump({"k": k, "epochs": args.epochs, "lr": args.lr, "init": args.init,
                   "runs": results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
