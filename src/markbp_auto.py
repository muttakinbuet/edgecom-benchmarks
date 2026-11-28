"""
MarkBP-auto: the complete, label-free method.

Two tests decide how the edge features are used:

1. Homophily test (no fitted model): do edges that share a node fall into the
   same edge-feature cluster more often than chance? If not, the features are
   unrelated to the graph and are ignored: the result is structure-only NOCD.
2. Alignment test: do the feature clusters agree with the communities that
   structure-only NOCD assigns to the edges?
     - yes: fit MarkBP with coupled marks from the feature-led start;
     - no:  the structure model cannot use the features (typically because
            structure itself carries little community signal), so the result
            is clustering the edge features alone, read out per edge.

3. Optional held-out check (holdout_check=True, off by default): the MarkBP
   fit is replaced by the structure-only fit when it predicts the held-out
   edges and non-edges significantly worse: a paired z statistic of the
   per-pair loss differences on the same held-out pairs (both fits use the
   seed's split) above Z_THRESHOLD. It catches the collapse on the yeast
   benchmark (non-textual evidence scores); on textual edges it fires only
   with sparse outside features and then helps or hurts depending on the
   ground truth (hurts on DBLP-EF venues and patent subclasses, helps on patent
   assignees), so the method does not use it. The z is
   always computed and logged.

Both tests use the same threshold z > Z_THRESHOLD. No community labels are
used. The NOCD fit is deterministic given the graph, K and the seed, so it can
be cached on disk (cache_dir) and shared by runs that differ only in their edge
features, e.g. real and shuffled.

With input_features=True the MarkBP branch also gives its encoder each node's
mean incident edge feature as input (as NOCD with averaged edge features does),
so the features enter both through the encoder and through the marks. The tests
and the other two branches are unchanged.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .feature_test import Z_THRESHOLD, alignment_test, feature_clusters, homophily_test
    from .ocd_models import OverlappingCommunityModel
    from .run_experiment import edge_feature_node_mean
except ImportError:  # Fallback for direct script execution
    from feature_test import Z_THRESHOLD, alignment_test, feature_clusters, homophily_test
    from ocd_models import OverlappingCommunityModel
    from run_experiment import edge_feature_node_mean


class _EdgeClustering:
    """Communities from clustering edge features alone (edge readout)."""

    def __init__(self, labels: np.ndarray, num_nodes: int, edge_index: np.ndarray,
                 k: int, share: float = 0.25):
        self.labels = labels
        has = labels >= 0
        u, v = edge_index[:, has]
        counts = np.zeros((num_nodes, k))
        np.add.at(counts, (u, labels[has]), 1)
        np.add.at(counts, (v, labels[has]), 1)
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        # Expose memberships as F so that a threshold readout equals the edge readout
        self.F = (frac >= share).astype(np.float64)
        self.threshold = 0.5

    def edge_assignments(self, edge_index, edge_features=None):
        return self.labels


class MarkBPAuto:
    def __init__(self, num_communities: int, seed: int = 0,
                 cache_dir: Optional[str] = None, input_features: bool = False,
                 mark_weight: float = 0.1, coupled_init: str = "features",
                 feature_clusters: Optional[int] = None, holdout_check: bool = False,
                 joint_init_seed: Optional[int] = None, **model_kwargs):
        self.k = num_communities
        self.holdout_check = holdout_check
        self.input_features = input_features
        # Settings of the MarkBP branch (mark_weight 0 makes it NOCD, with
        # averaged edge features as input when input_features is set). The
        # default 0.1 was chosen on CS-Areas (variant_screen.py) and confirmed
        # on CS-Systems and Random-30; earlier runs used 1.0
        self.mark_weight = mark_weight
        self.coupled_init = coupled_init
        self.feature_clusters = feature_clusters
        self.seed = seed
        # Initialisation seed of the joint (MarkBP) fit only: restarts vary it
        # while the tests, the NOCD fit and the held-out split keep `seed`
        self.joint_init_seed = joint_init_seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_kwargs = model_kwargs

    def _structure_model(self, num_nodes: int, edge_index: np.ndarray):
        """NOCD fit, loaded from the cache when this graph/K/seed was fitted before."""
        model = OverlappingCommunityModel(self.k, seed=self.seed, refit=False,
                                          **self.model_kwargs)
        path = None
        if self.cache_dir is not None:
            key = hashlib.sha1(np.ascontiguousarray(edge_index).tobytes()).hexdigest()[:16]
            path = self.cache_dir / f"nocd_{key}_n{num_nodes}_k{self.k}_s{self.seed}.npz"
            if path.exists():
                cached = np.load(path)
                model.F = cached["F"]
                model.best_val = float(cached["best_val"])
                model.best_epoch = int(cached["best_epoch"])
                return model
        model.fit(num_nodes, edge_index)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Per-process temporary name: parallel workers may fit the same model
            tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(tmp, F=model.F, best_val=model.best_val, best_epoch=model.best_epoch)
            try:
                tmp.replace(path)
            except PermissionError:
                # Another worker is writing the identical result; keep theirs
                tmp.unlink(missing_ok=True)
        return model

    def fit(self, num_nodes: int, edge_index: np.ndarray,
            edge_features: Optional[np.ndarray] = None) -> np.ndarray:
        self.homophily = self.alignment = None
        self.joint_val = self.holdout_z = None
        self.choice = "structure"
        self._features = None
        self.structure = self._structure_model(num_nodes, edge_index)
        self.model = self.structure

        if edge_features is not None:
            self.homophily = homophily_test(edge_index, edge_features, self.k, self.seed)
            if self.homophily["features_consistent"]:
                labels = self.structure.edge_assignments(edge_index)
                self.alignment = alignment_test(labels, edge_features, self.k, self.seed)
                if self.alignment["use_features"]:
                    self.choice = "markbp"
                    self.model = OverlappingCommunityModel(
                        self.k, seed=self.seed, init_seed=self.joint_init_seed,
                        refit=False, mark_weight=self.mark_weight,
                        mark_model="coupled", coupled_init=self.coupled_init,
                        feature_clusters=self.feature_clusters, **self.model_kwargs)
                    # Node inputs from training edges only (no held-out text)
                    self.model.fit(num_nodes, edge_index, edge_features=edge_features,
                                   input_edge_features=(edge_features if self.input_features
                                                        else None))
                    self._features = edge_features
                    self.joint_val = float(self.model.best_val)
                    self.holdout_z = self._holdout_z(self.model.val_pairs)
                    if self.holdout_check and self.holdout_z > Z_THRESHOLD:
                        self.choice = "structure_holdout"
                        self.model, self._features = self.structure, None
                else:
                    self.choice = "edge_clustering"
                    self.model = _EdgeClustering(
                        feature_clusters(edge_features, self.k, self.seed),
                        num_nodes, edge_index, self.k)

        self.F = self.model.F
        self.threshold = self.model.threshold
        return self.F

    def _holdout_z(self, val) -> float:
        """Paired z of (joint - structure) held-out loss; > 0: joint is worse."""
        (ej, nj) = OverlappingCommunityModel.heldout_terms(self.model.F, val)
        (es, ns) = OverlappingCommunityModel.heldout_terms(self.structure.F, val)
        de, dn = ej - es, nj - ns
        diff = 0.5 * (de.mean() + dn.mean())
        se = 0.5 * np.sqrt(de.var() / len(de) + dn.var() / len(dn))
        return float(diff / se) if se > 0 else 0.0

    @property
    def uses_features(self) -> bool:
        return not self.choice.startswith("structure")

    @property
    def test(self) -> dict:
        """Test statistics for logging."""
        return {"homophily_z": (self.homophily or {}).get("homophily_z"),
                "z": (self.alignment or {}).get("z"),
                "joint_val": self.joint_val, "holdout_z": self.holdout_z,
                "structure_val": float(self.structure.best_val),
                "choice": self.choice}

    def edge_assignments(self, edge_index: np.ndarray, edge_features=None) -> np.ndarray:
        """Edge communities (-1 for edges without features under edge clustering)."""
        return self.model.edge_assignments(edge_index, self._features)
