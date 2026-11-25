"""
Neural Overlapping Community Detection Models

* NOCD (Shchur & Guennemann, 2019): a GCN encoder outputs non-negative
  affiliations F, trained on the balanced Bernoulli-Poisson likelihood.
* MarkBP: NOCD plus edge features as marks. Community c creates edge (u, v)
  with rate F_uc F_vc, so the edge came from c with probability
  r_uvc = F_uc F_vc / F_u.F_v, and its features follow a von Mises-Fisher
  distribution with community mean direction mu_c. The mark term
  log sum_c r_uvc exp(kappa mu_c . x_uv) ties edge features to the
  community that explains each edge.

Edge features must be L2-normalised; the vMF normaliser is constant for a
fixed shared kappa and is dropped.
"""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

logger = logging.getLogger(__name__)


def normalized_adjacency(num_nodes: int, edge_index: np.ndarray) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a sparse tensor."""
    u, v = edge_index
    rows = np.concatenate([u, v, np.arange(num_nodes)])
    cols = np.concatenate([v, u, np.arange(num_nodes)])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(num_nodes, num_nodes)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    adj = (d_inv_sqrt @ adj @ d_inv_sqrt).tocoo()
    indices = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(adj.data.astype(np.float32)),
                                   (num_nodes, num_nodes)).coalesce()


def vmf_log_normalizer(kappa: torch.Tensor, dim: int) -> torch.Tensor:
    """
    kappa-dependent part of log C_d(kappa), the von Mises-Fisher normaliser, via
    the uniform asymptotic (Debye) expansion of the Bessel function I_v,
    v = d/2 - 1, which is accurate for the large v of text embeddings and
    stable as kappa -> 0. Constants shared by all kappa are dropped.
    """
    v = dim / 2.0 - 1.0
    s = torch.sqrt(v ** 2 + kappa ** 2)
    return -s + v * torch.log(v + s) + 0.5 * torch.log(s)


class GCNEncoder(nn.Module):
    """NOCD's encoder: GCN layers with batch norm and dropout, ReLU output."""

    def __init__(self, in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.5,
                 batch_norm: bool = True):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for d in hidden_dims) \
            if batch_norm else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if x.is_sparse:
                # Sparse input (e.g. the adjacency as features): no dropout
                h = torch.sparse.mm(x, layer.weight.T) + layer.bias
            else:
                if self.dropout:
                    x = F_nn.dropout(x, self.dropout, self.training)
                h = layer(x)
            x = torch.sparse.mm(adj, h)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                if self.norms is not None:
                    x = self.norms[i](x)
        return torch.relu(x)


def _project(layer: nn.Linear, x: torch.Tensor, dropout: float, training: bool):
    """layer(x), for dense or sparse x (no dropout on sparse input)."""
    if x.is_sparse:
        return torch.sparse.mm(x, layer.weight.T) + layer.bias
    if dropout:
        x = F_nn.dropout(x, dropout, training)
    return layer(x)


class _EdgeListEncoder(nn.Module):
    """Shared parts of encoders that pass messages along an explicit edge list."""

    def __init__(self, in_dim, hidden_dims, out_dim, src, dst, dropout, batch_norm):
        super().__init__()
        self.dims = [in_dim, *hidden_dims, out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b)
                                    for a, b in zip(self.dims[:-1], self.dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for d in hidden_dims) \
            if batch_norm else None
        self.dropout = dropout
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)

    def _finish(self, i, x):
        if i < len(self.layers) - 1:
            x = torch.relu(x)
            if self.norms is not None:
                x = self.norms[i](x)
        return x


class GINEEncoder(_EdgeListEncoder):
    """
    Edge-aware encoder in the style of GINE (Hu et al., 2020): a message from v
    to u is ReLU(W h_v + W_e x_uv), passed through the nonlinearity per edge
    before aggregation, so the encoder sees each edge's own features rather than
    their average. h_u' = W h_u + mean of incoming messages. ReLU output.
    """

    def __init__(self, in_dim, hidden_dims, out_dim, src, dst, edge_feats, num_nodes,
                 dropout=0.5, batch_norm=True):
        super().__init__(in_dim, hidden_dims, out_dim, src, dst, dropout, batch_norm)
        self.edge_layers = nn.ModuleList(nn.Linear(edge_feats.shape[1], b)
                                         for b in self.dims[1:])
        self.register_buffer("edge_feats", edge_feats)
        deg = torch.zeros(num_nodes).index_add_(0, dst, torch.ones(len(dst)))
        self.register_buffer("inv_deg", 1.0 / deg.clamp(min=1.0))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            hw = _project(layer, x, self.dropout, self.training)
            msg = torch.relu(hw[self.src] + self.edge_layers[i](self.edge_feats))
            agg = torch.zeros_like(hw).index_add_(0, self.dst, msg)
            agg = agg * self.inv_deg[:, None]
            x = self._finish(i, hw + agg)
        return torch.relu(x)


class GATEncoder(_EdgeListEncoder):
    """
    Single-head graph attention (Velickovic et al., 2018) over neighbours and
    the node itself, as used for overlapping communities by Sismanis et al.
    (2025). ReLU output.
    """

    def __init__(self, in_dim, hidden_dims, out_dim, src, dst, num_nodes, dropout=0.5,
                 batch_norm=True):
        loops = torch.arange(num_nodes)
        super().__init__(in_dim, hidden_dims, out_dim, torch.cat([src, loops]),
                         torch.cat([dst, loops]), dropout, batch_norm)
        self.att_src = nn.ParameterList(nn.Parameter(torch.randn(b) * 0.1)
                                        for b in self.dims[1:])
        self.att_dst = nn.ParameterList(nn.Parameter(torch.randn(b) * 0.1)
                                        for b in self.dims[1:])

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            hw = _project(layer, x, self.dropout, self.training)
            score = F_nn.leaky_relu(hw[self.src] @ self.att_src[i]
                                    + hw[self.dst] @ self.att_dst[i], 0.2)
            weight = torch.exp(score - score.max())
            norm = torch.zeros(hw.shape[0]).index_add_(0, self.dst, weight)
            alpha = weight / norm[self.dst].clamp(min=1e-16)
            out = torch.zeros_like(hw).index_add_(0, self.dst, alpha[:, None] * hw[self.src])
            x = self._finish(i, out)
        return torch.relu(x)


class EdgeSampler:
    """
    Mini-batches of edges plus an equal number of random non-edges. Non-edges
    avoid every edge in all_edge_index (default: the sampled edges).
    """

    def __init__(self, num_nodes: int, edge_index: np.ndarray, batch_size: int,
                 rng: np.random.Generator, all_edge_index: Optional[np.ndarray] = None):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.batch_size = min(batch_size, edge_index.shape[1])
        self.rng = rng
        u, v = edge_index if all_edge_index is None else all_edge_index
        self.edge_keys = set((np.minimum(u, v) * num_nodes + np.maximum(u, v)).tolist())

    def sample(self):
        e = self.rng.choice(self.edge_index.shape[1], self.batch_size, replace=False)
        n = 2 * self.batch_size
        a = self.rng.integers(0, self.num_nodes, n)
        b = self.rng.integers(0, self.num_nodes, n)
        keys = np.minimum(a, b) * self.num_nodes + np.maximum(a, b)
        keep = (a != b) & np.fromiter((k not in self.edge_keys for k in keys.tolist()),
                                      dtype=bool, count=n)
        a, b = a[keep][:self.batch_size], b[keep][:self.batch_size]
        return e, np.vstack([a, b])


class OverlappingCommunityModel:
    """
    Train NOCD (mark_weight = 0) or MarkBP (mark_weight > 0, edge_features given).
    """

    def __init__(self, num_communities: int, hidden_dims=(128,), dropout: float = 0.5,
                 lr: float = 1e-3, weight_decay: float = 1e-2, epochs: int = 500,
                 batch_size: int = 20000, mark_weight: float = 0.0, kappa: float = 10.0,
                 patience: int = 50, threshold: float = 0.5, seed: int = 0,
                 mu_init: str = "random", val_fraction: float = 0.1,
                 refit: bool = True, init_seed: Optional[int] = None,
                 min_epochs: int = 100, learn_kappa: bool = False,
                 mark_model: str = "vmf", feature_clusters: Optional[int] = None,
                 coupled_init: str = "uniform", pretrain_epochs: int = 100,
                 encoder: str = "gcn", mark_start: int = 100):
        self.k = num_communities
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.mark_weight = mark_weight
        self.kappa = kappa
        self.patience = patience
        self.threshold = threshold
        self.seed = seed
        self.mu_init = mu_init
        self.val_fraction = val_fraction
        self.refit = refit
        # The held-out loss can rise before communities form (seen on DBLP:
        # up until ~epoch 50, then a sharp drop), so never stop before this
        self.min_epochs = min_epochs
        # seed fixes the held-out split; init_seed the weights and batches, so
        # restarts can share one split
        self.init_seed = seed if init_seed is None else init_seed
        # MarkBP v2: one concentration per community, learned with the proper
        # vMF normaliser, so uninformative features get kappa -> 0
        self.learn_kappa = learn_kappa
        # MarkBP v3 ("coupled"): edge features are first clustered on their own
        # into M clusters; a learned coupling Pi (K x M, initialised uniform)
        # links communities to feature clusters. Unrelated features keep Pi
        # uniform and exert no pull; the per-edge influence is bounded by
        # log(1 / min cluster share) whatever the feature dimension.
        if mark_model not in ("vmf", "coupled"):
            raise ValueError(mark_model)
        self.mark_model = mark_model
        self.feature_clusters = feature_clusters
        # "features": start communities at the edge-feature clusters (encoder
        # pretrained to their edge readout, coupling near-identity); structure
        # can still override it because the feature influence is bounded
        # "structure": train without marks for mark_start epochs, then set the
        # coupling from how those communities' edges fall into the feature
        # clusters and continue with marks, so the features refine structural
        # communities instead of replacing them
        if coupled_init not in ("uniform", "features", "structure"):
            raise ValueError(coupled_init)
        self.coupled_init = coupled_init
        self.pretrain_epochs = pretrain_epochs
        self.mark_start = mark_start
        # Encoder: "gcn" (NOCD), "gine" (edge-aware messages, needs edge
        # features) or "gat" (graph attention, Sismanis et al. 2025)
        # or "dynares" (official DynaResGCN encoder)
        if encoder not in ("gcn", "gine", "gat", "dynares"):
            raise ValueError(encoder)
        self.encoder_type = encoder

    # -- likelihood terms ---------------------------------------------------
    @staticmethod
    def _bp_loss(F: torch.Tensor, edges: torch.Tensor, non_edges: torch.Tensor):
        """Balanced Bernoulli-Poisson negative log-likelihood (NOCD)."""
        x_e = (F[edges[0]] * F[edges[1]]).sum(dim=1)
        loss_edges = -torch.log(-torch.expm1(-x_e.clamp(min=1e-6))).mean()
        loss_non_edges = (F[non_edges[0]] * F[non_edges[1]]).sum(dim=1).mean()
        return 0.5 * (loss_edges + loss_non_edges)

    def _cluster_posteriors(self, feats: torch.Tensor) -> torch.Tensor:
        """log q(g | x) - log w_g for the fixed feature clusters (coupled model)."""
        logits = self.cluster_kappa * feats @ self.cluster_dirs.T + self.log_w
        return torch.log_softmax(logits, dim=1) - self.log_w

    def _mark_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """log p(x_uv | c) up to a constant shared by all communities."""
        if self.mark_model == "coupled":
            log_pi = torch.log_softmax(self.coupling, dim=1)              # K x M
            ratio = self._cluster_posteriors(feats)                       # E x M
            # Exact path up to K x M = 10^4, which covers every DBLP-EF run
            # (at most 85 x 85), so their results stay bit-identical
            if log_pi.numel() <= 10_000:
                return torch.logsumexp(ratio[:, None, :] + log_pi[None], dim=2)  # E x K
            # Large K x M (e.g. hundreds of small complexes): the same quantity
            # as a matrix product, without the E x K x M tensor
            shift = ratio.max(dim=1, keepdim=True).values
            return torch.log((ratio - shift).exp() @ log_pi.exp().T + 1e-30) + shift
        mu = F_nn.normalize(self.mu, dim=1)
        if self.learn_kappa:
            kappa = F_nn.softplus(self.rho)
            return vmf_log_normalizer(kappa, feats.shape[1]) + kappa * (feats @ mu.T)
        return self.kappa * (feats @ mu.T)

    def _mark_loss(self, F: torch.Tensor, edges: torch.Tensor, feats: torch.Tensor):
        """-log sum_c r_uvc p(x_uv | c), averaged over edges that have features."""
        has = feats.norm(dim=1) > 0  # edges without features carry no mark
        if not has.any():
            return F.sum() * 0.0
        edges, feats = edges[:, has], feats[has]
        rates = F[edges[0]] * F[edges[1]] + 1e-10
        log_r = torch.log(rates) - torch.log(rates.sum(dim=1, keepdim=True))
        return -torch.logsumexp(log_r + self._mark_logits(feats), dim=1).mean()

    # -- training -----------------------------------------------------------
    def fit(self, num_nodes: int, edge_index: np.ndarray,
            node_features: Optional[np.ndarray] = None,
            edge_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Unsupervised training. A fraction of the edges (and as many non-edges)
        is held out: the model is trained on the rest, stopped when the
        held-out Bernoulli-Poisson loss stops improving, and that loss is kept
        as self.best_val for label-free model selection. With refit=True the
        model is then retrained on all edges for the selected number of epochs.
        No community labels are used.
        """
        if (self.mark_weight > 0 or self.encoder_type == "gine") and edge_features is None:
            raise ValueError("MarkBP needs edge features")
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(edge_index.shape[1])
        num_val = max(1, int(self.val_fraction * edge_index.shape[1]))
        val_idx, train_idx = perm[:num_val], perm[num_val:]
        _, val_non_edges = EdgeSampler(num_nodes, edge_index[:, val_idx], num_val, rng,
                                       all_edge_index=edge_index).sample()
        val = (edge_index[:, val_idx], val_non_edges)
        self.val_pairs = val  # kept for paired comparisons of fits (markbp_auto)

        self.best_epoch, self.best_val = self._train(
            num_nodes, edge_index, train_idx, node_features, edge_features,
            self.epochs, val, np.random.default_rng(self.init_seed))
        if self.refit:
            self._train(num_nodes, edge_index, np.arange(edge_index.shape[1]),
                        node_features, edge_features, self.best_epoch, None,
                        np.random.default_rng(self.init_seed))

        self.encoder.eval()
        with torch.no_grad():
            self.F = self.encoder(self._x, self._adj).numpy()
        return self.F

    def _fit_feature_clusters(self, train_feats: np.ndarray, rng):
        """Cluster edge features once (spherical k-means); fixed during training."""
        try:
            from .kmeans_cache import cached_kmeans
        except ImportError:  # Fallback for direct script execution
            from kmeans_cache import cached_kmeans
        has = np.linalg.norm(train_feats, axis=1) > 0
        pool = train_feats[has]
        sample = rng.choice(len(pool), min(50000, len(pool)), replace=False)
        m = self.feature_clusters or self.k
        centers, labels = cached_kmeans(pool[sample], m, self.init_seed)
        centers = centers.astype(np.float32)
        r_bar = np.linalg.norm(centers, axis=1)
        share = np.bincount(labels, minlength=m) / len(labels)
        d = pool.shape[1]
        r = float(np.clip((share * r_bar).sum(), 1e-3, 0.999))
        self.cluster_dirs = torch.from_numpy(centers / np.maximum(r_bar[:, None], 1e-8))
        self.cluster_kappa = float(r * (d - r ** 2) / (1 - r ** 2))  # Banerjee et al.
        self.log_w = torch.from_numpy(np.log(np.maximum(share, 1e-6)).astype(np.float32))

    def _pretrain_to_feature_clusters(self, num_nodes, train_edges, feats, x, adj,
                                      share: float = 0.25):
        """Fit the encoder so F reproduces the memberships read out from the
        edges' feature clusters (a node is in c if >= share of its edges are)."""
        has = feats.norm(dim=1) > 0
        labels = torch.argmax(self._cluster_posteriors(feats[has]), dim=1).numpy()
        u, v = train_edges[:, has.numpy()]
        counts = np.zeros((num_nodes, self.k))
        np.add.at(counts, (u, labels), 1)
        np.add.at(counts, (v, labels), 1)
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        target = torch.from_numpy((frac >= share).astype(np.float32))
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr * 10)
        self.encoder.train()
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            loss = F_nn.mse_loss(self.encoder(x, adj), target)
            loss.backward()
            optimizer.step()

    def _couple_to_structure(self, F, train_edges, feats, smoothing: float = 1.0):
        """Coupling Pi_cm proportional to the edges of community c in feature cluster m."""
        has = feats.norm(dim=1) > 0
        edges = torch.from_numpy(train_edges)[:, has]
        community = torch.argmax(F[edges[0]] * F[edges[1]], dim=1)
        cluster = torch.argmax(self._cluster_posteriors(feats[has]), dim=1)
        counts = torch.full((self.k, len(self.log_w)), smoothing)
        counts.index_put_((community, cluster), torch.ones(len(cluster)), accumulate=True)
        with torch.no_grad():
            self.coupling.copy_(torch.log(counts / counts.sum(dim=1, keepdim=True)))

    def _train(self, num_nodes, all_edges, train_idx, node_features, edge_features,
               max_epochs, val, rng):
        """Train from scratch on all_edges[:, train_idx]; returns (best epoch, val loss)."""
        torch.manual_seed(self.init_seed)
        train_edges = all_edges[:, train_idx]
        adj = normalized_adjacency(num_nodes, train_edges)
        # NOCD-G uses the adjacency rows as node features
        x = adj if node_features is None else torch.from_numpy(
            node_features.astype(np.float32))
        self._adj, self._x = adj, x

        if self.encoder_type == "gcn":
            self.encoder = GCNEncoder(x.shape[1], self.hidden_dims, self.k, self.dropout)
        elif self.encoder_type == "dynares":
            try:
                from .dynaresgcn import DynaResGCNEncoder
            except ImportError:  # Fallback for direct script execution
                from dynaresgcn import DynaResGCNEncoder
            if node_features is None:
                # Official DynaResGCN-G: one-hot node identities as input
                idx = torch.arange(num_nodes)
                x = torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(num_nodes),
                                            (num_nodes, num_nodes)).coalesce()
                self._x = x
            self.encoder = DynaResGCNEncoder(x.shape[1], self.hidden_dims, self.k, num_nodes,
                                             train_edges, seed=self.init_seed)
        else:
            # Messages flow along training edges only, in both directions
            u, v = (torch.from_numpy(a) for a in train_edges)
            src, dst = torch.cat([u, v]), torch.cat([v, u])
            if self.encoder_type == "gine":
                e = torch.from_numpy(edge_features[train_idx].astype(np.float32))
                self.encoder = GINEEncoder(x.shape[1], self.hidden_dims, self.k, src, dst,
                                           torch.cat([e, e]), num_nodes, self.dropout)
            else:
                self.encoder = GATEncoder(x.shape[1], self.hidden_dims, self.k, src, dst,
                                          num_nodes, self.dropout)
        params = [{"params": self.encoder.parameters(), "weight_decay": self.weight_decay}]
        use_marks = self.mark_weight > 0
        if use_marks and self.mark_model == "coupled":
            train_feats = edge_features[train_idx]
            feats = torch.from_numpy(train_feats.astype(np.float32))
            self._fit_feature_clusters(train_feats, rng)
            m = len(self.log_w)
            coupling0 = torch.zeros(self.k, m)
            if self.coupled_init == "features":
                if m != self.k:
                    raise ValueError("feature-led init needs as many clusters as communities")
                coupling0 = torch.log(0.9 * torch.eye(self.k) + 0.1 / m)
                self._pretrain_to_feature_clusters(num_nodes, train_edges, feats, x, adj)
            self.coupling = nn.Parameter(coupling0)
            params.append({"params": [self.coupling], "weight_decay": 0.0})
        elif use_marks:
            train_feats = edge_features[train_idx]
            feats = torch.from_numpy(train_feats.astype(np.float32))
            if self.mu_init == "kmeans":
                # Start community directions at clusters of the edge features
                from sklearn.cluster import KMeans
                sample = rng.choice(len(train_feats), min(50000, len(train_feats)),
                                    replace=False)
                km = KMeans(self.k, n_init=3, random_state=self.init_seed)
                km.fit(train_feats[sample])
                mu0 = torch.from_numpy(km.cluster_centers_.astype(np.float32))
            else:
                mu0 = torch.randn(self.k, feats.shape[1])
            self.mu = nn.Parameter(mu0)
            params.append({"params": [self.mu], "weight_decay": 0.0})
            if self.learn_kappa:
                # Banerjee et al. (2005) estimate from the initial clusters
                d = feats.shape[1]
                r_bar = mu0.norm(dim=1).clamp(1e-3, 0.999)
                kappa0 = (r_bar * (d - r_bar ** 2) / (1 - r_bar ** 2)).clamp(0.1, 1000.0)
                self.rho = nn.Parameter(kappa0 + torch.log(-torch.expm1(-kappa0)))
                params.append({"params": [self.rho], "weight_decay": 0.0})
        optimizer = torch.optim.Adam(params, lr=self.lr)
        sampler = EdgeSampler(num_nodes, train_edges, self.batch_size, rng,
                              all_edge_index=all_edges)
        if val is not None:
            val_edges, val_non_edges = (torch.from_numpy(v) for v in val)

        best_val, best_epoch, best_state, wait = np.inf, max_epochs, None, 0
        for epoch in range(1, max_epochs + 1):
            self.encoder.train()
            optimizer.zero_grad()
            F = self.encoder(x, adj)
            e, ne = sampler.sample()
            edges = torch.from_numpy(train_edges[:, e])
            loss = self._bp_loss(F, edges, torch.from_numpy(ne))
            delayed = use_marks and self.coupled_init == "structure"
            if delayed and epoch == self.mark_start + 1:
                self._couple_to_structure(F.detach(), train_edges, feats)
                # Only models with marks may be returned
                best_val, best_epoch, best_state, wait = np.inf, max_epochs, None, 0
            if use_marks and not (delayed and epoch <= self.mark_start):
                loss = loss + self.mark_weight * self._mark_loss(F, edges, feats[e])
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                logger.info(f"epoch {epoch}: loss {loss.item():.4f}")
            if val is None:
                continue

            self.encoder.eval()
            with torch.no_grad():
                score = self._bp_loss(self.encoder(x, adj), val_edges, val_non_edges).item()
            if score < best_val:
                best_val, best_epoch, wait = score, epoch, 0
                best_state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
                best_mu = (self.mu.detach().clone()
                           if use_marks and self.mark_model == "vmf" else None)
                best_rho = (self.rho.detach().clone()
                            if use_marks and self.learn_kappa else None)
                best_coupling = (self.coupling.detach().clone()
                                 if use_marks and self.mark_model == "coupled" else None)
            else:
                wait += 1
                if wait >= self.patience and epoch >= self.min_epochs:
                    logger.info(f"early stop at epoch {epoch} (best {best_epoch})")
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state)
            if use_marks and self.mark_model == "coupled":
                self.coupling = nn.Parameter(best_coupling)
            elif use_marks:
                self.mu = nn.Parameter(best_mu)
                if self.learn_kappa:
                    self.rho = nn.Parameter(best_rho)
        return best_epoch, best_val

    def communities(self):
        member = self.F > self.threshold
        return [np.flatnonzero(member[:, c]) for c in range(self.k)
                if member[:, c].any()]

    @staticmethod
    def heldout_terms(F: np.ndarray, val):
        """Per-pair terms of the balanced held-out Bernoulli-Poisson loss:
        (losses of held-out edges, losses of held-out non-edges); the loss is
        0.5 * (mean of the first + mean of the second)."""
        edges, non_edges = val
        x = np.clip((F[edges[0]] * F[edges[1]]).sum(axis=1), 1e-6, None)
        return -np.log(-np.expm1(-x)), (F[non_edges[0]] * F[non_edges[1]]).sum(axis=1)

    def edge_assignments(self, edge_index: np.ndarray,
                         edge_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Most likely community of each edge, using marks when available."""
        F = torch.from_numpy(self.F)
        edges = torch.from_numpy(edge_index)
        rates = F[edges[0]] * F[edges[1]] + 1e-10
        logits = torch.log(rates)
        if edge_features is not None and (hasattr(self, "mu") or hasattr(self, "coupling")):
            feats = torch.from_numpy(edge_features.astype(np.float32))
            has = feats.norm(dim=1) > 0  # edges without features: structure only
            with torch.no_grad():
                logits[has] = logits[has] + self._mark_logits(feats[has])
        return logits.argmax(dim=1).numpy()

    def concentrations(self) -> Optional[np.ndarray]:
        """Learned per-community vMF concentrations (MarkBP v2), if any."""
        if not (self.learn_kappa and hasattr(self, "rho")):
            return None
        return F_nn.softplus(self.rho.detach()).numpy()
