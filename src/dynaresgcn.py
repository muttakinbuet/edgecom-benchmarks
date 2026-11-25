"""
DynaResGCN Baseline

Deep residual GCN with dynamic dilated aggregation (DynaResGCN, 2022), from the
official code (the Deep-DynaResGCN-community repository on GitHub, cloned to
external/dynaresgcn). The encoder class GCNRes is used unchanged, with the
settings of the official run-deep.py: 3 layers of 128 units, batch norm, no
dropout, dilation d = 2, neighbourhood augmentation with one random two-hop
neighbour per neighbour.

Three adaptations, none of which changes the model:
* augment_graph_neighbors_with_degree densifies the adjacency (N x N), which
  does not fit for DBLP-EF; augment() below is the same procedure on a sparse
  graph.
* networkx 3 removed to_scipy_sparse_matrix; it is mapped to
  to_scipy_sparse_array, converted to the scipy matrix type the code expects.
* Tensors stay on the CPU.
Training uses our protocol (held-out edges, early stopping) like every other
NOCD-based method, so the encoder is the only difference from NOCD.
"""

import random
import sys
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent / "external" / "dynaresgcn"

if not hasattr(nx, "to_scipy_sparse_matrix"):
    nx.to_scipy_sparse_matrix = lambda g, nodelist=None, **kw: \
        sp.csr_matrix(nx.to_scipy_sparse_array(g, nodelist=nodelist, **kw))


def _official():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import nocd.nn.models_deep as models_deep
    import nocd.utils
    cpu = partial(nocd.utils.to_sparse_tensor, cuda=False)
    models_deep.to_sparse_tensor = cpu
    nocd.utils.to_sparse_tensor = cpu
    return models_deep


def augment(num_nodes: int, edge_index: np.ndarray) -> nx.Graph:
    """augment_graph_neighbors_with_degree(adj, 2) of the official code, sparse.

    For each node i and each original neighbour n, add one random neighbour of n
    that is not yet a neighbour of i.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    g.remove_edges_from(nx.selfloop_edges(g))
    g_old = g.copy()
    for i in range(num_nodes):
        old_neighbors = list(g_old.neighbors(i))
        neighbors = set(old_neighbors)
        for n in old_neighbors:
            n_neighbors = set(g_old.neighbors(n)) - neighbors
            if n_neighbors:
                neighbors.add(random.choice(list(n_neighbors)))
        for j in neighbors - set(old_neighbors):
            g.add_edge(i, j)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


class DynaResGCNEncoder(nn.Module):
    """Official GCNRes behind our encoder interface forward(x, adj)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, edge_index,
                 num_layers=3, d=2, seed=0):
        super().__init__()
        models_deep = _official()
        random.seed(seed)       # augmentation
        np.random.seed(seed)    # dilation (fixed per layer at construction)
        g = augment(num_nodes, edge_index)
        self.gnn = models_deep.GCNRes(in_dim, hidden_dim, out_dim, dropout=0,
                                      batch_norm=True, layer_num=num_layers, g=g, d=d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        # The official model ignores the plain adjacency: every layer has its
        # own dilated, augmented graph
        return torch.relu(self.gnn(x))
