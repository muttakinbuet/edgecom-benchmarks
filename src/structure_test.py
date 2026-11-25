"""
Label-free check of whether graph structure carries community signal.

Two structure-only models are fitted on the same held-out edge split: NOCD with
K communities and the same model with a single community, which can express
only how strongly each node connects (its degree), not communities. The
structure gain is the relative reduction of the held-out Bernoulli-Poisson loss
that K communities achieve over one:

    gain = (loss_1 - loss_K) / loss_1

A gain close to zero means communities do not help predict which edges exist,
so a structure-only fallback cannot recover communities either.
"""

import numpy as np

try:
    from .ocd_models import OverlappingCommunityModel
except ImportError:  # Fallback for direct script execution
    from ocd_models import OverlappingCommunityModel


def structure_gain(num_nodes: int, edge_index: np.ndarray, structure_model, seed: int,
                   **model_kwargs) -> dict:
    """structure_model: the fitted K-community NOCD model (provides best_val)."""
    single = OverlappingCommunityModel(1, seed=seed, refit=False, **model_kwargs)
    single.fit(num_nodes, edge_index)
    loss_k, loss_1 = float(structure_model.best_val), float(single.best_val)
    return {"loss_k": loss_k, "loss_1": loss_1,
            "structure_gain": (loss_1 - loss_k) / loss_1}
