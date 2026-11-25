"""
Disk cache for k-means on edge features.

Single-threaded k-means (needed for reproducibility) is the slowest step on
DBLP-EF, and several methods cluster the same features with the same seed.
Results are cached under a key derived from the data itself, the number of
clusters, the seed and n_init, so a cache hit returns exactly what KMeans would.

Set EDGECOM_KMEANS_CACHE to change the cache folder (default data/cache/kmeans).
"""

import hashlib
import os
from pathlib import Path

import numpy as np


def cached_kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 3):
    """Return (centers, labels) of KMeans(k, n_init, random_state=seed) on x."""
    x = np.ascontiguousarray(x)
    digest = hashlib.sha1(x.tobytes()).hexdigest()[:16]
    folder = Path(os.environ.get("EDGECOM_KMEANS_CACHE", "data/cache/kmeans"))
    path = folder / f"{digest}_{x.shape[0]}x{x.shape[1]}_k{k}_s{seed}_n{n_init}.npz"
    if path.exists():
        cached = np.load(path)
        return cached["centers"], cached["labels"]

    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=n_init, random_state=seed).fit(x)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, centers=km.cluster_centers_, labels=km.labels_)
    os.replace(tmp, path)  # atomic, so parallel workers never read a partial file
    return km.cluster_centers_, km.labels_
