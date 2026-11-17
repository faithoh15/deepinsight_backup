"""Euclidean clustering (verbatim from proposed_pipeline/cluster_windows.py: euclid_clusters)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def euclid_clusters(P, tol, k=12):
    """Connected components of the 'distance <= tol' graph. tol: scalar or per-point array (edge if d <= max(tol_i, tol_j)).
    Each point is linked to its k nearest neighbours within tol (memory O(N k) even where points are 2 cm apart, e.g.
    next to the device); for clustering this is equivalent to the full radius graph."""
    n = len(P)
    if n == 0:
        return np.zeros(0, int)
    t = cKDTree(P)
    tmax = float(np.max(tol))
    d, j = t.query(P, k=min(k + 1, n), distance_upper_bound=tmax)
    d, j = d[:, 1:], j[:, 1:]
    i = np.repeat(np.arange(n), d.shape[1])
    d, j = d.ravel(), j.ravel()
    ok = np.isfinite(d)
    if np.ndim(tol):
        ok &= d <= np.maximum(np.asarray(tol)[i], np.asarray(tol)[np.minimum(j, n - 1)])
    g = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    return connected_components(g, directed=False)[1]
