"""Map-only pose correction (docs/2026-11-16-pose-refine-design.md).

Correction field C(t): x_corr = R(t) (x_legacy - c(t)) + c(t) + u(t), c(t) = legacy device position at t; R (slerp) and u (linear) are
interpolated between graph nodes (every 2 s). Node k's submap is expressed relative to c_k; Y_k = [R_k | c_k + u_k] maps those local
legacy coordinates into the corrected frame. An edge (i, j, T) says Y_i^-1 Y_j = T (T maps j-local points onto i-local points).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_refine_config(path: str | None = None) -> dict:
    if path:
        return yaml.safe_load(open(path, encoding="utf-8"))
    from pipeline.common.config import load_config
    return load_config()["pose"]              # configs/pipeline.yaml pose (= pose_refine.yaml)


# ---------------------------------------------------------------------------------------------------------------- correction field
class PoseCorrection:
    def __init__(self, t_us, c, rotvec, u):
        self.t = np.asarray(t_us, np.int64)
        self.c = np.asarray(c, float).reshape(-1, 3)
        self.rv = np.asarray(rotvec, float).reshape(-1, 3)
        self.u = np.asarray(u, float).reshape(-1, 3)
        self._slerp = Slerp(self.t.astype(float), Rotation.from_rotvec(self.rv)) if len(self.t) > 1 else None

    @classmethod
    def identity(cls, t_us, c):
        n = len(t_us)
        return cls(t_us, c, np.zeros((n, 3)), np.zeros((n, 3)))

    def at(self, t_us):
        """-> R (N,3,3), u (N,3) at the given times (clamped to the node range)."""
        t = np.clip(np.asarray(t_us, float), self.t[0], self.t[-1])
        if self._slerp is None:
            return np.repeat(Rotation.from_rotvec(self.rv[0]).as_matrix()[None], len(t), 0), np.repeat(self.u[:1], len(t), 0)
        R = self._slerp(t).as_matrix()
        u = np.stack([np.interp(t, self.t.astype(float), self.u[:, k]) for k in range(3)], 1)
        return R, u

    def apply(self, X, t_us, dev):
        """X (N,3) legacy odometry coordinates measured at t_us (N,), dev (N,3) = legacy device position at those times."""
        X, dev = np.asarray(X, float), np.asarray(dev, float)
        if len(X) == 0:
            return X.copy()
        R, u = self.at(t_us)
        return np.einsum("nij,nj->ni", R, X - dev) + dev + u

    def node_transform(self, k):
        """4x4 rigid correction of node k: x -> R_k (x - c_k) + c_k + u_k (the field at t_k, applied to a whole cloud)."""
        R = Rotation.from_rotvec(self.rv[k]).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, self.c[k] + self.u[k] - R @ self.c[k]
        return T

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "t_us", "cx", "cy", "cz", "rx", "ry", "rz", "ux", "uy", "uz"])
            for k in range(len(self.t)):
                w.writerow([k, int(self.t[k])] + [f"{v:.9f}" for v in (*self.c[k], *self.rv[k], *self.u[k])])

    @classmethod
    def from_csv(cls, path):
        rows = list(csv.DictReader(open(path)))
        g = lambda ks: np.array([[float(r[k]) for k in ks] for r in rows])  # noqa: E731
        return cls(np.array([int(r["t_us"]) for r in rows], np.int64), g(["cx", "cy", "cz"]), g(["rx", "ry", "rz"]), g(["ux", "uy", "uz"]))


# ---------------------------------------------------------------------------------------------------------------- edges
@dataclass
class Edge:
    i: int
    j: int
    T: np.ndarray            # 4x4, Y_i^-1 Y_j measured
    W: np.ndarray            # 6x6 whitener of xi = [rotvec, translation] of T^-1 Y_i^-1 Y_j
    robust: bool             # loop edge: Cauchy reweighting, may be dropped as outlier
    kind: str                # icp | prior | loop


def plane_info(P, N):
    """Mean point-to-plane information (6x6, order rotation, translation) of source points P (local frame) with normals N."""
    J = np.hstack([np.cross(P, N), N])
    return J.T @ J / max(len(P), 1)


def edge_whitener(info_mean, n_eff, sigma_p):
    w, V = np.linalg.eigh(info_mean * n_eff)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T / sigma_p


def icp_edge(src, dst, T0, icfg):
    """6-DOF point-to-plane ICP of open3d point clouds (src -> dst, both with normals). -> (T, ok, stats incl. 'info' = mean
    point-to-plane information of the inlier source points with source-frame normals, overlap = inlier fraction within inlier_d,
    rmse = point-to-plane rms of the inliers)."""
    import open3d as o3d
    reg = o3d.pipelines.registration
    T = np.asarray(T0, float).copy()
    for d in icfg["radii"]:
        r = reg.registration_icp(src, dst, d, T, reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d)),
                                 reg.ICPConvergenceCriteria(max_iteration=icfg["iters"]))
        T = np.asarray(r.transformation)
    S = np.asarray(src.points)
    D, Nd = np.asarray(dst.points), np.asarray(dst.normals)
    St = S @ T[:3, :3].T + T[:3, 3]
    dist, idx = cKDTree(D).query(St, distance_upper_bound=icfg["inlier_d"])
    inl = np.isfinite(dist)
    overlap = float(inl.mean()) if len(S) else 0.0
    # point-to-plane rms of the inliers: independent of where the two voxelised views happened to sample the surfaces
    rmse = float(np.sqrt(np.mean(np.einsum("ij,ij->i", St[inl] - D[idx[inl]], Nd[idx[inl]]) ** 2))) if inl.any() else np.inf
    n_src = Nd[idx[inl]] @ T[:3, :3]                                  # target normals in the source frame
    info = plane_info(S[inl], n_src) if inl.any() else np.zeros((6, 6))
    ok = overlap >= icfg["min_overlap"] and rmse <= icfg["max_rmse"]
    return T, bool(ok), dict(overlap=overlap, rmse=rmse, n_corr=int(inl.sum()), info=info)


def loop_candidates(t_s, centres, min_gap_s, max_dist_m, max_per_node, existing):
    """Node pairs (i < j) at least min_gap_s apart in time and within max_dist_m in space, nearest first, at most max_per_node per node."""
    t_s, centres = np.asarray(t_s, float), np.asarray(centres, float)
    tree = cKDTree(centres)
    pairs = []
    for i, j in tree.query_pairs(max_dist_m):
        i, j = min(i, j), max(i, j)
        if t_s[j] - t_s[i] >= min_gap_s and (i, j) not in existing:
            pairs.append((float(np.linalg.norm(centres[i] - centres[j])), i, j))
    cnt = np.zeros(len(centres), int)
    out = []
    for _, i, j in sorted(pairs):
        if cnt[i] < max_per_node and cnt[j] < max_per_node:
            out.append((i, j))
            cnt[i] += 1
            cnt[j] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------- graph
def _Y(rv, u, c):
    Y = np.eye(4)
    Y[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    Y[:3, 3] = c + u
    return Y


def _edge_xi(e, Yi, Yj):
    E = np.linalg.inv(e.T) @ np.linalg.inv(Yi) @ Yj
    return np.r_[Rotation.from_matrix(E[:3, :3]).as_rotvec(), E[:3, 3]]


def solve_graph(c, edges, up, gcfg, x0=None):
    """Nodes with legacy centres c (K,3); node 0 fixed. up (K,3) = mean measured up direction in the legacy frame (or None).
    -> rotvec (K,3), u (K,3), stats (outliers = dropped loop edges (i, j), up = common up vector, cost)."""
    c = np.asarray(c, float)
    K = len(c)
    ng = 2 if up is not None else 0
    n_var = 6 * (K - 1) + ng
    if up is not None:
        up = np.asarray(up, float) / np.linalg.norm(up, axis=1, keepdims=True)
        m = up.mean(0)
        m /= np.linalg.norm(m)
        g0 = np.array([np.arctan2(m[1], m[0]), np.arcsin(np.clip(m[2], -1, 1))])
    x = np.zeros(n_var) if x0 is None else np.asarray(x0, float).copy()
    if up is not None and x0 is None:
        x[-2:] = g0
    sg = np.radians(gcfg["gravity_sigma_deg"])
    active = list(edges)
    outliers = []

    def unpack(x):
        rv = np.vstack([np.zeros(3), x[:6 * (K - 1)].reshape(-1, 6)[:, :3]])
        u = np.vstack([np.zeros(3), x[:6 * (K - 1)].reshape(-1, 6)[:, 3:]])
        return rv, u

    def U_of(x):
        a, b = x[-2], x[-1]
        return np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])

    def edge_res(x, es):
        rv, u = unpack(x)
        Ys = [_Y(rv[k], u[k], c[k]) for k in range(K)]
        return [e.W @ _edge_xi(e, Ys[e.i], Ys[e.j]) for e in es]

    def solve(es, wts, x):
        def fun(x):
            r = [np.sqrt(w) * v for w, v in zip(wts, edge_res(x, es))]
            if up is not None:
                rv, _ = unpack(x)
                U = U_of(x)
                r.append(((Rotation.from_rotvec(rv).apply(up) - U) / sg).ravel())
            return np.concatenate(r)
        n_res = 6 * len(es) + (3 * K if up is not None else 0)
        sp = lil_matrix((n_res, n_var), dtype=int)
        for q, e in enumerate(es):
            for k in (e.i, e.j):
                if k > 0:
                    sp[6 * q:6 * q + 6, 6 * (k - 1):6 * k] = 1
        if up is not None:
            for k in range(K):
                r0 = 6 * len(es) + 3 * k
                if k > 0:
                    sp[r0:r0 + 3, 6 * (k - 1):6 * (k - 1) + 3] = 1
                sp[r0:r0 + 3, n_var - 2:] = 1
        res = least_squares(fun, x, jac_sparsity=sp, method="trf", x_scale="jac", max_nfev=200)
        return res.x, float(res.cost)

    s = gcfg["robust_scale"]
    while True:
        wts = np.ones(len(active))
        for _ in range(gcfg["irls_iters"]):
            x, cost = solve(active, wts, x)
            nr = np.array([np.linalg.norm(r) for r in edge_res(x, active)])
            wts = np.array([1.0 / (1.0 + (n / s) ** 2) if e.robust else 1.0 for e, n in zip(active, nr)])
        nr = np.array([np.linalg.norm(r) for r in edge_res(x, active)])
        bad = [q for q, (e, n) in enumerate(zip(active, nr)) if e.robust and n > gcfg["outlier_whitened"]]
        if not bad:
            break
        worst = max(bad, key=lambda q: nr[q])                         # drop the worst loop, solve again
        outliers.append((active[worst].i, active[worst].j))
        active = [e for q, e in enumerate(active) if q != worst]
    rv, u = unpack(x)
    nr = np.array([np.linalg.norm(r) for r in edge_res(x, active)])
    return rv, u, dict(outliers=outliers, up=U_of(x) if up is not None else None, cost=cost, x=x,
                       edge_norm={(e.i, e.j, e.kind): float(n) for e, n in zip(active, nr)})


# ---------------------------------------------------------------------------------------------------------------- comparison figures
def slice_occupancy(xyz, h, extent, h_lo, h_hi, px):
    """Occupied top-view cells (bool, rows from y1 downwards) of the points with height above floor in (h_lo, h_hi]. A wall seen twice at
    slightly different places occupies more cells than one crisp wall."""
    x0, y0, x1, y1 = extent
    W, H = int((x1 - x0) / px) + 1, int((y1 - y0) / px) + 1
    occ = np.zeros((H, W), bool)
    m = (np.asarray(h) > h_lo) & (np.asarray(h) <= h_hi)
    u = ((xyz[m, 0] - x0) / px).astype(int)
    v = ((y1 - xyz[m, 1]) / px).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    occ[v[ok], u[ok]] = True
    return occ

