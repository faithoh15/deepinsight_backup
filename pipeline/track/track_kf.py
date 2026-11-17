"""Constant-velocity Kalman filter on the horizontal odometry plane (state x, y, vx, vy) and the RTS smoother."""
from __future__ import annotations

import numpy as np

H = np.hstack([np.eye(2), np.zeros((2, 2))])


def cv_F(dt_s: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt_s
    return F


def cv_Q(dt_s: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise acceleration: sigma_a^2 G G^T, G = [[dt^2/2, 0], [0, dt^2/2], [dt, 0], [0, dt]]."""
    G = np.array([[dt_s ** 2 / 2, 0.0], [0.0, dt_s ** 2 / 2], [dt_s, 0.0], [0.0, dt_s]])
    return sigma_a ** 2 * G @ G.T


class KF:
    def __init__(self, x: np.ndarray, P: np.ndarray, t_us: int):
        self.x, self.P, self.t_us = np.asarray(x, float).copy(), np.asarray(P, float).copy(), int(t_us)

    @classmethod
    def init(cls, z, sigma_z: float, v0, v0_sigma: float, t_us: int) -> "KF":
        return cls(np.r_[np.asarray(z, float), np.asarray(v0, float)], np.diag([sigma_z ** 2, sigma_z ** 2, v0_sigma ** 2, v0_sigma ** 2]), t_us)

    def copy(self) -> "KF":
        return KF(self.x, self.P, self.t_us)

    def predict_to(self, t_us: int, sigma_a: float) -> None:
        dt = (int(t_us) - self.t_us) / 1e6
        assert dt >= -1e-9, "a track state is never moved back in time"
        if dt > 0:
            F = cv_F(dt)
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + cv_Q(dt, sigma_a)
            self.t_us = int(t_us)

    def innovation(self, z, sigma_z: float):
        nu = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + sigma_z ** 2 * np.eye(2)
        return nu, S

    def d2(self, z, sigma_z: float, pos_sigma_max: float | None = None) -> float:
        """Mahalanobis d^2 of z; with pos_sigma_max the predicted position covariance is capped (eigenvalues <= pos_sigma_max^2) for
        gating only, P itself is not changed."""
        nu, S = self.innovation(z, sigma_z)
        if pos_sigma_max is not None:
            w, V = np.linalg.eigh(H @ self.P @ H.T)
            S = V @ np.diag(np.minimum(w, pos_sigma_max ** 2)) @ V.T + sigma_z ** 2 * np.eye(2)
        return float(nu @ np.linalg.solve(S, nu))

    def update(self, z, sigma_z: float) -> float:
        nu, S = self.innovation(z, sigma_z)
        nis = float(nu @ np.linalg.solve(S, nu))
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ (sigma_z ** 2 * np.eye(2)) @ K.T     # Joseph form
        return nis

    def sigma_pos(self) -> float:
        return float(np.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))


def run_filter(kf0: KF, steps: list, sigma_a: float) -> list:
    """steps: [(t_us, z or None, sigma_z or None)] in time order. -> [dict(t_us, x_f, P_f, x_p, P_p, F)], first = kf0 itself."""
    kf = kf0.copy()
    out = [dict(t_us=kf.t_us, x_f=kf.x.copy(), P_f=kf.P.copy(), x_p=kf.x.copy(), P_p=kf.P.copy(), F=np.eye(4))]
    for t, z, sz in steps:
        dt = (int(t) - kf.t_us) / 1e6
        kf.predict_to(t, sigma_a)
        x_p, P_p = kf.x.copy(), kf.P.copy()
        if z is not None:
            kf.update(z, sz)
        out.append(dict(t_us=int(t), x_f=kf.x.copy(), P_f=kf.P.copy(), x_p=x_p, P_p=P_p, F=cv_F(max(dt, 0.0))))
    return out


def rts(filtered: list) -> list:
    """Rauch-Tung-Striebel smoother over run_filter's output -> [(x_s, P_s)]."""
    n = len(filtered)
    xs, Ps = [None] * n, [None] * n
    xs[-1], Ps[-1] = filtered[-1]["x_f"].copy(), filtered[-1]["P_f"].copy()
    for k in range(n - 2, -1, -1):
        f, nxt = filtered[k], filtered[k + 1]
        C = f["P_f"] @ nxt["F"].T @ np.linalg.pinv(nxt["P_p"])
        xs[k] = f["x_f"] + C @ (xs[k + 1] - nxt["x_p"])
        Ps[k] = f["P_f"] + C @ (Ps[k + 1] - nxt["P_p"]) @ C.T
    return list(zip(xs, Ps))
