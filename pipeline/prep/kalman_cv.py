"""Constant-velocity Kalman filter of the 2D polar tracker (verbatim from proposed_pipeline/track_3d.py: KalmanCV, _F_Q)."""
from __future__ import annotations

import math

import numpy as np


# ======================================================================================================================
# Kalman filter
# ======================================================================================================================
def _F_Q(dt, sigma_a, model, tau):
    """State transition and process noise for dt seconds. cv: constant velocity. damped: velocity decays with time
    constant tau (first-order Markov), F and Q from the SAME model (Q integrated numerically)."""
    if model == "cv" or dt <= 0:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q11, q12, q22 = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = sigma_a ** 2 * np.array([[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]])
        return F, Q
    a = math.exp(-dt / tau)
    b = tau * (1 - a)
    F = np.array([[1, 0, b, 0], [0, 1, 0, b], [0, 0, a, 0], [0, 0, 0, a]], float)
    # Q = ∫_0^dt F(dt - s) G G^T F(dt - s)^T ds, G = [0, 0, 1, 1]-wise acceleration input (per axis)
    n = max(8, int(dt / 0.05))
    s = np.linspace(0, dt, n + 1)
    Qa = np.zeros((2, 2))
    for k in range(n):
        for sm in ((s[k] + s[k + 1]) / 2,):
            r = dt - sm
            ar = math.exp(-r / tau)
            g = np.array([tau * (1 - ar), ar])        # effect of a unit acceleration impulse at time sm on [pos, vel]
            Qa += np.outer(g, g) * (s[k + 1] - s[k])
    Qa *= sigma_a ** 2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Qa
    Q[np.ix_([1, 3], [1, 3])] = Qa
    return F, Q


class KalmanCV:
    def __init__(self, xy, sigma_xy, v0_sigma, P=None):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([sigma_xy ** 2, sigma_xy ** 2, v0_sigma ** 2, v0_sigma ** 2]) if P is None else P.copy()

    def copy(self):
        k = KalmanCV(self.x[:2], 1.0, 1.0)
        k.x, k.P = self.x.copy(), self.P.copy()
        return k

    def predict(self, dt, sigma_a, model="cv", tau=4.0):
        if dt <= 0:
            return
        F, Q = _F_Q(dt, sigma_a, model, tau)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation_xy(self, z, R, sigma_cap=None):
        """nu, S, d² for a position measurement; sigma_cap limits the PREDICTED position uncertainty used for gating
        (the internal P is not modified)."""
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        Ppos = self.P[:2, :2]
        if sigma_cap is not None:
            w, V = np.linalg.eigh(Ppos)
            w = np.minimum(w, sigma_cap ** 2)
            Ppos = V @ np.diag(w) @ V.T
        S = Ppos + R
        nu = np.asarray(z, float) - self.x[:2]
        d2 = float(nu @ np.linalg.solve(S, nu))
        return nu, S, d2

    def update_xy(self, z, R):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        nu = np.asarray(z, float) - self.x[:2]
        self.x = self.x + K @ nu
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return float(nu @ np.linalg.solve(S, nu))
