"""Neuroprism: EEG dipole & distributed source localization engine.

Implements:
  - 3-sphere analytic leadfield (Berg–Scherg / Ary approximation style)
  - Single equivalent current dipole (ECD) nonlinear least-squares fit
  - Distributed MNE (minimum-norm) and sLORETA inverse
  - Synthetic forward simulation + localization error metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class SphereHeadModel:
    """Three concentric spheres (brain, skull, scalp) in meters."""

    r_brain: float = 0.080
    r_skull: float = 0.085
    r_scalp: float = 0.090
    sigma_brain: float = 0.33
    sigma_skull: float = 0.0042  # ~1/80 of brain
    sigma_scalp: float = 0.33


def standard_10_20_electrodes(radius: float = 0.090) -> Tuple[np.ndarray, list[str]]:
    """Approximate 10–20 electrode positions on a sphere (unit sphere scaled).

    Returns positions (n_elec, 3) and labels.
    """
    # Spherical coords: theta = azimuth, phi = inclination from +z
    # Simplified 19-channel montage
    montage = {
        "Fp1": ( -0.3, 0.9),
        "Fp2": (  0.3, 0.9),
        "F7":  ( -0.8, 0.5),
        "F3":  ( -0.4, 0.55),
        "Fz":  (  0.0, 0.6),
        "F4":  (  0.4, 0.55),
        "F8":  (  0.8, 0.5),
        "T3":  ( -1.0, 0.0),
        "C3":  ( -0.5, 0.0),
        "Cz":  (  0.0, 0.0),
        "C4":  (  0.5, 0.0),
        "T4":  (  1.0, 0.0),
        "T5":  ( -0.8, -0.5),
        "P3":  ( -0.4, -0.55),
        "Pz":  (  0.0, -0.6),
        "P4":  (  0.4, -0.55),
        "T6":  (  0.8, -0.5),
        "O1":  ( -0.3, -0.9),
        "O2":  (  0.3, -0.9),
    }
    labels = list(montage.keys())
    pos = []
    for az, el in montage.values():
        # map az,el in [-1,1] to spherical
        theta = az * (np.pi / 2)  # azimuth
        phi = (1.0 - el) * (np.pi / 2) * 0.5 + 0.15  # keep upper hemisphere-ish
        # Better: treat as (x,y) on unit disk then lift to sphere
        x, y = az, el
        r_xy = np.sqrt(x * x + y * y)
        if r_xy > 1.0:
            x, y = x / r_xy, y / r_xy
            r_xy = 1.0
        z = np.sqrt(max(0.0, 1.0 - r_xy * r_xy))
        p = np.array([x, y, z], dtype=float)
        p = p / (np.linalg.norm(p) + 1e-30) * radius
        pos.append(p)
    return np.asarray(pos, dtype=float), labels


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def _berg_scherg_factors(model: SphereHeadModel) -> Tuple[float, float]:
    """Return approximate radial/tangential scaling for 3-sphere (heuristic).

    Full series expansion is heavy; we use a conductivity-weighted scale that
    preserves qualitative skull attenuation (~order 0.2–0.5 radial gain).
    """
    sb, ss, sc = model.sigma_brain, model.sigma_skull, model.sigma_scalp
    # skull attenuation factor
    atten = (ss / sb) ** 0.5 * (model.r_brain / model.r_scalp) ** 2
    radial_gain = float(np.clip(0.15 + 0.7 * atten / (atten + 0.05), 0.05, 1.0))
    tang_gain = radial_gain * 0.7
    return radial_gain, tang_gain


def leadfield_dipole(
    electrode_pos: np.ndarray,
    dipole_pos: np.ndarray,
    dipole_moment: np.ndarray,
    model: Optional[SphereHeadModel] = None,
) -> np.ndarray:
    """Analytic infinite-medium dipole field with 3-sphere gain correction.

    V_i = (1/(4 pi sigma)) * (m · r_hat) / r^2   with r = electrode - dipole,
    then multiply by Berg–Scherg-like radial/tangential gains.
    """
    model = model or SphereHeadModel()
    sigma = model.sigma_brain
    g_r, g_t = _berg_scherg_factors(model)
    ep = np.asarray(electrode_pos, dtype=float)
    rp = np.asarray(dipole_pos, dtype=float).reshape(3)
    m = np.asarray(dipole_moment, dtype=float).reshape(3)

    # Ensure dipole inside brain
    r_norm = np.linalg.norm(rp)
    if r_norm >= model.r_brain * 0.99:
        rp = rp * (model.r_brain * 0.95 / (r_norm + 1e-30))

    R = ep - rp  # (n, 3)
    r = np.linalg.norm(R, axis=1) + 1e-12
    Rhat = R / r[:, None]

    # Decompose moment into radial (w.r.t. dipole location from origin) and tangential
    if r_norm < 1e-9:
        m_rad = m
        m_tang = np.zeros(3)
        radial_dir = np.array([0.0, 0.0, 1.0])
    else:
        radial_dir = rp / r_norm
        m_rad_comp = np.dot(m, radial_dir) * radial_dir
        m_tang = m - m_rad_comp
        m_rad = m_rad_comp

    m_eff = g_r * m_rad + g_t * m_tang
    # Infinite homogeneous sphere-ish: V = m·R / (4πσ r^3)
    V = (R @ m_eff) / (4.0 * np.pi * sigma * (r**3))
    # Average-reference
    V = V - np.mean(V)
    return V.astype(float)


def forward_matrix(
    electrode_pos: np.ndarray,
    source_pos: np.ndarray,
    model: Optional[SphereHeadModel] = None,
) -> np.ndarray:
    """Build leadfield L of shape (n_elec, n_src * 3) for free-orientation sources."""
    model = model or SphereHeadModel()
    ep = np.asarray(electrode_pos, dtype=float)
    sp = np.asarray(source_pos, dtype=float)
    if sp.ndim == 1:
        sp = sp.reshape(1, 3)
    n_e = ep.shape[0]
    n_s = sp.shape[0]
    L = np.zeros((n_e, n_s * 3), dtype=float)
    eye = np.eye(3)
    for j in range(n_s):
        for k in range(3):
            L[:, 3 * j + k] = leadfield_dipole(ep, sp[j], eye[k], model)
    return L


# ---------------------------------------------------------------------------
# Inverse: ECD
# ---------------------------------------------------------------------------

@dataclass
class DipoleFitResult:
    position: np.ndarray
    moment: np.ndarray
    gof: float  # goodness of fit in [0,1]
    residual_var: float
    n_iter: int


def fit_equivalent_current_dipole(
    data: np.ndarray,
    electrode_pos: np.ndarray,
    model: Optional[SphereHeadModel] = None,
    n_restarts: int = 8,
    max_iter: int = 40,
    seed: int = 0,
) -> DipoleFitResult:
    """Nonlinear least-squares ECD fit (position on grid + closed-form moment)."""
    model = model or SphereHeadModel()
    y = np.asarray(data, dtype=float).ravel()
    y = y - np.mean(y)
    ep = np.asarray(electrode_pos, dtype=float)
    rng = np.random.default_rng(seed)

    # Candidate positions: spherical grid inside brain
    radii = np.linspace(0.01, model.r_brain * 0.85, 5)
    thetas = np.linspace(0, np.pi, 6)
    phis = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    grid = []
    for r in radii:
        for th in thetas:
            for ph in phis:
                grid.append([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
    grid = np.asarray(grid)

    def best_moment_and_err(pos: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        L = np.column_stack([
            leadfield_dipole(ep, pos, np.array([1.0, 0, 0]), model),
            leadfield_dipole(ep, pos, np.array([0, 1.0, 0]), model),
            leadfield_dipole(ep, pos, np.array([0, 0, 1.0]), model),
        ])
        # least squares moment
        m, *_ = np.linalg.lstsq(L, y, rcond=None)
        pred = L @ m
        err = float(np.sum((y - pred) ** 2))
        return m, err, pred

    # coarse grid search
    best_pos = grid[0]
    best_m = np.zeros(3)
    best_err = np.inf
    best_pred = np.zeros_like(y)
    for pos in grid[:: max(1, len(grid) // 80)]:  # subsample
        m, err, pred = best_moment_and_err(pos)
        if err < best_err:
            best_err, best_pos, best_m, best_pred = err, pos.copy(), m, pred

    # local refinement: random walk / coordinate descent
    step = 0.008
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        improved = False
        for d in range(3):
            for sgn in (-1.0, 1.0):
                cand = best_pos.copy()
                cand[d] += sgn * step
                if np.linalg.norm(cand) >= model.r_brain * 0.9:
                    continue
                m, err, pred = best_moment_and_err(cand)
                if err < best_err * 0.999:
                    best_err, best_pos, best_m, best_pred = err, cand, m, pred
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
        # occasional random restart nudge
        if it % 10 == 9:
            cand = best_pos + rng.normal(0, 0.005, size=3)
            if np.linalg.norm(cand) < model.r_brain * 0.9:
                m, err, pred = best_moment_and_err(cand)
                if err < best_err:
                    best_err, best_pos, best_m, best_pred = err, cand, m, pred

    ss_tot = float(np.sum(y**2)) + 1e-30
    gof = float(max(0.0, 1.0 - best_err / ss_tot))
    return DipoleFitResult(
        position=best_pos.astype(float),
        moment=best_m.astype(float),
        gof=gof,
        residual_var=float(best_err / ss_tot),
        n_iter=n_iter,
    )


# ---------------------------------------------------------------------------
# Distributed inverse: MNE & sLORETA
# ---------------------------------------------------------------------------

@dataclass
class DistributedInverseResult:
    method: str
    source_pos: np.ndarray
    moments: np.ndarray  # (n_src, 3) or (n_src,) if fixed orientation
    power: np.ndarray  # (n_src,)
    peak_index: int
    peak_position: np.ndarray


def _source_grid(model: SphereHeadModel, n_radial: int = 3, n_ang: int = 8) -> np.ndarray:
    radii = np.linspace(0.02, model.r_brain * 0.8, n_radial)
    pts = []
    for r in radii:
        # Fibonacci-ish sphere samples
        n = n_ang * n_ang
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r_xy = np.sqrt(max(0.0, 1 - z * z))
            phi = i * 2.399963  # golden angle
            pts.append([r * r_xy * np.cos(phi), r * r_xy * np.sin(phi), r * z])
    return np.asarray(pts, dtype=float)


def inverse_mne(
    data: np.ndarray,
    electrode_pos: np.ndarray,
    source_pos: Optional[np.ndarray] = None,
    model: Optional[SphereHeadModel] = None,
    snr: float = 3.0,
) -> DistributedInverseResult:
    """Tikhonov-regularized minimum-norm estimate (free orientation)."""
    model = model or SphereHeadModel()
    y = np.asarray(data, dtype=float).ravel()
    y = y - np.mean(y)
    ep = np.asarray(electrode_pos, dtype=float)
    sp = np.asarray(source_pos) if source_pos is not None else _source_grid(model)
    L = forward_matrix(ep, sp, model)  # (n_e, 3 n_s)
    # noise cov ~ I; source cov ~ I; lambda = trace(L L^T)/(n_e snr^2)
    llt = L @ L.T
    lam = float(np.trace(llt)) / (L.shape[0] * (snr**2) + 1e-30)
    # W = L.T @ inv(L L.T + lam I)
    A = llt + lam * np.eye(L.shape[0])
    try:
        x = L.T @ np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        x = L.T @ np.linalg.lstsq(A, y, rcond=None)[0]
    moments = x.reshape(-1, 3)
    power = np.sum(moments**2, axis=1)
    peak = int(np.argmax(power))
    return DistributedInverseResult(
        method="MNE",
        source_pos=sp,
        moments=moments,
        power=power,
        peak_index=peak,
        peak_position=sp[peak].copy(),
    )


def inverse_sloreta(
    data: np.ndarray,
    electrode_pos: np.ndarray,
    source_pos: Optional[np.ndarray] = None,
    model: Optional[SphereHeadModel] = None,
    snr: float = 3.0,
) -> DistributedInverseResult:
    """sLORETA: MNE standardized by resolution-matrix diagonal blocks."""
    model = model or SphereHeadModel()
    y = np.asarray(data, dtype=float).ravel()
    y = y - np.mean(y)
    ep = np.asarray(electrode_pos, dtype=float)
    sp = np.asarray(source_pos) if source_pos is not None else _source_grid(model)
    L = forward_matrix(ep, sp, model)
    llt = L @ L.T
    lam = float(np.trace(llt)) / (L.shape[0] * (snr**2) + 1e-30)
    A = llt + lam * np.eye(L.shape[0])
    # T = L.T @ inv(A)   operator from data to sources
    try:
        Ainv_L = np.linalg.solve(A, L)
    except np.linalg.LinAlgError:
        Ainv_L = np.linalg.lstsq(A, L, rcond=None)[0]
    T = L.T @ Ainv_L  # (3ns, 3ns) resolution-ish; actually L.T inv A L
    # MNE weights applied to y:
    try:
        x_mne = L.T @ np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        x_mne = L.T @ np.linalg.lstsq(A, y, rcond=None)[0]
    n_s = sp.shape[0]
    power = np.zeros(n_s)
    moments = x_mne.reshape(n_s, 3).copy()
    for j in range(n_s):
        sl = slice(3 * j, 3 * j + 3)
        Rjj = T[sl, sl]
        # standardize: m^T inv(Rjj) m
        try:
            power[j] = float(moments[j] @ np.linalg.solve(Rjj + 1e-12 * np.eye(3), moments[j]))
        except np.linalg.LinAlgError:
            power[j] = float(np.sum(moments[j] ** 2))
    peak = int(np.argmax(power))
    return DistributedInverseResult(
        method="sLORETA",
        source_pos=sp,
        moments=moments,
        power=power,
        peak_index=peak,
        peak_position=sp[peak].copy(),
    )


def localization_error(true_pos: np.ndarray, est_pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(true_pos) - np.asarray(est_pos)))


def simulate_eeg_from_dipole(
    dipole_pos: np.ndarray,
    dipole_moment: np.ndarray,
    electrode_pos: Optional[np.ndarray] = None,
    model: Optional[SphereHeadModel] = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Forward-simulate average-referenced EEG for one dipole."""
    model = model or SphereHeadModel()
    if electrode_pos is None:
        electrode_pos, labels = standard_10_20_electrodes(model.r_scalp)
    else:
        labels = [f"E{i}" for i in range(len(electrode_pos))]
        electrode_pos = np.asarray(electrode_pos, dtype=float)
    V = leadfield_dipole(electrode_pos, dipole_pos, dipole_moment, model)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        V = V + rng.normal(0.0, noise_std, size=V.shape)
        V = V - np.mean(V)
    return V, electrode_pos, labels
