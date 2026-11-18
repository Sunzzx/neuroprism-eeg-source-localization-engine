"""Tests for Neuroprism EEG source/dipole localization."""

from __future__ import annotations

import numpy as np
import pytest

from neuroprism import (
    SphereHeadModel,
    fit_equivalent_current_dipole,
    forward_matrix,
    inverse_mne,
    inverse_sloreta,
    leadfield_dipole,
    localization_error,
    simulate_eeg_from_dipole,
    standard_10_20_electrodes,
)


def test_electrodes_on_sphere():
    pos, labels = standard_10_20_electrodes(0.09)
    assert pos.shape == (len(labels), 3)
    norms = np.linalg.norm(pos, axis=1)
    assert np.allclose(norms, 0.09, rtol=1e-6)


def test_leadfield_finite_and_avg_ref():
    ep, _ = standard_10_20_electrodes()
    V = leadfield_dipole(ep, np.array([0.0, 0.0, 0.04]), np.array([0.0, 0.0, 1e-7]))
    assert V.shape[0] == ep.shape[0]
    assert np.all(np.isfinite(V))
    assert abs(np.mean(V)) < 1e-12


def test_forward_matrix_shape():
    ep, _ = standard_10_20_electrodes()
    sp = np.array([[0.0, 0.0, 0.03], [0.02, 0.0, 0.03]])
    L = forward_matrix(ep, sp)
    assert L.shape == (ep.shape[0], 6)


def test_ecd_recovers_dipole_near_true():
    true_pos = np.array([0.02, -0.01, 0.045])
    true_m = np.array([0.0, 0.0, 5e-7])
    V, ep, _ = simulate_eeg_from_dipole(true_pos, true_m, noise_std=0.0)
    fit = fit_equivalent_current_dipole(V, ep, max_iter=50, seed=1)
    err = localization_error(true_pos, fit.position)
    assert fit.gof > 0.85
    assert err < 0.035  # < 3.5 cm on 9 cm head


def test_mne_peak_near_true():
    true_pos = np.array([0.0, 0.03, 0.04])
    true_m = np.array([1e-7, 0.0, 2e-7])
    V, ep, _ = simulate_eeg_from_dipole(true_pos, true_m)
    res = inverse_mne(V, ep, snr=5.0)
    err = localization_error(true_pos, res.peak_position)
    assert res.power[res.peak_index] == np.max(res.power)
    assert err < 0.05


def test_sloreta_peak_near_true():
    true_pos = np.array([-0.02, 0.01, 0.05])
    true_m = np.array([0.0, 1e-7, 1e-7])
    V, ep, _ = simulate_eeg_from_dipole(true_pos, true_m)
    res = inverse_sloreta(V, ep, snr=5.0)
    err = localization_error(true_pos, res.peak_position)
    assert err < 0.055
    assert res.method == "sLORETA"


def test_noise_degrades_gof_gracefully():
    true_pos = np.array([0.01, 0.01, 0.04])
    true_m = np.array([0.0, 0.0, 4e-7])
    V0, ep, _ = simulate_eeg_from_dipole(true_pos, true_m, noise_std=0.0)
    scale = float(np.std(V0) + 1e-15)
    Vn, _, _ = simulate_eeg_from_dipole(true_pos, true_m, noise_std=0.5 * scale, seed=2)
    fit0 = fit_equivalent_current_dipole(V0, ep, seed=0)
    fitn = fit_equivalent_current_dipole(Vn, ep, seed=0)
    assert fit0.gof >= fitn.gof - 1e-6
