#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-checking test for the cone-beam phase-contrast pipeline
(``toupy.tomo.cone_phase``).

Runs as a plain script::

    python test/cone_phase_test.py

and is also discoverable by pytest (the ``test_*`` functions use ``assert``).
A small phantom keeps the runtime to a few seconds on CPU.
"""

import os
import sys

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

# Prefer the local package over any installed toupy (matches tutorial scripts).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toupy.simulation import phantom3d
from toupy.tomo import (ConeBeamGeometry, cone_project, fdk_reconstruct,
                        cone_phase_retrieval_fdk, effective_fresnel_distance)

# ---------------------------------------------------------------------------
# Shared small fixture (built once)
# ---------------------------------------------------------------------------
WAVELENGTH = 1.23984193e-6 / (17.0 * 1e3)
DELTA_BETA = 50.0
_N, _NV, _NANG = 48, 16, 60
_MARGIN, _ZMARGIN = 7, 3


def _build():
    """Return (geometry, ground-truth volume, clean projected phase, intensity)."""
    core, core_z = _N - 2 * _MARGIN, _NV - 2 * _ZMARGIN
    vc = phantom3d(N=core, n_v=core_z, phantom_type="Modified Shepp-Logan")
    vol = np.zeros((_NV, _N, _N))
    zo, o = (_NV - core_z) // 2, (_N - core) // 2
    vol[zo:zo + core_z, o:o + core, o:o + core] = vc

    geom = ConeBeamGeometry(
        SOD=0.15, SDD=0.30, det_pixel_size=2e-6, n_u=_N, n_v=_NV,
        angles=np.linspace(0, 360, _NANG, endpoint=False),
    )
    proj = cone_project(vol, geom)
    proj *= 1.0 / proj.max()                       # peak projected phase ~1 rad

    z_eff = effective_fresnel_distance(geom)
    px_eff = geom.effective_pixel_size
    c = (-1.0 / DELTA_BETA + 1j)

    def fresnel(phi, pad=2):
        nv, nu = phi.shape
        my, mx = nv * pad, nu * pad
        oy, ox = (my - nv) // 2, (mx - nu) // 2
        fy, fx = fftfreq(my, d=px_eff), fftfreq(mx, d=px_eff)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        H = np.exp(1j * np.pi * WAVELENGTH * z_eff * (FY**2 + FX**2))
        field = np.ones((my, mx), dtype=complex)
        field[oy:oy + nv, ox:ox + nu] = np.exp(c * phi)
        return (np.abs(ifft2(fft2(field) * H))**2)[oy:oy + nv, ox:ox + nu]

    rng = np.random.default_rng(0)
    I = np.array([fresnel(proj[k]) for k in range(_NANG)])
    I = np.maximum(I + rng.normal(0, 0.005 * I.mean(), I.shape), 0)
    return geom, vol, proj, I


def _nrm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-30)


def _rmse(a, b):
    return float(np.sqrt(np.mean((_nrm(a) - _nrm(b))**2)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_effective_fresnel_distance():
    geom = ConeBeamGeometry(SOD=0.15, SDD=0.30, det_pixel_size=2e-6,
                            n_u=8, n_v=8, angles=np.array([0.0]))
    R1, R2 = 0.15, 0.15
    assert np.isclose(effective_fresnel_distance(geom), R1 * R2 / (R1 + R2))
    assert np.isclose(geom.magnification, 2.0)
    assert np.isclose(geom.effective_pixel_size, 1e-6)


def test_pipeline_shapes_and_recovery():
    geom, vol, proj, I = _build()

    volume, phase = cone_phase_retrieval_fdk(
        I, geom, WAVELENGTH, DELTA_BETA, n_iter=60, reg_tv=2e-3,
        return_phase=True)

    # shapes
    assert volume.shape == (_NV, _N, _N)
    assert phase.shape == I.shape

    # retrieved projected phase tracks the clean projected phase
    corr = np.corrcoef(phase.ravel(), proj.ravel())[0, 1]
    assert corr > 0.9, "retrieved-vs-clean projection corr too low: {:.3f}".format(corr)

    # the retrieval must not degrade the reconstruction beyond the FDK floor
    rmse_clean = _rmse(fdk_reconstruct(proj, geom), vol)
    rmse_iter = _rmse(volume, vol)
    assert rmse_iter < rmse_clean + 0.05, \
        "iterative RMSE {:.3f} far above clean-phase floor {:.3f}".format(
            rmse_iter, rmse_clean)


def test_shape_mismatch_raises():
    geom = ConeBeamGeometry(SOD=0.15, SDD=0.30, det_pixel_size=2e-6,
                            n_u=16, n_v=8, angles=np.linspace(0, 360, 4,
                                                              endpoint=False))
    bad = np.ones((4, 8, 99))            # wrong n_u
    try:
        cone_phase_retrieval_fdk(bad, geom, WAVELENGTH, DELTA_BETA, n_iter=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  {}".format(name))
            except Exception as exc:                      # noqa: BLE001
                failures += 1
                print("FAIL  {}: {}".format(name, exc))
    print("\n{} passed, {} failed".format(
        sum(1 for n in globals() if n.startswith("test_")) - failures, failures))
    raise SystemExit(1 if failures else 0)
