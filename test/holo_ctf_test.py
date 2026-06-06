#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-checking test for the magnified inline-holography CTF pipeline
(``toupy.restoration.holo_ctf``).

    python test/holo_ctf_test.py

Also pytest-discoverable (``test_*`` functions use ``assert``).
"""

import os
import sys

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import affine_transform, fourier_shift, gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toupy.simulation import phantom
from toupy.restoration import (holo_geometry, flat_field_correct,
                               eigenflat_correct, rescale_to_common_pixel,
                               align_holograms, holo_ctf_reconstruct)

WAVELENGTH = 1.23984193e-6 / (17.0 * 1e3)
DELTA_BETA = 50.0
ZD, DET_PX = 0.10, 1.0e-6
ZS = np.array([12.5, 16.7, 25.0, 40.0]) * 1e-3
N_DET, N_FINE = 96, 384


def test_geometry():
    g = holo_geometry(ZS, ZD, DET_PX)
    assert np.allclose(g.magnification, ZD / ZS)
    assert np.allclose(g.effective_pixel_size, DET_PX / g.magnification)
    assert np.allclose(g.effective_distance, ZS * (ZD - ZS) / ZD)
    # closest-to-focus position has the finest effective pixel
    assert np.argmin(g.effective_pixel_size) == 0


def test_flat_field():
    rng = np.random.default_rng(1)
    beam = 1.0 + 0.5 * rng.random((16, 16))
    dark = 10.0
    sample_true = 0.7 + 0.2 * rng.random((16, 16))
    S = beam * sample_true + dark
    R = beam + dark
    out = flat_field_correct(S, R, dark)
    assert np.allclose(out, sample_true, atol=1e-9)


def test_eigenflat_beats_simple_on_drift():
    # empty-beam transmission (no sample) -> a perfect flat-field should give 1
    rng = np.random.default_rng(2)
    M = 48
    yy, xx = np.mgrid[0:M, 0:M]
    beam0 = 1.0 + 0.4 * np.sin(2 * np.pi * xx / M) * np.cos(2 * np.pi * yy / M)
    mode = 0.2 * np.sin(2 * np.pi * yy / M * 2)
    dark = 5.0
    # flats at random drift states
    flats = np.array([beam0 + rng.normal(0, 1) * mode + dark for _ in range(8)])
    # a sample frame: empty beam at a *particular* drift state (no object -> T=1)
    sample = beam0 + 0.9 * mode + dark
    simple = flat_field_correct(sample, flats.mean(0), dark)
    eigen = eigenflat_correct(sample, flats, dark, n_components=3)
    # the eigen-corrected empty frame is flatter (closer to 1)
    assert eigen.std() < 0.5 * simple.std(), (eigen.std(), simple.std())


def test_align_recovers_shift():
    img = gaussian_filter(phantom(N_DET, "Modified Shepp-Logan"), 2.0)
    shift = np.array([1.4, -0.8])
    moved = np.real(ifft2(fourier_shift(fft2(img), shift)))
    _, est = align_holograms(np.array([img, moved]), reference_index=0,
                             upsample=20, blur=2.0, return_shifts=True)
    assert np.allclose(est[1], -shift, atol=0.2), est[1]


def _build_dataset():
    g = holo_geometry(ZS, ZD, DET_PX)
    px1 = g.effective_pixel_size.min()
    core = 60
    ph = phantom(core, "Modified Shepp-Logan") * 0.3
    phf = np.zeros((N_FINE, N_FINE)); o = (N_FINE - core) // 2
    phf[o:o + core, o:o + core] = ph
    c = (-1.0 / DELTA_BETA + 1j)

    def prop(psi, z, px, pad=2):
        n = psi.shape[0]; m = n * pad; off = (m - n) // 2
        f = fftfreq(m, d=px); FY, FX = np.meshgrid(f, f, indexing="ij")
        H = np.exp(1j * np.pi * WAVELENGTH * z * (FY**2 + FX**2))
        fld = np.ones((m, m), complex); fld[off:off + n, off:off + n] = psi
        return (np.abs(ifft2(fft2(fld) * H))**2)[off:off + n, off:off + n]

    def down(img, s):
        img = gaussian_filter(img, 0.5 * s) if s > 1 else img
        ni = img.shape[0]; off = (ni - 1) / 2.0 - s * (N_DET - 1) / 2.0
        return affine_transform(img, [s, s], offset=[off, off],
                                output_shape=(N_DET, N_DET), order=3, mode="nearest")

    shifts = np.array([[0, 0], [1.3, -0.7], [-0.8, 1.1], [0.6, 2.2]])
    beam = 1.0 + 0.2 * np.linspace(0, 1, N_FINE)[None, :]
    dark = 50.0
    S, R = [], []
    rng = np.random.default_rng(0)
    for i in range(4):
        fld = prop(np.exp(c * phf), g.effective_distance[i], px1)
        fld = np.real(ifft2(fourier_shift(fft2(fld), shifts[i])))
        s = g.effective_pixel_size[i] / px1
        bi = down(beam, s)
        S.append(bi * down(fld, s) + dark); R.append(bi + dark)
    S = np.array(S) + rng.normal(0, 0.003, (4, N_DET, N_DET))
    gt = phf[(N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2,
             (N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2]
    return np.array(S), np.array(R), dark, gt


def test_full_pipeline_recovers_phase():
    S, R, dark, gt = _build_dataset()
    phase = holo_ctf_reconstruct(S, R, dark, ZS, ZD, DET_PX, WAVELENGTH,
                                 alpha=1e-2, delta_beta=DELTA_BETA,
                                 align=True, align_blur=2.0)
    m = gt > 0.01 * gt.max()
    phase = phase - phase[~m].mean()
    corr = np.corrcoef(phase[m], gt[m])[0, 1]
    assert corr > 0.8, "full-pipeline phase correlation too low: {:.3f}".format(corr)


def test_nonlinear_and_refine_align():
    S, R, dark, gt = _build_dataset()
    m = gt > 0.01 * gt.max()

    def corr(method, refine=False):
        p = holo_ctf_reconstruct(S, R, dark, ZS, ZD, DET_PX, WAVELENGTH,
                                 alpha=1e-4, delta_beta=DELTA_BETA,
                                 method=method, refine_align=refine,
                                 nl_n_iter=80, nl_reg_tv=1e-3)
        p = p - p[~m].mean()
        return np.corrcoef(p[m], gt[m])[0, 1]

    # the non-linear refinement should not hurt, and recovers a good phase
    assert corr("nonlinear") >= corr("ctf") - 1e-3
    assert corr("nonlinear", refine=True) > 0.8


def test_alignment_helps():
    S, R, dark, gt = _build_dataset()
    m = gt > 0.01 * gt.max()

    def corr(align):
        p = holo_ctf_reconstruct(S, R, dark, ZS, ZD, DET_PX, WAVELENGTH,
                                 alpha=1e-2, delta_beta=DELTA_BETA, align=align)
        p = p - p[~m].mean()
        return np.corrcoef(p[m], gt[m])[0, 1]

    assert corr(True) >= corr(False) - 1e-6


if __name__ == "__main__":
    failures = 0
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
            print("PASS  {}".format(name))
        except Exception as exc:                          # noqa: BLE001
            failures += 1
            print("FAIL  {}: {}".format(name, exc))
    print("\n{} passed, {} failed".format(len(names) - failures, failures))
    raise SystemExit(1 if failures else 0)
