#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-way FSC comparison: FBP vs FBaP vs two-pass

Computes Fourier Shell Correlation resolution estimates for all three
reconstruction methods on the same dataset, using the half-dataset splits.

Prerequisites
-------------
Run these first to generate the half-dataset reconstructions:

  # Two-pass method (you already have these)
  python tutorial/twopass_real_data.py  # with FSC_HALF = 0
  python tutorial/twopass_real_data.py  # with FSC_HALF = 1

  # FBaP method (run these now)
  python tutorial/fbap_recon.py  # with FSC_HALF = 0
  python tutorial/fbap_recon.py  # with FSC_HALF = 1

Expected directory structure:
  twopass_real_figures_half0/twopass_reconstruction.npz
  twopass_real_figures_half1/twopass_reconstruction.npz
  twopass_real_figures_fbap_half0/fbap_reconstruction.npz
  twopass_real_figures_fbap_half1/fbap_reconstruction.npz

Usage
-----
  python tutorial/fsc_threeway_comparison.py

Outputs (in tutorial/fsc_threeway/)
-------
  fsc_curves.png        — FSC curves for all three methods
  fsc_resolution.txt    — resolution estimates at 0.5 and 0.143 thresholds
  fsc_data.npz          — raw FSC arrays for further analysis
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Input directories (half-dataset reconstructions)
TWOPASS_HALF0_DIR = os.path.join(_HERE, 'twopass_real_figures_half0')
TWOPASS_HALF1_DIR = os.path.join(_HERE, 'twopass_real_figures_half1')
FBAP_HALF0_DIR    = os.path.join(_HERE, 'twopass_real_figures_fbap_half0')
FBAP_HALF1_DIR    = os.path.join(_HERE, 'twopass_real_figures_fbap_half1')

# Output directory
OUT_DIR = os.path.join(_HERE, 'fsc_threeway')
os.makedirs(OUT_DIR, exist_ok=True)

# Physical parameters (from PXCTalignedprojections.npz metadata)
PIXEL_SIZE = 28.64e-9  # [m]

# ---------------------------------------------------------------------------
# FSC computation
# ---------------------------------------------------------------------------

def register_volumes(vol1, vol2, max_shift_px=5):
    """
    Register vol2 to vol1 via 3-D cross-correlation (phase correlation).

    Returns
    -------
    vol2_registered : ndarray
        Shifted vol2 aligned to vol1.
    shift : tuple (sz, sy, sx)
        Applied shift in pixels.
    """
    f1 = np.fft.fftn(vol1)
    f2 = np.fft.fftn(vol2)

    # Phase correlation
    cross_power = (f1 * f2.conj()) / (np.abs(f1 * f2.conj()) + 1e-10)
    xcorr = np.fft.ifftn(cross_power).real

    # Find peak
    peak_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    shift = np.array(peak_idx, dtype=float)

    # Wrap around for negative shifts
    for i in range(3):
        if shift[i] > xcorr.shape[i] // 2:
            shift[i] -= xcorr.shape[i]

    # Clamp to max_shift_px
    shift = np.clip(shift, -max_shift_px, max_shift_px)

    # Apply shift in Fourier domain
    vol2_reg = fourier_shift(np.fft.fftn(vol2), shift)
    vol2_reg = np.fft.ifftn(vol2_reg).real

    return vol2_reg, tuple(shift)


def compute_fsc(vol1, vol2, pixel_size, register=True):
    """
    Compute Fourier Shell Correlation between two volumes.

    Parameters
    ----------
    vol1, vol2 : ndarray (Nz, Ny, Nx)
        Reconstructed volumes (must have identical shape).
    pixel_size : float [m]
        Isotropic voxel size.
    register : bool
        If True, register vol2 to vol1 before FSC (corrects small shifts).

    Returns
    -------
    freq_nm : ndarray
        Spatial frequency [1/nm].
    fsc : ndarray
        FSC values (range [0, 1]).
    """
    assert vol1.shape == vol2.shape, "Volumes must have identical shape"

    if register:
        vol2, shift = register_volumes(vol1, vol2)
        print(f'    Registration shift: {shift} px')

    Nz, Ny, Nx = vol1.shape

    # Fourier transforms
    F1 = np.fft.fftn(vol1)
    F2 = np.fft.fftn(vol2)

    # Frequency grids
    fz = np.fft.fftfreq(Nz, d=pixel_size)
    fy = np.fft.fftfreq(Ny, d=pixel_size)
    fx = np.fft.fftfreq(Nx, d=pixel_size)
    FZ, FY, FX = np.meshgrid(fz, fy, fx, indexing='ij')
    freq_mag = np.sqrt(FZ**2 + FY**2 + FX**2)

    # Maximum frequency (Nyquist)
    f_nyquist = 0.5 / pixel_size
    n_shells = min(Nz, Ny, Nx) // 2
    freq_bins = np.linspace(0, f_nyquist, n_shells + 1)

    # Shell-average FSC
    fsc = np.zeros(n_shells)
    freq_centers = np.zeros(n_shells)

    for i in range(n_shells):
        mask = (freq_mag >= freq_bins[i]) & (freq_mag < freq_bins[i+1])

        if not mask.any():
            continue

        numerator   = np.sum((F1[mask] * F2[mask].conj()).real)
        denom1      = np.sum(np.abs(F1[mask])**2)
        denom2      = np.sum(np.abs(F2[mask])**2)

        if denom1 > 0 and denom2 > 0:
            fsc[i] = numerator / np.sqrt(denom1 * denom2)

        freq_centers[i] = 0.5 * (freq_bins[i] + freq_bins[i+1])

    # Convert frequency to 1/nm
    freq_nm = freq_centers * 1e-9

    return freq_nm, fsc


def estimate_resolution(freq_nm, fsc, threshold=0.5):
    """
    Estimate resolution from FSC curve at a given threshold.

    Returns
    -------
    resolution_nm : float
        Spatial resolution [nm] (1/frequency at threshold crossing).
    """
    # Find first crossing below threshold
    idx = np.where(fsc < threshold)[0]

    if len(idx) == 0:
        return np.inf  # No crossing (FSC always above threshold)

    i_cross = idx[0]

    if i_cross == 0:
        return np.nan  # FSC starts below threshold

    # Linear interpolation between i_cross-1 and i_cross
    f0, fsc0 = freq_nm[i_cross - 1], fsc[i_cross - 1]
    f1, fsc1 = freq_nm[i_cross], fsc[i_cross]

    f_thresh = f0 + (f1 - f0) * (threshold - fsc0) / (fsc1 - fsc0)

    if f_thresh > 0:
        return 1.0 / f_thresh  # [nm]
    else:
        return np.inf


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

print('='*70)
print('Three-way FSC comparison: FBP vs FBaP vs two-pass')
print('='*70)

# ── 1. Load half-dataset volumes ───────────────────────────────────────────

print('\n1. Loading half-dataset reconstructions ...\n')

methods = ['FBP', 'FBaP', 'Two-pass']
volumes = {}

for method in methods:
    if method == 'FBP':
        # FBP stored inside twopass_reconstruction.npz
        path0 = os.path.join(TWOPASS_HALF0_DIR, 'twopass_reconstruction.npz')
        path1 = os.path.join(TWOPASS_HALF1_DIR, 'twopass_reconstruction.npz')
        key = 'delta_fbp'
    elif method == 'FBaP':
        path0 = os.path.join(FBAP_HALF0_DIR, 'fbap_reconstruction.npz')
        path1 = os.path.join(FBAP_HALF1_DIR, 'fbap_reconstruction.npz')
        key = 'delta_fbap'
    else:  # Two-pass
        path0 = os.path.join(TWOPASS_HALF0_DIR, 'twopass_reconstruction.npz')
        path1 = os.path.join(TWOPASS_HALF1_DIR, 'twopass_reconstruction.npz')
        key = 'delta_tp'

    if not os.path.exists(path0):
        print(f'  ✗ {method} half-0: {path0} not found')
        sys.exit(1)
    if not os.path.exists(path1):
        print(f'  ✗ {method} half-1: {path1} not found')
        sys.exit(1)

    data0 = np.load(path0, allow_pickle=True)
    data1 = np.load(path1, allow_pickle=True)

    vol0 = data0[key]
    vol1 = data1[key]

    print(f'  ✓ {method:10s}  half-0: {vol0.shape}  half-1: {vol1.shape}')

    volumes[method] = (vol0, vol1)

# ── 2. Compute FSC for each method ─────────────────────────────────────────

print('\n2. Computing FSC curves ...\n')

fsc_results = {}

for method in methods:
    print(f'  {method} ...')
    vol0, vol1 = volumes[method]
    freq_nm, fsc = compute_fsc(vol0, vol1, PIXEL_SIZE, register=True)
    fsc_results[method] = (freq_nm, fsc)

# ── 3. Estimate resolution at 0.5 and 0.143 thresholds ─────────────────────

print('\n3. Resolution estimates:\n')

resolutions = {}

for method in methods:
    freq_nm, fsc = fsc_results[method]
    res_50  = estimate_resolution(freq_nm, fsc, threshold=0.5)
    res_143 = estimate_resolution(freq_nm, fsc, threshold=0.143)
    resolutions[method] = (res_50, res_143)

    print(f'  {method:10s}  FSC=0.5: {res_50:6.2f} nm    FSC=0.143: {res_143:6.2f} nm')

# ── 4. Save resolution table ───────────────────────────────────────────────

res_file = os.path.join(OUT_DIR, 'fsc_resolution.txt')
with open(res_file, 'w') as f:
    f.write('Three-way FSC resolution comparison\n')
    f.write('=' * 60 + '\n')
    f.write(f'Pixel size: {PIXEL_SIZE*1e9:.2f} nm\n')
    f.write(f'Nyquist limit: {2*PIXEL_SIZE*1e9:.2f} nm\n\n')
    f.write('Method       FSC=0.5 [nm]    FSC=0.143 [nm]\n')
    f.write('-' * 60 + '\n')
    for method in methods:
        res_50, res_143 = resolutions[method]
        f.write(f'{method:10s}   {res_50:8.2f}        {res_143:8.2f}\n')

print(f'\n  → Saved: {res_file}')

# ── 5. Plot FSC curves ─────────────────────────────────────────────────────

print('\n4. Plotting FSC curves ...\n')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

colors = {'FBP': 'C0', 'FBaP': 'C1', 'Two-pass': 'C2'}
linestyles = {'FBP': '--', 'FBaP': '-.', 'Two-pass': '-'}

for method in methods:
    freq_nm, fsc = fsc_results[method]
    ax.plot(freq_nm, fsc, label=method, color=colors[method],
            linestyle=linestyles[method], linewidth=2)

# Threshold lines
ax.axhline(0.5, color='k', linestyle=':', linewidth=1, label='FSC = 0.5')
ax.axhline(0.143, color='gray', linestyle=':', linewidth=1, label='FSC = 0.143')

# Nyquist frequency
f_nyquist_nm = 0.5 / PIXEL_SIZE * 1e-9
ax.axvline(f_nyquist_nm, color='red', linestyle=':', linewidth=1,
           label=f'Nyquist ({2*PIXEL_SIZE*1e9:.1f} nm)')

ax.set_xlabel('Spatial frequency [1/nm]', fontsize=12)
ax.set_ylabel('Fourier Shell Correlation', fontsize=12)
ax.set_title('FSC comparison: FBP vs FBaP vs Two-pass', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, f_nyquist_nm * 1.1)
ax.set_ylim(-0.1, 1.05)

fig.tight_layout()
fig_file = os.path.join(OUT_DIR, 'fsc_curves.png')
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'  → Saved: {fig_file}')

# ── 6. Save raw FSC data ───────────────────────────────────────────────────

npz_file = os.path.join(OUT_DIR, 'fsc_data.npz')
np.savez_compressed(
    npz_file,
    freq_nm_fbp=fsc_results['FBP'][0],
    fsc_fbp=fsc_results['FBP'][1],
    freq_nm_fbap=fsc_results['FBaP'][0],
    fsc_fbap=fsc_results['FBaP'][1],
    freq_nm_twopass=fsc_results['Two-pass'][0],
    fsc_twopass=fsc_results['Two-pass'][1],
    pixel_size=PIXEL_SIZE,
    resolutions=resolutions,
)

print(f'  → Saved: {npz_file}')

print('\n' + '='*70)
print('Done.')
print('='*70)
