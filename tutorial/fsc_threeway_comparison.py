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

# Optional variant runs of twopass_real_data.py.  Each is auto-included only
# if BOTH its half-volumes are present; otherwise it is silently skipped.
# Directory tags (set by twopass_real_data.py):
#   _snr       -> ANGLE_WEIGHT='snr'         (per-angle noise weighting)
#   _grid      -> FBP_METHOD='gridding'      (NUFFT back-projector)
#   _grid_snr  -> both
def _hdirs(tag):
    return (os.path.join(_HERE, f'twopass_real_figures{tag}_half0'),
            os.path.join(_HERE, f'twopass_real_figures{tag}_half1'))

VARIANT_DIRS = {
    'snr':      _hdirs('_snr'),
    'grid':     _hdirs('_grid'),
    'grid_snr': _hdirs('_grid_snr'),
}

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

# (method -> (npz_filename, half0_dir, half1_dir, key, required))
_TP = 'twopass_reconstruction.npz'
_FB = 'fbap_reconstruction.npz'
_method_spec = {
    'FBP':       (_TP, *_hdirs(''),                'delta_fbp', True),
    'FBaP':      (_FB, FBAP_HALF0_DIR, FBAP_HALF1_DIR, 'delta_fbap', True),
    'Two-pass':  (_TP, *_hdirs(''),                'delta_tp', True),
    # Optional variants (auto-included if present):
    'Two-pass (snr)':      (_TP, *VARIANT_DIRS['snr'],      'delta_tp', False),
    'Two-pass (grid)':     (_TP, *VARIANT_DIRS['grid'],     'delta_tp', False),
    'Two-pass (grid+snr)': (_TP, *VARIANT_DIRS['grid_snr'], 'delta_tp', False),
    # The gridding back-projector also changes the *FBP init* itself; compare
    # it directly to the iradon FBP (isolates the back-projector, pre-refine):
    'FBP (grid)':          (_TP, *VARIANT_DIRS['grid'],     'delta_fbp', False),
}

methods = []
volumes = {}

for method, (fname, d0, d1, key, required) in _method_spec.items():
    path0 = os.path.join(d0, fname)
    path1 = os.path.join(d1, fname)

    if not (os.path.exists(path0) and os.path.exists(path1)):
        if required:
            print(f'  ✗ {method}: missing {path0} or {path1}')
            sys.exit(1)
        else:
            print(f'  – {method:20s} not found (optional) — skipped')
            continue

    data0 = np.load(path0, allow_pickle=True)
    data1 = np.load(path1, allow_pickle=True)
    vol0, vol1 = data0[key], data1[key]
    print(f'  ✓ {method:20s} half-0: {vol0.shape}  half-1: {vol1.shape}')
    methods.append(method)
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

    print(f'  {method:20s} FSC=0.5: {res_50:6.2f} nm    FSC=0.143: {res_143:6.2f} nm')

# ── 3b. Before/after diffs for each present variant ─────────────────────────
# Each entry: (after_label, before_label, what_changed).  Reported only when
# BOTH labels are present.  Positive % = the variant improved resolution.
_COMPARISONS = [
    ('Two-pass (snr)',      'Two-pass', 'per-angle weighting (uniform -> snr)'),
    ('Two-pass (grid)',     'Two-pass', 'FBP back-projector (iradon -> gridding)'),
    ('Two-pass (grid+snr)', 'Two-pass', 'gridding + snr weighting (combined)'),
    ('FBP (grid)',          'FBP',      'FBP back-projector, pre-refinement '
                                        '(iradon -> gridding)'),
]


def _delta_line(label_thr, before, after):
    if not (np.isfinite(before) and np.isfinite(after)):
        return f'  {label_thr}: {before:6.2f} -> {after:6.2f} nm  (n/a)'
    d = before - after                      # >0 => improved (smaller nm)
    pct = 100.0 * d / before
    sign = '+' if d >= 0 else ''
    return (f'  {label_thr}: {before:6.2f} -> {after:6.2f} nm  '
            f'({sign}{d:.2f} nm, {sign}{pct:.1f}%)')


diff_summaries = []
for after, before, what in _COMPARISONS:
    if after in resolutions and before in resolutions:
        b50, b143 = resolutions[before]
        a50, a143 = resolutions[after]
        block = (f'\n{what}:\n'
                 f'{_delta_line("FSC=0.5  ", b50, a50)}\n'
                 f'{_delta_line("FSC=0.143", b143, a143)}')
        diff_summaries.append(block)

if diff_summaries:
    print('\nBefore/after effects (positive % = improved resolution):')
    for b in diff_summaries:
        print(b)

# ── 4. Save resolution table ───────────────────────────────────────────────

res_file = os.path.join(OUT_DIR, 'fsc_resolution.txt')
with open(res_file, 'w') as f:
    f.write('FSC resolution comparison\n')
    f.write('=' * 60 + '\n')
    f.write(f'Pixel size: {PIXEL_SIZE*1e9:.2f} nm\n')
    f.write(f'Nyquist limit: {2*PIXEL_SIZE*1e9:.2f} nm\n\n')
    f.write('Method                 FSC=0.5 [nm]    FSC=0.143 [nm]\n')
    f.write('-' * 60 + '\n')
    for method in methods:
        res_50, res_143 = resolutions[method]
        f.write(f'{method:20s}   {res_50:8.2f}        {res_143:8.2f}\n')
    if diff_summaries:
        f.write('\nBefore/after effects (positive % = improved resolution):\n')
        for b in diff_summaries:
            f.write(b + '\n')

print(f'\n  → Saved: {res_file}')

# ── 5. Plot FSC curves ─────────────────────────────────────────────────────

print('\n4. Plotting FSC curves ...\n')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Fixed style for the core three; variants get distinct colours from the cycle.
_fixed_color = {'FBP': 'C0', 'FBaP': 'C1', 'Two-pass': 'C2'}
_fixed_ls    = {'FBP': '--', 'FBaP': '-.', 'Two-pass': '-'}
_variant_colors = ['C3', 'C4', 'C5', 'C6', 'C7']
_vc = iter(_variant_colors)

for method in methods:
    freq_nm, fsc = fsc_results[method]
    color = _fixed_color.get(method) or next(_vc, 'k')
    # FBP variants dashed (compare to FBP); two-pass variants solid.
    ls = _fixed_ls.get(method) or (':' if method.startswith('FBP') else '-')
    ax.plot(freq_nm, fsc, label=method, color=color,
            linestyle=ls, linewidth=2)

# Threshold lines
ax.axhline(0.5, color='k', linestyle=':', linewidth=1, label='FSC = 0.5')
ax.axhline(0.143, color='gray', linestyle=':', linewidth=1, label='FSC = 0.143')

# Nyquist frequency
f_nyquist_nm = 0.5 / PIXEL_SIZE * 1e-9
ax.axvline(f_nyquist_nm, color='red', linestyle=':', linewidth=1,
           label=f'Nyquist ({2*PIXEL_SIZE*1e9:.1f} nm)')

ax.set_xlabel('Spatial frequency [1/nm]', fontsize=12)
ax.set_ylabel('Fourier Shell Correlation', fontsize=12)
_n_variants = sum(m not in ('FBP', 'FBaP', 'Two-pass') for m in methods)
_title = 'FSC comparison: FBP vs FBaP vs Two-pass'
if _n_variants:
    _title += f' (+{_n_variants} variant{"s" if _n_variants > 1 else ""})'
ax.set_title(_title, fontsize=13, fontweight='bold')
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
_save = {'pixel_size': PIXEL_SIZE, 'resolutions': resolutions}


def _safe_tag(label):
    return (label.lower().replace(' ', '_').replace('(', '')
            .replace(')', '').replace('+', '_').replace('-', ''))


for method in methods:
    t = _safe_tag(method)
    _save[f'freq_nm_{t}'] = fsc_results[method][0]
    _save[f'fsc_{t}'] = fsc_results[method][1]
np.savez_compressed(npz_file, **_save)

print(f'  → Saved: {npz_file}')

print('\n' + '='*70)
print('Done.')
print('='*70)
