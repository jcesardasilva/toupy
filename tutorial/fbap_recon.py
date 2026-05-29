#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtered Back-Propagation (FBaP) reconstruction — Aidukas et al. (2024)

Applies the depth-of-field-extended tomographic reconstruction algorithm of

  T. Aidukas et al., "High-performance 4-nm-resolution X-ray tomography
  using burst ptychography," Nature 632, 81–88 (2024).
  DOI: 10.1038/s41586-024-07615-6

to the same PXCT dataset used by the two-pass method, producing a third
reconstruction (alongside FBP and two-pass) for direct comparison.

Workflow
--------
1. Load phase projections from PXCTalignedprojections.npz (same as
   twopass_real_data.py, same crop).
2. Build complex exit waves: u = exp(i·φ)  [pure-phase approximation].
3. Run FBaP: propagate each projection to N_SLICES depths, extract local
   δ via phase finite-differences, apply ramp filter, 3-D back-project.
4. Load existing FBP and two-pass volumes for comparison.
5. Save FBaP volume and generate comparison figures.

Usage
-----
  python tutorial/fbap_recon.py

Outputs (in twopass_real_figures_fbap/ by default)
-------
  fbap_reconstruction.npz  — delta_fbap, delta_fbp, delta_tp, metadata
  fig1_orthogonal_cuts.png  — all three methods side by side
  fig2_difference_maps.png  — (FBaP − FBP) and (two-pass − FBaP)
  fig3_profiles.png         — line profiles through centre
"""

import os
import sys
import time
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_TOMO  = os.path.join(_HERE, '..', 'toupy', 'tomo')

# Load filtered_backprop directly (avoids toupy.__init__ fabio dependency)
def _load_module(dotted_name, fpath):
    spec = importlib.util.spec_from_file_location(dotted_name, fpath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

fbp_mod = _load_module(
    'toupy.tomo.filtered_backprop',
    os.path.join(_TOMO, 'filtered_backprop.py'))

filtered_back_propagation = fbp_mod.filtered_back_propagation

# ---------------------------------------------------------------------------
# 1.  User-tunable parameters
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(_HERE, 'PXCTalignedprojections.npz')

# Boundary crop — must match twopass_real_data.py for a fair comparison
CROP_X = 55
CROP_Y = 55

# Half-dataset mode (None = full dataset; 0 or 1 = FSC halves)
FSC_HALF = None

# Number of depth slices for FBaP (same as N_SLICES in twopass_real_data.py)
N_SLICES = 64

# Autofocus: disabled by default for pure-phase objects with very small δ.
# The intensity contrast developed by propagation is only ~1 % for δ~1e-7,
# making autofocus unreliable.  Set to True to enable (adds ~50 % runtime).
AUTO_FOCUS = False

# Existing two-pass result to load for comparison (None = skip comparison)
# Expected path: twopass_real_figures/twopass_reconstruction.npz
TWOPASS_NPZ = os.path.join(_HERE, 'twopass_real_figures',
                             'twopass_reconstruction.npz')

# Output directory
_out_suffix = f'_half{FSC_HALF}' if FSC_HALF is not None else ''
OUT_DIR = os.path.join(_HERE, f'twopass_real_figures_fbap{_out_suffix}')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2.  Load data
# ---------------------------------------------------------------------------
print(f'Loading data from: {DATA_FILE}')
data        = np.load(DATA_FILE, allow_pickle=True)
phase_stack = data['projections'].astype(np.float64)   # (N_ANGLES, Ny, Nx)
WAVELENGTH  = float(data['wavelen'])
PIXEL_SIZE  = float(data['psize'])
THETA       = data['theta'].astype(np.float64)

N_ANGLES, Ny, Nx = phase_stack.shape
K0 = 2.0 * np.pi / WAVELENGTH

print(f'  N_ANGLES = {N_ANGLES},  (Ny, Nx) = ({Ny}, {Nx})')
print(f'  λ = {WAVELENGTH*1e9:.4f} nm,  pixel = {PIXEL_SIZE*1e9:.2f} nm')

# Apply boundary crop
if CROP_X > 0 or CROP_Y > 0:
    _sy = slice(CROP_Y or None, -CROP_Y or None)
    _sx = slice(CROP_X or None, -CROP_X or None)
    phase_stack = np.ascontiguousarray(phase_stack[:, _sy, _sx])
    _, Ny, Nx   = phase_stack.shape
    Nz          = Nx
    print(f'  After crop ({CROP_Y}px top/bottom, {CROP_X}px left/right): '
          f'(Ny, Nx, Nz) = ({Ny}, {Nx}, {Nz})\n')
else:
    Nz = Nx

SAMPLE_THICKNESS = Nz * PIXEL_SIZE   # [m]
SLICE_DZ         = SAMPLE_THICKNESS / N_SLICES

# Half-dataset selection (for FSC)
if FSC_HALF is not None:
    _all_idx  = np.arange(N_ANGLES)
    _half_idx = _all_idx[FSC_HALF::2]
    theta_use  = THETA[_half_idx]
    phase_use  = phase_stack[_half_idx]
    N_use      = len(theta_use)
    print(f'  [FSC half-{FSC_HALF}] Using {N_use}/{N_ANGLES} angles\n')
else:
    theta_use = THETA
    phase_use = phase_stack
    N_use     = N_ANGLES

# ---------------------------------------------------------------------------
# 3.  Form complex exit waves
# ---------------------------------------------------------------------------
# Pure-phase object approximation: u = exp(i·φ)
# Amplitude = 1; adjust if you have an amplitude channel from ptychography.
print('Building complex exit waves from phase projections ...')
u_measured = np.exp(1j * phase_use.astype(np.complex128)).astype(np.complex64)
# shape: (N_use, Ny, Nx), complex64

# ---------------------------------------------------------------------------
# 4.  FBaP reconstruction
# ---------------------------------------------------------------------------
print('\n' + '='*65)
print('Filtered Back-Propagation (FBaP) — Aidukas et al. (2024)')
print('='*65)
print(f'  N_SLICES        = {N_SLICES}  (Δz = {SLICE_DZ*1e9:.1f} nm)')
print(f'  sample thickness = {SAMPLE_THICKNESS*1e6:.2f} µm  ({Nz} pixels)')
print(f'  auto_focus       = {AUTO_FOCUS}')
print(f'  filter           = ram-lak')

DoF    = PIXEL_SIZE**2 / WAVELENGTH
F_sl   = DoF / SLICE_DZ
print(f'  DoF              = {DoF*1e6:.2f} µm = {DoF/PIXEL_SIZE:.0f} px')
print(f'  F_slice          = {F_sl:.1f}  (>> 1 → slice approx. valid)')
print()

t0 = time.time()
delta_fbap, focus_dists = filtered_back_propagation(
    u_measured,
    theta_use,
    wavelength       = WAVELENGTH,
    pixel_size       = PIXEL_SIZE,
    sample_thickness = SAMPLE_THICKNESS,
    n_slices         = N_SLICES,
    auto_focus       = AUTO_FOCUS,
    focus_metric     = 'tv_intensity',
    filter_name      = 'ram-lak',
    verbose          = True,
)
print(f'\nFBaP total time: {time.time()-t0:.1f} s')
print(f'δ_FBaP range: [{delta_fbap.min():.3e}, {delta_fbap.max():.3e}]')

# ---------------------------------------------------------------------------
# 5.  Load FBP and two-pass volumes for comparison
# ---------------------------------------------------------------------------
delta_fbp = None
delta_tp  = None

# Try to load two-pass result
if os.path.exists(TWOPASS_NPZ):
    print(f'\nLoading FBP and two-pass volumes from: {TWOPASS_NPZ}')
    tp_data   = np.load(TWOPASS_NPZ)
    delta_fbp = tp_data['delta_fbp'].astype(np.float64)
    delta_tp  = tp_data['delta_tp'].astype(np.float64)
    print(f'  δ_FBP  range: [{delta_fbp.min():.3e}, {delta_fbp.max():.3e}]')
    print(f'  δ_TP   range: [{delta_tp.min():.3e},  {delta_tp.max():.3e}]')

    # Ensure all volumes have the same shape (FBaP might differ if different
    # crop settings were used)
    for name, vol in [('FBP', delta_fbp), ('two-pass', delta_tp)]:
        if vol.shape != delta_fbap.shape:
            print(f'  WARNING: {name} volume shape {vol.shape} ≠ '
                  f'FBaP shape {delta_fbap.shape}. Skipping comparison plots.')
            delta_fbp = None
            delta_tp  = None
            break
else:
    print(f'\nNo two-pass result found at {TWOPASS_NPZ}; '
          'only FBaP volume will be saved.')

# ---------------------------------------------------------------------------
# 6.  Save
# ---------------------------------------------------------------------------
save_dict = dict(
    delta_fbap  = delta_fbap.astype(np.float32),
    wavelength  = WAVELENGTH,
    psize       = PIXEL_SIZE,
    theta       = theta_use,
    n_slices    = N_SLICES,
    sample_thickness = SAMPLE_THICKNESS,
    focus_distances  = focus_dists,
    auto_focus  = AUTO_FOCUS,
)
if delta_fbp is not None:
    save_dict['delta_fbp'] = delta_fbp.astype(np.float32)
if delta_tp is not None:
    save_dict['delta_tp'] = delta_tp.astype(np.float32)

npz_path = os.path.join(OUT_DIR, 'fbap_reconstruction.npz')
np.savez_compressed(npz_path, **save_dict)
print(f'\nSaved: {npz_path}')

# ---------------------------------------------------------------------------
# 7.  Figures
# ---------------------------------------------------------------------------
print('\nGenerating figures ...')
CMAP    = 'gray'
Nz_vol  = delta_fbap.shape[0]
Ny_vol  = delta_fbap.shape[1]
Nx_vol  = delta_fbap.shape[2]
iz_mid  = Nz_vol // 2
iy_mid  = Ny_vol // 2
ix_mid  = Nx_vol // 2

# Shared colour scale (99.5th percentile of FBaP)
vmin = 0.0
_pos = delta_fbap[delta_fbap > 0]
vmax = float(np.percentile(_pos, 99.5)) if _pos.size > 0 else 1e-5
del _pos

# ── Figure 1: Orthogonal cuts ────────────────────────────────────────────
volumes = {'FBaP': delta_fbap}
if delta_fbp is not None:
    volumes['FBP']      = delta_fbp
if delta_tp is not None:
    volumes['Two-pass'] = delta_tp

n_methods = len(volumes)
cut_labels = [f'xy  (z={iz_mid})', f'xz  (y={iy_mid})', f'yz  (x={ix_mid})']

fig1, axes = plt.subplots(3, n_methods,
                           figsize=(5 * n_methods, 12),
                           gridspec_kw={'hspace': 0.40, 'wspace': 0.30})
if n_methods == 1:
    axes = axes[:, np.newaxis]

for row, label in enumerate(cut_labels):
    for col, (method, vol) in enumerate(volumes.items()):
        if row == 0:
            cut = vol[iz_mid]
        elif row == 1:
            cut = vol[:, iy_mid, :]
        else:
            cut = vol[:, :, ix_mid].T
        im = axes[row, col].imshow(cut, cmap=CMAP,
                                    vmin=vmin, vmax=vmax, origin='lower')
        axes[row, col].set_title(f'{method}  [{label}]', fontsize=8)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.04, label='δ')

fig1.suptitle(
    f'FBaP (Aidukas 2024) vs comparison methods\n'
    f'λ={WAVELENGTH*1e9:.3f} nm  |  pixel={PIXEL_SIZE*1e9:.1f} nm  '
    f'|  {N_use} angles  |  {N_SLICES} depth slices',
    fontsize=9)
p1 = os.path.join(OUT_DIR, 'fig1_orthogonal_cuts.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight')
print(f'  Saved: {p1}')

# ── Figure 2: Difference maps ────────────────────────────────────────────
if delta_fbp is not None and delta_tp is not None:
    diff_fbap_fbp = delta_fbap - delta_fbp
    diff_tp_fbap  = delta_tp   - delta_fbap

    fig2, axes2 = plt.subplots(2, 3, figsize=(13, 8),
                                gridspec_kw={'hspace': 0.40, 'wspace': 0.35})
    diffs = [
        (diff_fbap_fbp, 'FBaP − FBP'),
        (diff_tp_fbap,  'Two-pass − FBaP'),
    ]
    cut_keys = [
        (iz_mid, 'xy'),
        (slice(None), 'xz', iy_mid),
        (slice(None), 'yz', ix_mid),
    ]
    for row, (diff, title) in enumerate(diffs):
        dlim = float(np.percentile(np.abs(diff), 99.5))
        cuts = [diff[iz_mid],
                diff[:, iy_mid, :],
                diff[:, :, ix_mid].T]
        lbls = [f'xy (z={iz_mid})', f'xz (y={iy_mid})', f'yz (x={ix_mid})']
        for col, (img, lbl) in enumerate(zip(cuts, lbls)):
            im = axes2[row, col].imshow(img, cmap='bwr',
                                         vmin=-dlim, vmax=dlim, origin='lower')
            axes2[row, col].set_title(f'{title}  [{lbl}]', fontsize=8)
            axes2[row, col].axis('off')
            plt.colorbar(im, ax=axes2[row, col], fraction=0.04, label='Δδ')
    fig2.suptitle('Difference maps', fontsize=9)
    p2 = os.path.join(OUT_DIR, 'fig2_difference_maps.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f'  Saved: {p2}')

# ── Figure 3: Line profiles ──────────────────────────────────────────────
profile_x = np.arange(Nx_vol) * PIXEL_SIZE * 1e9   # [nm]
profile_z = np.arange(Nz_vol) * PIXEL_SIZE * 1e9   # [nm]

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={'wspace': 0.40})
colors = {'FBaP': 'green', 'FBP': 'red', 'Two-pass': 'blue'}
lstyles = {'FBaP': '-', 'FBP': '--', 'Two-pass': '-.'}

for method, vol in volumes.items():
    col = colors.get(method, 'k')
    ls  = lstyles.get(method, '-')
    axes3[0].plot(profile_x,
                  vol[iz_mid, iy_mid, :] * 1e6,
                  color=col, ls=ls, lw=1.5, label=method)
    axes3[1].plot(profile_z,
                  vol[:, iy_mid, ix_mid] * 1e6,
                  color=col, ls=ls, lw=1.5, label=method)

for ax, xlabel, title in zip(
        axes3,
        ['x [nm]', 'z [nm]'],
        ['Horizontal profile  (z=mid, y=mid)',
         'Vertical profile    (y=mid, x=mid)']):
    ax.set_xlabel(xlabel)
    ax.set_ylabel('δ  ×10⁻⁶')
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

fig3.suptitle('Central line profiles', fontsize=9)
p3 = os.path.join(OUT_DIR, 'fig3_profiles.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight')
print(f'  Saved: {p3}')

plt.close('all')

# ---------------------------------------------------------------------------
# 8.  Summary
# ---------------------------------------------------------------------------
print()
print('='*65)
print('FBAP RECONSTRUCTION SUMMARY')
print('='*65)
print(f'  Wavelength         : {WAVELENGTH*1e9:.4f} nm')
print(f'  Pixel size         : {PIXEL_SIZE*1e9:.2f} nm')
print(f'  Volume shape       : {delta_fbap.shape}  [Nz, Ny, Nx]')
print(f'  Angles used        : {N_use}  (of {N_ANGLES} total)')
print(f'  Depth slices       : {N_SLICES}  (Δz = {SLICE_DZ*1e9:.1f} nm)')
print(f'  Slice Fresnel no.  : {F_sl:.1f}')
print(f'  Auto-focus         : {AUTO_FOCUS}')
print(f'  δ_FBaP range       : [{delta_fbap.min():.3e}, {delta_fbap.max():.3e}]')
if delta_fbp is not None:
    print(f'  δ_FBP  range       : [{delta_fbp.min():.3e}, {delta_fbp.max():.3e}]')
if delta_tp is not None:
    print(f'  δ_TP   range       : [{delta_tp.min():.3e},  {delta_tp.max():.3e}]')
print(f'  Figures saved to   : {OUT_DIR}/')
print('='*65)
