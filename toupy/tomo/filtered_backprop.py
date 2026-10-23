#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtered Back-Propagation (FBaP) tomographic reconstruction.

Implements the depth-of-field-extended algorithm described in:

  T. Aidukas et al., "High-performance 4-nm-resolution X-ray tomography
  using burst ptychography," Nature 632, 81–88 (2024).
  DOI: 10.1038/s41586-024-07615-6

For each projection angle the complex ptychographic exit wave is processed
in five steps (Methods, Tomographic reconstruction):

  1. Digitally refocus the exit wave to the sample centre (autofocus).
  2. Propagate from the focused plane to n_slices depths spanning [−t/2, t/2].
  3. Extract phase at each depth → depth-resolved phase stack φ_j(x, y; θ).
  4. Compute local δ sinogram via forward finite differences in depth:
       δ_proj_j = (φ_{j+1} − φ_j) / (k₀ · Δz)
     Then apply the Ram-Lak ramp filter along the projection direction (x).
  5. Rotate the filtered 3-D stack by −θ and accumulate into the tomogram.

Physical motivation
-------------------
Standard filtered back-projection (FBP) ignores wavefront changes within
the sample, limiting the usable depth of field to T ≤ 5.2 r²/λ (equation 27
of the paper, where r is the resolution).  FBaP propagates each projection
to every depth slice before back-projecting, incorporating those wavefront
changes and extending the effective depth of field across the full sample
thickness.

The method assumes weak (single) scattering within each projection —
i.e. it uses single-slice ptychographic exit waves — and is therefore
complementary to the two-pass multislice refinement implemented in
:mod:`toupy.tomo.twopass`, which iteratively corrects the 3-D volume using
a multislice forward model.

Relation between phase stack and local δ
-----------------------------------------
Under the projection approximation, back-propagating the exit wave at
angle θ to depth z_j gives a wavefield whose phase approximates the
cumulative phase accumulated from depth z_j to the exit face:

  φ_j(x, y; θ) ≈ −k₀ ∫_{z_j}^{t/2} δ(R_{−θ}(x, y, z)) dz

The forward finite difference extracts the local contribution:

  φ_{j+1} − φ_j ≈ −k₀ · δ_j · Δz  →  δ_j = (φ_{j+1} − φ_j) / (k₀ · Δz)

Applying FBP to δ_j(x, y; θ) over all angles gives the 2-D δ map at
depth z_j; stacking all depths yields the 3-D volume.

Autofocus note
--------------
The paper finds the optimal propagation distance by maximising image
sharpness (standard deviation of a high-pass filtered image or total
variation — Extended Data Fig. 4, caption).  For pure-phase objects the
intensity contrast develops as the wave propagates away from the sample
plane (Transport-of-Intensity effect), so tv_intensity works even when
|ψ_exit| ≈ 1 everywhere.  For a very weakly scattering object (δ ≲ 10⁻⁷),
the developed contrast is very small (~1 %) and autofocus can be disabled
(auto_focus=False) with the focus fixed at the sample centre.
"""

# standard library
import time

# third-party
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import rotate as ndimage_rotate

__all__ = [
    "fresnel_propagate",
    "sharpness_metric",
    "find_focus",
    "filtered_back_propagation",
]


# ---------------------------------------------------------------------------
# Optional GPU backend (PyTorch)
# ---------------------------------------------------------------------------
#
# The reconstruction is pure NumPy/SciPy by default (CPU).  When PyTorch is
# available and a GPU (CUDA or Apple MPS) is present, the per-angle pipeline
# — Fresnel depth propagation, Ram-Lak filtering and the 3-D back-rotation —
# runs on the GPU instead, giving a 5–10× speed-up.  The GPU path reuses the
# validated ``rotate_volume_torch`` from ``multislice_torch.py`` so the
# rotation convention is identical to the two-pass reconstruction.

def _load_torch_helpers():
    """
    Lazily import torch and the rotation helper from ``multislice_torch.py``.

    Loads ``multislice_torch.py`` by file path (next to this module) so it
    works even when the full ``toupy`` package is not importable — the same
    strategy the tutorial scripts use to avoid the ``toupy.__init__`` fabio
    dependency.

    Returns
    -------
    (torch_module, multislice_torch_module)  or  (None, None) if torch missing.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return None, None

    import os
    import importlib.util

    _here = os.path.dirname(os.path.abspath(__file__))
    _mst_path = os.path.join(_here, "multislice_torch.py")
    spec = importlib.util.spec_from_file_location(
        "toupy.tomo._mst_for_fbap", _mst_path)
    mst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mst)
    return torch, mst


def _resolve_fbap_device(device):
    """
    Decide whether to use the torch GPU backend.

    Parameters
    ----------
    device : str or None or torch.device
        'auto'  → use GPU (CUDA/MPS) if available, else NumPy CPU.
        'cpu'   → force the NumPy CPU backend.
        'cuda' / 'mps' → force that GPU (errors if unavailable).
        torch.device → use as given.

    Returns
    -------
    (use_torch, torch_module, mst_module, torch_device)
        ``use_torch`` is False for the NumPy CPU path (the other three are
        then None).
    """
    if device is None or (isinstance(device, str) and device.lower() == "cpu"):
        return False, None, None, None

    torch, mst = _load_torch_helpers()
    if torch is None:
        # torch not installed — fall back to NumPy CPU regardless of request
        return False, None, None, None

    if isinstance(device, str) and device.lower() == "auto":
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            dev = mst.select_device()              # CUDA > MPS
        else:
            return False, None, None, None         # CPU → NumPy path
    else:
        dev = mst.select_device(device) if isinstance(device, str) else device
        if dev.type == "cpu":
            # explicit torch-CPU request: NumPy path is faster, use it
            return False, None, None, None

    return True, torch, mst, dev


def _sharpness_metric_torch(field, metric, torch):
    """Torch mirror of :func:`sharpness_metric` (higher = sharper)."""
    if metric == 'tv_intensity':
        I = field.abs() ** 2
        return float(torch.mean(torch.abs(torch.diff(I, dim=0))) +
                     torch.mean(torch.abs(torch.diff(I, dim=1))))
    elif metric == 'tv_phase':
        phi = torch.angle(field)
        return float(torch.mean(torch.abs(torch.diff(phi, dim=0))) +
                     torch.mean(torch.abs(torch.diff(phi, dim=1))))
    elif metric == 'std_intensity':
        return float((field.abs() ** 2).std())
    else:
        raise ValueError(f"Unknown sharpness metric: {metric!r}. "
                         "Choose 'tv_intensity', 'tv_phase', or 'std_intensity'.")


def _find_focus_torch(u_ft, f2, wavelength, z_range, n_search, metric, torch):
    """Torch mirror of :func:`find_focus`; operates on a precomputed FFT."""
    d_vals = np.linspace(z_range[0], z_range[1], n_search)
    best_score = -np.inf
    best_d = float(d_vals[0])
    for d in d_vals:
        H = torch.exp(1j * ((-np.pi * wavelength * float(d)) * f2))
        u_prop = torch.fft.ifft2(u_ft * H)
        s = _sharpness_metric_torch(u_prop, metric, torch)
        if s > best_score:
            best_score = s
            best_d = float(d)
    return best_d


def _filtered_back_propagation_torch(
        u_exit_stack, theta, wavelength, pixel_size, sample_thickness,
        Nz, auto_focus, focus_z_range, focus_n_search, focus_metric,
        filter_name, verbose, torch, mst, dev):
    """
    GPU implementation of :func:`filtered_back_propagation`.

    Mathematically identical to the NumPy path, with two GPU-friendly changes:

    * The incremental depth-propagation loop is replaced by a single
      **batched** kernel ``exp(−iπλ z_j f²)`` applied to the focused
      spectrum, propagating to all ``Nz`` depths in one ``ifft2`` call.
    * Ram-Lak filtering and the −θ back-rotation are batched over depth.
    """
    theta = np.asarray(theta, dtype=np.float64)
    N_angles, Ny, Nx = u_exit_stack.shape
    k0 = 2.0 * np.pi / wavelength

    z_slices = np.linspace(-sample_thickness / 2.0,
                            sample_thickness / 2.0, Nz)

    if focus_z_range is None:
        focus_z_range = (-sample_thickness * 1.5, 0.0)

    # Squared-frequency grid on device  (indexing='xy' → first index = x)
    fy = torch.fft.fftfreq(Ny, d=pixel_size, device=dev)
    fx = torch.fft.fftfreq(Nx, d=pixel_size, device=dev)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')           # (Ny, Nx)
    f2 = (FX ** 2 + FY ** 2).to(torch.float32)

    # Batched depth kernels: at slice j the total kernel relative to the
    # focused sample centre is exp(−iπλ z_j f²).  Equivalent to the NumPy
    # H_init · H_step^(Nz-1-j) incremental product (verified algebraically).
    z_t = torch.as_tensor(z_slices, dtype=torch.float32, device=dev)
    phase_kern = (-np.pi * wavelength) * z_t[:, None, None] * f2[None]   # (Nz,Ny,Nx)
    depth_kernels = torch.exp(1j * phase_kern).to(torch.complex64)

    # Ram-Lak ramp filter
    if filter_name == 'ram-lak':
        n_pad = max(64, int(2 ** np.ceil(np.log2(2 * Nx))))
        ramp = (2.0 * torch.fft.rfftfreq(n_pad, device=dev)).to(torch.float32)
    else:
        n_pad = None
        ramp = None

    delta_vol = torch.zeros((Nz, Ny, Nx), dtype=torch.float32, device=dev)
    focus_distances = np.zeros(N_angles)

    t_total = time.time()

    for ai in range(N_angles):
        t_iter = time.time()

        u = torch.as_tensor(u_exit_stack[ai], dtype=torch.complex64, device=dev)
        u_ft = torch.fft.fft2(u)

        # Step 1: digital refocus to sample centre
        if auto_focus:
            d_focus = _find_focus_torch(u_ft, f2, wavelength, focus_z_range,
                                        focus_n_search, focus_metric, torch)
        else:
            d_focus = -sample_thickness / 2.0
        focus_distances[ai] = d_focus

        u_focused_ft = u_ft * torch.exp(1j * ((-np.pi * wavelength * d_focus) * f2))

        # Steps 2 & 3: propagate to all depths at once, extract phase
        fields = torch.fft.ifft2(u_focused_ft[None] * depth_kernels,
                                 dim=(-2, -1))               # (Nz,Ny,Nx)
        phase_stack = torch.angle(fields)                    # float32

        # Step 4: Ram-Lak filter along x (batched over depth & y)
        if ramp is not None:
            row_f = torch.fft.rfft(phase_stack, n=n_pad, dim=-1)
            row_f = row_f * ramp[None, None, :]
            phase_stack = torch.fft.irfft(row_f, n=n_pad, dim=-1)[..., :Nx]

        # Step 5: rotate by −θ (same convention as scipy axes=(2,0)) & accumulate
        delta_vol += mst.rotate_volume_torch(phase_stack, -float(theta[ai]))

        if verbose:
            elapsed = time.time() - t_iter
            elapsed_total = time.time() - t_total
            done = ai + 1
            eta_s = elapsed_total / done * (N_angles - done)
            print(f"  FBaP {done:4d}/{N_angles}  θ={theta[ai]:7.2f}°  "
                  f"t={elapsed:.2f}s  "
                  f"elapsed={elapsed_total/60:.1f}min  "
                  f"ETA={eta_s/60:.1f}min",
                  flush=True)

    # Accumulated phase sinogram → δ  (FBP norm × phase-to-δ; see NumPy path)
    delta_vol *= -np.pi / (2.0 * N_angles * k0 * pixel_size)
    delta_vol = torch.clamp(delta_vol, min=0.0)

    delta_np = delta_vol.cpu().numpy().astype(np.float64)

    if verbose:
        print(f"FBaP done in {time.time()-t_total:.1f} s total.  "
              f"δ ∈ [{delta_np.min():.3e}, {delta_np.max():.3e}]")

    return delta_np, focus_distances


# ---------------------------------------------------------------------------
# Fresnel propagator
# ---------------------------------------------------------------------------

def fresnel_propagate(field, distance, wavelength, pixel_size):
    """
    Paraxial (Fresnel) propagation of a 2-D complex wavefield.

    Transfer function:  H(fx, fy; Δz) = exp(−iπλΔz(fx² + fy²))

    The paraxial approximation is valid when (λ f_max)² ≪ 1, i.e. when
    the maximum spatial frequency f_max = 1/(2Δx) satisfies
    (λ/2Δx)² ≪ 1.  For hard X-rays and Δx = 25 nm this factor is
    ~4×10⁻⁶, so the approximation is excellent.

    Parameters
    ----------
    field : ndarray, complex, shape (Ny, Nx)
    distance : float
        Propagation distance [m].  Positive = along beam; negative = backward.
    wavelength : float   [m]
    pixel_size : float   [m]

    Returns
    -------
    ndarray, complex, shape (Ny, Nx)
    """
    Ny, Nx = field.shape
    fy = fftfreq(Ny, d=pixel_size)
    fx = fftfreq(Nx, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    return ifft2(fft2(field) * H)


# ---------------------------------------------------------------------------
# Sharpness metrics for autofocus
# ---------------------------------------------------------------------------

def sharpness_metric(field, metric='tv_intensity'):
    """
    Compute a scalar sharpness score for the propagated wavefield.

    Parameters
    ----------
    field : ndarray, complex, shape (Ny, Nx)
    metric : str
        ``'tv_intensity'``  — total variation of |field|² (default).
            Works for both amplitude and pure-phase objects (intensity
            contrast develops via the transport-of-intensity effect as
            the field propagates away from the sample plane).
        ``'tv_phase'``      — total variation of angle(field).
        ``'std_intensity'`` — standard deviation of |field|².

    Returns
    -------
    float   (higher = sharper)
    """
    if metric == 'tv_intensity':
        I = np.abs(field) ** 2
        return float(np.mean(np.abs(np.diff(I, axis=0))) +
                     np.mean(np.abs(np.diff(I, axis=1))))
    elif metric == 'tv_phase':
        phi = np.angle(field)
        return float(np.mean(np.abs(np.diff(phi, axis=0))) +
                     np.mean(np.abs(np.diff(phi, axis=1))))
    elif metric == 'std_intensity':
        return float(np.std(np.abs(field) ** 2))
    else:
        raise ValueError(f"Unknown sharpness metric: {metric!r}. "
                         "Choose 'tv_intensity', 'tv_phase', or 'std_intensity'.")


# ---------------------------------------------------------------------------
# Autofocus
# ---------------------------------------------------------------------------

def find_focus(u_exit, wavelength, pixel_size,
               z_range=None, n_search=50, metric='tv_intensity',
               sample_thickness=None):
    """
    Find the propagation distance that maximises image sharpness.

    Scans propagation distances in *z_range*, computes the chosen
    sharpness metric at each distance, and returns the distance at the
    maximum.  Following Aidukas et al. (2024) the default metric is the
    total variation of the propagated intensity.

    Parameters
    ----------
    u_exit : ndarray, complex, shape (Ny, Nx)
        Complex exit wave from ptychography.
    wavelength : float [m]
    pixel_size : float [m]
    z_range : tuple (z_min, z_max) [m]
        Propagation-distance search range.
        Default: (−sample_thickness, 0) if sample_thickness is given,
                 (−5·pixel_size, 5·pixel_size) otherwise.
    n_search : int
        Number of distances to evaluate.  Default 50.
    metric : str
        Sharpness metric (see :func:`sharpness_metric`).
    sample_thickness : float [m], optional
        Used only to set the default z_range.

    Returns
    -------
    d_opt : float [m]   Optimal propagation distance.
    scores : ndarray (n_search,)   Sharpness scores.
    d_vals : ndarray (n_search,)   Evaluated distances.
    """
    if z_range is None:
        if sample_thickness is not None:
            z_range = (-sample_thickness, 0.0)
        else:
            z_range = (-5.0 * pixel_size, 5.0 * pixel_size)

    Ny, Nx = u_exit.shape
    fy = fftfreq(Ny, d=pixel_size)
    fx = fftfreq(Nx, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    f2 = (FX ** 2 + FY ** 2).astype(np.float64)

    u_ft = fft2(u_exit)
    d_vals = np.linspace(z_range[0], z_range[1], n_search)
    scores = np.empty(n_search)

    for k, d in enumerate(d_vals):
        H = np.exp(-1j * np.pi * wavelength * d * f2)
        u_prop = ifft2(u_ft * H)
        scores[k] = sharpness_metric(u_prop, metric)

    d_opt = d_vals[int(np.argmax(scores))]
    return d_opt, scores, d_vals


# ---------------------------------------------------------------------------
# Main FBaP reconstruction
# ---------------------------------------------------------------------------

def filtered_back_propagation(
    u_exit_stack,
    theta,
    wavelength,
    pixel_size,
    sample_thickness,
    n_slices=None,
    auto_focus=False,
    focus_z_range=None,
    focus_n_search=50,
    focus_metric='tv_intensity',
    filter_name='ram-lak',
    verbose=True,
    device='auto',
):
    """
    Filtered Back-Propagation (FBaP) 3-D tomographic reconstruction.

    Implements the algorithm of Aidukas et al. (2024), Nature 632, 81.

    Parameters
    ----------
    u_exit_stack : ndarray, complex, shape (N_angles, Ny, Nx)
        Complex ptychographic exit waves, one per projection angle.
        For a pure-phase object: u = exp(i·φ) where φ is the projected
        phase [rad] with sign convention φ = −k₀ · ∫δ dz.
    theta : array_like, shape (N_angles,)
        Projection angles [degrees].
    wavelength : float   [m]
    pixel_size : float   [m]   (isotropic, used for both propagation and FBP)
    sample_thickness : float   [m]   (= n_slices × pixel_size typically)
    n_slices : int, optional
        Number of depth slices.  Default = round(sample_thickness/pixel_size).
    auto_focus : bool
        If True, find the optimal focus distance per projection by maximising
        sharpness.  Default False (focus fixed at sample centre, i.e.
        propagation distance = −sample_thickness/2).
        For pure-phase objects with very small δ (≲10⁻⁷) the intensity
        contrast developed by propagation is only ~1 % and autofocus may
        be unreliable; disable it and use a fixed focus.
    focus_z_range : tuple (z_min, z_max) [m], optional
        Search range for autofocus propagation distance.
        Default: (−sample_thickness × 1.5, 0).
    focus_n_search : int
        Number of distances evaluated during autofocus.  Default 50.
    focus_metric : str
        Sharpness metric for autofocus.  Default 'tv_intensity'.
    filter_name : str
        Ramp filter: 'ram-lak' (default) or 'none' (no filter, for testing).
    verbose : bool
    device : str or torch.device
        Compute backend:
          'auto'  → GPU (CUDA/MPS) if PyTorch + a GPU are available,
                    otherwise the NumPy CPU path (default).
          'cpu'   → force the NumPy CPU path.
          'cuda' / 'mps' / torch.device → force that GPU.
        The GPU path is mathematically identical to the CPU path but
        5–10× faster (it batches the depth propagation and reuses the
        validated ``rotate_volume_torch`` rotation from the two-pass code).

    Returns
    -------
    delta_vol : ndarray, real, shape (n_slices, Ny, Nx)
        Reconstructed refractive-index decrement volume δ(r).
    focus_distances : ndarray, shape (N_angles,)
        Propagation distances used for each projection [m].
        All zero if auto_focus=False.

    Notes
    -----
    **Memory**: the depth stack per angle has shape (n_slices, Ny, Nx).
    On the GPU path a complex64 kernel stack of the same shape is held
    persistently (≈760 MB for 494×389×494) plus a transient field stack of
    the same size — comfortably within a 48 GB A40.

    **Speed**: For N_angles=450, n_slices=64, (Ny,Nx)=(350,333):
    ≈8 min on CPU.  Full-resolution (n_slices=Nz≈494) is ≈60–90 min on CPU
    but ≈5–8 min on an A40 GPU.

    **Normalization**: standard parallel-beam FBP factor π/(2·N_angles).
    """
    theta = np.asarray(theta, dtype=np.float64)
    N_angles, Ny, Nx = u_exit_stack.shape
    k0 = 2.0 * np.pi / wavelength

    if n_slices is None:
        n_slices = int(round(sample_thickness / pixel_size))
    Nz = n_slices

    # ── Dispatch to the GPU backend when available/requested ───────────────
    use_torch, _torch, _mst, _dev = _resolve_fbap_device(device)
    if use_torch:
        if verbose:
            print(f"  FBaP backend: {_mst.device_info(_dev)}", flush=True)
        return _filtered_back_propagation_torch(
            u_exit_stack, theta, wavelength, pixel_size, sample_thickness,
            Nz, auto_focus, focus_z_range, focus_n_search, focus_metric,
            filter_name, verbose, _torch, _mst, _dev)
    if verbose:
        print("  FBaP backend: NumPy CPU", flush=True)

    # Depth positions relative to sample centre (z=0 at centre)
    # z = -t/2 → entrance face,  z = +t/2 → exit face
    z_slices = np.linspace(-sample_thickness / 2.0,
                            sample_thickness / 2.0, Nz)
    dz = float(z_slices[1] - z_slices[0])   # positive [m]

    if focus_z_range is None:
        focus_z_range = (-sample_thickness * 1.5, 0.0)

    # Pre-compute squared-frequency grid (reused for all propagations)
    fy = fftfreq(Ny, d=pixel_size)
    fx = fftfreq(Nx, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    f2 = (FX ** 2 + FY ** 2).astype(np.float64)

    # Pre-compute incremental step kernel for depth propagation:
    # Moving one step Δz further from the exit face (deeper into the sample)
    # corresponds to propagating BACKWARD by Δz → distance = −Δz.
    # H_step = exp(+iπλΔz f²)
    H_step = np.exp(1j * np.pi * wavelength * dz * f2)

    # Ram-Lak ramp filter
    if filter_name == 'ram-lak':
        n_pad = max(64, int(2 ** np.ceil(np.log2(2 * Nx))))
        ramp = 2.0 * np.fft.rfftfreq(n_pad)    # |ω| on one-sided spectrum
    else:
        n_pad = None
        ramp = None

    delta_vol = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    focus_distances = np.zeros(N_angles)

    t_total = time.time()

    for ai in range(N_angles):
        t_iter = time.time()

        u = u_exit_stack[ai].astype(complex)
        u_ft = fft2(u)       # compute Fourier transform once

        # ------------------------------------------------------------------
        # Step 1: Digital refocus to sample centre
        # ------------------------------------------------------------------
        if auto_focus:
            d_focus, _, _ = find_focus(
                u, wavelength, pixel_size,
                z_range=focus_z_range,
                n_search=focus_n_search,
                metric=focus_metric,
                sample_thickness=sample_thickness,
            )
        else:
            # Fixed focus at sample centre: propagate backward by t/2
            d_focus = -sample_thickness / 2.0

        focus_distances[ai] = d_focus

        # Apply focus propagation in Fourier domain
        H_focus = np.exp(-1j * np.pi * wavelength * d_focus * f2)
        u_focused_ft = u_ft * H_focus   # focused at sample centre (z=0)

        # ------------------------------------------------------------------
        # Steps 2 & 3: Propagate to all depth slices, extract phase
        #
        # The focused wavefield is at z=0 (sample centre).
        # To reach depth z_j we propagate by z_j:
        #   z_j > 0 → forward (toward exit face)
        #   z_j < 0 → backward (toward entrance face)
        #
        # Strategy: use incremental propagation starting from the exit face
        # end (j = Nz-1, z = +t/2) and step backward toward j=0.
        # At j=Nz-1: additional propagation from centre = +t/2.
        # At j=0:    additional propagation from centre = -t/2.
        #
        # H_at_exit = exp(−iπλ(+t/2) f²)
        # Then multiply by H_step = exp(+iπλΔz f²) for each backward step.
        # ------------------------------------------------------------------

        # Start at exit face (j = Nz-1, z = +t/2)
        H_init = np.exp(-1j * np.pi * wavelength * (sample_thickness / 2.0) * f2)
        current_ft = u_focused_ft * H_init

        phase_stack = np.empty((Nz, Ny, Nx), dtype=np.float64)

        for j in range(Nz - 1, -1, -1):
            if j < Nz - 1:
                # One backward step of Δz (from exit toward entrance)
                current_ft = current_ft * H_step
            phase_stack[j] = np.angle(ifft2(current_ft))

        # ------------------------------------------------------------------
        # Step 4: Ram-Lak ramp filter the depth-resolved phase images.
        #
        # The back-propagated phase φ_j(x, y; θ) = angle(ψ(z_j)) is used
        # directly as the sinogram for depth z_j.  This is correct because:
        #   • At depth z_j features at that depth are in-focus in φ_j.
        #   • Features at other depths are defocused (blurred).
        #   • Coherent FBP summation across angles reinforces in-focus
        #     features at their correct 3-D position and averages out the
        #     defocused contributions from other depths.
        #
        # This is the method described in Aidukas et al. (2024), steps 3–4
        # ("extract phase images … filter as in FBP").  The conversion from
        # back-projected phase to δ is done after accumulation (step 5)
        # using the same −1/(k₀·pixel) factor as standard FBP.
        #
        # Note: for a sample smaller than the depth of field (DoF > t), all
        # φ_j ≈ φ_total and FBaP degenerates to standard FBP.  For a sample
        # larger than DoF (as in Aidukas et al.), the depth-dependent focus
        # provides genuine 3-D resolution improvement.
        # ------------------------------------------------------------------
        if ramp is not None:
            for j in range(Nz):
                row_f = np.fft.rfft(phase_stack[j], n=n_pad, axis=-1)
                row_f *= ramp[None, :]
                phase_stack[j] = np.fft.irfft(row_f, n=n_pad, axis=-1)[:, :Nx]

        # ------------------------------------------------------------------
        # Step 5: Rotate filtered phase stack and accumulate (3-D FBP).
        #
        # The Nz×Ny×Nx stack is in the beam frame (z-axis = beam direction).
        # Rotating by −θ around y places each depth slice at the correct
        # 3-D position in the lab frame.  At each voxel (x, y, z), exactly
        # one depth slice (at beam-frame depth x sinθ + z cosθ) contributes,
        # making this a proper 3-D back-projection.
        # ------------------------------------------------------------------
        delta_rot = ndimage_rotate(
            phase_stack, -theta[ai],
            axes=(2, 0), reshape=False, order=1,
            mode='constant', cval=0.0,
        )
        delta_vol += delta_rot

        if verbose:
            elapsed  = time.time() - t_iter
            elapsed_total = time.time() - t_total
            done     = ai + 1
            eta_s    = elapsed_total / done * (N_angles - done)
            eta_min  = eta_s / 60.0
            print(f"  FBaP {done:4d}/{N_angles}  θ={theta[ai]:7.2f}°  "
                  f"t={elapsed:.1f}s  "
                  f"elapsed={elapsed_total/60:.1f}min  "
                  f"ETA={eta_min:.1f}min",
                  flush=True)

    # Convert accumulated phase sinogram → δ volume.
    #
    # The accumulated delta_vol is a ramp-filtered, back-projected phase sum.
    # Standard FBP normalisation: multiply by π / (2 · N_angles).
    # Phase–to–δ conversion: divide by −(k₀ · pixel_size), because the
    # phase projection is φ = −k₀ · pixel_size · Σ_z δ  (sign convention).
    #
    # Combined factor: −π / (2 · N_angles · k₀ · pixel_size)
    delta_vol *= -np.pi / (2.0 * N_angles * k0 * pixel_size)

    # Physical constraint: δ ≥ 0
    np.clip(delta_vol, 0.0, None, out=delta_vol)

    if verbose:
        print(f"FBaP done in {time.time()-t_total:.1f} s total.  "
              f"δ ∈ [{delta_vol.min():.3e}, {delta_vol.max():.3e}]")

    return delta_vol, focus_distances
