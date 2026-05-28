#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch backend for the multislice wave-propagation engine.

This module mirrors ``multislice.py`` but uses PyTorch as the compute
backend, giving transparent access to:

    * CUDA  — NVIDIA GPUs via cuFFT / cuBLAS
    * MPS   — Apple Silicon GPU via Metal Performance Shaders
    * CPU   — multi-threaded torch (comparable to NumPy at small sizes;
              faster at large sizes due to better thread-pool utilisation)

Automatic device selection
--------------------------
    device = select_device()          # CUDA > MPS > CPU  (automatic)
    device = select_device("cpu")     # force CPU
    device = select_device("mps")     # force Apple GPU
    device = select_device("cuda")    # force NVIDIA GPU

Gradient checkpointing
-----------------------
For volumes >= 512³ the intermediate wavefields stored during the forward
pass start to dominate GPU memory.  ``TorchMultisliceEngine`` supports
gradient checkpointing: only the entry wavefield of each of the
``n_checkpoints`` segments is retained; within-segment forward passes are
recomputed during backpropagation.  Memory scales as
O(n_checkpoints × N²) instead of O(N_slices × N²).

MPS rotation workaround
------------------------
PyTorch's ``grid_sampler_3d`` is not yet implemented on MPS (as of 2.x).
``rotate_volume_torch`` uses batched 2D ``grid_sample`` over the y-axis
instead, which is mathematically equivalent for a rotation in the xz-plane
and runs natively on MPS, CUDA, and CPU.

Precision
---------
This module uses **float32 / complex64** throughout (even on CPU) for
consistency with GPU paths.  float32 gives ~7 significant digits, more than
sufficient for refractive-index values in the range 10⁻⁶ – 10⁻³.  The
NumPy module (``multislice.py``) retains float64 for users who prefer it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "select_device",
    "device_info",
    "warmup_device",
    "rotate_volume_torch",
    "TorchFresnelPropagator",
    "TorchMultisliceEngine",
    "extract_slices_torch",
    "scatter_gradient_torch",
    "tv_grad_torch",
    "TorchAdamState",
]


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def select_device(prefer: Optional[str] = None) -> torch.device:
    """
    Return the best available torch device.

    Priority (when ``prefer`` is ``None``): CUDA > MPS > CPU.

    Parameters
    ----------
    prefer : str, optional
        ``'cuda'``, ``'mps'``, or ``'cpu'``.  Raises ``RuntimeError`` if
        the requested device is unavailable.

    Returns
    -------
    torch.device
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if prefer == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable one-line description of the device."""
    name = str(device)
    if name.startswith("cuda"):
        gpu   = torch.cuda.get_device_name(device)
        mem   = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA  — {gpu}  ({mem:.0f} GB VRAM)"
    if name == "mps":
        return "MPS   — Apple Silicon GPU (Metal Performance Shaders)"
    return f"CPU   — torch {torch.__version__}  ({torch.get_num_threads()} threads)"


def warmup_device(device: torch.device, size: int = 64) -> None:
    """
    Run a tiny FFT and element-wise op to trigger JIT compilation.

    The first operation on a GPU device (especially MPS) can take several
    hundred milliseconds for shader/kernel compilation.  Call this once
    before timing-sensitive code.
    """
    x = torch.randn(size, size, dtype=torch.complex64, device=device)
    _ = torch.fft.ifft2(torch.fft.fft2(x) * x)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Volume rotation  (MPS-compatible via batched 2-D grid_sample)
# ---------------------------------------------------------------------------

def rotate_volume_torch(vol: torch.Tensor, theta_deg: float) -> torch.Tensor:
    """
    Rotate a 3-D volume in the (x, z) plane.

    Equivalent to ``scipy.ndimage.rotate(vol, theta_deg, axes=(2, 0),
    reshape=False, order=1, mode='constant', cval=0)`` in the volume
    interior.  Near the boundary the two implementations differ slightly:
    scipy returns a hard 0 for any source coordinate outside the domain,
    while PyTorch bilinear still weights the nearest valid pixel; in practice
    this only affects voxels within ~1 pixel of the rotated cube's edge and
    is inconsequential for tomographic volumes where boundary voxels are air.

    Implementation note
    -------------------
    ``torch.nn.functional.grid_sampler_3d`` is not available on MPS.
    We work around this by treating each (Nz, Nx) face (constant y-slice)
    as an independent 2-D image and applying batched ``grid_sample`` over
    the Ny slices.  The result matches scipy in the interior to < 1e-4 and
    runs on MPS, CUDA, and CPU.

    The inverse-mapping matrix (output → input, as required by
    ``affine_grid``) follows the scipy ``axes=(2, 0)`` convention::

        ⎡ x_in ⎤   ⎡ cos θ  −sin θ ⎤   ⎡ x_out ⎤
        ⎣ z_in ⎦ = ⎣ sin θ   cos θ ⎦ × ⎣ z_out ⎦

    Parameters
    ----------
    vol : torch.Tensor, real float32, shape (Nz, Ny, Nx)
    theta_deg : float
        Rotation angle in degrees (same sign convention as scipy).

    Returns
    -------
    torch.Tensor, same shape, dtype, and device as ``vol``.
    """
    Nz, Ny, Nx = vol.shape
    device = vol.device
    dtype_in = vol.dtype

    theta = math.radians(theta_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    # 2×3 affine matrix  [x_in, z_in] = R @ [x_out, z_out, 1]
    # PyTorch affine_grid encodes the OUTPUT→INPUT mapping.
    #
    # scipy.ndimage.rotate(axes=(2,0), angle=θ) uses the rotation matrix
    #   [[cos, sin], [-sin, cos]]
    # acting on (axis2, axis0) = (x, z) in the OUTPUT→INPUT sense:
    #   x_in_c =  c·x_out_c + s·z_out_c          (centered pixel coords)
    #   z_in_c = -s·x_out_c + c·z_out_c
    # Forward (INPUT→OUTPUT):
    #   x_out_c =  c·x_in_c - s·z_in_c
    #   z_out_c =  s·x_in_c + c·z_in_c
    #
    # affine_grid works in NORMALISED coords [-1, 1].  The mapping between
    # normalised and pixel centred coords is:
    #   x_norm = 2·x_c / (Nx-1),   z_norm = 2·z_c / (Nz-1)
    #
    # Substituting the OUTPUT→INPUT mapping into normalised coords gives:
    #   x_in_norm =  c·x_out_norm + s·(Nz-1)/(Nx-1)·z_out_norm
    #   z_in_norm = -s·(Nx-1)/(Nz-1)·x_out_norm + c·z_out_norm
    #
    # When Nz == Nx the aspect-ratio factors cancel and we get [[c,+s],[-s,c]].
    #
    # NOTE: an earlier version of this matrix had the signs of the off-diagonal
    # elements flipped ([[c,-s*az],[+s*ax,c]]), which caused the GPU rotation to
    # go in the OPPOSITE direction from scipy.  This produced a mirror-image
    # reconstruction on GPU vs CPU and has now been corrected.
    az = (Nz - 1) / (Nx - 1) if Nx != 1 else 1.0   # aspect ratio z/x
    ax = (Nx - 1) / (Nz - 1) if Nz != 1 else 1.0   # aspect ratio x/z
    R = torch.tensor([[c,     s * az, 0.0],
                      [-s * ax,  c,  0.0]], dtype=torch.float32, device=device)

    # Compute the sampling grid for a SINGLE y-slice (the Nz×Nx plane).
    # All y-slices share the same xz-rotation, so we only need shape (1,1,Nz,Nx).
    # This allocates ~2 MB instead of the ~766 MB that would result from passing
    # a full (Ny, 2, 3) R_batch to affine_grid.
    R_1     = R.unsqueeze(0)                                          # 1,2,3
    grid_1  = F.affine_grid(R_1, (1, 1, Nz, Nx), align_corners=True) # 1,Nz,Nx,2

    # Reshape volume to batch of 2-D images: (Ny, 1, Nz, Nx)
    vol_batch = vol.to(torch.float32).permute(1, 0, 2).unsqueeze(1)  # Ny,1,Nz,Nx

    # Expand grid to (Ny, Nz, Nx, 2) — zero-copy view; grid_sample accepts
    # non-contiguous grids on CUDA, CPU, and MPS (PyTorch ≥ 2.0).
    grid    = grid_1.expand(Ny, -1, -1, -1)                          # Ny,Nz,Nx,2

    rotated = F.grid_sample(vol_batch, grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True)                       # Ny,1,Nz,Nx

    out = rotated.squeeze(1).permute(1, 0, 2)                        # Nz,Ny,Nx
    return out.to(dtype_in)


# ---------------------------------------------------------------------------
# Fresnel propagator
# ---------------------------------------------------------------------------

class TorchFresnelPropagator:
    """
    Paraxial Fresnel propagator in the Fourier domain (PyTorch backend).

    Mirrors :class:`~toupy.tomo.multislice.FresnelPropagator` with
    complex64 arithmetic on the selected device.

    Parameters
    ----------
    shape : tuple (Ny, Nx)
    pixel_size : float   [m]
    wavelength : float   [m]
    device : torch.device
    """

    def __init__(self, shape: Tuple[int, int], pixel_size: float,
                 wavelength: float, device: torch.device) -> None:
        self.shape      = shape
        self.pixel_size = float(pixel_size)
        self.wavelength = float(wavelength)
        self.device     = device

        Ny, Nx = shape
        fx = torch.fft.fftfreq(Nx, d=self.pixel_size, device=device)
        fy = torch.fft.fftfreq(Ny, d=self.pixel_size, device=device)
        # indexing='xy': first index = x (cols), second = y (rows)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")
        self._f2 = (FX ** 2 + FY ** 2).to(torch.float32)   # (Ny, Nx)

    # ------------------------------------------------------------------
    def _kernel(self, distance: float, delta_mean: float = 0.0) -> torch.Tensor:
        n_mean  = 1.0 - float(delta_mean)
        lam_eff = self.wavelength / n_mean
        phase   = -math.pi * lam_eff * float(distance) * self._f2
        return torch.exp(1j * phase)                        # complex64

    def propagate(self, wavefield: torch.Tensor,
                  distance: float, delta_mean: float = 0.0) -> torch.Tensor:
        H = self._kernel(distance, delta_mean)
        return torch.fft.ifft2(torch.fft.fft2(wavefield) * H)

    def propagate_adjoint(self, wavefield: torch.Tensor,
                          distance: float, delta_mean: float = 0.0) -> torch.Tensor:
        return self.propagate(wavefield, -distance, delta_mean)


# ---------------------------------------------------------------------------
# Multislice engine
# ---------------------------------------------------------------------------

class TorchMultisliceEngine:
    """
    Forward and adjoint multislice engine (PyTorch backend).

    Mirrors :class:`~toupy.tomo.multislice.MultisliceEngine` with float32
    arithmetic.  All tensors live on ``device``.

    Parameters
    ----------
    shape : tuple (Ny, Nx)
    pixel_size : float   [m]
    wavelength : float   [m]
    slice_thickness : float   [m]
    device : torch.device, optional
        Auto-selected if ``None``.
    n_checkpoints : int, optional
        Number of gradient-checkpointing segments (default 1 = no
        checkpointing).  Increase to reduce GPU memory at the cost of one
        extra partial forward pass per segment.
    """

    def __init__(self, shape: Tuple[int, int], pixel_size: float,
                 wavelength: float, slice_thickness: float,
                 device: Optional[torch.device] = None,
                 n_checkpoints: int = 1) -> None:
        self.shape          = shape
        self.device         = device or select_device()
        self.pixel_size     = float(pixel_size)
        self.wavelength     = float(wavelength)
        self.slice_thickness = float(slice_thickness)
        self.k0             = 2.0 * math.pi / self.wavelength
        self.n_checkpoints  = max(1, int(n_checkpoints))
        self._prop          = TorchFresnelPropagator(
                                  shape, pixel_size, wavelength, self.device)

    # ------------------------------------------------------------------
    def _to(self, arr) -> torch.Tensor:
        """Convert numpy array or torch tensor to complex64 on device."""
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return arr.to(device=self.device, dtype=torch.complex64)

    def _to_real(self, arr) -> torch.Tensor:
        """Convert numpy array or torch tensor to float32 on device."""
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return arr.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    def transmission(self, delta_slice: torch.Tensor,
                     beta_slice: torch.Tensor) -> torch.Tensor:
        """
        T(x,y) = exp(−k₀ (β + iδ) Δz)
        """
        dz = self.slice_thickness
        d  = self._to_real(delta_slice)
        b  = self._to_real(beta_slice)
        return torch.exp(-self.k0 * dz * (b + 1j * d))     # complex64

    # ------------------------------------------------------------------
    def forward(self, delta_slices, beta_slices, probe,
                delta_means=None, store_wavefields: bool = False):
        """
        Forward multislice propagation.

        Parameters
        ----------
        delta_slices : array-like, shape (n_slices, Ny, Nx)
        beta_slices  : array-like, shape (n_slices, Ny, Nx)
        probe        : array-like, complex, shape (Ny, Nx)
        delta_means  : array-like of float, length n_slices, optional
        store_wavefields : bool

        Returns
        -------
        U_exit : torch.Tensor, complex64, shape (Ny, Nx)
        wavefields : dict   (only when ``store_wavefields=True``)
        """
        if isinstance(delta_slices, np.ndarray):
            delta_slices = torch.from_numpy(delta_slices)
        if isinstance(beta_slices, np.ndarray):
            beta_slices = torch.from_numpy(beta_slices)

        n_slices = delta_slices.shape[0]
        if delta_means is None:
            delta_means = torch.zeros(n_slices, dtype=torch.float32)
        elif isinstance(delta_means, np.ndarray):
            delta_means = torch.from_numpy(delta_means)

        # Move delta_means to CPU *once* as a plain Python list.
        # Calling float(delta_means[j]) inside the loop forces a device→host
        # sync on every slab (one Metal command-buffer flush per call on MPS,
        # or a cudaDeviceSynchronize equivalent on CUDA).  With n_slices=16 and
        # 450 angles that is 14 400 stalls per iteration — the dominant runtime
        # cost on both MPS and CUDA.  One .cpu().tolist() costs a single sync.
        if isinstance(delta_means, torch.Tensor) and delta_means.device.type != 'cpu':
            _dm = delta_means.detach().cpu().tolist()
        elif isinstance(delta_means, torch.Tensor):
            _dm = delta_means.tolist()
        else:
            _dm = [float(x) for x in delta_means]

        dz = self.slice_thickness
        U  = self._to(probe).clone()

        if store_wavefields:
            W_list, T_list, U_list = [], [], [U.clone()]

        for j in range(n_slices):
            T_j = self.transmission(delta_slices[j], beta_slices[j])
            W_j = T_j * U
            U   = self._prop.propagate(W_j, dz, delta_mean=_dm[j])
            if store_wavefields:
                T_list.append(T_j)
                W_list.append(W_j)
                U_list.append(U.clone())

        if store_wavefields:
            return U, {"W": W_list, "T": T_list, "U": U_list}
        return U

    # ------------------------------------------------------------------
    def gradient(self, wavefields: Dict, u_measured,
                 delta_means=None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Adjoint (backward) pass — same formula as the NumPy version.

        Returns
        -------
        grad_delta : float32, shape (n_slices, Ny, Nx)
        grad_beta  : float32, shape (n_slices, Ny, Nx)
        loss       : float
        """
        W_list   = wavefields["W"]
        T_list   = wavefields["T"]
        U_list   = wavefields["U"]
        n_slices = len(W_list)

        if delta_means is None:
            delta_means = torch.zeros(n_slices, dtype=torch.float32)
        elif isinstance(delta_means, np.ndarray):
            delta_means = torch.from_numpy(delta_means)

        dz    = self.slice_thickness
        scale = self.k0 * dz

        U_exit   = U_list[n_slices]
        u_meas_t = self._to(u_measured)
        residual = U_exit - u_meas_t
        loss     = 0.5 * float(residual.abs().pow(2).sum().cpu())

        # One sync to get all delta_means on CPU (avoids per-slab sync in loop)
        if isinstance(delta_means, torch.Tensor) and delta_means.device.type != 'cpu':
            _dm = delta_means.detach().cpu().tolist()
        elif isinstance(delta_means, torch.Tensor):
            _dm = delta_means.tolist()
        else:
            _dm = [float(x) for x in delta_means]

        grad_delta = torch.zeros((n_slices,) + self.shape,
                                 dtype=torch.float32, device=self.device)
        grad_beta  = torch.zeros((n_slices,) + self.shape,
                                 dtype=torch.float32, device=self.device)
        r = residual.clone()

        for j in range(n_slices - 1, -1, -1):
            s_j = self._prop.propagate_adjoint(r, dz, delta_mean=_dm[j])
            cross           = W_list[j] * s_j.conj()
            grad_delta[j]   =  scale * cross.imag
            grad_beta[j]    = -scale * cross.real
            r = T_list[j].conj() * s_j

        return grad_delta, grad_beta, loss

    # ------------------------------------------------------------------
    def loss_and_gradient(self, delta_slices, beta_slices, probe,
                          u_measured, delta_means=None):
        """Forward + backward in one call (no checkpointing)."""
        u_exit, wf = self.forward(delta_slices, beta_slices, probe,
                                  delta_means=delta_means,
                                  store_wavefields=True)
        gd, gb, loss = self.gradient(wf, u_measured, delta_means=delta_means)
        return loss, gd, gb, u_exit

    # ------------------------------------------------------------------
    def loss_and_gradient_checkpointed(self, delta_slices, beta_slices,
                                       probe, u_measured,
                                       delta_means=None,
                                       n_checkpoints: Optional[int] = None):
        """
        Memory-efficient forward + backward via gradient checkpointing.

        Divides the slice stack into ``n_checkpoints`` segments.  Only the
        entry wavefield of each segment is stored; within-segment forward
        passes are recomputed during the backward pass.

        Memory: O(n_checkpoints × N²)  instead of O(N_slices × N²).
        Compute: ~2× the forward-pass cost (one extra partial forward pass
                 per segment).

        Parameters
        ----------
        n_checkpoints : int, optional
            Overrides ``self.n_checkpoints`` for this call.
        """
        if isinstance(delta_slices, np.ndarray):
            delta_slices = torch.from_numpy(delta_slices)
        if isinstance(beta_slices, np.ndarray):
            beta_slices = torch.from_numpy(beta_slices)

        n_slices = delta_slices.shape[0]
        if delta_means is None:
            delta_means = torch.zeros(n_slices, dtype=torch.float32)
        elif isinstance(delta_means, np.ndarray):
            delta_means = torch.from_numpy(delta_means)

        ncp   = n_checkpoints if n_checkpoints is not None else self.n_checkpoints
        seg   = max(1, n_slices // ncp)               # slices per segment
        n_seg = math.ceil(n_slices / seg)

        dz    = self.slice_thickness
        scale = self.k0 * dz

        # One sync to get all delta_means on CPU (avoids per-slab sync in loop)
        if isinstance(delta_means, torch.Tensor) and delta_means.device.type != 'cpu':
            _dm = delta_means.detach().cpu().tolist()
        elif isinstance(delta_means, torch.Tensor):
            _dm = delta_means.tolist()
        else:
            _dm = [float(x) for x in delta_means]

        # ── Forward pass: store only segment-entry wavefields ──────────
        checkpoints: List[torch.Tensor] = []
        U = self._to(probe).clone()

        for s in range(n_seg):
            checkpoints.append(U.clone())
            j0 = s * seg
            j1 = min((s + 1) * seg, n_slices)
            for j in range(j0, j1):
                T_j = self.transmission(delta_slices[j], beta_slices[j])
                U   = self._prop.propagate(T_j * U, dz, delta_mean=_dm[j])

        u_meas_t = self._to(u_measured)
        residual = U - u_meas_t
        loss     = 0.5 * float(residual.abs().pow(2).sum().cpu())

        grad_delta = torch.zeros((n_slices,) + self.shape,
                                 dtype=torch.float32, device=self.device)
        grad_beta  = torch.zeros((n_slices,) + self.shape,
                                 dtype=torch.float32, device=self.device)
        r = residual.clone()

        # ── Backward pass: recompute each segment on the fly ───────────
        for s in range(n_seg - 1, -1, -1):
            j0 = s * seg
            j1 = min((s + 1) * seg, n_slices)

            # Recompute forward within segment (cheap)
            U_s = checkpoints[s].clone()
            W_seg: List[torch.Tensor] = []
            T_seg: List[torch.Tensor] = []
            for j in range(j0, j1):
                T_j = self.transmission(delta_slices[j], beta_slices[j])
                W_j = T_j * U_s
                U_s = self._prop.propagate(W_j, dz, delta_mean=_dm[j])
                W_seg.append(W_j)
                T_seg.append(T_j)

            # Backward through segment
            for j in range(j1 - 1, j0 - 1, -1):
                jj  = j - j0
                s_j = self._prop.propagate_adjoint(r, dz, delta_mean=_dm[j])
                cross         = W_seg[jj] * s_j.conj()
                grad_delta[j] =  scale * cross.imag
                grad_beta[j]  = -scale * cross.real
                r = T_seg[jj].conj() * s_j

        return loss, grad_delta, grad_beta


# ---------------------------------------------------------------------------
# Volume ↔ slice utilities  (torch versions of the numpy helpers)
# ---------------------------------------------------------------------------

def extract_slices_torch(
    delta_vol: torch.Tensor,
    beta_vol:  torch.Tensor,
    theta_deg: float,
    n_slices:  Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch version of ``extract_slices_from_volume``.

    Rotates the 3-D volume in the (x, z) plane and extracts ``n_slices``
    perpendicular 2-D slabs.

    Parameters
    ----------
    delta_vol : torch.Tensor, float32, shape (Nz, Ny, Nx)
    beta_vol  : torch.Tensor, float32, shape (Nz, Ny, Nx)
    theta_deg : float
    n_slices  : int, optional

    Returns
    -------
    delta_slices : torch.Tensor, shape (n_slices, Ny, Nx)
    beta_slices  : torch.Tensor, shape (n_slices, Ny, Nx)
    delta_means  : torch.Tensor, shape (n_slices,)
    """
    Nz, Ny, Nx = delta_vol.shape

    delta_rot = rotate_volume_torch(delta_vol, theta_deg)
    beta_rot  = rotate_volume_torch(beta_vol,  theta_deg)

    if n_slices is None or n_slices == Nz:
        delta_slices = delta_rot
        beta_slices  = beta_rot
    else:
        k = Nz // n_slices
        if k == 0:
            raise ValueError(
                f"n_slices={n_slices} > Nz={Nz}: cannot have more slices than z-pixels.")
        n_use = k * n_slices
        delta_slices = delta_rot[:n_use].reshape(n_slices, k, Ny, Nx).mean(dim=1)
        beta_slices  = beta_rot[:n_use].reshape(n_slices, k, Ny, Nx).mean(dim=1)

    delta_means = delta_slices.mean(dim=(1, 2))   # (n_slices,)
    return delta_slices, beta_slices, delta_means


def scatter_gradient_torch(
    grad_delta_slices: torch.Tensor,
    grad_beta_slices:  torch.Tensor,
    theta_deg:         float,
    volume_shape:      Tuple[int, int, int],
    n_slices:          Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch version of ``scatter_gradient_to_volume``.

    Adjoint of :func:`extract_slices_torch`: back-rotates per-slice
    gradient maps into the 3-D volume.

    Parameters
    ----------
    grad_delta_slices : torch.Tensor, shape (n_slices, Ny, Nx)
    grad_beta_slices  : torch.Tensor, shape (n_slices, Ny, Nx)
    theta_deg : float
    volume_shape : tuple (Nz, Ny, Nx)
    n_slices : int, optional

    Returns
    -------
    grad_delta_vol : torch.Tensor, shape (Nz, Ny, Nx)
    grad_beta_vol  : torch.Tensor, shape (Nz, Ny, Nx)
    """
    Nz, Ny, Nx = volume_shape
    device = grad_delta_slices.device

    if n_slices is not None and n_slices < Nz:
        k     = Nz // n_slices
        n_use = k * n_slices
        # Each slab gradient is distributed equally over k z-layers
        gd_full = grad_delta_slices.repeat_interleave(k, dim=0) / k
        gb_full = grad_beta_slices.repeat_interleave(k, dim=0) / k
        if n_use < Nz:
            pad = torch.zeros(Nz - n_use, Ny, Nx,
                              dtype=gd_full.dtype, device=device)
            gd_full = torch.cat([gd_full, pad], dim=0)
            gb_full = torch.cat([gb_full, pad], dim=0)
    else:
        gd_full = grad_delta_slices
        gb_full = grad_beta_slices

    # Back-rotate by −theta (adjoint of the forward rotation)
    grad_delta_vol = rotate_volume_torch(gd_full, -theta_deg)
    grad_beta_vol  = rotate_volume_torch(gb_full, -theta_deg)

    return grad_delta_vol, grad_beta_vol


# ---------------------------------------------------------------------------
# TV regulariser
# ---------------------------------------------------------------------------

def tv_grad_torch(vol: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gradient of the smooth isotropic 3-D total-variation functional.

    Computes  ∇R_TV(v) = −div(∇v / |∇v|_ε)  where

        R_TV(v) = Σ_r  √(|∇v(r)|² + ε)
        |∇v|_ε  = √(gz² + gy² + gx² + ε)

    Torch mirror of the NumPy ``_tv_gradient_3d`` in ``twopass.py``;
    both use the same isotropic formula and the same ``eps`` convention
    so that switching between the NumPy and PyTorch backends produces
    identical results.

    Forward differences with zero-Neumann boundary conditions are used
    for ∇v; their adjoint (backward differences) gives −div(p).

    Parameters
    ----------
    vol : torch.Tensor, real float32, shape (Nz, Ny, Nx)
    eps : float
        Smoothing constant that makes R_TV differentiable at flat regions.
        Default 1e-8 (matches ``_tv_gradient_3d``).

    Returns
    -------
    torch.Tensor, same shape, dtype, and device as ``vol``
    """
    # --- forward differences with zero-Neumann BC (last diff = 0) ----------
    gz = torch.zeros_like(vol); gz[:-1]       = vol[1:]       - vol[:-1]
    gy = torch.zeros_like(vol); gy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    gx = torch.zeros_like(vol); gx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]

    # --- smoothed local norm, normalise in-place ---------------------------
    norm = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + eps)
    gz = gz / norm
    gy = gy / norm
    gx = gx / norm

    # --- adjoint backward differences  →  −div(p) = ∇R_TV -----------------
    # (D_d^* p)_i = p_{i-1} − p_i   with  p_{-1} = 0
    #
    # z-axis
    grad = torch.empty_like(vol)
    grad[0,  :, :]  = -gz[0,  :, :]
    grad[1:, :, :]  =  gz[:-1, :, :] - gz[1:, :, :]
    # y-axis (add in-place)
    grad[:,  0, :]  = grad[:,  0, :]  - gy[:,  0, :]
    grad[:, 1:, :]  = grad[:, 1:, :] + gy[:, :-1, :] - gy[:, 1:, :]
    # x-axis (add in-place)
    grad[:, :,  0]  = grad[:, :,  0]  - gx[:, :,  0]
    grad[:, :, 1:]  = grad[:, :, 1:]  + gx[:, :, :-1] - gx[:, :, 1:]

    return grad


# ---------------------------------------------------------------------------
# Minimal Adam optimiser
# ---------------------------------------------------------------------------

class TorchAdamState:
    """
    Lightweight Adam optimiser for a single float32 tensor parameter.

    Mirrors the NumPy ``_Adam`` class in the tutorial but keeps all state
    on ``device`` for zero-copy updates.

    Parameters
    ----------
    shape : tuple
    lr    : float   learning rate (can be updated between steps via ``self.lr``)
    b1    : float   first-moment decay  (default 0.9)
    b2    : float   second-moment decay (default 0.999)
    eps   : float   numerical stabiliser (default 1e-8)
    device : torch.device
    """

    def __init__(self, shape: tuple, lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 device: Optional[torch.device] = None) -> None:
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.t   = 0
        kw = {"dtype": torch.float32, "device": device}
        self.m = torch.zeros(shape, **kw)
        self.v = torch.zeros(shape, **kw)

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """In-place Adam update of ``param``.  Returns ``param``."""
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad ** 2
        m_hat  = self.m / (1.0 - self.b1 ** self.t)
        v_hat  = self.v / (1.0 - self.b2 ** self.t)
        param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return param
