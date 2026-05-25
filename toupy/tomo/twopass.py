#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Two-pass tomographic reconstruction with multislice wave-propagation correction.

Overview
--------
Standard ptychographic X-ray CT (PXCT) reconstructs a 3-D refractive-index
volume in two sequential steps that are each internally consistent:

  1. Ptychography (2-D, per projection angle) → complex exit waves
  2. FBP (Filtered Back-Projection, assuming straight-ray line integrals)

This works beautifully for hard X-rays and thin/low-Z samples, where the
projection approximation is excellent.  When the sample is thick enough or
energetic enough that diffraction *within* the sample is non-negligible,
however, the chain breaks down:

  * FBP assumes straight-ray propagation (projection approximation).
  * Multislice ptychography correctly models bent rays but uses the
    vacuum Fresnel propagator between slices — inconsistent with a
    refracting medium.
  * Feeding multislice-derived projections into FBP is a model mismatch.

The :class:`TwoPassReconstructor` resolves these issues:

**Pass 1 — FBP initialisation**
    Standard filtered back-projection of the phase (and optionally
    amplitude) projections from ptychography, using existing Toupy
    infrastructure.  This delivers a high-quality initial guess for
    δ(r) and β(r) in O(N³ log N) time.

**Pass 2 — Multislice refinement**
    Iterative gradient-descent optimisation of the 3-D volume using the
    multislice forward model with a matter-corrected Fresnel propagator.
    At each outer iteration:

      a. The mean refractive index per depth slice δ̄(z) is computed from
         the current 3-D estimate → used to build the matter-corrected
         propagator (eliminates the vacuum-propagator inconsistency).
      b. For each projection angle θ, the 3-D volume is rotated into the
         beam frame, the multislice forward model propagates the probe
         through the rotated slices, and the simulated exit wave is
         compared to the measured one.
      c. The adjoint multislice (exact gradient of the squared-residual
         loss) accumulates the contribution of angle θ into the 3-D
         gradient volume (correct even for non-straight rays).
      d. The 3-D volume is updated with an Adam optimiser step.
      e. Optional: non-negativity projection (δ, β ≥ 0).

Since Pass 1 already delivers a near-correct solution, Pass 2 is a
*perturbative* correction — convergence is fast and the ill-posedness
of a cold-start multislice inversion is avoided.  The 3-D
parameterisation (voxel grid) also eliminates the slice-decomposition
ambiguity that plagues multislice ptychography.

Quick start
-----------
::

    rec = TwoPassReconstructor(
        wavelength  = 0.1e-9,    # 0.1 nm  (12.4 keV)
        pixel_size  = 5e-9,      # 5 nm voxel
        n_slices    = 20,        # slices per projection
    )

    # Pass 1: FBP
    delta_vol, beta_vol = rec.pass1(phase_stack, theta, **fbp_params)

    # Pass 2: multislice refinement
    delta_vol, beta_vol = rec.pass2(
        u_measured_stack, theta, delta_vol, beta_vol,
        probe       = ptychographic_probe,  # or None for plane wave
        n_iter      = 30,
        lr          = 5e-4,
        matter_correction = True,
    )

Parameters guide
----------------
n_slices
    How many thin slabs to use per projection.  The physical slice
    thickness becomes ``Nz * pixel_size / n_slices``.  More slices →
    better diffraction accuracy but higher memory and compute cost.
    Good starting value: 20–40.  For hard X-rays with pixel_size ≈ 5 nm
    and Nz = 256, n_slices = 20 gives Δz = 64 nm per slice.
lr (learning rate)
    Step size for the Adam optimiser.  Start with 1e-4 to 1e-3.
    Values > 1e-2 may cause instability.
matter_correction
    If True (default), use the matter-corrected Fresnel propagator
    instead of the vacuum propagator.  Disable only for debugging.
regularisation
    ``"none"`` (default), ``"tv"`` (total-variation; experimental),
    or ``"positivity"`` (clip δ, β to [0, +∞) after each step).
"""

# standard library
import time
import warnings

# third-party
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

# local toupy
from ..utils import tqdm
from .iradon import backprojector
from .tomorecons import tomo_recons
from .multislice import (
    MultisliceEngine,
    extract_slices_from_volume,
    scatter_gradient_to_volume,
    mean_refractive_index_profile,
)

__all__ = ["TwoPassReconstructor"]


# ---------------------------------------------------------------------------
# Adam optimiser (minimal, dependency-free implementation)
# ---------------------------------------------------------------------------

class _AdamState:
    """Per-variable Adam momentum state."""

    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(shape, dtype=np.float64)   # first moment
        self.v     = np.zeros(shape, dtype=np.float64)   # second moment
        self.t     = 0                                    # step counter

    def step(self, param, grad):
        """
        Update ``param`` in-place with an Adam step.

        Returns
        -------
        param : ndarray  (updated in-place)
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        m_hat  = self.m / (1.0 - self.beta1 ** self.t)
        v_hat  = self.v / (1.0 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param


# ---------------------------------------------------------------------------
# Two-pass reconstructor
# ---------------------------------------------------------------------------

class TwoPassReconstructor:
    """
    Two-pass tomographic reconstruction combining FBP initialisation with
    multislice wave-propagation refinement.

    Parameters
    ----------
    wavelength : float
        X-ray wavelength [m].
    pixel_size : float
        Isotropic voxel side length [m].
    n_slices : int, optional
        Number of multislice slabs per projection angle.  Default 20.
    verbose : bool, optional
        Print progress during Pass 2.  Default True.
    """

    def __init__(self, wavelength, pixel_size, n_slices=20, verbose=True):
        self.wavelength  = float(wavelength)
        self.pixel_size  = float(pixel_size)
        self.n_slices    = int(n_slices)
        self.verbose     = bool(verbose)

    # ==================================================================
    # Pass 1: FBP reconstruction
    # ==================================================================

    def pass1(self, phase_stack, theta, beta_stack=None, **fbp_params):
        """
        Pass 1: FBP reconstruction from projected phase (and optionally
        projected amplitude attenuation) maps.

        This is a thin wrapper around :func:`toupy.tomo.tomo_recons`; all
        keyword arguments are forwarded to it unchanged.

        Parameters
        ----------
        phase_stack : ndarray, real, shape (n_angles, Ny, Nx)
            Projected phase maps φ_θ(x,y) = −k₀∫δ dz [radians].
            One 2-D image per projection angle.
        theta : array_like, shape (n_angles,)
            Projection angles [degrees].
        beta_stack : ndarray, real, shape (n_angles, Ny, Nx), optional
            Projected amplitude attenuation maps −k₀∫β dz [nepers].
            If provided, a separate FBP is run to recover β(r).
        **fbp_params
            Parameters forwarded to :func:`toupy.tomo.tomo_recons`,
            e.g. ``filtertype``, ``freqcutoff``, ``circle``, ``derivatives``.

        Returns
        -------
        delta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Refractive-index decrement volume from FBP.
            δ(r) = −φ_FBP(r) / k₀  (divided by k₀ after reconstruction).
        beta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Absorption volume.  All-zeros if ``beta_stack`` is None.

        Notes
        -----
        The FBP output is the projected *phase* (or attenuation), not yet
        divided by k₀.  The conversion φ_3D(r) → δ(r) = −φ_3D(r)/k₀ is
        performed here.  Ensure that ``phase_stack`` carries the correct
        sign convention (negative for refracting matter).
        """
        n_angles, Ny, Nx = phase_stack.shape
        theta = np.asarray(theta, dtype=np.float64)

        # Default FBP parameters compatible with Toupy conventions
        default_params = dict(
            algorithm       = "FBP",
            filtertype      = "ram-lak",
            freqcutoff      = 1.0,
            circle          = True,
            derivatives     = False,
            calc_derivatives= False,
            weight_angles   = False,
            opencl          = False,
        )
        default_params.update(fbp_params)

        if self.verbose:
            print("Pass 1 — FBP reconstruction")
            t0 = time.time()

        # Reconstruct δ slice-by-slice from phase sinograms
        # Phase stack shape: (n_angles, Ny, Nx) → sinogram shape: (Nx, n_angles)
        # (Toupy convention: sinogram[detector_bin, angle])
        sinogram0 = np.transpose(phase_stack[:, 0, :])  # slice 0 for size estimation
        recons0 = tomo_recons(sinogram0, theta, **default_params)
        Nz_recons, _ = recons0.shape

        # We reconstruct all horizontal slices (axis 1 = y = row)
        delta_vol = np.zeros((Ny, Nz_recons, Nz_recons), dtype=np.float32)
        delta_vol[0] = recons0

        for iy in tqdm(range(1, Ny), desc="Pass 1 FBP (δ)", disable=not self.verbose):
            sino = np.transpose(phase_stack[:, iy, :])
            delta_vol[iy] = tomo_recons(sino, theta, **default_params)

        # Convert from projected phase to δ: φ_FBP = −k₀ · δ_FBP · pixel_size
        # ⟹  δ = −φ_FBP / (k₀ · pixel_size)
        k0  = 2.0 * np.pi / self.wavelength
        delta_vol = (-delta_vol / (k0 * self.pixel_size)).astype(np.float64)
        np.clip(delta_vol, 0.0, None, out=delta_vol)   # physical constraint δ ≥ 0

        # Reconstruct β if amplitude projections are given
        beta_vol = np.zeros_like(delta_vol)
        if beta_stack is not None:
            for iy in tqdm(range(Ny), desc="Pass 1 FBP (β)",
                           disable=not self.verbose):
                sino = np.transpose(beta_stack[:, iy, :])
                beta_vol[iy] = tomo_recons(sino, theta, **default_params)
            beta_vol = (-beta_vol / (k0 * self.pixel_size)).astype(np.float64)
            np.clip(beta_vol, 0.0, None, out=beta_vol)

        # FBP returns (Ny, Nz, Nx) — height · depth · transverse.
        # The multislice engine expects (Nz, Ny, Nx) — depth · height · transverse.
        # Transpose to the canonical convention used throughout Pass 2.
        delta_vol = np.ascontiguousarray(np.transpose(delta_vol, (1, 0, 2)))
        beta_vol  = np.ascontiguousarray(np.transpose(beta_vol,  (1, 0, 2)))

        if self.verbose:
            print(f"  Pass 1 done in {time.time()-t0:.1f} s.  "
                  f"δ: [{delta_vol.min():.3e}, {delta_vol.max():.3e}]")

        return delta_vol, beta_vol

    # ==================================================================
    # Pass 2: multislice refinement
    # ==================================================================

    def pass2(
        self,
        u_measured_stack,
        theta,
        delta_vol,
        beta_vol,
        probe             = None,
        n_iter            = 30,
        lr                = 1e-3,
        matter_correction = True,
        regularisation    = "positivity",
        batch_size        = None,
        callback          = None,
    ):
        """
        Pass 2: iterative multislice wave-propagation refinement.

        Uses the Pass-1 volume as initialisation and refines it by
        minimising the squared residual between simulated and measured
        exit waves, using the multislice forward model with an optional
        matter-corrected Fresnel propagator.

        Parameters
        ----------
        u_measured_stack : ndarray, complex, shape (n_angles, Ny, Nx)
            Measured complex exit waves, one per projection angle.
            Typically the ptychographic reconstructions divided by the
            probe (i.e. the complex transmission function of the object).
        theta : array_like, shape (n_angles,)
            Projection angles in degrees (same order as ``u_measured_stack``).
        delta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Initial δ(r) volume from Pass 1.  Modified in-place and returned.
        beta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Initial β(r) volume from Pass 1.  Modified in-place and returned.
        probe : ndarray, complex, shape (Ny, Nx), optional
            Incident probe wavefield.  If None, a unit plane wave is used.
        n_iter : int, optional
            Number of outer (volume-update) iterations.  Default 30.
        lr : float, optional
            Adam learning rate.  Default 1e-3.
        matter_correction : bool, optional
            Use the matter-corrected Fresnel propagator.  Default True.
        regularisation : str, optional
            ``"none"``       — no regularisation.
            ``"positivity"`` — clip δ, β to [0, +∞) after each step.
            Default ``"positivity"``.
        batch_size : int, optional
            Number of angles to process per gradient-accumulation step
            (stochastic gradient).  Default None = all angles (full batch).
        callback : callable, optional
            Called as ``callback(iteration, loss, delta_vol, beta_vol)``
            at the end of each iteration.  Useful for monitoring or saving
            intermediate results.

        Returns
        -------
        delta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Refined refractive-index decrement volume.
        beta_vol : ndarray, real, shape (Nz, Ny, Nx)
            Refined absorption volume.
        history : dict
            ``history["loss"]`` : list of per-iteration total loss values.
            ``history["time"]`` : list of per-iteration wall-clock times [s].

        Notes
        -----
        Memory usage
            Intermediate wavefields (one per slice, per angle in batch)
            are stored on the CPU as complex128 arrays.  Peak memory per
            angle ≈ 2 · n_slices · Ny · Nx · 16 bytes.  For a 256² grid
            with n_slices = 20: ≈ 42 MB per angle.  With batch_size = 5:
            ≈ 210 MB.  Reduce n_slices or batch_size for large volumes.

        Convergence
            For hard X-rays, Pass 1 is already very close to the truth
            and the correction in Pass 2 is small.  Ten to twenty
            iterations at lr = 1e-3 typically suffice to remove the
            residual model-mismatch artefacts.
        """
        theta   = np.asarray(theta, dtype=np.float64)
        n_angles, Ny, Nx = u_measured_stack.shape
        Nz = delta_vol.shape[0]

        # Initialise probe (plane wave if not provided)
        if probe is None:
            probe = np.ones((Ny, Nx), dtype=complex)
        else:
            probe = np.asarray(probe, dtype=complex)

        # Multislice engine
        slice_thickness = Nz * self.pixel_size / self.n_slices
        engine = MultisliceEngine(
            shape          = (Ny, Nx),
            pixel_size     = self.pixel_size,
            wavelength     = self.wavelength,
            slice_thickness= slice_thickness,
        )

        # Adam optimiser states for δ and β
        adam_delta = _AdamState(delta_vol.shape, lr=lr)
        adam_beta  = _AdamState(beta_vol.shape,  lr=lr)

        # Batch configuration
        angle_indices = np.arange(n_angles)
        if batch_size is None:
            batch_size = n_angles

        history = {"loss": [], "time": []}

        if self.verbose:
            print(f"\nPass 2 — Multislice refinement")
            print(f"  n_iter={n_iter}, lr={lr}, n_slices={self.n_slices}, "
                  f"matter_correction={matter_correction}, "
                  f"batch_size={batch_size}")
            print(f"  slice_thickness = {slice_thickness*1e9:.2f} nm")

        t_total = time.time()

        for it in range(n_iter):
            t_iter = time.time()

            # Recompute matter-corrected propagator using current volume
            # (updated once per outer iteration — cheap and sufficient)
            # delta_means_per_angle is computed inside _angle_gradient

            total_loss  = 0.0
            grad_delta  = np.zeros_like(delta_vol)
            grad_beta   = np.zeros_like(beta_vol)

            # Shuffle angles for stochastic gradient
            batch_idx = np.random.choice(angle_indices, size=batch_size, replace=False)

            for ai in batch_idx:
                theta_i   = theta[ai]
                u_meas_i  = u_measured_stack[ai]

                loss_i, gd_i, gb_i = self._angle_gradient(
                    delta_vol, beta_vol, theta_i,
                    probe, u_meas_i, engine,
                    matter_correction=matter_correction,
                )
                total_loss += loss_i
                grad_delta += gd_i
                grad_beta  += gb_i

            # Normalise gradient by batch size
            grad_delta /= batch_size
            grad_beta  /= batch_size

            # Adam update
            adam_delta.step(delta_vol, grad_delta)
            adam_beta.step( beta_vol,  grad_beta)

            # Regularisation
            if regularisation == "positivity":
                np.clip(delta_vol, 0.0, None, out=delta_vol)
                np.clip(beta_vol,  0.0, None, out=beta_vol)

            elapsed = time.time() - t_iter
            history["loss"].append(total_loss)
            history["time"].append(elapsed)

            if self.verbose:
                print(f"  Iter {it+1:3d}/{n_iter}  loss={total_loss:.4e}  "
                      f"t={elapsed:.1f}s  "
                      f"δ∈[{delta_vol.min():.2e},{delta_vol.max():.2e}]")

            if callback is not None:
                callback(it + 1, total_loss, delta_vol, beta_vol)

        if self.verbose:
            print(f"Pass 2 done in {time.time()-t_total:.1f} s total.")

        return delta_vol, beta_vol, history

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _angle_gradient(self, delta_vol, beta_vol, theta_deg,
                        probe, u_measured, engine, matter_correction=True):
        """
        Forward + adjoint pass for a single projection angle.

        Extracts oblique slices from the 3-D volume at ``theta_deg``,
        runs the multislice forward and backward passes, then back-rotates
        the slice gradients into the 3-D volume gradient.

        Parameters
        ----------
        delta_vol : ndarray (Nz, Ny, Nx)
        beta_vol  : ndarray (Nz, Ny, Nx)
        theta_deg : float
        probe : ndarray, complex (Ny, Nx)
        u_measured : ndarray, complex (Ny, Nx)
        engine : MultisliceEngine
        matter_correction : bool

        Returns
        -------
        loss : float
        grad_delta_vol : ndarray (Nz, Ny, Nx)
        grad_beta_vol  : ndarray (Nz, Ny, Nx)
        """
        # 1. Extract slices perpendicular to the beam at this angle
        delta_slices, beta_slices, delta_means = extract_slices_from_volume(
            delta_vol, beta_vol, theta_deg, n_slices=self.n_slices
        )

        # 2. Build matter-corrected propagator (or vacuum)
        d_means = delta_means if matter_correction else None

        # 3. Forward + adjoint multislice
        loss, grad_delta_slices, grad_beta_slices, _ = engine.loss_and_gradient(
            delta_slices, beta_slices, probe, u_measured, delta_means=d_means
        )

        # 4. Back-rotate slice gradients → 3-D volume gradients
        grad_delta_vol, grad_beta_vol = scatter_gradient_to_volume(
            grad_delta_slices, grad_beta_slices,
            theta_deg, delta_vol.shape, n_slices=self.n_slices,
        )

        return loss, grad_delta_vol, grad_beta_vol

    # ==================================================================
    # Diagnostics / utilities
    # ==================================================================

    def simulate_exit_wave(self, delta_vol, beta_vol, theta_deg,
                           probe=None, matter_correction=True):
        """
        Simulate the exit wave at a given projection angle.

        Useful for assessing convergence: compare with the measured exit
        wave to see how well the current volume explains the data.

        Parameters
        ----------
        delta_vol : ndarray (Nz, Ny, Nx)
        beta_vol  : ndarray (Nz, Ny, Nx)
        theta_deg : float
        probe : ndarray, complex (Ny, Nx), optional
        matter_correction : bool, optional

        Returns
        -------
        u_sim : ndarray, complex (Ny, Nx)
            Simulated exit wave.
        """
        Nz, Ny, Nx = delta_vol.shape
        if probe is None:
            probe = np.ones((Ny, Nx), dtype=complex)

        slice_thickness = Nz * self.pixel_size / self.n_slices
        engine = MultisliceEngine(
            shape          = (Ny, Nx),
            pixel_size     = self.pixel_size,
            wavelength     = self.wavelength,
            slice_thickness= slice_thickness,
        )

        delta_slices, beta_slices, delta_means = extract_slices_from_volume(
            delta_vol, beta_vol, theta_deg, n_slices=self.n_slices
        )
        d_means = delta_means if matter_correction else None
        u_sim = engine.forward(delta_slices, beta_slices, probe, delta_means=d_means)
        return u_sim

    def projection_residuals(self, u_measured_stack, theta,
                             delta_vol, beta_vol, probe=None,
                             matter_correction=True):
        """
        Compute per-angle residual norms ‖U_sim(θ) − U_meas(θ)‖.

        Parameters
        ----------
        u_measured_stack : ndarray, complex (n_angles, Ny, Nx)
        theta : array_like (n_angles,)
        delta_vol : ndarray (Nz, Ny, Nx)
        beta_vol  : ndarray (Nz, Ny, Nx)
        probe : ndarray, complex (Ny, Nx), optional
        matter_correction : bool, optional

        Returns
        -------
        residuals : ndarray, real, shape (n_angles,)
        """
        theta = np.asarray(theta)
        n_angles, Ny, Nx = u_measured_stack.shape
        if probe is None:
            probe = np.ones((Ny, Nx), dtype=complex)

        residuals = np.empty(n_angles)
        for ai, th in enumerate(tqdm(theta, desc="Computing residuals",
                                     disable=not self.verbose)):
            u_sim = self.simulate_exit_wave(delta_vol, beta_vol, th,
                                            probe=probe,
                                            matter_correction=matter_correction)
            residuals[ai] = np.linalg.norm(u_sim - u_measured_stack[ai])
        return residuals
