#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multislice wave-propagation engine for tomographic reconstruction.

This module provides the building blocks for the two-pass tomographic
reconstruction method implemented in :mod:`toupy.tomo.twopass`.  It
contains a Fresnel propagator that can be corrected for the mean
refractive index of the traversed material (vacuum propagator is the
standard multislice approximation; matter-corrected propagator is the
physically consistent version), and a MultisliceEngine that computes:

  * the **forward pass**: probe → exit wave through a stack of thin slices
  * the **backward pass** (adjoint): residual → gradient of the loss
    w.r.t. the projected refractive-index decrements and absorptions of
    every slice — used by the gradient-descent optimiser in Pass 2.

Physical convention
-------------------
Refractive index:       n(r) = 1 − δ(r) + iβ(r)     (δ, β ≥ 0)
Transmission per slice: T_j  = exp(−k₀ (β_j + iδ_j) Δz)
Fresnel propagator:

  Vacuum:
    H_vac(f_x, f_y; Δz) = exp(−iπλΔz (f_x² + f_y²))

  Matter-corrected (paraxial, n_mean = 1 − δ̄):
    H_mat(f_x, f_y; Δz) = exp(−iπ(λ/n_mean)Δz (f_x² + f_y²))
                         = H_vac · exp(−iπλΔzδ̄ (f_x² + f_y²))

where f_x, f_y are spatial frequencies in cycles m⁻¹.

The correction is tiny for hard X-rays (δ̄ ~ 10⁻⁶) but becomes
significant for soft X-rays and electrons.

Gradient derivation (Wirtinger calculus)
-----------------------------------------
Loss:   L = ½ ‖U_N − U_meas‖²

Forward pass:
    W_j  = T_j  ⊙  U_{j−1}      (transmission)
    U_j  = P_Δz[W_j]             (propagation)

Backward pass (using adjoint operators):
    r_N  = U_N − U_meas
    For j = N, …, 1:
        s_j       = P_{−Δz}[r_j]                       (adjoint propagation)
        ∂L/∂δ_j   = +2 k₀ Δz  Im[ W_j · conj(s_j) ]
        ∂L/∂β_j   = −2 k₀ Δz  Re[ W_j · conj(s_j) ]
        r_{j−1}   = conj(T_j) ⊙ s_j                   (adjoint transmission)

References
----------
Goodman (2005) "Introduction to Fourier Optics", 3rd ed.
Paganin (2006) "Coherent X-Ray Optics", Oxford.
Kirkland (2010) "Advanced Computing in Electron Microscopy", Springer.
"""

# standard library
import warnings

# third-party
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import rotate as ndimage_rotate

__all__ = [
    "FresnelPropagator",
    "AngularSpectrumPropagator",
    "get_propagator",
    "MultisliceEngine",
    "extract_slices_from_volume",
    "scatter_gradient_to_volume",
    "mean_refractive_index_profile",
]


# ---------------------------------------------------------------------------
# Fresnel propagator
# ---------------------------------------------------------------------------

class _PropagatorBase:
    """
    Base class for Fourier-domain (angular-spectrum-type) propagators.

    Pre-computes the transverse squared-frequency grid ``f² = f_x² + f_y²``
    at construction time (shape ``(Ny, Nx)``) so that repeated propagation
    calls (same grid, different distance or δ̄) only rebuild the cheap kernel.
    Concrete subclasses implement :meth:`_kernel`; the shared
    :meth:`propagate` / :meth:`propagate_adjoint` apply it in the Fourier
    domain.

    Matter (refractive-index) correction is handled uniformly for every
    subclass: a non-zero ``delta_mean`` δ̄ rescales the wavelength to the
    in-medium value ``λ_eff = λ / (1 − δ̄)``.  δ̄ = 0 recovers the vacuum
    propagator.

    Parameters
    ----------
    shape : tuple (Ny, Nx)
        Transverse grid size.
    pixel_size : float
        Real-space pixel size [m].
    wavelength : float
        X-ray wavelength [m].
    """

    def __init__(self, shape, pixel_size, wavelength):
        self.shape = shape
        self.pixel_size = float(pixel_size)
        self.wavelength = float(wavelength)

        Ny, Nx = shape
        fx = fftfreq(Nx, d=self.pixel_size)   # cycles m⁻¹
        fy = fftfreq(Ny, d=self.pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        # f² grid stored for reuse; shape (Ny, Nx)
        self._f2 = (FX ** 2 + FY ** 2).astype(np.float64)

    # ------------------------------------------------------------------
    def _lambda_eff(self, delta_mean):
        """In-medium wavelength ``λ_eff = λ / (1 − δ̄)`` (δ̄ = ``delta_mean``)."""
        return self.wavelength / (1.0 - float(delta_mean))

    # ------------------------------------------------------------------
    def _kernel(self, distance, delta_mean=0.0):
        """
        Build the transfer function ``H(f; Δz, δ̄)`` — subclass responsibility.

        Parameters
        ----------
        distance : float
            Propagation distance [m].  Positive → forward, negative → backward.
        delta_mean : float, optional
            Mean refractive-index decrement of the traversed layer.
            δ̄ = 0 gives the vacuum propagator; the layer-averaged value from
            the step-1 volume gives the matter-corrected propagator.

        Returns
        -------
        H : ndarray, complex128, shape (Ny, Nx)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def propagate(self, wavefield, distance, delta_mean=0.0):
        """
        Propagate a complex wavefield by ``distance`` metres.

        Parameters
        ----------
        wavefield : ndarray, complex, shape (Ny, Nx)
        distance : float
            [m]; positive = along the optical axis (forward).
        delta_mean : float, optional
            Mean δ of the traversed layer for matter-corrected propagation.

        Returns
        -------
        ndarray, complex, shape (Ny, Nx)
        """
        H = self._kernel(distance, delta_mean=delta_mean)
        return ifft2(fft2(wavefield) * H)

    def propagate_adjoint(self, wavefield, distance, delta_mean=0.0):
        """
        Adjoint propagation: backward propagation by ``distance``.

        For a lossless medium the adjoint equals forward propagation by
        ``−distance`` (phase conjugation / time reversal).  For the
        band-masked angular-spectrum propagator the evanescent mask is a
        real projection, so propagation by ``−Δz`` remains the exact adjoint
        of the masked forward operator.

        Parameters
        ----------
        wavefield : ndarray, complex, shape (Ny, Nx)
        distance : float   [m]
        delta_mean : float, optional

        Returns
        -------
        ndarray, complex, shape (Ny, Nx)
        """
        return self.propagate(wavefield, -distance, delta_mean=delta_mean)


class FresnelPropagator(_PropagatorBase):
    """
    Paraxial Fresnel propagator in the Fourier domain.

    Transfer function::

        H(f; Δz, δ̄) = exp(−iπ λ_eff Δz f²),   λ_eff = λ / (1 − δ̄)

    This is the paraxial (small-angle) limit of the exact
    :class:`AngularSpectrumPropagator`.  With δ̄ = 0 it is the standard
    vacuum multislice propagator; with δ̄ set to the layer-mean
    refractive-index decrement it is the matter-corrected propagator.
    """

    def _kernel(self, distance, delta_mean=0.0):
        lam_eff = self._lambda_eff(delta_mean)
        phase = -np.pi * lam_eff * float(distance) * self._f2
        return np.exp(1j * phase)


class AngularSpectrumPropagator(_PropagatorBase):
    r"""
    Exact (non-paraxial) angular-spectrum propagator in the Fourier domain.

    Transfer function, with the on-axis piston removed so that ``H(0) = 1``
    (matching the Fresnel convention)::

        H(f; Δz, δ̄) = exp( i (2π Δz / λ_eff) [ √(1 − (λ_eff f)²) − 1 ] ),
                       for (λ_eff f)² ≤ 1     (propagating waves)
                     = 0,                      otherwise  (evanescent, masked)

    with ``λ_eff = λ / (1 − δ̄)``.  Expanding the square root to first order,
    ``√(1 − (λ_eff f)²) − 1 ≈ −½ (λ_eff f)²``, reproduces the Fresnel kernel
    ``exp(−iπ λ_eff Δz f²)`` exactly; the two therefore agree for
    ``λ_eff f ≪ 1`` and diverge only at high spatial frequencies / large
    numerical aperture, where Fresnel's parabolic approximation to the true
    spherical dispersion breaks down.

    The global piston ``exp(i 2π Δz / λ_eff)`` is dropped so that this class
    is a drop-in alternative to :class:`FresnelPropagator` (both satisfy
    ``H(0) = 1`` and differ only in the higher-order angular terms).
    Evanescent components are set to zero rather than amplified, which keeps
    the operator bounded and makes :meth:`propagate_adjoint` (propagation by
    ``−Δz``) the exact adjoint of the masked forward operator.

    For hard X-rays (λ ~ 0.1 nm, sub-µrad scattering angles) the paraxial
    approximation is normally excellent, so this propagator is primarily a
    physically exact reference against which to validate the default Fresnel
    engine; it becomes necessary only at very small pixel sizes / high NA.
    """

    def _kernel(self, distance, delta_mean=0.0):
        lam_eff = self._lambda_eff(delta_mean)
        # (λ_eff f)²  — dimensionless direction-cosine squared
        s = (lam_eff ** 2) * self._f2
        propagating = s <= 1.0
        root = np.sqrt(np.where(propagating, 1.0 - s, 0.0))
        phase = (2.0 * np.pi * float(distance) / lam_eff) * (root - 1.0)
        H = np.exp(1j * phase)
        # mask evanescent band (do not amplify)
        return np.where(propagating, H, 0.0).astype(np.complex128)


# ---------------------------------------------------------------------------
# Propagator registry / factory
# ---------------------------------------------------------------------------

_PROPAGATORS = {
    "fresnel": FresnelPropagator,
    "angular_spectrum": AngularSpectrumPropagator,
    "asm": AngularSpectrumPropagator,   # alias
}


def get_propagator(spec, shape, pixel_size, wavelength):
    """
    Resolve a propagator specification into a ready-to-use instance.

    Parameters
    ----------
    spec : str, or a :class:`_PropagatorBase` subclass, or an instance
        * ``str`` — one of ``"fresnel"``, ``"angular_spectrum"`` (alias
          ``"asm"``); case-insensitive.
        * a propagator **class** — instantiated with the grid parameters.
        * a propagator **instance** — returned unchanged (its grid must
          already match ``shape``/``pixel_size``/``wavelength``).
    shape : tuple (Ny, Nx)
    pixel_size : float   [m]
    wavelength : float   [m]

    Returns
    -------
    _PropagatorBase
    """
    if isinstance(spec, _PropagatorBase):
        return spec
    if isinstance(spec, type) and issubclass(spec, _PropagatorBase):
        return spec(shape, pixel_size, wavelength)
    if isinstance(spec, str):
        key = spec.lower()
        if key not in _PROPAGATORS:
            raise ValueError(
                f"Unknown propagator {spec!r}. Available: "
                f"{sorted(set(_PROPAGATORS))!r}."
            )
        return _PROPAGATORS[key](shape, pixel_size, wavelength)
    raise TypeError(
        "propagator must be a name string, a _PropagatorBase subclass, or "
        f"an instance; got {type(spec).__name__}."
    )


# ---------------------------------------------------------------------------
# Multislice engine
# ---------------------------------------------------------------------------

class MultisliceEngine:
    """
    Forward and adjoint multislice wave-propagation engine.

    The object is discretised into ``n_slices`` thin slabs, each of
    thickness ``slice_thickness`` metres.  For every slice the engine:

      1. Multiplies the current wavefield by the transmission function
         T_j = exp(−k₀(β_j + iδ_j)Δz).
      2. Propagates the resulting field to the next slice using a
         Fresnel propagator (vacuum or matter-corrected).

    The adjoint of this forward operator provides exact gradients of the
    squared-residual loss with respect to the slice δ and β maps.

    Parameters
    ----------
    shape : tuple (Ny, Nx)
        Transverse grid size (must match the exit-wave shape).
    pixel_size : float
        Real-space pixel size [m].
    wavelength : float
        X-ray wavelength [m].
    slice_thickness : float
        Thickness of each slice [m].  For isotropic voxels set this equal
        to ``pixel_size``.
    propagator : str or propagator instance/class, optional
        Which inter-slice propagator to use.  ``"fresnel"`` (default) is the
        paraxial Fresnel propagator; ``"angular_spectrum"`` (alias ``"asm"``)
        is the exact non-paraxial angular-spectrum propagator.  A custom
        :class:`_PropagatorBase` subclass or instance may also be passed.
        See :func:`get_propagator`.
    """

    def __init__(self, shape, pixel_size, wavelength, slice_thickness,
                 propagator="fresnel"):
        self.shape = shape
        self.pixel_size = float(pixel_size)
        self.wavelength = float(wavelength)
        self.slice_thickness = float(slice_thickness)
        self.k0 = 2.0 * np.pi / self.wavelength
        self._prop = get_propagator(propagator, shape, pixel_size, wavelength)

    # ------------------------------------------------------------------
    def transmission(self, delta_slice, beta_slice):
        """
        Compute the complex transmission for a single slice.

        T(x,y) = exp(−k₀ (β(x,y) + iδ(x,y)) Δz)

        Parameters
        ----------
        delta_slice : ndarray, real, shape (Ny, Nx)
        beta_slice  : ndarray, real, shape (Ny, Nx)

        Returns
        -------
        T : ndarray, complex128, shape (Ny, Nx)
        """
        dz = self.slice_thickness
        return np.exp(-self.k0 * dz * (beta_slice + 1j * delta_slice))

    # ------------------------------------------------------------------
    def forward(self, delta_slices, beta_slices, probe,
                delta_means=None, store_wavefields=False):
        """
        Forward multislice propagation.

        Parameters
        ----------
        delta_slices : ndarray, real, shape (n_slices, Ny, Nx)
            Refractive-index decrement maps, one per slice.
        beta_slices  : ndarray, real, shape (n_slices, Ny, Nx)
            Absorption maps, one per slice.
        probe : ndarray, complex, shape (Ny, Nx)
            Incident wavefield entering the first slice.
        delta_means : array_like of float, length n_slices, optional
            Mean δ in each slice for the matter-corrected propagator.
            If None, the vacuum propagator is used (δ̄ = 0 for all slices).
        store_wavefields : bool, optional
            If True, return the intermediate wavefields needed for the
            adjoint pass.  Required for gradient computation.

        Returns
        -------
        U_exit : ndarray, complex, shape (Ny, Nx)
            Exit wavefield after the last slice.
        wavefields : dict  (only when ``store_wavefields=True``)
            ``wavefields["W"][j]``: wavefield *after* transmission, *before*
            propagation, at slice j (0-indexed).
            ``wavefields["U"][j]``: wavefield *after* propagation at slice j.
            ``wavefields["U"][-1]`` == ``wavefields["U"][0]`` == probe (entry).
            ``wavefields["T"][j]``: transmission function at slice j.
        """
        n_slices = delta_slices.shape[0]
        if delta_means is None:
            delta_means = np.zeros(n_slices)

        dz = self.slice_thickness
        U = probe.astype(complex, copy=True)

        if store_wavefields:
            W_list = []   # after transmission, before propagation
            T_list = []   # transmission functions
            U_list = [U.copy()]  # U_list[0] = probe entering slice 0

        for j in range(n_slices):
            T_j = self.transmission(delta_slices[j], beta_slices[j])
            W_j = T_j * U              # apply transmission
            U   = self._prop.propagate(W_j, dz, delta_mean=delta_means[j])
            if store_wavefields:
                T_list.append(T_j)
                W_list.append(W_j)
                U_list.append(U.copy())   # U_list[j+1] = after propagation j

        if store_wavefields:
            return U, {"W": W_list, "T": T_list, "U": U_list}
        return U

    # ------------------------------------------------------------------
    def gradient(self, wavefields, u_measured, delta_means=None):
        """
        Compute gradients via the adjoint (backward) pass.

        Must be called with ``wavefields`` returned by :meth:`forward`
        (using ``store_wavefields=True``).

        Parameters
        ----------
        wavefields : dict
            As returned by :meth:`forward` with ``store_wavefields=True``.
        u_measured : ndarray, complex, shape (Ny, Nx)
            Reference (measured) exit wave.
        delta_means : array_like of float, optional
            Same values used during the forward pass.

        Returns
        -------
        grad_delta : ndarray, real, shape (n_slices, Ny, Nx)
            ∂L/∂δ_j for each slice.
        grad_beta  : ndarray, real, shape (n_slices, Ny, Nx)
            ∂L/∂β_j for each slice.
        loss : float
            Squared-residual loss ½‖U_exit − U_meas‖².
        """
        W_list = wavefields["W"]
        T_list = wavefields["T"]
        U_list = wavefields["U"]    # length n_slices + 1

        n_slices = len(W_list)
        if delta_means is None:
            delta_means = np.zeros(n_slices)

        dz = self.slice_thickness
        k0 = self.k0
        scale = k0 * dz   # common factor in gradient formulas

        # Residual at exit plane
        U_exit = U_list[n_slices]    # = U_list[-1]
        residual = U_exit - u_measured
        loss = 0.5 * float(np.sum(np.abs(residual) ** 2))

        grad_delta = np.zeros((n_slices,) + self.shape, dtype=np.float64)
        grad_beta  = np.zeros((n_slices,) + self.shape, dtype=np.float64)

        # Initialise adjoint state r_N = residual
        r = residual.astype(complex, copy=True)

        for j in range(n_slices - 1, -1, -1):
            # Adjoint of propagation: backward propagation by -Δz
            s_j = self._prop.propagate_adjoint(r, dz, delta_mean=delta_means[j])

            # Gradient contributions from slice j
            # ∂L/∂δ_j = +2k₀Δz · Im[W_j · conj(s_j)]
            # ∂L/∂β_j = −2k₀Δz · Re[W_j · conj(s_j)]
            cross = W_list[j] * np.conj(s_j)
            grad_delta[j] =  scale * np.imag(cross)
            grad_beta[j]  = -scale * np.real(cross)

            # Adjoint of transmission: r_{j-1} = conj(T_j) ⊙ s_j
            r = np.conj(T_list[j]) * s_j

        return grad_delta, grad_beta, loss

    # ------------------------------------------------------------------
    def loss_and_gradient(self, delta_slices, beta_slices, probe,
                          u_measured, delta_means=None):
        """
        Convenience: forward + backward in one call.

        Returns
        -------
        loss : float
        grad_delta : ndarray  (n_slices, Ny, Nx)
        grad_beta  : ndarray  (n_slices, Ny, Nx)
        u_exit : ndarray, complex  (Ny, Nx)
            Simulated exit wave (useful for monitoring convergence).
        """
        u_exit, wf = self.forward(delta_slices, beta_slices, probe,
                                  delta_means=delta_means,
                                  store_wavefields=True)
        gd, gb, loss = self.gradient(wf, u_measured, delta_means=delta_means)
        return loss, gd, gb, u_exit


# ---------------------------------------------------------------------------
# Volume ↔ slice utilities
# ---------------------------------------------------------------------------

def extract_slices_from_volume(delta_vol, beta_vol, theta_deg, n_slices=None):
    """
    Extract 2D transverse slices from a 3D volume at a given rotation angle.

    The volume is rotated about the y-axis (axis 1) by ``theta_deg``
    degrees so that the beam direction aligns with axis 0.  The resulting
    rotated array is then sliced along axis 0 to obtain ``n_slices``
    2D (Ny × Nx) maps of δ and β perpendicular to the beam.

    If ``n_slices`` < Nz, adjacent z-layers are averaged to form thicker
    effective slices; the corresponding slice thickness then becomes
    Nz/n_slices × pixel_size.

    Parameters
    ----------
    delta_vol : ndarray, real, shape (Nz, Ny, Nx)
    beta_vol  : ndarray, real, shape (Nz, Ny, Nx)
    theta_deg : float
        Rotation angle in degrees.
    n_slices  : int, optional
        Number of output slices.  Defaults to Nz (one per z-pixel).

    Returns
    -------
    delta_slices : ndarray, shape (n_slices, Ny, Nx)
    beta_slices  : ndarray, shape (n_slices, Ny, Nx)
    delta_means  : ndarray, shape (n_slices,)
        Transverse-plane mean of δ per slice (for matter-corrected propagator).
    """
    Nz, Ny, Nx = delta_vol.shape

    # Rotate in the xz-plane (axes 2 and 0) about the y-axis.
    # order=1 (trilinear) is fast and sufficient; use order=3 for higher accuracy.
    kwargs = dict(axes=(2, 0), reshape=False, order=1, mode="constant", cval=0.0)
    delta_rot = ndimage_rotate(delta_vol.astype(np.float64), theta_deg, **kwargs)
    beta_rot  = ndimage_rotate(beta_vol.astype(np.float64),  theta_deg, **kwargs)

    if n_slices is None or n_slices == Nz:
        delta_slices = delta_rot
        beta_slices  = beta_rot
    else:
        # Group consecutive z-layers by averaging
        k = Nz // n_slices
        if k == 0:
            raise ValueError(
                f"n_slices={n_slices} > Nz={Nz}.  "
                "Cannot have more slices than z-pixels."
            )
        n_use = k * n_slices
        delta_slices = delta_rot[:n_use].reshape(n_slices, k, Ny, Nx).mean(axis=1)
        beta_slices  = beta_rot[:n_use].reshape(n_slices, k, Ny, Nx).mean(axis=1)

    delta_means = delta_slices.mean(axis=(1, 2))   # shape (n_slices,)
    return delta_slices, beta_slices, delta_means


def scatter_gradient_to_volume(grad_delta_slices, grad_beta_slices,
                               theta_deg, volume_shape, n_slices=None):
    """
    Accumulate 2D slice gradients back into the 3D volume gradient.

    This is the adjoint of :func:`extract_slices_from_volume`: it
    back-rotates the per-slice gradient maps by ``−theta_deg`` and scatters
    them into the corresponding z-layers of the 3D volume.

    Parameters
    ----------
    grad_delta_slices : ndarray, real, shape (n_slices, Ny, Nx)
    grad_beta_slices  : ndarray, real, shape (n_slices, Ny, Nx)
    theta_deg : float
    volume_shape : tuple (Nz, Ny, Nx)
    n_slices : int, optional
        Must match the value used in :func:`extract_slices_from_volume`.

    Returns
    -------
    grad_delta_vol : ndarray, real, shape (Nz, Ny, Nx)
    grad_beta_vol  : ndarray, real, shape (Nz, Ny, Nx)
    """
    Nz, Ny, Nx = volume_shape

    if n_slices is not None and n_slices < Nz:
        k = Nz // n_slices
        n_use = k * n_slices
        # Expand: each slice gradient distributes equally over k z-layers
        gd_full = np.repeat(grad_delta_slices, k, axis=0) / k
        gb_full = np.repeat(grad_beta_slices,  k, axis=0) / k
        # Pad remaining layers (if Nz not divisible by n_slices) with zeros
        if n_use < Nz:
            pad = np.zeros((Nz - n_use, Ny, Nx))
            gd_full = np.concatenate([gd_full, pad], axis=0)
            gb_full = np.concatenate([gb_full, pad], axis=0)
    else:
        gd_full = grad_delta_slices
        gb_full = grad_beta_slices

    # Back-rotate by −theta (adjoint of the forward rotation)
    kwargs = dict(axes=(2, 0), reshape=False, order=1, mode="constant", cval=0.0)
    grad_delta_vol = ndimage_rotate(gd_full, -theta_deg, **kwargs)
    grad_beta_vol  = ndimage_rotate(gb_full, -theta_deg, **kwargs)

    return grad_delta_vol, grad_beta_vol


def mean_refractive_index_profile(delta_vol, beta_vol, theta_deg, n_slices=None):
    """
    Compute mean δ̄ and β̄ per slice at angle ``theta_deg``.

    A lightweight wrapper around :func:`extract_slices_from_volume` that
    only returns the per-slice means (no need to rotate beta for the
    propagator, since only δ̄ enters).

    Parameters
    ----------
    delta_vol : ndarray, shape (Nz, Ny, Nx)
    beta_vol  : ndarray, shape (Nz, Ny, Nx)  (accepted but not used here)
    theta_deg : float
    n_slices  : int, optional

    Returns
    -------
    delta_means : ndarray, shape (n_slices,)
    """
    _, _, delta_means = extract_slices_from_volume(
        delta_vol, beta_vol, theta_deg, n_slices=n_slices
    )
    return delta_means
