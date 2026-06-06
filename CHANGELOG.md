# Changelog

All notable changes to **toupy** are documented in this file.

## Unreleased

### Added — magnified inline-holography CTF (`toupy.restoration`)

- **`holo_ctf_reconstruct`** — ESRF ID16A-style multi-distance CTF pipeline for
  focused-beam (cone-beam) inline holography: flat-field correction
  `(S−dark)/(ref−dark)` → rescale the magnified holograms onto a common
  effective pixel (Fresnel scaling theorem, centre-preserving affine resample)
  → sub-pixel alignment → multi-distance CTF inversion (`ctf_retrieve`).
- Helpers **`flat_field_correct`**, **`eigenflat_correct`** (dynamic eigen
  flat-field correction, robust to beam drift), **`holo_geometry`**
  (magnification, effective distance/pixel per defocus position),
  **`rescale_to_common_pixel`**, **`align_holograms`** (cross-correlation with
  low-pass + `normalization=None`, robust to the differing Fresnel fringes
  between distances).
- `holo_ctf_reconstruct` exposes `flat_method='simple'|'eigen'`; the default
  CTF `alpha` is `1e-4` (must be ≲ `(2β/δ)²` so the absorption term carries the
  low frequencies — a larger alpha bows the interior grey levels / cupping).
- Example `tutorial/example_holo_ctf_id16a.py` (with a CTF frequency-coverage
  diagnostic) and test `test/holo_ctf_test.py` (6 tests).

## 0.4.0 — 2026-05-31 — phase retrieval, ring correction, TV reconstruction

New methods for propagation-based / holotomographic X-ray nanotomography.
All new compute functions accept `cuda=True` for optional GPU acceleration via
CuPy, with automatic CPU fallback when CuPy is not installed.

### Added — phase retrieval (`toupy.restoration`)

- **`tie_hom`** — single-distance TIE-Hom (Paganin) phase retrieval for a
  homogeneous object (fixed δ/β).
- **`ctf_retrieve`** — CTF phase retrieval, unified for:
  - single distance (2-D image) and batches of independent projections;
  - **multi-distance / holotomographic** inversion (pass a list of distances +
    a stack), which fills the single-distance CTF zeros and — through the
    absorption term (`delta_beta`) — recovers low frequencies / DC.
  - Sign convention fixed: pure-phase contrast `I−1 = −2 sin(χ) φ` for the
    `exp(+iπλz|u|²)` propagator (previously the retrieved phase was inverted).
- **`iterative_phase_retrieval`** — nonlinear iterative retrieval inverting the
  **exact** Fresnel forward model by conjugate-gradient minimisation
  (analytic adjoint gradient, finite-difference verified). Warm-started from
  TIE-Hom. Single- or multi-distance (nonlinear holotomography, valid for
  strong phase where linear CTF fails). Tikhonov (`reg_smooth`) and
  edge-preserving total-variation (`reg_tv`) regularisation. Handles a 1-D
  projection line exactly (used by the tomography pipeline).
- **`suggest_holo_distances`** — builds a geometrically-spaced, gap-free
  multi-distance series from the CTF-zero interleaving rules, parametrised by
  the Fresnel-number span (`nf_short`/`nf_long`); verifies frequency coverage.

### Added — ring artifact correction (`toupy.restoration`)

- **`remove_rings_wavelet_fft`** — multi-scale wavelet stripe removal. Uses a
  safe angular-mean (median-baseline) stripe subtraction on every band; the
  subtracted stripe is constant across angle so real signal is never removed.
- **`remove_rings_titarenko`** — single-scale median-profile correction.
- **`remove_rings_stack`** — apply either method to a 3-D sinogram stack.

### Added — total-variation reconstruction (`toupy.tomo`)

- **`tv_reconstruction`** / **`chambolle_pock_tv`** — TV-regularised
  reconstruction via the Chambolle–Pock primal–dual algorithm.

### Added — examples (`tutorial/`)

- `example_phase_retrieval_tv_ring.py` — phase retrieval (TIE-Hom + single- and
  multi-distance CTF) + ring correction + FBP/TV reconstruction.
- `example_iterative_phase_retrieval.py` — TIE-Hom vs nonlinear iterative
  (single & multi-distance), with quantitative line profiles.
- `example_iterative_phase_tomography.py` — full phase-contrast tomography
  pipeline (per-angle retrieval → sinogram → FBP) plus a CTF frequency-coverage
  diagnostic showing why multi-distance fills single-distance gaps.

### Changed

- Added GPU (CuPy) acceleration paths across the new modules, with automatic
  CPU fallback when CuPy is not installed.

## 0.3.0 — 2026-05-22

### Added

- FDK cone-beam reconstruction pipeline (geometry, projector, backprojector).
- Gradient-descent (Adam optimiser) vertical and horizontal registration.
- `LocalFSC` and `LocalResolution` for spatially-resolved resolution estimation,
  including half-period resolution attributes.
- `phantom3D()` — 3-D Shepp-Logan and modified Shepp-Logan phantom generator.
- Interactive figure picker in `GUI_tracker`.

### Fixed — compatibility

- Python 3.14: `ValueError` from mathtext/pyparsing in `show_ssnr_curve`.
- NumPy 2.0: replace deprecated `np.trapz` with `np.trapezoid` in FDK.
- SciPy 2.0: replace deprecated `scipy.ndimage.filters` / `scipy.ndimage.fourier`
  imports with the flat `scipy.ndimage` namespace.

### Fixed — display / Jupyter

- Rewrite static figure rendering to the OO matplotlib API
  (`Figure` + `FigureCanvasAgg`), eliminating `AttributeError` from the ipympl
  `manager=None` issue across all display paths.
- `tomoconsistency_multiple` blank/error display under `%matplotlib widget`.
- `show_fsc_images` blank on 2nd+ call (SSNRPlot, LocalFSC).
- `show_ssnr_curve` tick-label crash (mathtext ParseException).
- tqdm progress bars accumulating in notebook cell output.
- Axis limits clipping data (zero padding) in `RegisterPlot`.

### Changed — dependencies

- Remove unused dependencies: numexpr, joblib, pyopencl, silx, decorator.
- Add ipywidgets as an explicit runtime dependency (was already used internally).
- ipympl is now optional (`extras_require["notebook"]`).
- Raise minimum Python to 3.8; numpy ≥ 1.20.0, scipy ≥ 1.7.0,
  scikit-image ≥ 0.18.0.

## 0.1.2

### Changed

- Remove dependency: roipoly.
- Silx requirement relaxed to ≥ 0.9.0.
- Better holotomography templates.
- Documentation improvements.
