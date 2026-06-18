#!/usr/bin/env bash
# =============================================================================
# SLURM job script — two-pass multislice reconstruction (Toupy)
# =============================================================================
#
# USAGE
# -----
# 1. Copy this file, twopass_real_data.py, and cluster_diagnostic.py to your
#    cluster scratch directory alongside PXCTalignedprojections.npz.
#
# 2. Run the diagnostic first (< 2 min, no queue needed):
#
#       python cluster_diagnostic.py
#
#    The diagnostic prints recommended values for N_SLICES and --time.
#    Edit the SBATCH directives and USER SETTINGS below accordingly.
#
# 3. Submit:
#
#       sbatch slurm_twopass.sh
#
# 4. Monitor:
#
#       squeue -u $USER
#       tail -f twopass_$SLURM_JOB_ID.log
#
# =============================================================================
#
# ── SBATCH directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=twopass_ms
#SBATCH --output=twopass_%j.log        # %j = job ID; one log file per run
#SBATCH --error=twopass_%j.err
#
# --- Compute resources --------------------------------------------------------
# Partition / GPU selection — ESRF GPU5a40 cluster
# ---------------------------------------------------------------
# Partition guide (from sinfo):
#   gpu          1:00:00   A40 / A100  — too short for 100-iter run
#   low-gpu      2:00:00   A40 / A100  — recommended for ≤ 100 iters on A40
#   gpu-long  1-00:00:00   A40 / A100  — use for FSC half-runs or >100 iters
#
# GRES guide:
#   gpu:nvidia_a40:1   — request one A40  (45 GB VRAM, fits N_SLICES=64)
#   gpu:nvidia_a100:1  — request one A100 (40 GB VRAM, fits N_SLICES=64)
#   gpu:1              — any available GPU
#SBATCH --partition=low-gpu
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=8             # SLURM-allocated CPUs (used by GPU FBP fallback)
#SBATCH --mem=128G                    # host RAM. 32G is fine for the small tutorial
                                      # volume, but large volumes hold several
                                      # full float32 arrays + figures on the CPU
                                      # (a 988^3 float32 volume is 3 GB; the
                                      # comparison figures peak at ~5-6 of them).
                                      # Use 128G for big data; reduce for small.
#
# --- Wall-clock time ----------------------------------------------------------
# Benchmarks at N_SLICES=64, 450 angles, CROP_X=80/CROP_Y=20 (333×354×333):
#   A40  48 GB :  ~40 s/iter  → 100 iters ≈  67 min  →  request 90 min (low-gpu)
#   A100 40 GB :  ~35 s/iter  → 100 iters ≈  58 min  →  request 90 min (low-gpu)
#   V100 16 GB :  ~70 s/iter  → 100 iters ≈ 117 min  →  request gpu-long 2:30:00
# At N_SLICES=16 (quick test run), divide times by ~4.
# GPU FBP (Pass 1) adds only ~5 s; data loading ~1 min.
# Run cluster_diagnostic.py first to get dataset-specific estimates.
#SBATCH --time=01:30:00               # 90 min — safe for A40 100-iter run on low-gpu
#
# --- Notifications (optional) -------------------------------------------------
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.email@institution.edu
#
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " SLURM job : $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo " Node      : $SLURMD_NODENAME"
echo " GPUs      : ${CUDA_VISIBLE_DEVICES:-none visible}"
echo " Started   : $(date)"
echo "============================================================"

# =============================================================================
# ── USER SETTINGS — edit these ───────────────────────────────────────────────
# =============================================================================

# Directory containing twopass_real_data.py, cluster_diagnostic.py, and the
# data file.  Defaults to the directory from which sbatch was called.
WORK_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"

# Data file
DATA_FILE="${WORK_DIR}/PXCTalignedprojections.npz"

# Python interpreter.  Point to the interpreter in your conda env / venv.
# Examples:
#   PYTHON="python"                                    (env already activated)
#   PYTHON="$HOME/miniconda3/envs/toupy/bin/python"   (absolute path)
PYTHON="/home/esrf/jdasilva/micromamba/envs/myvenv/bin/python"

# ── Half-dataset mode for FSC ─────────────────────────────────────────────
# None = full dataset  |  0 = even angles  |  1 = odd angles
# Submit two jobs (FSC_HALF=0 and FSC_HALF=1) in parallel, then run
# fsc_analysis.py on the two result files to get the FSC resolution curves.
FSC_HALF=None

# ── Reconstruction parameters (override the defaults in twopass_real_data.py)
N_SLICES=64      # multislice slabs — A40/A100: 64 fits in 48 GB, ~40 s/iter
N_ITER=100       # Pass 2 gradient iterations  (50–100 for production quality)
LR=5e-6          # Adam peak learning rate  (hard X-ray data: δ ~ 1e-5–1e-6)
LAMBDA_TV=1e-5   # TV regularisation weight  (0 to disable)
WARMUP_ITERS=5   # linear LR warm-up iterations
ANGLE_STEP=1     # 1 = all angles;  >1 = subsample for fast prototyping
ANGLE_WEIGHT=uniform  # per-angle noise weighting: 'uniform' | 'snr'
FBP_METHOD=auto       # Pass-1 back-projector: 'auto'|'iradon'|'gpu'|'gridding'
OPTIMIZE_BETA=True    # False = freeze beta (saves ~3 volumes of VRAM; use for
                      # large volumes that OOM on a 40-48 GB GPU)

# ── Projection boundary crop (removes ptychography edge noise) ─────────────
# Ptychography has insufficient probe overlap near the scan boundaries,
# producing noisy phases that contaminate the multislice loss.
# Set based on the self-consistency residual plot (where residuals are large).
CROP_X=55        # pixels to remove from each side in the column (x) direction
CROP_Y=55        # pixels to remove from each side in the row (y) direction
                 # Calibrated for PXCTalignedprojections.npz

# Output directory — must match the tagged path twopass_real_data.py computes
# (and that fsc_threeway_comparison.py looks for):
#   twopass_real_figures[_DATA][_grid][_snr][_halfN]
#     _DATA : derived from DATA_FILE so different inputs (big volume, jittered
#             copies, ...) NEVER overwrite each other; empty for the canonical
#             PXCTalignedprojections.npz.
#     _grid : FBP_METHOD=gridding
#     _snr  : ANGLE_WEIGHT=snr (or other non-uniform mode)
#     _halfN: FSC_HALF=0 / 1
_BASENAME="$(basename "${DATA_FILE}" .npz)"
_DATA_TAG="${_BASENAME#PXCTalignedprojections}"     # strip canonical prefix
_DATA_TAG="${_DATA_TAG#_}"                            # strip a leading underscore
_DTAG=""
if [ -n "${_DATA_TAG}" ]; then _DTAG="_${_DATA_TAG}"; fi
_FTAG=""
if [ "${FBP_METHOD}" = "gridding" ]; then _FTAG="_grid"; fi
_WTAG=""
if [ "${ANGLE_WEIGHT}" != "uniform" ]; then _WTAG="_${ANGLE_WEIGHT}"; fi
_HTAG=""
if [ "${FSC_HALF}" != "None" ]; then _HTAG="_half${FSC_HALF}"; fi
OUT_DIR="${WORK_DIR}/twopass_real_figures${_DTAG}${_FTAG}${_WTAG}${_HTAG}"

# =============================================================================
# ── Environment setup — uncomment the block matching your cluster ─────────────
# =============================================================================

# --- Option A: conda / micromamba environment ---------------------------------
# ESRF cluster: PYTHON path is set above to the full micromamba env path,
# so no module load or activate is needed.  If your cluster requires a CUDA
# module to expose libcuda.so (needed by PyTorch), uncomment the line below:
# module load cuda/13.2.1  # Uncomment ONLY if PyTorch can't find libcuda.so at runtime.
# conda/micromamba PyTorch already bundles the CUDA runtime — no module load needed
# in most cases.  Available versions: cuda/12.9.1  cuda/13.1.2  cuda/13.2.1

# --- Option B: module + virtualenv -------------------------------------------
# module load python/3.11 cuda/12.1 cudnn/8.9
# source "$HOME/venvs/toupy/bin/activate"

# --- Option C: Singularity / Apptainer container -----------------------------
# CONTAINER="$HOME/containers/toupy_cuda.sif"
# PYTHON="singularity exec --nv $CONTAINER python"

# --- Option D: EasyBuild / Lmod module (common at European HPC sites) --------
# module load Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# source "$HOME/venvs/toupy/bin/activate"

# =============================================================================
# ── Pre-flight checks ────────────────────────────────────────────────────────
# =============================================================================

echo ""
echo "Python     : $($PYTHON --version 2>&1)"
echo "Interpreter: $(which "$PYTHON" 2>/dev/null || echo 'resolved from PATH')"
echo ""

# Quick GPU check (non-fatal if torch not installed)
$PYTHON - <<'PYCHECK'
import sys
print(f"sys.prefix : {sys.prefix}")
try:
    import torch
    print(f"PyTorch    : {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            mem_gb = p.total_memory / 1e9
            print(f"  GPU {i}: {p.name}  {mem_gb:.1f} GB  "
                  f"cc={p.major}.{p.minor}")
    else:
        print("  CUDA not available — will use CPU torch backend")
except ImportError:
    print("PyTorch    : NOT found — reconstruction will use NumPy backend")
PYCHECK

echo ""

# Check data file exists
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: data file not found: ${DATA_FILE}"
    echo "       Edit DATA_FILE in this script or copy the .npz to WORK_DIR."
    exit 1
fi

mkdir -p "${OUT_DIR}"

# =============================================================================
# ── Patch the tutorial script and run ────────────────────────────────────────
# =============================================================================
# Strategy: create a temporary copy of twopass_real_data.py with the parameter
# lines replaced by the values set above.  sed is used for simple, reliable
# substitution without requiring any changes to the tutorial script itself.

PATCHED="${WORK_DIR}/twopass_real_data_${SLURM_JOB_ID}.py"
cp "${WORK_DIR}/twopass_real_data.py" "${PATCHED}"

# Replace the tunable parameter assignments (match the exact variable names)
sed -i \
    -e "s|^N_SLICES\s*=.*|N_SLICES    = ${N_SLICES}|" \
    -e "s|^N_ITER\s*=.*|N_ITER       = ${N_ITER}|" \
    -e "s|^LR\s*=.*|LR           = ${LR}|" \
    -e "s|^LAMBDA_TV\s*=.*|LAMBDA_TV    = ${LAMBDA_TV}|" \
    -e "s|^WARMUP_ITERS\s*=.*|WARMUP_ITERS = ${WARMUP_ITERS}|" \
    -e "s|^ANGLE_STEP\s*=.*|ANGLE_STEP   = ${ANGLE_STEP}|" \
    -e "s|^ANGLE_WEIGHT\s*=.*|ANGLE_WEIGHT = '${ANGLE_WEIGHT}'|" \
    -e "s|^FBP_METHOD\s*=.*|FBP_METHOD  = '${FBP_METHOD}'|" \
    -e "s|^OPTIMIZE_BETA\s*=.*|OPTIMIZE_BETA = ${OPTIMIZE_BETA}|" \
    -e "s|^CROP_X\s*=.*|CROP_X = ${CROP_X}|" \
    -e "s|^CROP_Y\s*=.*|CROP_Y = ${CROP_Y}|" \
    -e "s|^FSC_HALF\s*=.*|FSC_HALF = ${FSC_HALF}|" \
    -e "s|^DATA_FILE\s*=.*|DATA_FILE = \"${DATA_FILE}\"|" \
    -e "s|^OUT_DIR\s*=.*|OUT_DIR = \"${OUT_DIR}\"|" \
    "${PATCHED}"

echo "Reconstruction parameters (as patched):"
grep -E "^(N_SLICES|N_ITER|LR|LAMBDA_TV|WARMUP_ITERS|ANGLE_STEP|ANGLE_WEIGHT|FBP_METHOD|OPTIMIZE_BETA|CROP_X|CROP_Y|FSC_HALF|DATA_FILE|OUT_DIR)" \
     "${PATCHED}" | sed 's/^/  /'
echo ""

# =============================================================================
# ── Run ──────────────────────────────────────────────────────────────────────
# =============================================================================

# expandable_segments can REDUCE fragmentation OOM but may SLOW large runs that
# churn big alloc/free.  Leave it OFF by default (the GPU OOM was solved by
# OPTIMIZE_BETA=False).  Uncomment ONLY if a CUDA OOM recurs from fragmentation:
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

T_START=$(date +%s)

$PYTHON "${PATCHED}"
EXIT_CODE=$?

T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

# Clean up the patched copy
rm -f "${PATCHED}"

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " Job SUCCEEDED"
else
    echo " Job FAILED (exit code ${EXIT_CODE})"
fi
echo " Finished  : $(date)"
echo " Wall time : ${ELAPSED} s  ($(( ELAPSED / 60 )) min $(( ELAPSED % 60 )) s)"
echo " Results   : ${OUT_DIR}"
echo "============================================================"

exit $EXIT_CODE
