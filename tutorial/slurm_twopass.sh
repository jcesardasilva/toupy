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
# Adjust partition, GPU type and count for your cluster.
# Common partition names:  gpu  gpu_short  gpu-a100  gpu_p100  etc.
# Common gres syntax:  gpu:1   gpu:a100:1   gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8             # for joblib-parallel FBP (Pass 1)
#SBATCH --mem=32G                     # host RAM; ~4 GB model + data, 32 GB is safe
#
# --- Wall-clock time ----------------------------------------------------------
# Rough guide (full 450-angle run, 394×493 volume, 50 iters):
#   A100 40 GB :  ~15 s/iter  →  50 iters ≈  13 min  →  request 1 h
#   V100 16 GB :  ~25 s/iter  →  50 iters ≈  22 min  →  request 1 h
#   RTX 3090   :  ~30 s/iter  →  50 iters ≈  27 min  →  request 1 h
#   T4 16 GB   :  ~55 s/iter  →  50 iters ≈  50 min  →  request 2 h
# Add ~10 min for FBP (Pass 1) and data loading.
# Run cluster_diagnostic.py to get a dataset-specific estimate first.
#SBATCH --time=02:00:00               # HH:MM:SS — adjust after diagnostics
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
PYTHON="python"

# Output directory for figures and the .npz result file
OUT_DIR="${WORK_DIR}/results_${SLURM_JOB_ID}"

# ── Reconstruction parameters (override the defaults in twopass_real_data.py)
N_SLICES=16      # multislice slabs — increase for higher accuracy (slower)
N_ITER=50        # Pass 2 gradient iterations
LR=5e-4          # Adam peak learning rate
LAMBDA_TV=1e-5   # TV regularisation weight  (0 to disable)
WARMUP_ITERS=5   # linear LR warm-up iterations
ANGLE_STEP=1     # 1 = all angles;  >1 = subsample for fast prototyping

# =============================================================================
# ── Environment setup — uncomment the block matching your cluster ─────────────
# =============================================================================

# --- Option A: conda environment ----------------------------------------------
# module load anaconda3            # or: miniconda3 / conda, depends on cluster
# conda activate toupy

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
    -e "s|^DATA_FILE\s*=.*|DATA_FILE = \"${DATA_FILE}\"|" \
    -e "s|^OUT_DIR\s*=.*|OUT_DIR = \"${OUT_DIR}\"|" \
    "${PATCHED}"

echo "Reconstruction parameters (as patched):"
grep -E "^(N_SLICES|N_ITER|LR|LAMBDA_TV|WARMUP_ITERS|ANGLE_STEP|DATA_FILE|OUT_DIR)" \
     "${PATCHED}" | sed 's/^/  /'
echo ""

# =============================================================================
# ── Run ──────────────────────────────────────────────────────────────────────
# =============================================================================

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
