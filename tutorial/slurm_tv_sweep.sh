#!/usr/bin/env bash
# =============================================================================
# TV sweep — is the two-pass gain physics, or just denoising?
# =============================================================================
#
# WHY
# ---
# The two-pass gain (FBP 72.7 nm -> two-pass 64.9 nm) was measured by FSC, and
# FSC is BLIND to smoothing: a deterministic filter appears identically in the
# numerator and denominator of the normalised correlation and cancels exactly.
# Decorrelation analysis, which is sensitive to it, reports the two-pass volume
# as having only ~52 % of FBP's power above 0.3 cyc/px, with low frequencies
# untouched -- the signature of the TV regulariser.
#
# So: is the gain the multislice physics, or is TV just denoising?
#
# Worse, every FSC run so far used LAMBDA_TV = 5e-5 WITHOUT SAYING SO.
# twopass_real_data.py silently replaces N_ITER/LR/LAMBDA_TV on any FSC_HALF run
# (the "auto-gentle" block), and slurm_twopass.sh's sed only matches the
# top-level `^LAMBDA_TV` assignment, not that indented one.  The published gain
# is therefore a strong-TV number.  This sweep sets FSC_AUTO_GENTLE=False so the
# explicit LAMBDA_TV survives, and holds N_ITER/LR at the gentle values
# (30 / 2e-6) so that TV IS THE ONLY VARIABLE.
#
# WHAT TO READ FROM IT
# --------------------
#   TV = 0  is the decisive point ("run A").
#     * two-pass still beats FBP by ~10 % at TV=0  -> the gain is the multislice
#       physics.  The TV worry is dead; report the TV=0 number.
#     * the gain collapses at TV=0                 -> the "two-pass gain" is
#       really a TV-denoising gain.  That is a different, weaker claim, and you
#       need to know before a referee finds out.
#
#   The shape of resolution vs LAMBDA_TV:
#     * keeps improving as TV grows -> FSC is being gamed by smoothing, and FSC
#       alone cannot justify the TV choice.  (At extreme TV the volume becomes a
#       blob, so it MUST turn over eventually -- if it does not, FSC is not
#       measuring resolution here.)
#     * turns over at some TV       -> a real optimum; check the default sits at
#       it, and report the curve.
#
# The FBP baseline is free: every run writes delta_fbp alongside delta_tp, and
# FBP does not depend on LAMBDA_TV, so it is the fixed reference across the
# sweep (a useful self-check: FBP's FSC should be constant to within run-to-run
# scatter).
#
# USAGE
# -----
#   cd <scratch dir with twopass_real_data.py, slurm_twopass.sh, the .npz>
#   bash slurm_tv_sweep.sh              # submit the whole sweep
#   bash slurm_tv_sweep.sh --dry-run    # print what would be submitted
#
# Each TV value costs 2 GPU jobs (half0 + half1) plus a short CPU job for the
# FSC.  With the default 5 values that is 10 GPU jobs; they queue in parallel.
#
# Then collect:
#   python collect_tv_sweep.py
# =============================================================================

set -euo pipefail

# ── Sweep configuration ──────────────────────────────────────────────────────
# 0 is the decisive point; 5e-5 reproduces the historical FSC runs; 5e-4 is
# deliberately over-smoothed so the curve has to turn over somewhere.
TV_VALUES="${TV_VALUES:-0 1e-6 1e-5 5e-5 5e-4}"

# Held fixed so TV is the only variable.  These are the "gentle" values the
# auto-gentle block used to impose; keeping them avoids re-introducing the
# half-angle overfitting that drove corr(FBP, two-pass) down to ~0.69.
SWEEP_N_ITER="${SWEEP_N_ITER:-30}"
SWEEP_LR="${SWEEP_LR:-2e-6}"
SWEEP_N_SLICES="${SWEEP_N_SLICES:-32}"   # 32 == 64 in quality; 2x faster

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON="${PYTHON:-/home/esrf/jdasilva/micromamba/envs/myvenv/bin/python}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

for f in slurm_twopass.sh twopass_real_data.py fsc_analysis_normfreq.py; do
    [ -f "${WORK_DIR}/${f}" ] || { echo "ERROR: ${f} not found in ${WORK_DIR}"; exit 1; }
done

# Refuse to run against a twopass_real_data.py that cannot honour FSC_AUTO_GENTLE
# -- otherwise every point in the sweep would silently be TV=5e-5 and the whole
# experiment would be worthless while looking fine.
if ! grep -q "^FSC_AUTO_GENTLE" "${WORK_DIR}/twopass_real_data.py"; then
    echo "ERROR: twopass_real_data.py has no top-level FSC_AUTO_GENTLE flag."
    echo "       Without it the auto-gentle block overrides LAMBDA_TV to 5e-5"
    echo "       on every FSC_HALF run and the sweep measures nothing."
    exit 1
fi

echo "============================================================"
echo " TV sweep — LAMBDA_TV in: ${TV_VALUES}"
echo " fixed: N_ITER=${SWEEP_N_ITER} LR=${SWEEP_LR} N_SLICES=${SWEEP_N_SLICES}"
echo "        FSC_AUTO_GENTLE=False  (explicit LAMBDA_TV survives)"
echo " work dir: ${WORK_DIR}"
echo "============================================================"

SUMMARY=""
for TV in ${TV_VALUES}; do
    TAG="_tv${TV}"
    JOBIDS=""
    for HALF in 0 1; do
        EXPORTS="ALL,FSC_HALF=${HALF},LAMBDA_TV=${TV},FSC_AUTO_GENTLE=False"
        EXPORTS="${EXPORTS},N_ITER=${SWEEP_N_ITER},LR=${SWEEP_LR}"
        EXPORTS="${EXPORTS},N_SLICES=${SWEEP_N_SLICES},RUN_TAG=${TAG}"
        if [ "${DRY_RUN}" = "1" ]; then
            echo "  sbatch --export=${EXPORTS} slurm_twopass.sh"
            JOBIDS="${JOBIDS}:000000"
        else
            JID=$(sbatch --parsable --export="${EXPORTS}" \
                         "${WORK_DIR}/slurm_twopass.sh")
            echo "  TV=${TV} half=${HALF} -> job ${JID}"
            JOBIDS="${JOBIDS}:${JID}"
        fi
    done
    JOBIDS="${JOBIDS#:}"

    # FSC for this TV, once both halves finish (afterok = only if both succeed)
    H0="${WORK_DIR}/twopass_real_figures${TAG}_half0/twopass_reconstruction.npz"
    H1="${WORK_DIR}/twopass_real_figures${TAG}_half1/twopass_reconstruction.npz"
    FSC_OUT="${WORK_DIR}/fsc_tv${TV}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  sbatch --dependency=afterok:${JOBIDS} <fsc for TV=${TV}>"
    else
        FJID=$(sbatch --parsable --dependency="afterok:${JOBIDS}" \
                      --job-name="fsc_tv${TV}" --output="fsc_tv${TV}_%j.log" \
                      --partition=nice --time=00:30:00 --mem=64G \
                      --wrap="${PYTHON} ${WORK_DIR}/fsc_analysis_normfreq.py \
                              ${H0} ${H1} --out-dir ${FSC_OUT}")
        echo "  TV=${TV} FSC -> job ${FJID} (after ${JOBIDS})"
    fi
    SUMMARY="${SUMMARY}  TV=${TV}: ${FSC_OUT}/fsc_data_normfreq.npz\n"
done

echo ""
echo "Submitted. When everything finishes, collect with:"
echo "  ${PYTHON} collect_tv_sweep.py --root ${WORK_DIR}"
echo ""
echo "Expected FSC outputs:"
printf "%b" "${SUMMARY}"
