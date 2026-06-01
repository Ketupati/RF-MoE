#!/usr/bin/env bash
# run_all.sh  —  Complete RF-MoE pipeline for DF40 benchmark on campus GPU
#
# USAGE:
#   bash run_all.sh           # run full pipeline
#   bash run_all.sh setup     # one-time setup only
#   bash run_all.sh train_fs  # train FS-only model
#   bash run_all.sh eval_all  # run all evaluations (assumes training done)
#
# This script runs from wherever it's called but DOES NOT cd permanently.
# Each phase is clearly labelled — you can run them individually.
#
# Prerequisites:
#   - Data already at /home/ibubu/ketupati/data/  (DF40, DF40_train, ff_real, cdf_real, dataset_json, weights)
#   - Python 3.8+ with pip

set -euo pipefail

BASE="/home/ibubu/ketupati"
REPO="${BASE}/DeepfakeBench_DF40"
OUTPUTS="${BASE}/outputs/rf_moe"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "RF-MoE DF40 Pipeline"
echo "  BASE:    ${BASE}"
echo "  REPO:    ${REPO}"
echo "  SCRIPTS: ${SCRIPTS_DIR}"
echo "================================================================"

# Helper: find newest checkpoint for a given mode
find_ckpt() {
    local mode="$1"
    local first_ds
    case "$mode" in
        fs|fr|efs) first_ds="FSAll_ff" ;;
        joint)     first_ds="simswap_ff" ;;
        *) echo "Unknown mode: $mode"; exit 1 ;;
    esac
    # Find newest ckpt_best.pth under outputs
    find "${OUTPUTS}" -path "*/test/${first_ds}/ckpt_best.pth" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}'
}

# ────────────────────────────────────────────────────────────────
# PHASE 1: One-time setup
# ────────────────────────────────────────────────────────────────
phase_setup() {
    echo ""
    echo "━━━ PHASE 1: Setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/setup.py"
}

# ────────────────────────────────────────────────────────────────
# PHASE 2: Generate training JSONs
# ────────────────────────────────────────────────────────────────
phase_generate_jsons() {
    echo ""
    echo "━━━ PHASE 2: Generate Training JSONs ━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/generate_jsons.py"
}

# ────────────────────────────────────────────────────────────────
# PHASE 3: Training
# (Run one at a time — each takes hours on A100)
# ────────────────────────────────────────────────────────────────
phase_train_fs() {
    echo ""
    echo "━━━ PHASE 3a: Train FS-only (Tables 3/4/5) ━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/run_training.py" --mode fs
}

phase_train_fr() {
    echo ""
    echo "━━━ PHASE 3b: Train FR-only (Tables 3/4/5) ━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/run_training.py" --mode fr
}

phase_train_efs() {
    echo ""
    echo "━━━ PHASE 3c: Train EFS-only (Tables 3/4/5) ━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/run_training.py" --mode efs
}

phase_train_joint() {
    echo ""
    echo "━━━ PHASE 3d: Train Joint model (Table 6) ━━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/run_training.py" --mode joint
}

# ────────────────────────────────────────────────────────────────
# PHASE 4: Evaluation (Tables 3, 4, 5)
# Run AFTER each single-type training completes.
# ────────────────────────────────────────────────────────────────
phase_eval_fs() {
    echo ""
    echo "━━━ PHASE 4a: Evaluate FS model (Tables 3/4/5) ━━━━━━━━━━━━"
    FS_CKPT=$(find_ckpt fs)
    if [[ -z "${FS_CKPT}" ]]; then
        echo "ERROR: No FS checkpoint found under ${OUTPUTS}"
        echo "       Run:  python run_training.py --mode fs"
        exit 1
    fi
    echo "  Checkpoint: ${FS_CKPT}"
    python "${SCRIPTS_DIR}/run_evaluation.py" \
        --trained_on fs \
        --checkpoint "${FS_CKPT}" \
        --tables 3 4 5
}

phase_eval_fr() {
    echo ""
    echo "━━━ PHASE 4b: Evaluate FR model (Tables 3/4/5) ━━━━━━━━━━━━"
    FR_CKPT=$(find_ckpt fr)
    if [[ -z "${FR_CKPT}" ]]; then
        echo "ERROR: No FR checkpoint found."
        echo "       Run:  python run_training.py --mode fr"
        exit 1
    fi
    echo "  Checkpoint: ${FR_CKPT}"
    python "${SCRIPTS_DIR}/run_evaluation.py" \
        --trained_on fr \
        --checkpoint "${FR_CKPT}" \
        --tables 3 4 5
}

phase_eval_efs() {
    echo ""
    echo "━━━ PHASE 4c: Evaluate EFS model (Tables 3/4/5) ━━━━━━━━━━━"
    EFS_CKPT=$(find_ckpt efs)
    if [[ -z "${EFS_CKPT}" ]]; then
        echo "ERROR: No EFS checkpoint found."
        echo "       Run:  python run_training.py --mode efs"
        exit 1
    fi
    echo "  Checkpoint: ${EFS_CKPT}"
    python "${SCRIPTS_DIR}/run_evaluation.py" \
        --trained_on efs \
        --checkpoint "${EFS_CKPT}" \
        --tables 3 4 5
}

phase_eval_joint() {
    echo ""
    echo "━━━ PHASE 4d: Evaluate Joint model (Table 6) ━━━━━━━━━━━━━━"
    JOINT_CKPT=$(find_ckpt joint)
    if [[ -z "${JOINT_CKPT}" ]]; then
        echo "ERROR: No joint checkpoint found."
        echo "       Run:  python run_training.py --mode joint"
        exit 1
    fi
    echo "  Checkpoint: ${JOINT_CKPT}"
    python "${SCRIPTS_DIR}/run_evaluation.py" \
        --trained_on joint \
        --checkpoint "${JOINT_CKPT}" \
        --tables 6
}

# ────────────────────────────────────────────────────────────────
# PHASE 5: Compile results
# ────────────────────────────────────────────────────────────────
phase_compile() {
    echo ""
    echo "━━━ PHASE 5: Compile Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPTS_DIR}/compile_results.py"
}

# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────
ACTION="${1:-all}"

case "$ACTION" in
    setup)
        phase_setup
        ;;
    generate_jsons)
        phase_generate_jsons
        ;;
    train_fs)
        phase_train_fs
        ;;
    train_fr)
        phase_train_fr
        ;;
    train_efs)
        phase_train_efs
        ;;
    train_joint)
        phase_train_joint
        ;;
    eval_fs)
        phase_eval_fs
        ;;
    eval_fr)
        phase_eval_fr
        ;;
    eval_efs)
        phase_eval_efs
        ;;
    eval_joint)
        phase_eval_joint
        ;;
    eval_all)
        phase_eval_fs
        phase_eval_fr
        phase_eval_efs
        phase_eval_joint
        ;;
    compile)
        phase_compile
        ;;
    all)
        echo ""
        echo "Running FULL pipeline (this will take many GPU-hours)."
        echo "Consider running phases individually for better control."
        echo ""
        phase_setup
        phase_generate_jsons
        phase_train_fs
        phase_eval_fs
        phase_train_fr
        phase_eval_fr
        phase_train_efs
        phase_eval_efs
        phase_train_joint
        phase_eval_joint
        phase_compile
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Valid: setup | generate_jsons | train_fs | train_fr | train_efs | train_joint"
        echo "       eval_fs | eval_fr | eval_efs | eval_joint | eval_all | compile | all"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Done: ${ACTION}"
echo "================================================================"
