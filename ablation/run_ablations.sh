#!/bin/bash
# run_ablations.sh — Run all 4 RF-MoE ablations (FS protocol) on GPU server.
#
# Run this ON the server: bash /home/ibubu/ketupati/model7_ablation/run_ablations.sh
#
# Each ablation trains for 20 epochs on FSAll (FS forgery type).
# Checkpoints saved to: /home/ibubu/ketupati/outputs/rf_moe_abl_{name}_fs/epoch_ckpts/
# Logs saved to:        /home/ibubu/ketupati/model7_ablation/logs/
#
# Estimated runtime: ~8–10 hrs per ablation on RTX 4080 Super (--no_test)
# Total: ~32–40 hrs (run sequentially or screen each separately)

BASE=/home/ibubu/ketupati
REPO=$BASE/DeepfakeBench_DF40
ABLATION_DIR=$BASE/model7_ablation
LOG_DIR=$ABLATION_DIR/logs
PYTHON=$BASE/venv/bin/python3

# JSON paths (confirmed: symlink at DeepfakeBench_DF40/preprocessing/dataset_json → here)
JSON_DIR=/home/ibubu/ketupati/data/dataset_json
TRAIN_JSON=FSAll_ff_train90
VAL_JSON=$JSON_DIR/FSAll_ff_val10.json

mkdir -p $LOG_DIR

# ── Step 1: Install ablation detector into DF40 repo ───────────────────────

echo "===== Setting up ablation detector ====="

DETECTOR_DST=$REPO/training/detectors/rfmoe_ablation_detector.py
cp $ABLATION_DIR/rfmoe_ablation_detector.py $DETECTOR_DST
echo "  [OK] Copied rfmoe_ablation_detector.py → $DETECTOR_DST"

INIT=$REPO/training/detectors/__init__.py
if ! grep -q "rfmoe_ablation_detector" $INIT; then
    echo "from .rfmoe_ablation_detector import RFMoEAblationDetector" >> $INIT
    echo "  [OK] Registered RFMoEAblationDetector in __init__.py"
else
    echo "  [SKIP] Already registered"
fi

# ── Step 2: Verify val JSON exists for val_auc logging ─────────────────────
# (train.py already has val_auc_patch_v3 from Model7 training;
#  it uses the FS val JSON hardcoded in that patch)

if [ ! -f "$VAL_JSON" ]; then
    echo "  [WARN] Val JSON not found: $VAL_JSON"
    echo "  [WARN] Val AUC will not be computed. Check JSON_DIR above."
fi

# ── Step 3: Run each ablation ───────────────────────────────────────────────

ABLATIONS=(no_spectral no_region uniform_gate frozen_bb)

for ABL in "${ABLATIONS[@]}"; do
    YAML_SRC=$ABLATION_DIR/rfmoe_abl_${ABL}.yaml
    YAML_DST=$REPO/training/config/detector/rfmoe_ablation.yaml
    LOG_FILE=$LOG_DIR/train_abl_${ABL}_fs.log

    if [ ! -f "$YAML_SRC" ]; then
        echo "  [ERROR] Yaml not found: $YAML_SRC — skipping $ABL"
        continue
    fi

    cp $YAML_SRC $YAML_DST
    echo ""
    echo "======================================================="
    echo "  Starting ablation: $ABL"
    echo "  Log: $LOG_FILE"
    echo "======================================================="

    cd $REPO
    $PYTHON training/train.py \
        --detector_path training/config/detector/rfmoe_ablation.yaml \
        --train_dataset $TRAIN_JSON \
        2>&1 | tee $LOG_FILE

    echo "  [DONE] $ABL  (exit code: $?)"
done

echo ""
echo "======================================================="
echo "ALL ABLATIONS COMPLETE"
echo "Logs: $LOG_DIR"
echo "Next: python find_best_epoch_ablation.py  (or check logs manually)"
echo "======================================================="
