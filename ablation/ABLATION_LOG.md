# RF-MoE Ablation Study Log

## Overview

**Date started:** 2026-05-27  
**Base model:** Model7 (RF-MoE Phase 2)  
**Purpose:** Quantify contribution of each architectural component by removing it one at a time.  
**Protocol:** FS training only (face-swapping), evaluated on Tables 3, 4, 5.  
**Comparison baseline:** Model7 FS Epoch 8 (T3 avg=0.969, T4 avg=0.805, T5 avg=0.796)

---

## Architecture Being Ablated (Model7)

```
Input (B, 3, 224, 224)
    ↓
CLIP-Large ViT-L/14 backbone (303M, blocks 0-21 frozen, blocks 22-23 unfrozen at backbone_lr=1e-5)
    ├─→ cls_token  (B, 1024)  → GatingNetwork → gate_weights (B, 3)   [A3]
    └─→ patch_tokens (B, 256, 1024)
             ↓
    RegionAttentionPool                                                 [A2]
         ├─→ boundary_feat (B, 1024)  → Expert 0
         ├─→ interior_feat (B, 1024)  → Expert 1
         └─→ global_feat   (B, 1024)  → Expert 2

SpectralBranch (Sobel-H/V/Diag + Laplacian fixed filters + CNN)       [A1]
    → spec_feat (B, 256)  — shared across all 3 experts

3 Expert MLPs: Linear(1280→512) → ReLU → Dropout(0.5) → Linear(512→2)

Weighted sum: (expert_logits × gate_weights) → combined logits → AUC
Loss: cross_entropy + 0.1 × load_balance_loss
```

---

## Training Configuration (identical across all ablations)

| Parameter | Value | Note |
|-----------|-------|------|
| Optimizer | Adam | |
| lr (adapters) | 0.0003 | matches Model7 |
| weight_decay | 0.0001 | matches Model7 |
| backbone_lr | 0.00001 | matches Model7 (unused for A4) |
| lr_scheduler | cosine | T_max=30, eta_min=1e-5 |
| nEpochs | 20 | Model7 FS best was ep8; 20 is sufficient |
| train_batchSize | 32 | |
| test_batchSize | 16 | |
| Protocol | FS only | FSAll_ff_train90 (90% split) |
| Val set | FSAll_ff_val10 | for checkpoint selection |
| Checkpoint selection | best val_auc | same as Model7 |
| Per-epoch testing | disabled (`--no_test`) | saves ~3 hrs per run |
| GPU | RTX 4080 Super 16GB | |

---

## Ablation Definitions

### A1 — No SpectralBranch (`no_spectral`)
**What changes:** `spec_feat = torch.zeros(B, 256)` instead of running Sobel+Laplacian CNN  
**What it tests:** How much do frequency-domain edge features contribute?  
**Hypothesis:** Drop in cross-domain (T4) and unknown-domain (T5) generalization, since spectral features provide frequency-level cues beyond CLIP's semantic features.  
**Log:** `logs/train_abl_no_spectral_fs.log`  
**Checkpoints:** `outputs/rf_moe_abl_no_spectral_fs/epoch_ckpts/`

### A2 — No RegionAttentionPool (`no_region`)
**What changes:** All 3 experts receive `patches.mean(1)` (simple mean pool) instead of learned boundary/interior/global attention  
**What it tests:** How much does spatially-differentiated feature routing contribute?  
**Hypothesis:** Drop in FS performance specifically (boundary attention is key for swap-edge artifacts). EFS may be less affected since it already uses mean pool.  
**Log:** `logs/train_abl_no_region_fs.log`  
**Checkpoints:** `outputs/rf_moe_abl_no_region_fs/epoch_ckpts/`

### A3 — Uniform Gating (`uniform_gate`)
**What changes:** `gate_weights = [1/3, 1/3, 1/3]` (fixed, no GatingNetwork); load_balance_loss = 0  
**What it tests:** How much does the learned per-sample expert routing contribute?  
**Hypothesis:** Minor drop — the expert specialization (region views) matters more than the routing. If gating is important, we'd see a large drop especially on cross-forgery (T3).  
**Log:** `logs/train_abl_uniform_gate_fs.log`  
**Checkpoints:** `outputs/rf_moe_abl_uniform_gate_fs/epoch_ckpts/`

### A4 — Frozen Backbone (`frozen_backbone`)
**What changes:** CLIP blocks 22-23 are NOT unfrozen; all 303M CLIP params frozen  
**What it tests:** How much does partial CLIP fine-tuning contribute?  
**Hypothesis:** Significant drop across all tables. Unfreezing blocks 22-23 was the key change from Model1→Model4/7, and it improved cross-forgery generalization noticeably.  
**Log:** `logs/train_abl_frozen_bb_fs.log`  
**Checkpoints:** `outputs/rf_moe_abl_frozen_bb_fs/epoch_ckpts/`

---

## File Locations

### Mac (local)
```
/Users/ketupatiswargiary/final/Model7/ablation/
  rfmoe_ablation_detector.py     — single detector for all 4 ablations (registered as rfmoe_ablation)
  rfmoe_abl_no_spectral.yaml     — A1 config
  rfmoe_abl_no_region.yaml       — A2 config
  rfmoe_abl_uniform_gate.yaml    — A3 config
  rfmoe_abl_frozen_bb.yaml       — A4 config
  run_ablations.sh               — server launch script
  sync_ablation.sh               — Mac→server sync script
  find_best_epoch_ablation.py    — post-training: find best epoch per ablation
  ABLATION_LOG.md                — this file
  results/                       — evaluation results (to be filled in)
```

### Server (`<user>@<your-server>`)
```
/home/ibubu/ketupati/model7_ablation/
  [same files as above, synced]
  logs/
    ablations_master.log         — master log for the run_ablations.sh session
    train_abl_no_spectral_fs.log — per-ablation training logs
    train_abl_no_region_fs.log
    train_abl_uniform_gate_fs.log
    train_abl_frozen_bb_fs.log

/home/ibubu/ketupati/outputs/
  rf_moe_abl_no_spectral_fs/epoch_ckpts/    — A1 checkpoints
  rf_moe_abl_no_region_fs/epoch_ckpts/      — A2 checkpoints
  rf_moe_abl_uniform_gate_fs/epoch_ckpts/   — A3 checkpoints
  rf_moe_abl_frozen_bb_fs/epoch_ckpts/      — A4 checkpoints

/home/ibubu/ketupati/DeepfakeBench_DF40/training/detectors/
  rfmoe_ablation_detector.py     — copied here by run_ablations.sh at launch
```

---

## What to Do After Training

1. **Find best epoch per ablation** (on server):
   ```bash
   python3 /home/ibubu/ketupati/model7_ablation/find_best_epoch_ablation.py
   ```

2. **Evaluate best checkpoint** for each ablation (Tables 3, 4, 5):
   ```bash
   # Example for A1:
   cd /home/ibubu/ketupati/DeepfakeBench_DF40
   /home/ibubu/ketupati/venv/bin/python3 /home/ibubu/ketupati/model7/run_evaluation.py \
       --trained_on fs \
       --checkpoint /home/ibubu/ketupati/outputs/rf_moe_abl_no_spectral_fs/epoch_ckpts/ckpt_epoch_XX.pth \
       --tables 3 4 5
   ```
   Note: `run_evaluation.py` was written for the `rfmoe` detector. For ablations it may need a
   `--detector rfmoe_ablation` flag or the rfmoe_ablation yaml to be pointed at instead.

3. **Compile results** into the comparison table (see `results/` folder).

---

## Result Table

All metrics are AUC. Baseline = Model7 FS Epoch 8.  
Evaluated 2026-05-27. Best epoch selected by best val_auc on 10% holdout.

| Model | T3 FS(FF) | T3 FR(FF) | T3 EFS(FF) | T3 Avg | T4 FS(CDF) | T4 FR(CDF) | T4 EFS(CDF) | T4 Avg | T5 Avg |
|-------|-----------|-----------|------------|--------|------------|------------|-------------|--------|--------|
| **Model7 (full)** | 0.995 | 0.952 | 0.960 | **0.969** | 0.920 | 0.687 | 0.807 | **0.805** | **0.796** |
| A1: no_spectral   | 0.994 | 0.915 | 0.965 | 0.958 | 0.925 | 0.661 | 0.846 | 0.811 | 0.803 |
| A2: no_region     | 0.994 | 0.916 | 0.946 | 0.952 | 0.912 | 0.639 | 0.817 | 0.789 | 0.748 |
| A3: uniform_gate  | 0.994 | 0.915 | 0.958 | 0.956 | 0.932 | 0.689 | 0.831 | 0.817 | 0.788 |
| A4: frozen_bb     | 0.993 | 0.911 | 0.964 | 0.956 | 0.890 | 0.697 | 0.797 | 0.794 | 0.791 |

**Drop vs full model (T3 Avg / T4 Avg / T5 Avg):**
- A1 no_spectral:  −0.011 / +0.006 / +0.007  ← spectral branch helps T3 slightly; T4/T5 comparable
- A2 no_region:    −0.017 / −0.016 / −0.048  ← largest drop; RegionAttentionPool matters most
- A3 uniform_gate: −0.013 / +0.012 / −0.008  ← gating contributes modestly
- A4 frozen_bb:    −0.013 / −0.011 / −0.005  ← unfreezing blocks 22-23 helps across all tables

### T5 Full Breakdown (DFL / HeyGen / MidJ / WiR / SG / SG2 / SCLIP / e4e / CDiff)

| Model          | DFL   | HeyGen | MidJ  | WiR   | SG    | SG2   | SCLIP | e4e   | CDiff | Avg   |
|----------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| Model7 (full)  | 0.963 | 0.861  | 0.703 | 0.453 | 0.953 | 0.813 | 0.906 | 0.973 | 0.536 | 0.796 |
| A1 no_spectral | 0.977 | 0.851  | 0.750 | 0.484 | 0.875 | 0.969 | 0.844 | 0.963 | 0.515 | 0.803 |
| A2 no_region   | 0.982 | 0.697  | 0.766 | 0.422 | 0.812 | 0.891 | 0.688 | 0.938 | 0.540 | 0.748 |
| A3 uniform_gate| 0.970 | 0.734  | 0.781 | 0.516 | 0.797 | 0.969 | 0.781 | 0.952 | 0.595 | 0.788 |
| A4 frozen_bb   | 0.940 | 0.843  | 0.734 | 0.500 | 0.844 | 0.875 | 0.672 | 0.978 | 0.736 | 0.791 |

---

## Analysis and Findings

### Component Ranking by Importance

From largest to smallest contribution to the full model's performance:

1. **RegionAttentionPool (A2) — most important**
   - Removing it causes the largest drop across all three tables, especially T5 (−0.048).
   - The spatially-differentiated routing (boundary / interior / global) is the single biggest driver of open-set generalisation.

2. **Unfrozen CLIP backbone (A4) — consistently important**
   - Consistent drops across T3 (−0.013), T4 (−0.011), T5 (−0.005) when frozen.
   - Fine-tuning blocks 22–23 with backbone_lr=1e-5 allows the model to adapt CLIP's high-level features toward forgery-specific patterns.

3. **Learned gating / GatingNetwork (A3) — modest, mixed**
   - Helps T3 and T5 slightly, but removing it actually improves T4 (+0.012).
   - Uniform routing [1/3, 1/3, 1/3] is nearly as good as learned routing.
   - NOTE: A3 only ablates the *routing weights* — the 3 expert MLPs and their different region inputs are still present. The MoE *structure* itself (3 experts vs 1) was not ablated.

4. **SpectralBranch (A1) — domain-specific, does not generalise**
   - Helps within-domain detection (T3: −0.011 when removed).
   - Removing it actually improves T4 (+0.006) and T5 (+0.007).
   - Interpretation: the Sobel+Laplacian features learned frequency artifacts specific to the FF++ training domain that do not transfer to CDF or unseen forgeries.
   - The spectral branch was added to beat the CLIP-large baseline but the gains over CLIP came from RegionAttentionPool and the unfrozen backbone, not from frequency features.

### Comparison vs CLIP-large Baseline

The full model beats CLIP-large on T3 and T5 but not T4:

| | Model7 | CLIP-large | Gap |
|---|---|---|---|
| T3 Avg (FS) | 0.969 | 0.913 | +0.056 |
| T4 Avg (FS) | 0.805 | 0.813 | −0.008 |
| T5 Avg (FS) | 0.796 | 0.692 | +0.104 |

The largest gains over CLIP come from RegionAttentionPool (A2 shows −0.048 T5 drop without it) and the unfrozen backbone. The spectral branch did not contribute to beating CLIP on cross-domain or open-set evaluation.

### What Was Not Ablated

- **MoE structure itself**: A3 (uniform_gate) ablates the learned routing but keeps 3 experts with different region inputs. A true "single expert" ablation (one MLP on global mean-pool) was not run, so it is unknown how much the multi-expert structure contributes vs a simpler single-MLP head.

### Thesis Implications

- The ablation study supports the claim that RegionAttentionPool and partial CLIP fine-tuning are the key architectural contributions.
- For the spectral branch, the honest thesis statement is: *"The spectral branch improves within-domain cross-forgery detection (+0.011 T3 Avg) but does not improve cross-domain or open-set generalisation, suggesting the learned frequency features are domain-specific rather than forgery-generic."*
- The MoE routing ablation (A3) shows learned gating has modest impact; the stronger claim is about spatially-differentiated expert inputs (A2), not about the routing mechanism.

---

## Status

| Ablation | Training | Best Epoch | Evaluated | Notes |
|----------|----------|------------|-----------|-------|
| A1 no_spectral  | ✅ Done | Ep12 (val=0.9761) | ✅ Done | |
| A2 no_region    | ✅ Done | Ep14 (val=0.9768) | ✅ Done | |
| A3 uniform_gate | ✅ Done | Ep14 (val=0.9789) | ✅ Done | |
| A4 frozen_bb    | ✅ Done | Ep5  (val=0.9653) | ✅ Done | |
