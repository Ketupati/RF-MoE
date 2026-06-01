# Generalized Deepfake Detection: Region-Frequency Mixture of Experts

RF-MoE is a deepfake detector targeting the **DF40 (NeurIPS 2024)** benchmark. It combines a
partially fine-tuned **CLIP-Large ViT-L/14** backbone with learnable region-aware expert
routing and a spectral branch, to detect deepfakes across four forgery categories
(Face Swapping, Face Reenactment, Entire-Face Synthesis, Face Editing).

## Architecture

```
Input (B, 3, 224, 224)
    │
CLIP-Large ViT-L/14 backbone (blocks 0–21 frozen, blocks 22–23 fine-tuned)
    ├─ cls_token     (B, 1024)   ─→ GatingNetwork ─→ gate weights (B, 3)
    └─ patch_tokens  (B, 256, 1024)
            │
       RegionAttentionPool (learned spatial routing)
            ├─ boundary_feat (B, 1024)  ─→ Expert 0  (FS: swap-edge artifacts)
            ├─ interior_feat (B, 1024)  ─→ Expert 1  (FR: inner-face artifacts)
            └─ global_feat   (B, 1024)  ─→ Expert 2  (EFS: synthetic patterns)

SpectralBranch (fixed Sobel + Laplacian filters + CNN) ─→ spec_feat (B, 256)
   (shared input to all 3 experts)

3 Expert MLPs: concat(region_feat, spec_feat) ─→ logits (B, 2) each
Weighted sum by gate weights ─→ combined logits ─→ AUC

Loss: cross_entropy + 0.1·load_balance + 0.1·gating_entropy
```

## Files

| File | Purpose |
|------|---------|
| `rfmoe_detector.py` | The RF-MoE model (backbone, RegionAttentionPool, SpectralBranch, experts, gating) |
| `rfmoe.yaml` | Training/evaluation config for the DF40 framework |
| `setup.py` | Installs deps, clones the DF40 repo, patches bugs, registers the detector |
| `run_training.py` | Launch training (`--mode fs|fr|efs|joint`) |
| `run_evaluation.py` | Evaluate checkpoints for Protocols 1–4 (Tables 3–6) |
| `find_best_epoch.py` | Select best checkpoint by validation AUC |
| `generate_jsons.py`, `generate_test_jsons.py` | Build dataset metadata JSONs |
| `create_val_split.py` | Build the 10% video-level validation holdout |
| `check_dataset_stats.py`, `scan_checkpoints.py` | Utilities |
| `compile_results.py`, `make_final_tables.py`, `generate_html_tables.py` | Result compilation |
| `run_all.sh` | End-to-end pipeline |

## Usage

```bash
python setup.py                       # one-time setup
python generate_jsons.py              # build train-split metadata
python run_training.py --mode fs      # train (fs | fr | efs | joint)
python find_best_epoch.py --mode fs   # pick best checkpoint
python run_evaluation.py --trained_on fs --checkpoint <path> --tables 3 4 5
python compile_results.py             # formatted result tables
```

> Note: `BASE` paths in the scripts point to the training server layout
> (`/home/ibubu/ketupati`). Adjust these to your environment before running.

## Citation

DF40 benchmark: *DF40: Toward Next-Generation Deepfake Detection*, NeurIPS 2024.
