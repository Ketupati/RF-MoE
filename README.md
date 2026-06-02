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
| `ablation/` | Ablation study: detector variant (A1–A4), configs, runner, and `ABLATION_LOG.md` |
| `requirements.txt` | Python dependencies (`albumentations==1.3.1` is pinned) |

## Data & Directory Layout

The project expects the following layout on the training machine (root = `BASE`, e.g. `/home/ibubu/ketupati`):

```
BASE/
├── data/                         # all datasets
│   ├── DF40_train/               # training fake frames   (30 method folders)
│   ├── ff_real/                  # real frames (FaceForensics++)
│   ├── DF40/                     # test data              (40 method folders)
│   ├── cdf_real/                 # Celeb-DF real frames
│   └── dataset_json/             # nested-JSON split metadata (pointers into the trees above)
│
├── DeepfakeBench_DF40/           # cloned official DF40 framework (setup.py registers our detector here)
│   └── training/
│       ├── config/detector/rfmoe.yaml      # our config
│       ├── detectors/rfmoe_detector.py     # our model
│       ├── train.py   test.py
│
├── outputs/                      # one folder per training run
│   └── rf_moe_<mode>/            # mode = fs | fr | efs | joint
│       ├── epoch_ckpts/ckpt_epoch_XX.pth   # per-epoch checkpoints
│       ├── training.log
│       └── table{3,4,5,6}_<mode>_results.json
│
├── checkpoints/                  # best checkpoint copied out, per mode
└── weights/                      # CLIP-Large backbone weights
```

**Frame data is organized as `method → frames → video → image`:**

```
data/DF40_train/simswap/
├── frames/
│   └── <video_id>/               # e.g. 025_067  (target_source)
│       ├── 000.png
│       ├── 024.png               # frames sampled along the clip
│       └── ...
└── landmarks/                    # .npy facial landmarks (unused by RF-MoE)

data/ff_real/FaceForensics++/original_sequences/youtube/c23/frames/<id>/<frame>.png
```

Notes:
- **One folder per generation method** (simswap, StyleGAN2, …). Video-level grouping (`<video_id>/`) is what enables the leak-free **90/10 train/val split at the video level**.
- The model never scans raw folders directly — it reads the **nested JSONs** in `data/dataset_json/`, which store the relative frame paths and per-clip real/fake labels (label ending in `_Real` → real, bare method name → fake).
- Real frames are shared across all training modes (the FF++ source is identical).

## Usage

```bash
pip install -r requirements.txt       # dependencies
python setup.py                       # one-time setup (clones DF40, patches, registers detector)
python generate_jsons.py              # build train-split metadata
python run_training.py --mode fs      # train (fs | fr | efs | joint)
python find_best_epoch.py --mode fs   # pick best checkpoint
python run_evaluation.py --trained_on fs --checkpoint <path> --tables 3 4 5
python compile_results.py             # formatted result tables
```

> Note: `BASE` paths in the scripts point to the training server layout
> (`/home/ibubu/ketupati`). Adjust these to your environment before running.

## Pretrained Checkpoints

Trained RF-MoE checkpoints (FS, FR, EFS, and Joint modes) are available on Google Drive:

🔗 https://drive.google.com/drive/folders/1o5jaEnUcrKtyKxYXeJPzTZXgzdjTKguL?usp=sharing

Download the checkpoint for the desired mode and pass it to `run_evaluation.py` via `--checkpoint`.

## Citation

DF40 benchmark: *DF40: Toward Next-Generation Deepfake Detection*, NeurIPS 2024.
