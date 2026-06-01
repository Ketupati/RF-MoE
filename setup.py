#!/usr/bin/env python3
"""
setup.py  —  One-time setup for RF-MoE on campus GPU
Run once before training. Safe to re-run (idempotent).

Steps:
  1.  Install Python dependencies
  2.  Clone DF40 repo (skipped if already present)
  3.  Verify data directory structure
  4.  Create symlinks inside repo → data
  5.  Fix hardcoded /Youtu_Pangu_Security/ paths in YAML configs
  6.  Fix known bugs (logger.py, dataset/__init__.py)
  7.  Patch test.py with sys.setrecursionlimit(50000)
  8.  Add DF40 labels to train_config.yaml / test_config.yaml
  9.  Write rfmoe_detector.py → training/detectors/
 10.  Register RFMoEDetector in detectors/__init__.py
 11.  Write rfmoe.yaml → training/config/detector/

Usage:
  python setup.py
"""

import os
import sys
import shutil
import subprocess

# ============================================================
# PATHS  —  edit BASE if your data lives somewhere else
# ============================================================
BASE    = '/home/ibubu/ketupati'
DATA    = f'{BASE}/data'
REPO    = f'{BASE}/DeepfakeBench_DF40'
OUTPUTS = f'{BASE}/outputs'

TRAIN_DATA = f'{DATA}/DF40_train'
FF_REAL    = f'{DATA}/ff_real/FaceForensics++'
CDF_REAL   = f'{DATA}/cdf_real/Celeb-DF-v2'
JSON_DIR   = f'{DATA}/dataset_json'
WEIGHTS    = f'{BASE}/weights/df40_weights'

# ============================================================
# 1. Install dependencies
# ============================================================
def install_deps():
    print("\n[1/11] Installing dependencies...")
    # albumentations MUST be 1.3.1 — newer versions cause ZeroDivisionError
    pkgs = [
        'albumentations==1.3.1',  # MUST be 1.3.1 — newer causes ZeroDivisionError
        'timm',                   # xception + other CNN backbones in DF40 detectors
        'tensorboard',            # training logging
        'einops',                 # tensor ops used by several detectors
        'scikit-learn',           # AUC / metrics
        'scikit-image',           # image processing utilities
        'kornia',                 # geometric augmentation
        'opencv-python',          # cv2 image I/O
        'lmdb',                   # dataset loading
        'fvcore',                 # Facebook vision core utilities
        'simplejson',             # fast JSON
        'iopath',                 # path utilities
        'efficientnet_pytorch',   # EfficientNet backbone
        'transformers',           # CLIP (our backbone)
        'pandas',                 # data handling
        'tqdm',                   # progress bars
        'pyyaml',                 # YAML config parsing
        'imageio',                # image I/O fallback
    ]
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)
    if r.returncode != 0:
        print("  [WARN] pip returned non-zero — check output above")
    else:
        print("  [OK] All deps installed (albumentations==1.3.1)")


# ============================================================
# 2. Clone repo
# ============================================================
def clone_repo():
    print(f"\n[2/11] Checking repo at {REPO}...")
    if os.path.exists(f'{REPO}/training/test.py'):
        print("  [SKIP] Repo already present")
        return

    os.makedirs(BASE, exist_ok=True)
    print("  Cloning https://github.com/YZY-stack/DF40.git ...")
    r = subprocess.run(['git', 'clone', 'https://github.com/YZY-stack/DF40.git',
                        f'{BASE}/DF40_tmp'])
    if r.returncode != 0:
        raise RuntimeError("git clone failed")

    clone_dir = f'{BASE}/DF40_tmp'
    for item in os.listdir(clone_dir):
        inner = f'{clone_dir}/{item}'
        if os.path.isdir(inner) and os.path.exists(f'{inner}/training/test.py'):
            shutil.move(inner, REPO)
            print(f"  Moved DeepfakeBench_DF40 → {REPO}")
            break

    shutil.rmtree(clone_dir, ignore_errors=True)

    if not os.path.exists(f'{REPO}/training/test.py'):
        raise RuntimeError(f"Could not locate repo at {REPO} after clone")
    print("  [OK] Repo ready")


# ============================================================
# 3. Verify data
# ============================================================
def verify_data():
    print("\n[3/11] Verifying data structure...")
    checks = {
        'DF40 test data':  f'{DATA}/DF40',
        'DF40 train data': TRAIN_DATA,
        'FF++ real':       FF_REAL,
        'CDF real':        CDF_REAL,
        'JSON metadata':   JSON_DIR,
        'Weights':         WEIGHTS,
    }
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        n = len(os.listdir(path)) if exists and os.path.isdir(path) else 0
        status = 'OK' if exists else 'MISSING'
        print(f"  [{status:7s}] {name}: {path}  ({n} items)")
        if not exists:
            all_ok = False
    print("  [OK] All data present" if all_ok else
          "\n  [WARN] Fix MISSING items above before training")


# ============================================================
# 4. Create symlinks inside repo → actual data
# ============================================================
def create_symlinks():
    print("\n[4/11] Creating symlinks...")

    def mklink(src, dst, label):
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        ok = os.path.exists(dst)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")

    DDATA = f'{REPO}/deepfakes_detection_datasets'
    os.makedirs(DDATA, exist_ok=True)

    mklink(f'{DATA}/DF40',  f'{DDATA}/DF40',            'DF40 test data')
    mklink(TRAIN_DATA,      f'{DDATA}/DF40_train',       'DF40 train data')
    mklink(FF_REAL,         f'{DDATA}/FaceForensics++',  'FaceForensics++ real')
    mklink(CDF_REAL,        f'{DDATA}/Celeb-DF-v2',      'Celeb-DF-v2 real')
    mklink(JSON_DIR,        f'{REPO}/preprocessing/dataset_json', 'dataset_json (JSONs)')
    mklink(WEIGHTS,         f'{REPO}/training/df40_weights',      'df40_weights')

    os.makedirs(f'{OUTPUTS}/rf_moe', exist_ok=True)
    print("  [OK] Output directory ensured")


# ============================================================
# 5. Fix hardcoded Youtu_Pangu_Security paths in YAML configs
# ============================================================
def fix_yaml_paths():
    print("\n[5/11] Fixing hardcoded paths in YAML configs...")
    DDATA = f'{REPO}/deepfakes_detection_datasets'
    replacements = {
        '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets': DDATA,
        '/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json': JSON_DIR,
        '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json': JSON_DIR,
        '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/logs/df40_exps': OUTPUTS,
        '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/logs/benchv2': OUTPUTS,
        '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBench/training/pretrained': f'{REPO}/training/pretrained',
    }

    # CRITICAL: only walk specific safe directories with followlinks=False.
    # Using glob.glob('**/*.yaml', recursive=True) on the whole repo WILL follow
    # the deepfakes_detection_datasets symlink into 93 GB of test data.
    safe_dirs = [
        os.path.join(REPO, 'training', 'config'),
        os.path.join(REPO, 'preprocessing'),
    ]
    fixed = 0
    for safe_dir in safe_dirs:
        if not os.path.exists(safe_dir):
            continue
        for root, dirs, files in os.walk(safe_dir, followlinks=False):
            for fname in files:
                if not fname.endswith('.yaml'):
                    continue
                yf = os.path.join(root, fname)
                with open(yf) as f:
                    content = f.read()
                orig = content
                for old, new in replacements.items():
                    content = content.replace(old, new)
                if content != orig:
                    with open(yf, 'w') as f:
                        f.write(content)
                    fixed += 1

    print(f"  [OK] Fixed {fixed} YAML file(s)")


# ============================================================
# 6. Fix known bugs
# ============================================================
def fix_bugs():
    print("\n[6/11] Fixing known bugs...")

    # Bug 1: logger.py crashes on single-GPU (calls dist.get_rank() without DDP)
    lp = f'{REPO}/training/logger.py'
    if os.path.exists(lp):
        with open(lp) as f:
            c = f.read()
        if 'is_initialized' not in c:
            c = c.replace(
                'return dist.get_rank() == self.rank',
                'if dist.is_initialized():\n            return dist.get_rank() == self.rank\n        return True'
            )
            with open(lp, 'w') as f:
                f.write(c)
            print("  [OK] Fixed logger.py — added dist.is_initialized() guard")
        else:
            print("  [SKIP] logger.py already fixed")
    else:
        print("  [WARN] logger.py not found — skipping")

    # Bug 2: dataset/__init__.py imports SBI which needs imgaug (not installed)
    ip = f'{REPO}/training/dataset/__init__.py'
    needs_fix = True
    if os.path.exists(ip):
        with open(ip) as f:
            needs_fix = 'DeepfakeAbstractBaseDataset' not in f.read()
    if needs_fix:
        with open(ip, 'w') as f:
            f.write(
                'from .abstract_dataset import DeepfakeAbstractBaseDataset\n'
                'from .pair_dataset import pairDataset\n'
            )
        print("  [OK] Fixed dataset/__init__.py — removed SBI import")
    else:
        print("  [SKIP] dataset/__init__.py already fixed")


# ============================================================
# 7. Add sys.setrecursionlimit(50000) to test.py
# ============================================================
def fix_recursion_limit():
    print("\n[7/11] Patching recursion limit in test.py...")
    tp = f'{REPO}/training/test.py'
    if not os.path.exists(tp):
        print("  [WARN] test.py not found")
        return
    with open(tp) as f:
        c = f.read()
    if 'setrecursionlimit' not in c:
        c = 'import sys\nsys.setrecursionlimit(50000)\n' + c
        with open(tp, 'w') as f:
            f.write(c)
        print("  [OK] Added sys.setrecursionlimit(50000) to test.py")
    else:
        print("  [SKIP] test.py already has setrecursionlimit")


# ============================================================
# 8. Add DF40 method labels to train_config.yaml / test_config.yaml
# ============================================================
def add_df40_labels():
    print("\n[8/11] Adding DF40 labels to train/test configs...")
    methods = [
        'FSAll', 'FRAll', 'EFSAll', 'simswap', 'faceswap', 'facedancer',
        'blendface', 'inswap', 'facevid2vid', 'fomm', 'hyperreenact', 'mcnet',
        'sadtalker', 'wav2lip', 'StyleGAN3', 'VQGAN', 'pixart', 'ddim',
        'sd2.1', 'e4s', 'fsgan', 'mobileswap', 'uniface', 'danet', 'lia',
        'MRAA', 'one_shot_free', 'pirender', 'tpsm', 'e4e', 'StyleGAN2',
        'StyleGANXL', 'DiT', 'SiT', 'RDDM',
    ]
    label_yaml = '\n'.join(f'  {m}_Real: 0\n  {m}_Fake: 1' for m in methods)

    for cfg in ['training/config/train_config.yaml', 'training/config/test_config.yaml']:
        path = f'{REPO}/{cfg}'
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        with open(path) as f:
            content = f.read()
        if 'FSAll_Real' not in content:
            content += '\n' + label_yaml + '\n'
            with open(path, 'w') as f:
                f.write(content)
            print(f"  [OK] Added labels to {os.path.basename(cfg)}")
        else:
            print(f"  [SKIP] Labels already in {os.path.basename(cfg)}")


# ============================================================
# 9 & 10. Write rfmoe_detector.py and register it
# ============================================================
DETECTOR_CODE = r'''import os, logging, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from transformers import CLIPModel

logger = logging.getLogger(__name__)


class GatingLogger:
    """Accumulates per-batch gating weights for later analysis."""

    def __init__(self, num_experts=3, log_dir=None):
        self.num_experts = num_experts
        self.log_dir = log_dir
        self.reset()

    def reset(self):
        self.gate_weights  = []
        self.labels        = []
        self.dataset_names = []
        self.step_count    = 0

    def record(self, gw, labels, dataset_name=None):
        self.gate_weights.append(gw.cpu())
        self.labels.append(labels.cpu())
        if dataset_name:
            self.dataset_names.extend([dataset_name] * gw.size(0))
        self.step_count += 1

    def compute_stats(self):
        if not self.gate_weights:
            return {}
        all_gw     = torch.cat(self.gate_weights, 0)
        all_labels = torch.cat(self.labels, 0)
        mean_gw    = all_gw.mean(0).tolist()
        eps        = 1e-8
        entropy    = -(all_gw * (all_gw + eps).log()).sum(-1).mean().item()
        max_ent    = -(1 / self.num_experts) * self.num_experts * \
                      torch.tensor(1 / self.num_experts + eps).log().item()
        stats = {
            "total_samples":      int(all_gw.size(0)),
            "expert_utilization": {f"expert_{i}": round(v, 4) for i, v in enumerate(mean_gw)},
            "gating_entropy":     round(entropy, 4),
            "max_possible_entropy": round(max_ent, 4),
            "entropy_ratio":      round(entropy / max_ent, 4),
        }
        for lbl, name in [(0, "real"), (1, "fake")]:
            mask = all_labels == lbl
            if mask.any():
                sub = all_gw[mask]
                stats[f"{name}_expert_util"]     = {f"expert_{i}": round(v, 4) for i, v in enumerate(sub.mean(0).tolist())}
                stats[f"{name}_dominant_expert"] = int(sub.mean(0).argmax().item())
                stats[f"{name}_count"]           = int(mask.sum().item())
        if self.dataset_names:
            ds_stats = defaultdict(list)
            for i, ds in enumerate(self.dataset_names):
                if i < all_gw.size(0):
                    ds_stats[ds].append(all_gw[i])
            stats["per_dataset"] = {}
            for ds, gws in ds_stats.items():
                stacked = torch.stack(gws)
                stats["per_dataset"][ds] = {f"expert_{i}": round(v, 4) for i, v in enumerate(stacked.mean(0).tolist())}
                stats["per_dataset"][ds]["dominant_expert"] = int(stacked.mean(0).argmax().item())
                stats["per_dataset"][ds]["count"]           = len(gws)
        return stats

    def save_epoch(self, epoch, phase="train"):
        stats = self.compute_stats()
        stats.update({"epoch": epoch, "phase": phase,
                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"gating_{phase}_epoch{epoch:02d}.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
        return stats

    def save_raw(self, epoch, phase="train"):
        if self.log_dir and self.gate_weights:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"gating_raw_{phase}_epoch{epoch:02d}.pt")
            torch.save({
                "gate_weights":  torch.cat(self.gate_weights, 0),
                "labels":        torch.cat(self.labels, 0),
                "dataset_names": self.dataset_names,
            }, path)


class SpectralBranch(nn.Module):
    """Fixed edge/Laplacian filter bank + learnable CNN → frequency features."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.register_buffer("filters", self._build_filter_bank())
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def _build_filter_bank(self):
        f = torch.zeros(4, 1, 3, 3)
        f[0, 0] = torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])   # Sobel-H
        f[1, 0] = torch.tensor([[-1.,  0.,  1.], [-1., 0., 1.], [-1., 0., 1.]]) # Sobel-V
        f[2, 0] = torch.tensor([[ 0.,  1., -1.], [-1., 0., 1.], [1., -1., 0.]]) # diagonal
        f[3, 0] = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) # Laplacian
        return f

    def forward(self, images):
        gray = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]).unsqueeze(1)
        with torch.no_grad():
            filtered = F.conv2d(gray, self.filters, padding=1)
        return self.proj(self.cnn(filtered).flatten(1))


class RegionAttentionPool(nn.Module):
    """
    3 region views from CLIP patch tokens:
      boundary_feat  — learned boundary attention (FS expert: swap edges)
      interior_feat  — learned interior attention (FR expert: inner face)
      global_feat    — mean pool               (EFS expert: whole-image)
    """

    def __init__(self, d=1024):
        super().__init__()
        self.boundary_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))
        self.interior_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))

    def forward(self, patches):                     # patches: (B, N, 1024)
        bw = F.softmax(self.boundary_attn(patches), dim=1)
        bf = (patches * bw).sum(1)                  # (B, 1024)

        iw = F.softmax(self.interior_attn(patches), dim=1)
        if_ = (patches * iw).sum(1)                 # (B, 1024)

        gf = patches.mean(1)                        # (B, 1024)
        return bf, if_, gf


class Expert(nn.Module):
    def __init__(self, clip_dim=1024, spec_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim + spec_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, region_feat, spec_feat):
        return self.fc(torch.cat([region_feat, spec_feat], dim=1))


class GatingNetwork(nn.Module):
    """Switch Transformer-style router with load-balancing loss."""

    def __init__(self, d=1024, n_experts=3):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, n_experts))
        self.n = n_experts

    def forward(self, cls_token):
        weights         = F.softmax(self.router(cls_token), dim=1)   # (B, n)
        mean_load       = weights.mean(0)                             # (n,)
        load_balance    = self.n * (mean_load * mean_load).sum()      # scalar ≥ 1/n
        return weights, load_balance


@DETECTOR.register_module(module_name="rfmoe")
class RFMoEDetector(AbstractDetector):
    """
    RF-MoE: Region-Frequency Mixture of Experts for Deepfake Detection.

    Architecture (306M total, only 2.9M trainable):
      - Frozen CLIP-Large backbone          303M  (frozen)
      - RegionAttentionPool                   ~1M  (trainable)
      - SpectralBranch (4 fixed filters + CNN) ~0.3M (trainable)
      - 3 Experts (MLP each)                 ~1.5M (trainable)
      - GatingNetwork                        ~0.3M (trainable)
    """

    def __init__(self, config):
        super().__init__()
        self.config           = config
        self.backbone         = self.build_backbone(config)
        # Freeze entire CLIP backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.region_pool      = RegionAttentionPool(1024)
        self.spectral_branch  = SpectralBranch(256)
        self.experts          = nn.ModuleList([Expert() for _ in range(3)])
        self.gating           = GatingNetwork(1024, 3)
        self.loss_func        = self.build_loss(config)
        self.lb_weight        = config.get("load_balance_weight", 0.1)

        self._gating_logger       = None
        self._current_dataset_name = None

    @property
    def gating_logger(self):
        if self._gating_logger is None:
            log_dir = self.config.get("log_dir", ".")
            self._gating_logger = GatingLogger(
                num_experts=3,
                log_dir=os.path.join(log_dir, "gating_analysis"),
            )
        return self._gating_logger

    def set_dataset_name(self, name):
        self._current_dataset_name = name

    def build_backbone(self, config):
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

    def build_loss(self, config):
        return LOSSFUNC[config["loss_func"]]()

    def features(self, data_dict):
        with torch.no_grad():
            out = self.backbone(data_dict["image"], output_hidden_states=True)
        # cls_token: (B, 1024)  patch_tokens: (B, 196, 1024)
        return out.pooler_output, out.last_hidden_state[:, 1:, :]

    def classifier(self, features):
        pass  # routing handled in forward()

    def get_losses(self, data_dict, pred_dict):
        cls_loss = self.loss_func(pred_dict["cls"], data_dict["label"])
        lb_loss  = pred_dict.get("load_balance_loss", torch.tensor(0.0, device=cls_loss.device))
        return {
            "overall":  cls_loss + self.lb_weight * lb_loss,
            "cls_loss": cls_loss,
            "lb_loss":  lb_loss,
        }

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def forward(self, data_dict, inference=False):
        cls_token, patches = self.features(data_dict)               # (B,1024), (B,196,1024)
        spec_feat          = self.spectral_branch(data_dict["image"])  # (B,256)
        region_feats       = self.region_pool(patches)              # 3 × (B,1024)

        # Each expert sees its own spatial view + shared spectral features
        expert_logits = torch.stack(
            [expert(region_feats[i], spec_feat) for i, expert in enumerate(self.experts)],
            dim=1,
        )                                                           # (B, 3, 2)

        gate_weights, lb_loss = self.gating(cls_token)             # (B,3), scalar
        combined = (expert_logits * gate_weights.unsqueeze(-1)).sum(1)  # (B, 2)
        prob     = torch.softmax(combined, dim=1)[:, 1]            # (B,)

        if self.training and "label" in data_dict:
            self.gating_logger.record(
                gate_weights.detach(),
                data_dict["label"].detach(),
                dataset_name=self._current_dataset_name,
            )

        return {
            "cls":               combined,
            "prob":              prob,
            "feat":              cls_token,
            "load_balance_loss": lb_loss,
            "gate_weights":      gate_weights.detach(),
        }
'''


def write_detector():
    print("\n[9/11] Writing rfmoe_detector.py ...")
    det_path = f'{REPO}/training/detectors/rfmoe_detector.py'
    with open(det_path, 'w') as f:
        f.write(DETECTOR_CODE)
    print(f"  [OK] {det_path}")

    print("\n[10/11] Registering RFMoEDetector in detectors/__init__.py ...")
    ip = f'{REPO}/training/detectors/__init__.py'
    with open(ip) as f:
        c = f.read()
    if 'rfmoe_detector' not in c:
        with open(ip, 'w') as f:
            f.write(c.rstrip() + '\nfrom .rfmoe_detector import RFMoEDetector\n')
        print("  [OK] Registered")
    else:
        print("  [SKIP] Already registered")


# ============================================================
# 11. Write rfmoe.yaml
# ============================================================
def write_yaml():
    print("\n[11/11] Writing rfmoe.yaml ...")
    yaml_content = f"""model_name: rfmoe
all_dataset: [FaceForensics++, Celeb-DF-v2, DF40]
dataset_type: no_pair
compression: c23
frame_num:
  train: 4
  test: 8
  val: 8
train_batchSize: 64       # lower to 32 if GPU < 40GB
test_batchSize: 32        # lower to 16 if OOM during eval
optimizer:
  type: adam
  adam:
    lr: 0.0003
    weight_decay: 0.00001
    beta1: 0.9
    beta2: 0.999
    eps: 0.00000001
    amsgrad: false
  sgd:
    lr: 0.01
    weight_decay: 0.0005
    momentum: 0.9
lr_scheduler: null
nEpochs: 10
start_epoch: 0
loss_func: cross_entropy
load_balance_weight: 0.1
log_dir: {OUTPUTS}/rf_moe
save_feat: true
verbose: false
metric_scoring: auc
pretrained: null
manualSeed: 42
save_ckpt: true
cuda: true
cudnn: true
workers: 2              # do NOT increase — causes OOM on test sets
use_data_augmentation: true
data_aug:
  flip_prob: 0.5
  rotate_prob: 0.5
  rotate_limit: [-10, 10]
  blur_prob: 0.1
  blur_limit: [3, 7]
  brightness_prob: 0.2
  brightness_limit: [-0.1, 0.1]
  contrast_limit: [-0.1, 0.1]
  quality_lower: 40
  quality_upper: 100
resolution: 224
with_landmark: false
with_mask: false
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
"""
    yaml_path = f'{REPO}/training/config/detector/rfmoe.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"  [OK] {yaml_path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("RF-MoE Campus GPU Setup")
    print(f"  BASE:    {BASE}")
    print(f"  DATA:    {DATA}")
    print(f"  REPO:    {REPO}")
    print(f"  OUTPUTS: {OUTPUTS}")
    print("=" * 70)

    install_deps()
    clone_repo()
    verify_data()
    create_symlinks()
    fix_yaml_paths()
    fix_bugs()
    fix_recursion_limit()
    add_df40_labels()
    write_detector()
    write_yaml()

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("Next step:  python generate_jsons.py")
    print("=" * 70)
