"""
rfmoe_detector.py  —  RF-MoE: Region-Frequency Mixture of Experts  (Phase 2 revised v2)

Standalone copy.  setup.py writes this file to:
  {REPO}/training/detectors/rfmoe_detector.py

This version: Phase 0 architecture (Sobel+Laplacian) + unfreeze CLIP blocks 22-23.
Goal: test if Sobel's better cross-domain generalization combines with partial CLIP fine-tuning.

  - SpectralBranch (Sobel+Laplacian) — Phase 0 spectral branch restored
  - CLIP blocks 22-23 unfrozen, fine-tuned end-to-end. A gradient-scaling hook
    (backbone_lr/adapter_lr) is registered, but note this does NOT produce a
    reduced effective rate under Adam (Adam normalizes away the gradient scale);
    the unfrozen blocks effectively train at the adapter learning rate.
  - Contrastive loss disabled (weight=0.0)

Architecture (~306M total, ~28M trainable):
  - CLIP-Large (blocks 0-21 frozen, blocks 22-23 unfrozen)            303M
  - RegionAttentionPool                                                  ~1M  (trainable)
  - SpectralBranch (Sobel+Laplacian, fixed filters + CNN)             ~0.3M  (trainable)
  - 3 Expert MLPs                                                      ~1.5M  (trainable)
  - GatingNetwork (Switch Transformer style)                           ~0.3M  (trainable)
  - ContrastiveHead (projection MLP)                                   ~0.1M  (trainable)

Expert specialization (same as Phase 0):
  Expert 0 (FS)  — boundary attention   (face-swap edge artifacts)
  Expert 1 (FR)  — interior attention   (reenactment inner-face artifacts)
  Expert 2 (EFS) — global mean pool     (whole-image synthesis patterns)
"""

import math
import os
import logging
import json
import time
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


# ────────────────────────────────────────────────────────────────
# Gating statistics logger  (unchanged from Phase 0)
# ────────────────────────────────────────────────────────────────
class GatingLogger:
    """Accumulates per-batch gating weights for post-hoc analysis."""

    def __init__(self, num_experts: int = 3, log_dir: str = None):
        self.num_experts = num_experts
        self.log_dir     = log_dir
        self.reset()

    def reset(self):
        self.gate_weights  = []
        self.labels        = []
        self.dataset_names = []
        self.step_count    = 0

    def record(self, gw: torch.Tensor, labels: torch.Tensor, dataset_name: str = None):
        self.gate_weights.append(gw.cpu())
        self.labels.append(labels.cpu())
        if dataset_name:
            self.dataset_names.extend([dataset_name] * gw.size(0))
        self.step_count += 1

    def compute_stats(self) -> dict:
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
            "total_samples":        int(all_gw.size(0)),
            "expert_utilization":   {f"expert_{i}": round(v, 4) for i, v in enumerate(mean_gw)},
            "gating_entropy":       round(entropy, 4),
            "max_possible_entropy": round(max_ent, 4),
            "entropy_ratio":        round(entropy / max_ent, 4),
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
                stats["per_dataset"][ds] = {
                    f"expert_{i}": round(v, 4) for i, v in enumerate(stacked.mean(0).tolist())
                }
                stats["per_dataset"][ds]["dominant_expert"] = int(stacked.mean(0).argmax().item())
                stats["per_dataset"][ds]["count"]           = len(gws)
        return stats

    def save_epoch(self, epoch: int, phase: str = "train") -> dict:
        stats = self.compute_stats()
        stats.update({"epoch": epoch, "phase": phase,
                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"gating_{phase}_epoch{epoch:02d}.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Gating stats → {path}")
        return stats

    def save_raw(self, epoch: int, phase: str = "train"):
        if self.log_dir and self.gate_weights:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"gating_raw_{phase}_epoch{epoch:02d}.pt")
            torch.save({
                "gate_weights":  torch.cat(self.gate_weights, 0),
                "labels":        torch.cat(self.labels, 0),
                "dataset_names": self.dataset_names,
            }, path)
            logger.info(f"Raw gating data → {path}")


# ────────────────────────────────────────────────────────────────
# Phase 0 Spectral Branch: Sobel + Laplacian fixed filters
# ────────────────────────────────────────────────────────────────
class SpectralBranch(nn.Module):
    """
    Fixed-filter spectral feature extractor using Sobel and Laplacian kernels.
    4 channels: Sobel-H, Sobel-V, Diagonal, Laplacian — all fixed (no grad).
    Only the downstream CNN + projection layer are trainable.

    Input:  (B, 3, H, W)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        sobel_h  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_v  = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32)
        diagonal = torch.tensor([[ 0, 1, 2], [-1, 0, 1], [-2,-1, 0]], dtype=torch.float32)
        laplace  = torch.tensor([[ 0,-1, 0], [-1, 4,-1], [ 0,-1, 0]], dtype=torch.float32)

        kernels = torch.stack([sobel_h, sobel_v, diagonal, laplace]).unsqueeze(1)  # (4,1,3,3)
        self.register_buffer('kernels', kernels)

        self.cnn = nn.Sequential(
            nn.Conv2d(4,  32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(1, keepdim=True)                          # (B, 1, H, W)
        with torch.no_grad():
            feats = F.conv2d(gray, self.kernels, padding=1)     # (B, 4, H, W)
        out = self.cnn(feats)
        return self.proj(out.squeeze(-1).squeeze(-1))           # (B, embed_dim)


# ────────────────────────────────────────────────────────────────
# Phase 1 Spectral Branch: Block-wise DCT + 2-level db4 DWT
# ────────────────────────────────────────────────────────────────
class SpectralBranchV2(nn.Module):
    """
    NOTE: UNUSED in this model. The active spectral branch is SpectralBranch
    (Sobel+Laplacian) above; this DCT+DWT variant is kept for reference only
    and is never instantiated by RFMoEDetector.

    Fixed-filter spectral feature extractor combining:
      1. 2-level Daubechies db4 DWT  — multi-scale detail subbands (LH, HL, HH)
         capturing edge and texture artifacts at H/2 and H/4 resolution.
      2. 8×8 block DCT-II            — captures JPEG quantization artifacts and
         GAN frequency fingerprints (no pywt required; pure torch conv ops).

    Both filter sets are registered as buffers (fixed, no grad).
    Only the downstream CNN + projection layer are trainable.

    Channel layout fed to CNN  (all pooled to H/8 × W/8):
      Channels  0– 2 : DWT level-1 detail subbands  (LH1, HL1, HH1)
      Channels  3– 5 : DWT level-2 detail subbands  (LH2, HL2, HH2)
      Channels  6–21 : DCT-II AC coefficients  (indices 1–16, DC skipped)
      Total: 22 channels

    Input:  (B, 3, H, W)   RGB image (any size; 224×224 standard for CLIP)
    Output: (B, embed_dim)
    """

    _N_DCT  = 16   # AC DCT channels to keep (skip DC component at index 0)
    _N_DWT  = 6    # 3 subbands × 2 DWT levels
    _CNN_IN = _N_DCT + _N_DWT   # 22

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # ── db4 Daubechies wavelet filter coefficients ────────────────────────
        # Scaling (lo) and wavelet (hi) coefficients for decimation
        _lo = [0.48296291314469025,  0.8365163037378079,
               0.22414386804185735, -0.12940952255092145]
        _hi = [-0.12940952255092145, -0.22414386804185735,
                0.8365163037378079,  -0.48296291314469025]
        self.register_buffer('_dwt_lo',
                             torch.tensor(_lo, dtype=torch.float32).view(1, 1, 4))
        self.register_buffer('_dwt_hi',
                             torch.tensor(_hi, dtype=torch.float32).view(1, 1, 4))

        # ── Orthonormal DCT-II basis for 8×8 blocks ───────────────────────────
        self.register_buffer('_dct_basis', self._build_dct8())

        # ── Learnable CNN + projection ────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(self._CNN_IN, 32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,           64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,           128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_dct8() -> torch.Tensor:
        """Return orthonormal 8×8 DCT-II basis matrix."""
        N = 8
        D = torch.zeros(N, N, dtype=torch.float32)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    D[k, n] = math.sqrt(1.0 / N)
                else:
                    D[k, n] = math.sqrt(2.0 / N) * math.cos(
                        math.pi * k * (2 * n + 1) / (2 * N))
        return D   # (8, 8)

    # ── DWT helpers ───────────────────────────────────────────────────────────

    def _dwt2d_level(self, x: torch.Tensor):
        """
        One level of 2D DWT via separable db4 1D convolutions (stride 2).

        Uses left-padding of (filter_len - 1) = 3 so that the output length
        is exactly ceil(input_len / 2) — equivalent to periodic extension.

        x      : (B, 1, H, W)
        returns: LL, LH, HL, HH  — each (B, 1, H//2, W//2)
                 LL = low-low  (approximation)
                 LH = low-high (horizontal detail)
                 HL = high-low (vertical detail)
                 HH = high-high (diagonal detail)
        """
        B, C, H, W = x.shape
        lo, hi = self._dwt_lo, self._dwt_hi   # (1, 1, 4)
        p = lo.shape[-1] - 1                   # = 3

        # ── Step 1: filter along W (column direction) ─────────────────────
        xw     = x.reshape(B * C * H, 1, W)
        xw     = F.pad(xw, (p, 0))
        xL_w   = F.conv1d(xw, lo, stride=2).reshape(B, C, H, -1)  # lowpass
        xH_w   = F.conv1d(xw, hi, stride=2).reshape(B, C, H, -1)  # highpass

        W2 = xL_w.shape[-1]   # = ceil(W/2)

        # ── Step 2: filter along H (row direction) ────────────────────────
        def _filt_H(t):
            # t: (B, C, H, W2) → filter H dimension
            t2  = t.permute(0, 1, 3, 2).reshape(B * C * W2, 1, H)
            t2  = F.pad(t2, (p, 0))
            tL  = F.conv1d(t2, lo, stride=2)
            tH  = F.conv1d(t2, hi, stride=2)
            H2  = tL.shape[-1]
            tL  = tL.reshape(B, C, W2, H2).permute(0, 1, 3, 2)   # (B, C, H2, W2)
            tH  = tH.reshape(B, C, W2, H2).permute(0, 1, 3, 2)
            return tL, tH

        LL, LH = _filt_H(xL_w)   # from low-pass column: LL approx, LH horiz detail
        HL, HH = _filt_H(xH_w)   # from high-pass column: HL vert detail, HH diag
        return LL, LH, HL, HH

    # ── DCT helper ────────────────────────────────────────────────────────────

    def _block_dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        8×8 block 2D DCT-II applied to a grayscale image.

        x: (B, 1, H, W)  — H and W should be divisible by 8 (224 → 28 blocks/side)
        Returns: (B, _N_DCT, H//8, W//8)  — first _N_DCT AC coefficients (DC skipped)
        """
        B, _C, H, W = x.shape
        N  = 8
        Hb = H // N
        Wb = W // N

        # Pad to multiple of 8 if needed (handles edge cases)
        if H % N != 0 or W % N != 0:
            H_pad = (N - H % N) % N
            W_pad = (N - W % N) % N
            x  = F.pad(x, (0, W_pad, 0, H_pad))
            Hb = x.shape[2] // N
            Wb = x.shape[3] // N

        # Extract non-overlapping 8×8 blocks: (B, 64, Hb*Wb)
        blocks = F.unfold(x, kernel_size=N, stride=N)
        L      = blocks.shape[-1]
        blocks = blocks.view(B, N, N, L)              # (B, 8, 8, L)

        # 2D DCT-II via two matrix multiplications
        D      = self._dct_basis                       # (8, 8)
        blocks = torch.einsum('ki,bijl->bkjl', D, blocks)  # row DCT
        blocks = torch.einsum('mj,bkjl->bkml', D, blocks)  # col DCT

        # Reshape to spatial grid: (B, 64, Hb, Wb)
        blocks = blocks.reshape(B, N * N, Hb, Wb)

        # Drop DC (index 0); return first _N_DCT AC coefficients
        return blocks[:, 1:self._N_DCT + 1, :, :]    # (B, 16, Hb, Wb)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Grayscale conversion: (B, 3, H, W) → (B, 1, H, W)
        gray = (0.299 * images[:, 0]
                + 0.587 * images[:, 1]
                + 0.114 * images[:, 2]).unsqueeze(1)

        with torch.no_grad():
            # ── DWT level 1 → detail subbands at H/2 × W/2 ─────────────────
            LL1, LH1, HL1, HH1 = self._dwt2d_level(gray)

            # ── DWT level 2 → detail subbands at H/4 × W/4 ─────────────────
            LL2, LH2, HL2, HH2 = self._dwt2d_level(LL1)

            # ── 8×8 block DCT → AC coefficients at H/8 × W/8 ───────────────
            dct_feats = self._block_dct(gray)              # (B, 16, H/8, W/8)

        # Pool all feature maps to the DCT spatial resolution (H/8 × W/8)
        th, tw = dct_feats.shape[2], dct_feats.shape[3]
        pool   = lambda t: F.adaptive_avg_pool2d(t, (th, tw))

        dwt_feats = torch.cat([
            pool(LH1), pool(HL1), pool(HH1),   # 3 level-1 detail channels
            pool(LH2), pool(HL2), pool(HH2),   # 3 level-2 detail channels
        ], dim=1)                               # (B, 6, H/8, W/8)

        spectral = torch.cat([dwt_feats, dct_feats], dim=1)  # (B, 22, H/8, W/8)

        out = self.cnn(spectral).flatten(1)    # (B, 128)
        return self.proj(out)                  # (B, embed_dim)


# ────────────────────────────────────────────────────────────────
# Contrastive projection head  (Phase 1)
# ────────────────────────────────────────────────────────────────
class ContrastiveHead(nn.Module):
    """
    NOTE: instantiated but inactive in this model (contrastive_loss_weight=0.0),
    so it receives no gradients and does not affect training or inference.

    Two-layer MLP projection head for supervised contrastive learning.
    Projects CLIP CLS embeddings into a lower-dimensional hypersphere
    where same-class samples are pulled together and cross-class pushed apart.

    in_dim  : dimension of CLIP CLS token (1024 for ViT-L/14)
    proj_dim: dimension of projection space (128 standard per SimCLR/SupCon)
    """

    def __init__(self, in_dim: int = 1024, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(True),
            nn.Linear(in_dim // 2, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=1)   # L2-normalized (B, proj_dim)


# ────────────────────────────────────────────────────────────────
# Region attention pooling  (unchanged from Phase 0)
# ────────────────────────────────────────────────────────────────
class RegionAttentionPool(nn.Module):
    """
    Extracts 3 spatial region views from CLIP patch tokens:
      boundary_feat  — softmax-weighted attention (learns face boundary focus)
      interior_feat  — separate softmax attention (learns inner-face focus)
      global_feat    — simple mean pool (whole-image summary)

    Each view is fed to the corresponding expert.
    """

    def __init__(self, d: int = 1024):
        super().__init__()
        self.boundary_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))
        self.interior_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))

    def forward(self, patches: torch.Tensor):
        # patches: (B, N, 1024)   N = 256 for ViT-L/14 @ 224px (16x16 grid of 14px patches)
        bw = F.softmax(self.boundary_attn(patches), dim=1)
        boundary_feat = (patches * bw).sum(1)       # (B, 1024)

        iw = F.softmax(self.interior_attn(patches), dim=1)
        interior_feat = (patches * iw).sum(1)       # (B, 1024)

        global_feat = patches.mean(1)               # (B, 1024)

        return boundary_feat, interior_feat, global_feat


# ────────────────────────────────────────────────────────────────
# Expert MLP  (unchanged from Phase 0)
# ────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    """
    Fuses one spatial region view with spectral features.
    Input:  region_feat (B, 1024) + spec_feat (B, 256)  →  (B, 1280)
    Output: logits (B, 2)
    """

    def __init__(self, clip_dim: int = 1024, spec_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim + spec_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, region_feat: torch.Tensor, spec_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([region_feat, spec_feat], dim=1))


# ────────────────────────────────────────────────────────────────
# Gating network  (unchanged from Phase 0)
# ────────────────────────────────────────────────────────────────
class GatingNetwork(nn.Module):
    """
    Switch Transformer-style soft router.
    Computes per-sample expert weights and a load-balancing auxiliary loss.
    """

    def __init__(self, d: int = 1024, n_experts: int = 3):
        super().__init__()
        self.router    = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, n_experts))
        self.n_experts = n_experts

    def forward(self, cls_token: torch.Tensor):
        weights    = F.softmax(self.router(cls_token), dim=1)   # (B, n_experts)
        mean_load  = weights.mean(0)
        load_balance = self.n_experts * (mean_load * mean_load).sum()
        return weights, load_balance


# ────────────────────────────────────────────────────────────────
# RF-MoE Detector  (Phase 1)
# ────────────────────────────────────────────────────────────────
@DETECTOR.register_module(module_name="rfmoe")
class RFMoEDetector(AbstractDetector):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Backbone: CLIP-Large vision encoder
        self.backbone = self.build_backbone(config)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Phase 2 revised: unfreeze last 2 blocks (22-23) with scaled LR
        _adapter_lr  = config.get("optimizer", {}).get("adam", {}).get("lr", 3e-4)
        _backbone_lr = config.get("backbone_lr", 1e-5)
        _lr_scale    = _backbone_lr / _adapter_lr
        for layer in self.backbone.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
                p.register_hook(lambda g: g * _lr_scale)

        # Learnable components
        self.region_pool     = RegionAttentionPool(d=1024)
        self.spectral_branch = SpectralBranch(embed_dim=256)       # Phase 0 Sobel+Laplacian
        self.experts         = nn.ModuleList([Expert() for _ in range(3)])
        self.gating          = GatingNetwork(d=1024, n_experts=3)

        # Contrastive head (Phase 1)
        self.contrast_head   = ContrastiveHead(in_dim=1024, proj_dim=128)
        self.contrast_weight = config.get("contrastive_loss_weight", 0.1)
        self.contrast_temp   = config.get("contrastive_temperature",  0.07)

        self.loss_func    = self.build_loss(config)
        self.lb_weight    = config.get("load_balance_weight", 0.1)
        self.entropy_weight = config.get("entropy_reg_weight", 0.0)

        self._gating_logger        = None
        self._current_dataset_name = None

    # ── Gating logger (lazy init) ──────────────────────────────
    @property
    def gating_logger(self) -> GatingLogger:
        if self._gating_logger is None:
            log_dir = self.config.get("log_dir", ".")
            self._gating_logger = GatingLogger(
                num_experts=3,
                log_dir=os.path.join(log_dir, "gating_analysis"),
            )
        return self._gating_logger

    def set_dataset_name(self, name: str):
        self._current_dataset_name = name

    # ── AbstractDetector API ───────────────────────────────────
    def build_backbone(self, config):
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

    def build_loss(self, config):
        return LOSSFUNC[config["loss_func"]]()

    def features(self, data_dict: dict):
        """Extract CLIP patch tokens and CLS token (gradients flow through blocks 22-23)."""
        out = self.backbone(data_dict["image"], output_hidden_states=True)
        return out.pooler_output, out.last_hidden_state[:, 1:, :]

    def classifier(self, features):
        # Routing handled in forward(); stub required by AbstractDetector
        pass

    # ── Supervised contrastive loss ────────────────────────────
    def _supervised_contrastive_loss(self,
                                     proj:   torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        """
        Supervised contrastive loss (Khosla et al., NeurIPS 2020).

        proj  : (B, D) L2-normalized projected features
        labels: (B,)   0 = real, 1 = fake

        Pulls embeddings of the same class together and pushes apart
        across classes in the temperature-scaled cosine similarity space.
        """
        B = proj.shape[0]
        if B < 2:
            return proj.sum() * 0.0   # degenerate batch — skip gracefully

        # Temperature-scaled cosine similarity matrix
        sim = torch.mm(proj, proj.T) / self.contrast_temp   # (B, B)

        # Mask out the diagonal (self-similarity is uninformative)
        eye      = torch.eye(B, dtype=torch.bool, device=proj.device)
        sim      = sim.masked_fill(eye, float('-inf'))

        # Positive pairs: same label, different sample
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye   # (B, B)

        if not pos_mask.any():
            return torch.tensor(0.0, device=proj.device)   # all same class — skip

        # Log-softmax denominator (over all non-self samples)
        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)   # (B, 1)
        log_prob  = sim - log_denom                              # (B, B)

        # IMPORTANT: use torch.where instead of multiplying by float mask.
        # log_prob contains -inf at non-positive positions (diagonal was masked
        # to -inf above). -inf * 0.0 = NaN in IEEE 754 — corrupts the entire loss.
        # torch.where replaces those positions with 0.0 without touching -inf.
        log_prob_pos = torch.where(pos_mask, log_prob,
                                   torch.zeros_like(log_prob))  # (B, B)

        n_pos = pos_mask.float().sum(1)                          # (B,)
        loss  = -(log_prob_pos).sum(1) / n_pos.clamp(min=1)

        # Only average over anchors that have at least one positive
        return loss[n_pos > 0].mean()

    # ── Loss aggregation ───────────────────────────────────────
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        cls_loss = self.loss_func(pred_dict["cls"], data_dict["label"])
        lb_loss  = pred_dict.get("load_balance_loss",
                                 torch.tensor(0.0, device=cls_loss.device))
        con_loss = pred_dict.get("contrastive_loss",
                                 torch.tensor(0.0, device=cls_loss.device))

        # Per-sample gate entropy regularization — penalises routing collapse.
        # High entropy (uniform routing) is rewarded; low entropy (all-to-one) is penalised.
        gate_w = pred_dict.get("gate_weights", None)
        if gate_w is not None and self.entropy_weight > 0:
            eps = 1e-8
            entropy = -(gate_w * (gate_w + eps).log()).sum(dim=-1).mean()
            entropy_loss = -entropy  # we want to maximise entropy → minimise negative entropy
        else:
            entropy_loss = torch.tensor(0.0, device=cls_loss.device)

        total = (cls_loss
                 + self.lb_weight      * lb_loss
                 + self.contrast_weight * con_loss
                 + self.entropy_weight  * entropy_loss)
        return {
            "overall":          total,
            "cls_loss":         cls_loss,
            "lb_loss":          lb_loss,
            "contrastive_loss": con_loss,
            "entropy_loss":     entropy_loss,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    # ── Forward pass ──────────────────────────────────────────
    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        # 1. Extract CLIP features (gradients flow through blocks 22-23)
        cls_token, patches = self.features(data_dict)      # (B,1024), (B,256,1024)

        # 2. Spectral features (Sobel + Laplacian fixed filters + CNN)
        spec_feat = self.spectral_branch(data_dict["image"])           # (B, 256)

        # 3. Three region views
        region_feats = self.region_pool(patches)                       # 3 × (B, 1024)

        # 4. Expert logits
        expert_logits = torch.stack(
            [expert(region_feats[i], spec_feat)
             for i, expert in enumerate(self.experts)],
            dim=1,
        )                                                               # (B, 3, 2)

        # 5. Gating network
        gate_weights, lb_loss = self.gating(cls_token)                # (B, 3), scalar

        # 6. Weighted combination
        combined = (expert_logits * gate_weights.unsqueeze(-1)).sum(1) # (B, 2)
        prob     = torch.softmax(combined, dim=1)[:, 1]                # (B,)

        # 7. Supervised contrastive loss (training only)
        con_loss = torch.tensor(0.0, device=cls_token.device)
        if self.training and "label" in data_dict and self.contrast_weight > 0:
            proj     = self.contrast_head(cls_token)
            con_loss = self._supervised_contrastive_loss(proj, data_dict["label"])

        # 8. Log gating weights
        if self.training and "label" in data_dict:
            self.gating_logger.record(
                gate_weights.detach(),  # detach explicitly for logging
                data_dict["label"].detach(),
                dataset_name=self._current_dataset_name,
            )

        return {
            "cls":               combined,
            "prob":              prob,
            "feat":              cls_token,
            "load_balance_loss": lb_loss,
            "gate_weights":      gate_weights,          # keep grad for entropy reg
            "gate_weights_log":  gate_weights.detach(), # detached copy for logging
            "contrastive_loss":  con_loss,
        }
