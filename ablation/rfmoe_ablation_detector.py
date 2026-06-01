"""
rfmoe_ablation_detector.py  —  RF-MoE Ablation Study (based on Model7)

Registers as "rfmoe_ablation". Controls which component is ablated via
the 'ablation_mode' key in yaml:

  no_spectral    : SpectralBranch output replaced with zeros
  no_region      : RegionAttentionPool replaced with mean-pool for all 3 experts
  uniform_gate   : GatingNetwork replaced with fixed uniform weights [1/3, 1/3, 1/3]
  frozen_backbone: CLIP blocks 22-23 kept frozen (no backbone fine-tuning)

Everything else is identical to Model7 (Dropout 0.5, blocks 22-23 unfrozen).

Ablation A1 (no_spectral)   — quantifies SpectralBranch contribution
Ablation A2 (no_region)     — quantifies RegionAttentionPool contribution
Ablation A3 (uniform_gate)  — quantifies learned gating contribution
Ablation A4 (frozen_backbone) — quantifies CLIP fine-tuning contribution
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from transformers import CLIPModel

logger = logging.getLogger(__name__)


class SpectralBranch(nn.Module):
    """Fixed Sobel+Laplacian filter bank + learnable CNN (identical to Model7)."""

    def __init__(self, embed_dim=256):
        super().__init__()
        sobel_h  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_v  = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32)
        diagonal = torch.tensor([[ 0, 1, 2], [-1, 0, 1], [-2,-1, 0]], dtype=torch.float32)
        laplace  = torch.tensor([[ 0,-1, 0], [-1, 4,-1], [ 0,-1, 0]], dtype=torch.float32)
        kernels = torch.stack([sobel_h, sobel_v, diagonal, laplace]).unsqueeze(1)
        self.register_buffer('kernels', kernels)
        self.cnn = nn.Sequential(
            nn.Conv2d(4,  32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        with torch.no_grad():
            feats = F.conv2d(gray, self.kernels, padding=1)
        return self.proj(self.cnn(feats).squeeze(-1).squeeze(-1))


class RegionAttentionPool(nn.Module):
    """3 region views: boundary attn, interior attn, mean pool (identical to Model7)."""

    def __init__(self, d=1024):
        super().__init__()
        self.boundary_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))
        self.interior_attn = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1))

    def forward(self, patches):
        bw = F.softmax(self.boundary_attn(patches), dim=1)
        boundary_feat = (patches * bw).sum(1)
        iw = F.softmax(self.interior_attn(patches), dim=1)
        interior_feat = (patches * iw).sum(1)
        global_feat   = patches.mean(1)
        return boundary_feat, interior_feat, global_feat


class Expert(nn.Module):
    """Expert MLP: concat(region[1024], spectral[256]) → logits[2]. Dropout=0.5 (Model7)."""

    def __init__(self, clip_dim=1024, spec_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim + spec_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, region_feat, spec_feat):
        return self.fc(torch.cat([region_feat, spec_feat], dim=1))


class GatingNetwork(nn.Module):
    """Switch Transformer-style soft router with load-balancing loss."""

    def __init__(self, d=1024, n_experts=3):
        super().__init__()
        self.router    = nn.Sequential(nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, n_experts))
        self.n_experts = n_experts

    def forward(self, cls_token):
        weights      = F.softmax(self.router(cls_token), dim=1)
        mean_load    = weights.mean(0)
        load_balance = self.n_experts * (mean_load * mean_load).sum()
        return weights, load_balance


@DETECTOR.register_module(module_name="rfmoe_ablation")
class RFMoEAblationDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config       = config
        self.ablation_mode = config.get("ablation_mode", "full")
        logger.info(f"[RFMoEAblation] ablation_mode = {self.ablation_mode}")

        # Backbone: CLIP-Large vision encoder (start fully frozen)
        self.backbone = self.build_backbone(config)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # A4 frozen_backbone: keep ALL blocks frozen; else unfreeze last 2
        if self.ablation_mode != "frozen_backbone":
            _adapter_lr  = config.get("optimizer", {}).get("adam", {}).get("lr", 3e-4)
            _backbone_lr = config.get("backbone_lr", 1e-5)
            _lr_scale    = _backbone_lr / _adapter_lr
            for layer in self.backbone.encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
                    p.register_hook(lambda g: g * _lr_scale)

        self.region_pool     = RegionAttentionPool(d=1024)
        self.spectral_branch = SpectralBranch(embed_dim=256)
        self.experts         = nn.ModuleList([Expert() for _ in range(3)])
        self.gating          = GatingNetwork(d=1024, n_experts=3)

        self.loss_func = self.build_loss(config)
        self.lb_weight = config.get("load_balance_weight", 0.1)

    def build_backbone(self, config):
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

    def build_loss(self, config):
        return LOSSFUNC[config["loss_func"]]()

    def features(self, data_dict):
        out = self.backbone(data_dict["image"], output_hidden_states=True)
        return out.pooler_output, out.last_hidden_state[:, 1:, :]

    def classifier(self, features):
        pass

    def get_losses(self, data_dict, pred_dict):
        cls_loss = self.loss_func(pred_dict["cls"], data_dict["label"])
        lb_loss  = pred_dict.get("load_balance_loss",
                                 torch.tensor(0.0, device=cls_loss.device))
        total = cls_loss + self.lb_weight * lb_loss
        return {"overall": total, "cls_loss": cls_loss, "lb_loss": lb_loss}

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def forward(self, data_dict, inference=False):
        cls_token, patches = self.features(data_dict)      # (B,1024), (B,256,1024)
        B = cls_token.shape[0]

        # A1: no spectral branch — zeroed features
        if self.ablation_mode == "no_spectral":
            spec_feat = torch.zeros(B, 256, device=cls_token.device)
        else:
            spec_feat = self.spectral_branch(data_dict["image"])

        # A2: no region attention — mean pool for all 3 experts
        if self.ablation_mode == "no_region":
            mean_feat    = patches.mean(1)
            region_feats = (mean_feat, mean_feat, mean_feat)
        else:
            region_feats = self.region_pool(patches)

        expert_logits = torch.stack(
            [expert(region_feats[i], spec_feat) for i, expert in enumerate(self.experts)],
            dim=1,
        )                                                   # (B, 3, 2)

        # A3: uniform gate — equal 1/3 weight to each expert
        if self.ablation_mode == "uniform_gate":
            gate_weights = torch.ones(B, 3, device=cls_token.device) / 3
            lb_loss      = torch.tensor(0.0, device=cls_token.device)
        else:
            gate_weights, lb_loss = self.gating(cls_token)

        combined = (expert_logits * gate_weights.unsqueeze(-1)).sum(1)  # (B, 2)
        prob     = torch.softmax(combined, dim=1)[:, 1]                  # (B,)

        return {
            "cls":               combined,
            "prob":              prob,
            "feat":              cls_token,
            "load_balance_loss": lb_loss,
            "gate_weights":      gate_weights.detach(),
        }
