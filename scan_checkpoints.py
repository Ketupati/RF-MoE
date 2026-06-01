#!/usr/bin/env python3
"""
scan_checkpoints.py — Find best epoch checkpoint by val AUC (not val loss).

Runs inference on the held-out val10 JSON for a subset of epoch checkpoints
and picks the one with the highest AUC. No test data is touched.

Usage:
  python scan_checkpoints.py --mode fs
  python scan_checkpoints.py --mode fr
  python scan_checkpoints.py --mode efs
  python scan_checkpoints.py --mode joint
  python scan_checkpoints.py --mode fs --all_epochs        # scan all 31
  python scan_checkpoints.py --mode fs --epochs 2 5 10 15 20 25 30
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

BASE     = '/home/ibubu/ketupati'
REPO     = f'{BASE}/DeepfakeBench_DF40'
JSON_DIR = f'{BASE}/data/dataset_json'

VAL_JSON_MAP = {
    'fs':    'FSAll_ff_val10.json',
    'fr':    'FRAll_ff_val10.json',
    'efs':   'EFSAll_ff_val10.json',
    'joint': 'joint_val10.json',
}

# Checkpoints to scan by default (covers early, mid, late training)
DEFAULT_EPOCHS = [0, 2, 3, 5, 6, 8, 10, 11, 15, 20, 25, 30]


class ValDataset(Dataset):
    def __init__(self, json_path, frame_num=8, resolution=224,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        with open(json_path) as f:
            data = json.load(f)
        self.tf = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ])
        self.frame_num = frame_num
        self.repo = REPO
        self.samples = []
        for ds, label_dict in data.items():
            for label_name, split_dict in label_dict.items():
                label = 0 if 'Real' in label_name else 1
                for split_key, videos in split_dict.items():
                    for vname, vinfo in videos.items():
                        frames = vinfo.get('frames', [])
                        if frames:
                            self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        sel = frames[:self.frame_num]
        imgs = []
        for fp in sel:
            # Try path as-is, then relative to REPO
            for candidate in [fp, os.path.join(self.repo, fp)]:
                if os.path.exists(candidate):
                    try:
                        imgs.append(self.tf(Image.open(candidate).convert('RGB')))
                    except Exception:
                        pass
                    break
        if not imgs:
            return None
        return {'image': imgs[0], 'label': torch.tensor(label, dtype=torch.long)}


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'label': torch.stack([x['label'] for x in batch]),
    }


def load_config(yaml_path):
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path, config):
    sys.path.insert(0, os.path.join(REPO, 'training'))
    from detectors import DETECTOR
    model = DETECTOR[config['model_name']](config)
    state = torch.load(ckpt_path, map_location='cpu')
    # Handle DataParallel-wrapped state dicts
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def compute_auc(model, dataloader, device):
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            imgs   = batch['image'].to(device)
            labels = batch['label'].numpy()
            preds  = model({'image': imgs, 'label': batch['label'].to(device)})
            # preds is dict with 'cls' key (logits) or direct tensor
            if isinstance(preds, dict):
                logits = preds.get('cls', preds.get('logits', None))
            else:
                logits = preds
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    if len(np.unique(all_labels)) < 2:
        return float('nan')
    return roc_auc_score(all_labels, all_probs)


def find_ckpt(ckpt_dir, epoch):
    p1 = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch:02d}.pth')
    p2 = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None


def main():
    global OUTPUTS
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['fs', 'fr', 'efs', 'joint'])
    p.add_argument('--epochs', nargs='+', type=int, default=None,
                   help='Epochs to scan (default: subset)')
    p.add_argument('--all_epochs', action='store_true',
                   help='Scan all 31 checkpoints (slow)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--workers',    type=int, default=2)
    args = p.parse_args()

    # Resolve paths
    mode_dir = f'{BASE}/outputs/rf_moe_{args.mode}'
    OUTPUTS  = mode_dir if os.path.isdir(mode_dir) else f'{BASE}/outputs/rf_moe'
    ckpt_dir = os.path.join(OUTPUTS, 'epoch_ckpts')

    yaml_path = os.path.join(REPO, 'training', 'config', 'detector', 'rfmoe.yaml')
    val_json  = os.path.join(JSON_DIR, VAL_JSON_MAP[args.mode])

    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: No epoch_ckpts/ found in {OUTPUTS}")
        sys.exit(1)
    if not os.path.exists(val_json):
        print(f"ERROR: Val JSON not found: {val_json}")
        sys.exit(1)

    # Decide which epochs to scan
    if args.all_epochs:
        epochs = list(range(0, 31))
    elif args.epochs:
        epochs = args.epochs
    else:
        epochs = DEFAULT_EPOCHS

    # Filter to epochs that actually have checkpoints
    available = []
    for ep in epochs:
        ckpt = find_ckpt(ckpt_dir, ep)
        if ckpt:
            available.append((ep, ckpt))
        else:
            print(f"  [SKIP] No checkpoint for epoch {ep}")

    if not available:
        print("ERROR: No checkpoints found.")
        sys.exit(1)

    print(f"\nMode:      {args.mode}")
    print(f"Val JSON:  {os.path.basename(val_json)}")
    print(f"Scanning:  {len(available)} checkpoints\n")

    # Load config and dataset once
    config   = load_config(yaml_path)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset  = ValDataset(val_json,
                          frame_num=config.get('frame_num', {}).get('test', 8),
                          resolution=config.get('resolution', 224),
                          mean=config.get('mean', [0.485, 0.456, 0.406]),
                          std=config.get('std',  [0.229, 0.224, 0.225]))
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=collate_fn,
                          pin_memory=True)

    print(f"Val samples: {len(dataset)}")
    print(f"\n{'Epoch':>6}  {'Val AUC':>10}  {'Checkpoint'}")
    print('-' * 55)

    results = {}
    for epoch, ckpt_path in available:
        try:
            model = load_model(ckpt_path, config).to(device)
            auc   = compute_auc(model, loader, device)
            results[epoch] = auc
            marker = ''
            print(f"  {epoch:>4}   {auc:>10.6f}  {os.path.basename(ckpt_path)}{marker}",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {epoch:>4}   {'ERROR':>10}  {e}")

    if not results:
        print("No results obtained.")
        return

    valid   = {ep: auc for ep, auc in results.items() if not np.isnan(auc)}
    best_ep = max(valid, key=valid.get)

    print('\n' + '-' * 55)
    print(f"\nBest epoch: {best_ep}  (val_AUC={valid[best_ep]:.6f})")
    best_ckpt = find_ckpt(ckpt_dir, best_ep)
    print(f"Best checkpoint: {best_ckpt}")
    print(f"\nNext step:")
    print(f"  python run_evaluation.py --trained_on {args.mode} "
          f"--checkpoint {best_ckpt} --tables {'6' if args.mode == 'joint' else '3 4 5'}")


if __name__ == '__main__':
    main()
