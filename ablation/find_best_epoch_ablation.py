#!/usr/bin/env python3
"""
find_best_epoch_ablation.py — Find best checkpoint per ablation run.

Parses val_auc from training logs, prints top 5 per ablation.

Usage (on server):
  python3 /home/ibubu/ketupati/model7_ablation/find_best_epoch_ablation.py

  # Specific ablation:
  python3 find_best_epoch_ablation.py --ablation no_spectral
"""

import os
import re
import argparse

BASE     = '/home/ibubu/ketupati'
LOG_DIR  = f'{BASE}/model7_ablation/logs'
CKPT_BASE = f'{BASE}/outputs'

ABLATIONS = ['no_spectral', 'no_region', 'uniform_gate', 'frozen_bb']


def parse_log(log_path):
    """Extract {epoch: val_auc} from training log."""
    results = {}
    if not os.path.exists(log_path):
        return results
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[VAL\] Epoch (\d+) val_auc: ([0-9.]+)', line)
            if m:
                results[int(m.group(1))] = float(m.group(2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ablation', default=None, help='Single ablation to check')
    args = p.parse_args()

    ablations = [args.ablation] if args.ablation else ABLATIONS

    print("=" * 60)
    print("  RF-MoE Ablation — Best Checkpoint Summary")
    print("=" * 60)

    for abl in ablations:
        log_path = os.path.join(LOG_DIR, f'train_abl_{abl}_fs.log')
        results  = parse_log(log_path)

        print(f"\n  Ablation: {abl}")
        if not results:
            print(f"    [No val_auc found]  Log: {log_path}")
            continue

        top5 = sorted(results, key=results.get, reverse=True)[:5]
        ckpt_dir = os.path.join(CKPT_BASE, f'rf_moe_abl_{abl}_fs', 'epoch_ckpts')

        print(f"  {'Rank':<6} {'Epoch':<8} {'val_auc':<10} {'Checkpoint'}")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*40}")
        for rank, ep in enumerate(top5, 1):
            ckpt = os.path.join(ckpt_dir, f'ckpt_epoch_{ep:02d}.pth')
            exists = 'OK' if os.path.exists(ckpt) else 'MISSING'
            print(f"  {rank:<6} {ep:<8} {results[ep]:<10.4f} [{exists}] {ckpt}")

        best_ep  = top5[0]
        best_auc = results[best_ep]
        print(f"\n  --> Best: Epoch {best_ep}  val_auc={best_auc:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
