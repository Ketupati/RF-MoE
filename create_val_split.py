#!/usr/bin/env python3
"""
create_val_split.py — Create 90/10 train/val splits from RF-MoE training JSONs.

Splits at the VIDEO level (stratified by real/fake) so frames from the same
video never appear in both splits.

Outputs (written to same dir as input JSONs):
  FSAll_ff_train90.json  +  FSAll_ff_val10.json
  FRAll_ff_train90.json  +  FRAll_ff_val10.json
  EFSAll_ff_train90.json +  EFSAll_ff_val10.json

Usage:
  python create_val_split.py
  python create_val_split.py --val_frac 0.1 --seed 42
"""

import os
import json
import random
import argparse

BASE     = '/home/ibubu/ketupati'
JSON_DIR = f'{BASE}/data/dataset_json'

MODES = {
    'fs':  'FSAll_ff_train',
    'fr':  'FRAll_ff_train',
    'efs': 'EFSAll_ff_train',
}


def split_json(src_path: str, train_path: str, val_path: str,
               val_frac: float, seed: int, force: bool = False):
    """
    Read src JSON, split videos stratified by label, write train/val JSONs.
    Top-level key is renamed to match the output filename so the dataloader
    can find it by dataset name.
    Idempotent — skips if both output files already exist (unless --force).
    """
    if not force and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  [SKIP] Already exists: {os.path.basename(train_path)} + val")
        return

    with open(src_path) as f:
        data = json.load(f)

    rng = random.Random(seed)

    # Use output filenames (without .json) as top-level keys
    train_key = os.path.splitext(os.path.basename(train_path))[0]
    val_key   = os.path.splitext(os.path.basename(val_path))[0]

    train_data = {}
    val_data   = {}

    for dataset_name, label_dict in data.items():
        train_data[train_key] = {}
        val_data[val_key]     = {}

        for label_name, split_dict in label_dict.items():
            train_data[train_key][label_name] = {}
            val_data[val_key][label_name]     = {}

            # split_dict keys are split names ('train') → {video_name: video_info}
            for split_key, videos in split_dict.items():
                video_keys = list(videos.keys())
                rng.shuffle(video_keys)

                n_val      = max(1, int(len(video_keys) * val_frac))
                val_keys   = set(video_keys[:n_val])
                train_keys = set(video_keys[n_val:])

                train_data[train_key][label_name][split_key] = {
                    k: v for k, v in videos.items() if k in train_keys
                }
                val_data[val_key][label_name][split_key] = {
                    k: v for k, v in videos.items() if k in val_keys
                }

                n_total = len(video_keys)
                print(f"    {label_name}/{split_key}: "
                      f"{len(train_keys)} train + {len(val_keys)} val "
                      f"(total {n_total})")

    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    with open(val_path, 'w') as f:
        json.dump(val_data, f)

    print(f"  [OK] Written: {os.path.basename(train_path)}")
    print(f"  [OK] Written: {os.path.basename(val_path)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed',     type=int,   default=42)
    p.add_argument('--force',    action='store_true',
                   help='Overwrite existing split files')
    args = p.parse_args()

    for mode, base_name in MODES.items():
        src   = os.path.join(JSON_DIR, f'{base_name}.json')
        train = os.path.join(JSON_DIR, f'{base_name}90.json')
        val   = os.path.join(JSON_DIR, f'{base_name[:-6]}_val10.json')  # FSAll_ff_val10

        if not os.path.exists(src):
            print(f"[SKIP] Source not found: {src}")
            continue

        print(f"\n{mode.upper()} — splitting {base_name}.json")
        split_json(src, train, val, args.val_frac, args.seed, force=args.force)

    # Create combined joint_val10.json by merging all 3 val JSONs
    print("\nCreating joint_val10.json (merged FS+FR+EFS val)...")
    joint_val_path = os.path.join(JSON_DIR, 'joint_val10.json')
    if not args.force and os.path.exists(joint_val_path):
        print(f"  [SKIP] Already exists: joint_val10.json")
    else:
        merged = {}
        for mode, base_name in MODES.items():
            val_path = os.path.join(JSON_DIR, f'{base_name[:-6]}_val10.json')
            if os.path.exists(val_path):
                with open(val_path) as f:
                    merged.update(json.load(f))
        with open(joint_val_path, 'w') as f:
            json.dump(merged, f)
        print(f"  [OK] Written: joint_val10.json ({len(merged)} datasets)")

    print("\nDone. Val split JSONs written to:", JSON_DIR)


if __name__ == '__main__':
    main()
