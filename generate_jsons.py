#!/usr/bin/env python3
"""
generate_jsons.py  —  Generate nested-format training JSON files for RF-MoE.

Run AFTER setup.py (needs symlinks and repo in place).
Run BEFORE any training.  Safe to re-run (overwrites existing files).

Creates three files in data/dataset_json/:
  FSAll_ff_train.json   — 9 FS methods + FF++ real  (training split)
  FRAll_ff_train.json   — 12 FR methods + FF++ real  (training split)
  EFSAll_ff_train.json  — 10 EFS methods + FF++ real (training split)

CRITICAL NOTES:
  - Output uses NESTED format expected by abstract_dataset.py (NOT flat {path:label})
  - relpath is computed from the actual data directory, NOT via the repo symlink,
    to avoid getting garbage paths like ../../drive/MyDrive/...
  - Also patches label_dict in train_config.yaml / test_config.yaml

Usage:
  python generate_jsons.py
"""

import os
import json
import sys
from collections import defaultdict

# ============================================================
# PATHS  (must match setup.py)
# ============================================================
BASE       = '/home/ibubu/ketupati'
DATA       = f'{BASE}/data'
REPO       = f'{BASE}/DeepfakeBench_DF40'
JSON_DIR   = f'{DATA}/dataset_json'
TRAIN_DATA = f'{DATA}/DF40_train'
FF_REAL    = f'{DATA}/ff_real/FaceForensics++'

# ============================================================
# Method groups for DF40 training split
# ============================================================
FS_METHODS = [
    'simswap', 'faceswap', 'facedancer', 'blendface', 'inswap',
    'e4s', 'mobileswap', 'fsgan', 'uniface',
]
FR_METHODS = [
    'facevid2vid', 'fomm', 'hyperreenact', 'mcnet', 'sadtalker',
    'wav2lip', 'danet', 'lia', 'one_shot_free', 'pirender', 'tpsm', 'MRAA',
]
EFS_METHODS = [
    'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'ddim', 'DiT',
    'pixart', 'SiT', 'RDDM', 'sd2.1', 'VQGAN',
]

ALL_FAKE_LABELS = FS_METHODS + FR_METHODS + EFS_METHODS
ALL_REAL_LABELS = ['FSAll_Real', 'FRAll_Real', 'EFSAll_Real']


# ============================================================
# Helper: scan one fake method in DF40_train/
# ============================================================
def scan_fake_method(method_name, case_map):
    """
    Scan DF40_train/<method>/ and group image files by parent folder (= video_id).

    Video-based (FS/FR):  DF40_train/simswap/frames/001_870/000.png
      path: deepfakes_detection_datasets/DF40_train/simswap/frames/001_870/000.png

    Image-based (EFS):    DF40_train/StyleGAN2/001/seed0188.png
      path: deepfakes_detection_datasets/DF40_train/StyleGAN2/001/seed0188.png

    IMPORTANT: relpath is computed from TRAIN_DATA (the real filesystem path),
    not through the repo symlink, to avoid resolving symlinks and producing
    garbage paths like ../../drive/MyDrive/...
    """
    actual_folder = case_map.get(method_name.lower())
    if actual_folder is None:
        print(f"    WARNING: '{method_name}' not found in DF40_train/ — skipping")
        return {}

    method_dir = os.path.join(TRAIN_DATA, actual_folder)
    videos = defaultdict(list)

    for root, dirs, files in os.walk(method_dir, followlinks=False):
        for fname in sorted(files):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Compute relative path from TRAIN_DATA (real path, no symlink)
            rel_from_train = os.path.relpath(root, TRAIN_DATA)
            # Construct the path the dataloader will see (through the symlink inside repo)
            dataloader_path = f'deepfakes_detection_datasets/DF40_train/{rel_from_train}/{fname}'

            # Video ID = the immediate parent folder of the image file
            video_id = f'{os.path.basename(root)}_{method_name}'
            videos[video_id].append(dataloader_path)

    return dict(videos)


# ============================================================
# Helper: scan FF++ real data
# ============================================================
def scan_real_ff():
    """
    Scan FaceForensics++/original_sequences/youtube/c23/frames/<VID>/<frame>.png
    Groups by video folder.

    Returns dict of {video_id: [list of dataloader-relative paths]}
    """
    videos = defaultdict(list)

    if not os.path.exists(FF_REAL):
        print(f"  ERROR: FF++ real data not found at {FF_REAL}")
        sys.exit(1)

    for root, dirs, files in os.walk(FF_REAL, followlinks=False):
        for fname in sorted(files):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Compute path relative to the directory that the symlink points to
            # FF_REAL = .../ff_real/FaceForensics++
            # Symlink: repo/deepfakes_detection_datasets/FaceForensics++ → FF_REAL
            rel_from_ff = os.path.relpath(root, FF_REAL)
            dataloader_path = f'deepfakes_detection_datasets/FaceForensics++/{rel_from_ff}/{fname}'

            video_id = os.path.basename(root)
            videos[video_id].append(dataloader_path)

    return dict(videos)


# ============================================================
# Helper: write nested JSON
# ============================================================
def write_nested_json(filepath, dataset_name, real_label, real_videos, fake_methods_data):
    """
    Write the deeply nested JSON structure expected by abstract_dataset.py:

    {
      "dataset_name": {
        "class_label": {
          "train": {
            "video_id": {
              "label": "class_label",
              "frames": ["deepfakes_detection_datasets/...", ...]
            }
          }
        }
      }
    }

    Uses streaming writes to avoid holding entire structure in memory.
    Returns (total_videos, total_frames).
    """
    total_videos = 0
    total_frames = 0

    all_classes = [(real_label, real_videos)] + list(fake_methods_data)

    with open(filepath, 'w') as f:
        f.write('{\n')
        f.write(f'  {json.dumps(dataset_name)}: {{\n')

        for cls_idx, (class_label, class_videos) in enumerate(all_classes):
            f.write(f'    {json.dumps(class_label)}: {{\n')
            f.write(f'      "train": {{\n')

            video_items = list(class_videos.items())
            for vid_idx, (video_id, frames) in enumerate(video_items):
                comma       = ',' if vid_idx < len(video_items) - 1 else ''
                frames_json = json.dumps(frames)
                f.write(f'        {json.dumps(video_id)}: {{"label": {json.dumps(class_label)}, "frames": {frames_json}}}{comma}\n')
                total_frames += len(frames)
                total_videos += 1

            f.write('      }\n')  # close "train"
            cls_comma = ',' if cls_idx < len(all_classes) - 1 else ''
            f.write(f'    }}{cls_comma}\n')

        f.write('  }\n')
        f.write('}\n')

    return total_videos, total_frames


# ============================================================
# Helper: patch label_dict in YAML configs
# ============================================================
def patch_label_dict():
    """
    Add all label strings used in the generated JSONs to the label_dict
    sections of train_config.yaml and test_config.yaml.
    """
    try:
        import yaml
    except ImportError:
        print("  [WARN] pyyaml not installed — skipping label_dict patch")
        return

    config_files = [
        os.path.join(REPO, 'training', 'config', 'train_config.yaml'),
        os.path.join(REPO, 'training', 'config', 'test_config.yaml'),
    ]

    for cfg_path in config_files:
        if not os.path.exists(cfg_path):
            print(f"  [WARN] Not found: {cfg_path}")
            continue

        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        if 'label_dict' not in config:
            print(f"  [WARN] No label_dict in {os.path.basename(cfg_path)}")
            continue

        label_dict = config['label_dict']
        added      = []

        for lbl in ALL_REAL_LABELS:
            if lbl not in label_dict:
                label_dict[lbl] = 0
                added.append(f'{lbl}: 0')

        for lbl in ALL_FAKE_LABELS:
            if lbl not in label_dict:
                label_dict[lbl] = 1
                added.append(f'{lbl}: 1')

        if added:
            config['label_dict'] = label_dict
            with open(cfg_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"  [OK] {os.path.basename(cfg_path)}: added {len(added)} labels")
        else:
            print(f"  [SKIP] {os.path.basename(cfg_path)}: all labels already present")


# ============================================================
# Verify a sample of generated paths resolve through symlinks
# ============================================================
def verify_paths(json_path, n=5):
    """Grep a few paths from the JSON and check they resolve under REPO."""
    import subprocess
    result = subprocess.run(
        ['grep', '-oP', r'"deepfakes_detection_datasets/[^"]+\.png"', json_path],
        capture_output=True, text=True,
    )
    sample = [p.strip('"') for p in result.stdout.strip().split('\n') if p][:n]
    ok_count = 0
    for p in sample:
        abs_p  = os.path.join(REPO, p)
        exists = os.path.exists(abs_p)
        status = 'OK' if exists else 'MISSING'
        print(f"    [{status}] {p[:90]}")
        if exists:
            ok_count += 1
    return ok_count == len(sample)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("RF-MoE Training JSON Generator")
    print(f"  TRAIN_DATA: {TRAIN_DATA}")
    print(f"  FF_REAL:    {FF_REAL}")
    print(f"  JSON_DIR:   {JSON_DIR}")
    print("=" * 70)

    os.makedirs(JSON_DIR, exist_ok=True)

    # Build case-insensitive folder map for DF40_train/
    if not os.path.exists(TRAIN_DATA):
        print(f"ERROR: TRAIN_DATA not found: {TRAIN_DATA}")
        sys.exit(1)
    case_map = {f.lower(): f for f in os.listdir(TRAIN_DATA)}

    # Scan FF++ real data once (shared across all three JSONs)
    print("\nScanning FF++ real data ...")
    real_videos = scan_real_ff()
    n_real_frames = sum(len(v) for v in real_videos.values())
    print(f"  Found {len(real_videos)} real video folders, {n_real_frames} frames")

    # Generate one JSON per forgery type
    groups = [
        (FS_METHODS,  'FSAll_ff_train',  'FSAll_Real'),
        (FR_METHODS,  'FRAll_ff_train',  'FRAll_Real'),
        (EFS_METHODS, 'EFSAll_ff_train', 'EFSAll_Real'),
    ]

    for methods, group_name, real_label in groups:
        print(f"\n{'=' * 50}")
        print(f"Generating {group_name}.json  ({len(methods)} fake methods)")
        print('=' * 50)

        fake_methods_data = []
        for method in methods:
            print(f"  Scanning {method} ...", end=' ', flush=True)
            method_videos = scan_fake_method(method, case_map)
            n_v = len(method_videos)
            n_f = sum(len(v) for v in method_videos.values())
            print(f"{n_v} videos, {n_f} frames")
            if method_videos:
                fake_methods_data.append((method, method_videos))

        # Prefix real video keys to avoid collisions with fake video_ids
        real_prefixed = {f'{vid}_{real_label}': frames
                         for vid, frames in real_videos.items()}

        out_path = os.path.join(JSON_DIR, f'{group_name}.json')
        print(f"\n  Writing {out_path} ...", end=' ', flush=True)

        total_vids, total_frames = write_nested_json(
            out_path, group_name, real_label, real_prefixed, fake_methods_data
        )

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"done")
        print(f"  {total_vids} videos, {total_frames} frames, {size_mb:.1f} MB")

        # Spot-check a few paths
        print("  Path verification (5 sample frames):")
        verify_paths(out_path)

    # Patch label_dict in YAML configs
    print(f"\n{'=' * 50}")
    print("Patching label_dict in train/test configs ...")
    print('=' * 50)
    patch_label_dict()

    print("\n" + "=" * 70)
    print("JSON GENERATION COMPLETE")
    print("Generated files:")
    for name in ['FSAll_ff_train.json', 'FRAll_ff_train.json', 'EFSAll_ff_train.json']:
        p = os.path.join(JSON_DIR, name)
        size_mb = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0
        print(f"  {name}  ({size_mb:.1f} MB)")
    print("\nNext step:  python run_training.py --mode fs")
    print("=" * 70)


if __name__ == '__main__':
    main()
