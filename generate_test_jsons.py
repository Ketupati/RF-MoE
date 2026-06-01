#!/usr/bin/env python3
"""
generate_test_jsons.py — Generate test JSON files matched to your actual DF40 filesystem.

Replaces the author-provided JSONs (which have path mismatches causing None skips)
with freshly generated ones that only include files physically present on disk.

Generates (overwrites if already exists):

  Table 3 — FF domain (cross-forgery):
    FSAll_ff.json, FRAll_ff.json, EFSAll_ff.json

  Table 4 — CDF domain (cross-domain):
    FSAll_cdf.json, FRAll_cdf.json, EFSAll_cdf.json

  Training monitors — per-method FF (used during train.py testing):
    simswap_ff.json, faceswap_ff.json, facedancer_ff.json, blendface_ff.json,
    inswap_ff.json, facevid2vid_ff.json, wav2lip_ff.json,
    StyleGAN3_ff.json, pixart_ff.json  (+ all other methods)

  Tables 5/6 — Unknown/unseen domain:
    deepfacelab.json, heygen.json, MidJourney.json,
    stargan.json, starganv2.json, styleclip.json,
    e4e_ff.json, CollabDiff.json, whichisreal.json

Filesystem structures handled (confirmed from remote GPU):
  FS/FR  ff  : method/ff/frames/video_id/*.png
  FS/FR  cdf : method/cdf/frames/video_id/*.png
  EFS    ff  : method/ff/video_id/*.png          (no frames/ subfolder)
  EFS    cdf : method/cdf/Fake_from_Celeb-real/video_id/*.png
  e4e    ff  : e4e/ff/video_id/*.png             (EFS-style FE)
  stargan    : fake/fake/*.png  +  real/real/*.png
  StarGAN2, styleclip, MidJourney, Whichisreal : fake/*.png flat + real/*.png flat
  deepfacelab: fake/frames/vid_id/*.png + real/frames/vid_id/*.png
  heygen     : fake/frames/vid_id/*.png + real/vid_id/*.png
  CollabDiff : fake/vid_id/*.png + real/ (auto-detected)

Usage:
  python generate_test_jsons.py
"""

import os
import json
import sys
from collections import defaultdict

# ============================================================
# PATHS  (must match setup.py)
# ============================================================
BASE     = '/home/ibubu/ketupati'
DATA     = f'{BASE}/data'
REPO     = f'{BASE}/DeepfakeBench_DF40'
DF40_DIR = f'{DATA}/DF40'
FF_REAL  = f'{DATA}/ff_real/FaceForensics++'
CDF_REAL = f'{DATA}/cdf_real/Celeb-DF-v2'
JSON_DIR = f'{DATA}/dataset_json'

IMG_EXTS = {'.png', '.jpg', '.jpeg'}

# ============================================================
# Method groups  —  EXACT folder names in DF40/
# ============================================================
FS_METHODS  = ['simswap', 'faceswap', 'facedancer', 'blendface', 'inswap',
               'e4s', 'mobileswap', 'fsgan', 'uniface']
FR_METHODS  = ['facevid2vid', 'fomm', 'hyperreenact', 'mcnet', 'sadtalker',
               'wav2lip', 'danet', 'lia', 'one_shot_free', 'pirender', 'tpsm', 'MRAA']
EFS_METHODS = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'ddim', 'DiT',
               'pixart', 'SiT', 'RDDM', 'sd2.1', 'VQGAN']

# e4e lives in DF40/e4e/ff/video_id/ — EFS-style (no frames/ subfolder)
EFS_STYLE_EXTRA = ['e4e']

# Self-contained datasets: json_name → actual_folder_name_in_DF40
#   Each has its own fake/ and real/ subfolders
SELF_CONTAINED_FOLDER = {
    'deepfacelab': 'deepfacelab',
    'heygen':      'heygen',
    'MidJourney':  'MidJourney',
    'stargan':     'stargan',
    'starganv2':   'StarGAN2',
    'styleclip':   'styleclip',
    'CollabDiff':  'CollabDiff',
    'whichisreal': 'Whichisreal',
    # e4e_ff handled separately (has ff/cdf like EFS, not fake/real)
}


# ============================================================
# Core scanner
# ============================================================
def scan_images(scan_dir, data_root, dl_prefix, method_tag=''):
    """
    Walk scan_dir, group image files by their immediate parent folder name.
    Returns {video_id: [dataloader_path, ...]}

    video_id = parent folder name of each image, suffixed with method_tag.
    For flat dirs (images directly in scan_dir) → video_id = 'flat_all_{method_tag}'.

    data_root  : the real filesystem root that maps to dl_prefix symlink
    dl_prefix  : e.g. 'deepfakes_detection_datasets/DF40'
    """
    videos = defaultdict(list)

    if not os.path.exists(scan_dir):
        return {}

    for root, dirs, files in os.walk(scan_dir, followlinks=False):
        sorted_files = sorted(
            f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        if not sorted_files:
            continue

        # Video ID = immediate parent folder name
        if os.path.normpath(root) == os.path.normpath(scan_dir):
            vid = f'flat_all_{method_tag}' if method_tag else 'flat_all'
        else:
            vid = os.path.basename(root)
            if method_tag:
                vid = f'{vid}_{method_tag}'

        for fname in sorted_files:
            abs_path = os.path.join(root, fname)
            rel      = os.path.relpath(abs_path, data_root)
            videos[vid].append(f'{dl_prefix}/{rel}')

    return dict(videos)


# ============================================================
# Fake scanners per structure type
# ============================================================
def fake_ff_fs_fr(method):
    """FS/FR ff: method/ff/frames/video_id/"""
    d = os.path.join(DF40_DIR, method, 'ff', 'frames')
    return scan_images(d, DF40_DIR, 'deepfakes_detection_datasets/DF40', method)


def fake_cdf_fs_fr(method):
    """FS/FR cdf: method/cdf/frames/video_id/"""
    d = os.path.join(DF40_DIR, method, 'cdf', 'frames')
    return scan_images(d, DF40_DIR, 'deepfakes_detection_datasets/DF40', method)


def fake_ff_efs(method):
    """EFS ff: method/ff/video_id/  (no frames/ subfolder)"""
    d = os.path.join(DF40_DIR, method, 'ff')
    return scan_images(d, DF40_DIR, 'deepfakes_detection_datasets/DF40', method)


def fake_cdf_efs(method):
    """EFS cdf: method/cdf/<intermediate>/video_id/
    The intermediate subfolder (e.g. Fake_from_Celeb-real) is auto-detected.
    """
    cdf_dir = os.path.join(DF40_DIR, method, 'cdf')
    if not os.path.exists(cdf_dir):
        return {}
    # Find intermediate subdir (skip it for video_id grouping by scanning deeper)
    subdirs = sorted(
        d for d in os.listdir(cdf_dir)
        if os.path.isdir(os.path.join(cdf_dir, d))
    )
    if subdirs:
        scan_base = os.path.join(cdf_dir, subdirs[0])
    else:
        scan_base = cdf_dir
    return scan_images(scan_base, DF40_DIR, 'deepfakes_detection_datasets/DF40', method)


# ============================================================
# Real data scanners
# ============================================================
def real_ff():
    """FF++ real: original_sequences/youtube/c23/frames/video_id/"""
    d = os.path.join(FF_REAL, 'original_sequences', 'youtube', 'c23', 'frames')
    if not os.path.exists(d):
        # Fallback: scan entire FF_REAL
        print(f"  [WARN] Expected path not found: {d} — scanning all of FF_REAL")
        d = FF_REAL
    return scan_images(d, FF_REAL, 'deepfakes_detection_datasets/FaceForensics++', 'real')


def real_cdf():
    """Celeb-DF-v2 real: auto-detect structure"""
    if not os.path.exists(CDF_REAL):
        print(f"  [WARN] CDF real not found: {CDF_REAL}")
        return {}
    return scan_images(CDF_REAL, CDF_REAL, 'deepfakes_detection_datasets/Celeb-DF-v2', 'real')


# ============================================================
# Self-contained dataset scanners
# ============================================================
def scan_self_contained(json_name):
    """
    Scan a self-contained dataset (has its own fake/ and real/ in DF40/).
    Handles all observed structures:
      - flat:          fake/*.png   + real/*.png
      - double-nested: fake/fake/   + real/real/
      - frames_vid:    fake/frames/vid/ + real/frames/vid/
      - vid_direct:    fake/vid/   + real/vid/   (CollabDiff, heygen fake)
      - mixed_real:    fake/frames/vid/ + real/vid/ (heygen)
    Auto-detects structure from the filesystem.
    """
    folder = SELF_CONTAINED_FOLDER[json_name]
    method_dir = os.path.join(DF40_DIR, folder)

    if not os.path.exists(method_dir):
        print(f"  [WARN] Not found: {method_dir}")
        return {}, {}

    fake_dir = os.path.join(method_dir, 'fake')
    real_dir = os.path.join(method_dir, 'real')

    fake_videos = _scan_self_dir(fake_dir, folder, 'fake')
    real_videos = _scan_self_dir(real_dir, folder, 'real')

    return fake_videos, real_videos


def _scan_self_dir(base_dir, folder, side):
    """Scan fake/ or real/ of a self-contained method — auto-detect structure."""
    if not os.path.exists(base_dir):
        print(f"  [WARN] Dir not found: {base_dir}")
        return {}

    return scan_images(
        base_dir, DF40_DIR, 'deepfakes_detection_datasets/DF40', f'{folder}_{side}'
    )


# ============================================================
# JSON writer
# ============================================================
def write_test_json(filepath, dataset_name, classes):
    """
    Write nested test JSON.

    classes: list of (class_label, {video_id: [paths]}) tuples

    Resulting structure:
    {
      "dataset_name": {
        "class_label": {
          "test": {
            "video_id": {"label": "class_label", "frames": [...]}
          }
        }
      }
    }
    """
    total_videos = 0
    total_frames = 0

    with open(filepath, 'w') as f:
        f.write('{\n')
        f.write(f'  {json.dumps(dataset_name)}: {{\n')

        for cls_idx, (class_label, videos) in enumerate(classes):
            f.write(f'    {json.dumps(class_label)}: {{\n')
            f.write(f'      "test": {{\n')

            items = list(videos.items())
            for vid_idx, (video_id, frames) in enumerate(items):
                comma = ',' if vid_idx < len(items) - 1 else ''
                f.write(
                    f'        {json.dumps(video_id)}: '
                    f'{{"label": {json.dumps(class_label)}, '
                    f'"frames": {json.dumps(frames)}}}{comma}\n'
                )
                total_videos += 1
                total_frames += len(frames)

            f.write('      }\n')
            cls_comma = ',' if cls_idx < len(classes) - 1 else ''
            f.write(f'    }}{cls_comma}\n')

        f.write('  }\n}\n')

    return total_videos, total_frames


# ============================================================
# Label dict patcher
# ============================================================
def patch_label_dict(new_labels: dict):
    """
    Add new {label_string: int} entries to label_dict in train/test configs.
    new_labels: {label_string: 0_or_1}
    """
    try:
        import yaml
    except ImportError:
        print("  [WARN] pyyaml not available — skipping label_dict patch")
        return

    cfgs = [
        os.path.join(REPO, 'training', 'config', 'train_config.yaml'),
        os.path.join(REPO, 'training', 'config', 'test_config.yaml'),
    ]
    for cfg_path in cfgs:
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        if 'label_dict' not in config:
            continue
        added = []
        for lbl, val in new_labels.items():
            if lbl not in config['label_dict']:
                config['label_dict'][lbl] = val
                added.append(lbl)
        if added:
            with open(cfg_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"  label_dict: +{len(added)} labels in {os.path.basename(cfg_path)}")


# ============================================================
# High-level generators
# ============================================================
def generate_all_ff(methods, group_name, real_label, fake_scan_fn):
    """
    Generate FSAll_ff.json / FRAll_ff.json / EFSAll_ff.json
    All methods combined into one JSON, FF domain.
    """
    print(f"\n  [{group_name}.json]")
    real_videos = real_ff()
    real_prefixed = {f'{vid}_real': frames for vid, frames in real_videos.items()}
    print(f"    real: {len(real_prefixed)} videos")

    classes = [(real_label, real_prefixed)]
    new_labels = {real_label: 0}

    for method in methods:
        vids = fake_scan_fn(method)
        n = sum(len(v) for v in vids.values())
        print(f"    {method}: {len(vids)} videos, {n} frames")
        if vids:
            classes.append((method, vids))
            new_labels[method] = 1

    out = os.path.join(JSON_DIR, f'{group_name}.json')
    tv, tf = write_test_json(out, group_name, classes)
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"    → {out}  ({tv} videos, {tf} frames, {mb:.1f} MB)")
    patch_label_dict(new_labels)


def generate_all_cdf(methods, group_name, real_label, fake_scan_fn):
    """
    Generate FSAll_cdf.json / FRAll_cdf.json / EFSAll_cdf.json
    """
    print(f"\n  [{group_name}.json]")
    real_videos = real_cdf()
    real_prefixed = {f'{vid}_real': frames for vid, frames in real_videos.items()}
    print(f"    real: {len(real_prefixed)} videos")

    classes = [(real_label, real_prefixed)]
    new_labels = {real_label: 0}

    for method in methods:
        vids = fake_scan_fn(method)
        n = sum(len(v) for v in vids.values())
        print(f"    {method}: {len(vids)} videos, {n} frames")
        if vids:
            classes.append((method, vids))
            new_labels[method] = 1

    out = os.path.join(JSON_DIR, f'{group_name}.json')
    tv, tf = write_test_json(out, group_name, classes)
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"    → {out}  ({tv} videos, {tf} frames, {mb:.1f} MB)")
    patch_label_dict(new_labels)


def generate_per_method_ff(method, json_name, fake_scan_fn, real_label_suffix='Real'):
    """Generate per-method FF test JSON (e.g. simswap_ff.json)."""
    print(f"\n  [{json_name}.json]")
    real_videos   = real_ff()
    real_prefixed = {f'{vid}_real': frames for vid, frames in real_videos.items()}
    fake_videos   = fake_scan_fn(method)
    real_label    = f'{method}_{real_label_suffix}'
    classes = [(real_label, real_prefixed), (method, fake_videos)]
    out = os.path.join(JSON_DIR, f'{json_name}.json')
    tv, tf = write_test_json(out, json_name, classes)
    mb = os.path.getsize(out) / 1024 / 1024
    nf = sum(len(v) for v in fake_videos.values())
    print(f"    fake: {len(fake_videos)} videos, {nf} frames | real: {len(real_videos)} videos")
    print(f"    → {out}  ({tv} videos, {tf} frames, {mb:.1f} MB)")
    patch_label_dict({real_label: 0, method: 1})


def generate_self_contained(json_name):
    """Generate JSON for self-contained dataset (own fake/ + real/)."""
    print(f"\n  [{json_name}.json]")
    folder = SELF_CONTAINED_FOLDER[json_name]
    fake_videos, real_videos = scan_self_contained(json_name)

    nf = sum(len(v) for v in fake_videos.values())
    nr = sum(len(v) for v in real_videos.values())
    print(f"    fake: {len(fake_videos)} videos, {nf} frames")
    print(f"    real: {len(real_videos)} videos, {nr} frames")

    if not fake_videos and not real_videos:
        print(f"    [SKIP] No data found")
        return

    fake_label = folder
    real_label = f'{folder}_Real'
    classes = [(real_label, real_videos), (fake_label, fake_videos)]

    out = os.path.join(JSON_DIR, f'{json_name}.json')
    tv, tf = write_test_json(out, json_name, classes)
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"    → {out}  ({tv} videos, {tf} frames, {mb:.1f} MB)")
    patch_label_dict({fake_label: 1, real_label: 0})


def generate_e4e_ff():
    """e4e_ff.json — e4e has ff/cdf structure like EFS (no frames/ subfolder)."""
    print(f"\n  [e4e_ff.json]")
    fake_videos   = fake_ff_efs('e4e')
    real_videos   = real_ff()
    real_prefixed = {f'{vid}_real': frames for vid, frames in real_videos.items()}
    nf = sum(len(v) for v in fake_videos.values())
    print(f"    fake: {len(fake_videos)} videos, {nf} frames | real: {len(real_prefixed)} videos")
    classes = [('e4e_Real', real_prefixed), ('e4e', fake_videos)]
    out = os.path.join(JSON_DIR, 'e4e_ff.json')
    tv, tf = write_test_json(out, 'e4e_ff', classes)
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"    → {out}  ({tv} videos, {tf} frames, {mb:.1f} MB)")
    patch_label_dict({'e4e': 1, 'e4e_Real': 0})


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("RF-MoE Test JSON Generator")
    print(f"  DF40_DIR: {DF40_DIR}")
    print(f"  FF_REAL:  {FF_REAL}")
    print(f"  CDF_REAL: {CDF_REAL}")
    print(f"  JSON_DIR: {JSON_DIR}")
    print("=" * 70)

    os.makedirs(JSON_DIR, exist_ok=True)

    # ── TABLE 3: FF domain ─────────────────────────────────────
    print("\n━━━ TABLE 3: FF domain ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    generate_all_ff(FS_METHODS,  'FSAll_ff',  'FSAll_Real',  fake_ff_fs_fr)
    generate_all_ff(FR_METHODS,  'FRAll_ff',  'FRAll_Real',  fake_ff_fs_fr)
    generate_all_ff(EFS_METHODS, 'EFSAll_ff', 'EFSAll_Real', fake_ff_efs)

    # ── TABLE 4: CDF domain ────────────────────────────────────
    print("\n━━━ TABLE 4: CDF domain ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    generate_all_cdf(FS_METHODS,  'FSAll_cdf',  'FSAll_cdf_Real',  fake_cdf_fs_fr)
    generate_all_cdf(FR_METHODS,  'FRAll_cdf',  'FRAll_cdf_Real',  fake_cdf_fs_fr)
    generate_all_cdf(EFS_METHODS, 'EFSAll_cdf', 'EFSAll_cdf_Real', fake_cdf_efs)

    # ── PER-METHOD FF (used by train.py test evaluation) ───────
    print("\n━━━ Per-method FF JSONs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    fs_fr_to_generate = [
        ('simswap',    'simswap_ff',    fake_ff_fs_fr),
        ('faceswap',   'faceswap_ff',   fake_ff_fs_fr),
        ('facedancer', 'facedancer_ff', fake_ff_fs_fr),
        ('blendface',  'blendface_ff',  fake_ff_fs_fr),
        ('inswap',     'inswap_ff',     fake_ff_fs_fr),
        ('e4s',        'e4s_ff',        fake_ff_fs_fr),
        ('mobileswap', 'mobileswap_ff', fake_ff_fs_fr),
        ('fsgan',      'fsgan_ff',      fake_ff_fs_fr),
        ('uniface',    'uniface_ff',    fake_ff_fs_fr),
        ('facevid2vid','facevid2vid_ff',fake_ff_fs_fr),
        ('fomm',       'fomm_ff',       fake_ff_fs_fr),
        ('hyperreenact','hyperreenact_ff',fake_ff_fs_fr),
        ('mcnet',      'mcnet_ff',      fake_ff_fs_fr),
        ('sadtalker',  'sadtalker_ff',  fake_ff_fs_fr),
        ('wav2lip',    'wav2lip_ff',    fake_ff_fs_fr),
        ('danet',      'danet_ff',      fake_ff_fs_fr),
        ('lia',        'lia_ff',        fake_ff_fs_fr),
        ('one_shot_free','one_shot_free_ff',fake_ff_fs_fr),
        ('pirender',   'pirender_ff',   fake_ff_fs_fr),
        ('tpsm',       'tpsm_ff',       fake_ff_fs_fr),
        ('MRAA',       'MRAA_ff',       fake_ff_fs_fr),
    ]
    efs_to_generate = [
        ('StyleGAN2',  'StyleGAN2_ff',  fake_ff_efs),
        ('StyleGAN3',  'StyleGAN3_ff',  fake_ff_efs),
        ('StyleGANXL', 'StyleGANXL_ff', fake_ff_efs),
        ('ddim',       'ddim_ff',       fake_ff_efs),
        ('DiT',        'DiT_ff',        fake_ff_efs),
        ('pixart',     'pixart_ff',     fake_ff_efs),
        ('SiT',        'SiT_ff',        fake_ff_efs),
        ('RDDM',       'RDDM_ff',       fake_ff_efs),
        ('sd2.1',      'sd2.1_ff',      fake_ff_efs),
        ('VQGAN',      'VQGAN_ff',      fake_ff_efs),
    ]
    for method, json_name, fn in fs_fr_to_generate + efs_to_generate:
        generate_per_method_ff(method, json_name, fn)

    # ── TABLES 5/6: Unknown / unseen domain ───────────────────
    print("\n━━━ Tables 5/6: Unknown domain ━━━━━━━━━━━━━━━━━━━━━━━━━")
    for json_name in SELF_CONTAINED_FOLDER:
        generate_self_contained(json_name)
    generate_e4e_ff()

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — generated JSONs:")
    all_names = (
        ['FSAll_ff', 'FRAll_ff', 'EFSAll_ff',
         'FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf']
        + [jn for _, jn, _ in fs_fr_to_generate + efs_to_generate]
        + list(SELF_CONTAINED_FOLDER.keys()) + ['e4e_ff']
    )
    for name in all_names:
        p = os.path.join(JSON_DIR, f'{name}.json')
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1024 / 1024
            print(f"  ✓ {name}.json  ({mb:.1f} MB)")
        else:
            print(f"  ✗ {name}.json  MISSING")
    print("=" * 70)


if __name__ == '__main__':
    main()
