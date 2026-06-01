#!/usr/bin/env python3
"""
check_dataset_stats.py — Count videos and frames per forgery type.

Shows:
  - Train set  : DF40_train/ fake frames + FF++ real frames
  - Test set   : DF40/ ff domain + cdf domain
  - Totals per type (FS / FR / EFS) and grand total

NOTE: There is no separate val set in DF40 — test set is used for
      epoch-end checkpoint selection during training.

Usage:
  python check_dataset_stats.py
"""

import os
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
BASE       = '/home/ibubu/ketupati'
DATA       = f'{BASE}/data'
TRAIN_DATA = f'{DATA}/DF40_train'
TEST_DATA  = f'{DATA}/DF40'
FF_REAL    = f'{DATA}/ff_real/FaceForensics++'
CDF_REAL   = f'{DATA}/cdf_real/Celeb-DF-v2'

IMG_EXTS = {'.png', '.jpg', '.jpeg'}

# ============================================================
# Method groups  (exact folder names on disk)
# ============================================================
FS_METHODS  = ['simswap', 'faceswap', 'facedancer', 'blendface', 'inswap',
               'e4s', 'mobileswap', 'fsgan', 'uniface']
FR_METHODS  = ['facevid2vid', 'fomm', 'hyperreenact', 'mcnet', 'sadtalker',
               'wav2lip', 'danet', 'lia', 'one_shot_free', 'pirender', 'tpsm', 'MRAA']
EFS_METHODS = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'ddim', 'DiT',
               'pixart', 'SiT', 'RDDM', 'sd2.1', 'VQGAN']
FE_METHODS  = ['stargan', 'StarGAN2', 'styleclip', 'e4e', 'CollabDiff']
UNK_METHODS = ['deepfacelab', 'heygen', 'MidJourney', 'Whichisreal']


# ============================================================
# Helpers
# ============================================================
def count_images(directory):
    """Return (num_video_folders, num_image_files) by walking directory."""
    if not os.path.exists(directory):
        return 0, 0
    videos, frames = 0, 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if imgs:
            frames += len(imgs)
            videos += 1
    return videos, frames


def case_lookup(base, name):
    """Case-insensitive folder lookup."""
    if not os.path.exists(base):
        return None
    for f in os.listdir(base):
        if f.lower() == name.lower():
            return os.path.join(base, f)
    return None


def count_train_fake(method):
    d = case_lookup(TRAIN_DATA, method)
    return count_images(d) if d else (0, 0)


def count_test_ff(method):
    d = case_lookup(TEST_DATA, method)
    return count_images(os.path.join(d, 'ff')) if d else (0, 0)


def count_test_cdf(method):
    d = case_lookup(TEST_DATA, method)
    return count_images(os.path.join(d, 'cdf')) if d else (0, 0)


def count_test_self(method):
    """For FE/unknown domain: fake/ + real/ inside DF40/method/"""
    d = case_lookup(TEST_DATA, method)
    if not d:
        return 0, 0, 0, 0
    fv, ff = count_images(os.path.join(d, 'fake'))
    rv, rf = count_images(os.path.join(d, 'real'))
    return fv, ff, rv, rf


def count_real_ff():
    d = os.path.join(FF_REAL, 'original_sequences', 'youtube', 'c23', 'frames')
    if not os.path.exists(d):
        d = FF_REAL
    return count_images(d)


def count_real_cdf():
    return count_images(CDF_REAL)


# ============================================================
# Printers
# ============================================================
HDR  = f"  {'Method':<16} {'Train':>7} {'Train':>10} {'Test FF':>8} {'Test FF':>10} {'Test CDF':>9} {'Test CDF':>10}"
HDR2 = f"  {'':16} {'Vids':>7} {'Frames':>10} {'Vids':>8} {'Frames':>10} {'Vids':>9} {'Frames':>10}"
SEP  = '  ' + '-'*16 + ' ' + '-'*7 + ' ' + '-'*10 + ' ' + '-'*8 + ' ' + '-'*10 + ' ' + '-'*9 + ' ' + '-'*10


def print_type_header(title):
    print(f"\n{'━'*72}")
    print(f"  {title}")
    print('━'*72)
    print(HDR)
    print(HDR2)
    print(SEP)


def print_method_row(method, tv, tf, ffv, fff, cdfv, cdff):
    flags = []
    if tv == 0:   flags.append('no_train')
    if ffv == 0:  flags.append('no_ff')
    if cdfv == 0: flags.append('no_cdf')
    flag_str = f"  ← {', '.join(flags)}" if flags else ''
    print(f"  {method:<16} {tv:>7,} {tf:>10,} {ffv:>8,} {fff:>10,} {cdfv:>9,} {cdff:>10,}{flag_str}")


def print_type_total(label, t):
    print(SEP)
    print(f"  {label:<16} {t['tv']:>7,} {t['tf']:>10,} {t['ffv']:>8,} {t['fff']:>10,} {t['cdfv']:>9,} {t['cdff']:>10,}")


def accumulate(methods, scan_fn_ff=count_test_ff, scan_fn_cdf=count_test_cdf):
    totals = defaultdict(int)
    rows = []
    for m in methods:
        tv,  tf   = count_train_fake(m)
        ffv, fff  = scan_fn_ff(m)
        cdfv,cdff = scan_fn_cdf(m)
        rows.append((m, tv, tf, ffv, fff, cdfv, cdff))
        totals['tv']+=tv;  totals['tf']+=tf
        totals['ffv']+=ffv; totals['fff']+=fff
        totals['cdfv']+=cdfv; totals['cdff']+=cdff
    return rows, totals


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("  DF40 Dataset Statistics — Videos & Frames per Method")
    print(f"  TRAIN: {TRAIN_DATA}")
    print(f"  TEST:  {TEST_DATA}")
    print("=" * 72)
    print("\n  NOTE: DF40 has NO separate val set.")
    print("        Test set is used for epoch-end AUC monitoring during training.")

    grand = defaultdict(int)

    # ── Real data ──────────────────────────────────────────────
    print(f"\n{'━'*72}")
    print("  REAL DATA  (shared across all forgery types)")
    print('━'*72)
    ff_rv,  ff_rf  = count_real_ff()
    cdf_rv, cdf_rf = count_real_cdf()
    print(f"  {'FF++ real (train):':<28} {ff_rv:>7,} videos  {ff_rf:>10,} frames  → label=0 for training")
    print(f"  {'Celeb-DF-v2 real (test CDF):':<28} {cdf_rv:>7,} videos  {cdf_rf:>10,} frames  → label=0 for CDF eval")

    # ── FS ─────────────────────────────────────────────────────
    print_type_header("FACE SWAPPING (FS) — 9 methods")
    rows, t = accumulate(FS_METHODS)
    for row in rows:
        print_method_row(*row)
    print_type_total("TOTAL FS", t)
    for k in t: grand[k] += t[k]

    # ── FR ─────────────────────────────────────────────────────
    print_type_header("FACE REENACTMENT (FR) — 12 methods")
    rows, t = accumulate(FR_METHODS)
    for row in rows:
        print_method_row(*row)
    print_type_total("TOTAL FR", t)
    for k in t: grand[k] += t[k]

    # ── EFS ────────────────────────────────────────────────────
    print_type_header("ENTIRE FACE SYNTHESIS (EFS) — 10 methods")
    rows, t = accumulate(EFS_METHODS)
    for row in rows:
        print_method_row(*row)
    print_type_total("TOTAL EFS", t)
    for k in t: grand[k] += t[k]

    # ── FE (test only, self-contained) ─────────────────────────
    print(f"\n{'━'*72}")
    print("  FACE EDITING (FE) — 5 methods  [TEST ONLY — own fake/ + real/]")
    print('━'*72)
    print(f"  {'Method':<16} {'Fake Vids':>10} {'Fake Frms':>11} {'Real Vids':>10} {'Real Frms':>11}")
    print('  ' + '-'*16 + ' ' + '-'*10 + ' ' + '-'*11 + ' ' + '-'*10 + ' ' + '-'*11)
    fe_tot = defaultdict(int)
    for m in FE_METHODS:
        fv, ff, rv, rf = count_test_self(m)
        flags = []
        if fv == 0: flags.append('no_fake')
        if rv == 0: flags.append('no_real')
        flag_str = f"  ← {', '.join(flags)}" if flags else ''
        print(f"  {m:<16} {fv:>10,} {ff:>11,} {rv:>10,} {rf:>11,}{flag_str}")
        fe_tot['fv']+=fv; fe_tot['ff']+=ff; fe_tot['rv']+=rv; fe_tot['rf']+=rf
    print('  ' + '-'*16 + ' ' + '-'*10 + ' ' + '-'*11 + ' ' + '-'*10 + ' ' + '-'*11)
    print(f"  {'TOTAL FE':<16} {fe_tot['fv']:>10,} {fe_tot['ff']:>11,} {fe_tot['rv']:>10,} {fe_tot['rf']:>11,}")

    # ── Unknown domain (test only) ─────────────────────────────
    print(f"\n{'━'*72}")
    print("  UNKNOWN DOMAIN — [TEST ONLY — own fake/ + real/]")
    print('━'*72)
    print(f"  {'Method':<16} {'Fake Vids':>10} {'Fake Frms':>11} {'Real Vids':>10} {'Real Frms':>11}")
    print('  ' + '-'*16 + ' ' + '-'*10 + ' ' + '-'*11 + ' ' + '-'*10 + ' ' + '-'*11)
    unk_tot = defaultdict(int)
    for m in UNK_METHODS:
        fv, ff, rv, rf = count_test_self(m)
        flags = []
        if fv == 0: flags.append('no_fake')
        if rv == 0: flags.append('no_real')
        flag_str = f"  ← {', '.join(flags)}" if flags else ''
        print(f"  {m:<16} {fv:>10,} {ff:>11,} {rv:>10,} {rf:>11,}{flag_str}")
        unk_tot['fv']+=fv; unk_tot['ff']+=ff; unk_tot['rv']+=rv; unk_tot['rf']+=rf
    print('  ' + '-'*16 + ' ' + '-'*10 + ' ' + '-'*11 + ' ' + '-'*10 + ' ' + '-'*11)
    print(f"  {'TOTAL UNK':<16} {unk_tot['fv']:>10,} {unk_tot['ff']:>11,} {unk_tot['rv']:>10,} {unk_tot['rf']:>11,}")

    # ── Grand totals ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  GRAND TOTALS")
    print(f"  {'='*70}")
    print(f"  {'TRAIN  fake (FS+FR+EFS):':<35} {grand['tv']:>7,} videos  {grand['tf']:>12,} frames")
    print(f"  {'TRAIN  real (FF++):':<35} {ff_rv:>7,} videos  {ff_rf:>12,} frames")
    print(f"  {'TRAIN  total:':<35} {grand['tv']+ff_rv:>7,} videos  {grand['tf']+ff_rf:>12,} frames")
    print()
    print(f"  {'TEST   FF domain fake (FS+FR+EFS):':<35} {grand['ffv']:>7,} videos  {grand['fff']:>12,} frames")
    print(f"  {'TEST   CDF domain fake (FS+FR+EFS):':<35} {grand['cdfv']:>7,} videos  {grand['cdff']:>12,} frames")
    print(f"  {'TEST   FE (fake+real):':<35} {fe_tot['fv']+fe_tot['rv']:>7,} videos  {fe_tot['ff']+fe_tot['rf']:>12,} frames")
    print(f"  {'TEST   Unknown domain (fake+real):':<35} {unk_tot['fv']+unk_tot['rv']:>7,} videos  {unk_tot['ff']+unk_tot['rf']:>12,} frames")
    print(f"  {'TEST   real FF++ (for FF eval):':<35} {ff_rv:>7,} videos  {ff_rf:>12,} frames")
    print(f"  {'TEST   real CDF (for CDF eval):':<35} {cdf_rv:>7,} videos  {cdf_rf:>12,} frames")
    print('='*72)


if __name__ == '__main__':
    main()
