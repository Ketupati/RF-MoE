#!/usr/bin/env python3
"""
make_final_tables.py — Generate final result tables matching DF40 paper format.

Pulls RF-MoE results from JSON files, combines with hardcoded paper baselines,
marks values that beat ALL baselines with **, truncates to 3 decimal places.

Output: final_results_tables.txt
"""

import os
import sys
import json
import math
import argparse

BASE    = '/home/ibubu/ketupati'
OUTPUTS = f'{BASE}/outputs'

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_results_tables.txt')

COL_W = 13   # fixed column width — wide enough for **[0.999]** (11 chars) + padding


# ─── Hardcoded paper baselines ────────────────────────────────────────────────

T3_BASELINES = {
    'fs': {
        'Xception':   [0.991, 0.892, 0.810],
        'CLIP-large': [0.996, 0.908, 0.837],
        'RECCE':      [0.991, 0.855, 0.758],
        'RFM':        [0.992, 0.884, 0.821],
    },
    'fr': {
        'Xception':   [0.838, 0.996, 0.670],
        'CLIP-large': [0.932, 0.999, 0.798],
        'RECCE':      [0.865, 0.997, 0.716],
        'RFM':        [0.892, 0.999, 0.776],
    },
    'efs': {
        'Xception':   [0.665, 0.807, 0.999],
        'CLIP-large': [0.688, 0.889, 0.999],
        'RECCE':      [0.691, 0.801, 0.999],
        'RFM':        [0.653, 0.795, 0.999],
    },
}
T3_SBI = [0.810, 0.714, 0.678]

T4_BASELINES = {
    'fs': {
        'Xception':   [0.922, 0.657, 0.642],
        'CLIP-large': [0.967, 0.744, 0.730],
        'RECCE':      [0.926, 0.632, 0.610],
        'RFM':        [0.939, 0.637, 0.628],
    },
    'fr': {
        'Xception':   [0.481, 0.857, 0.369],
        'CLIP-large': [0.638, 0.933, 0.209],
        'RECCE':      [0.452, 0.881, 0.332],
        'RFM':        [0.492, 0.882, 0.359],
    },
    'efs': {
        'Xception':   [0.586, 0.594, 0.983],
        'CLIP-large': [0.617, 0.735, 0.988],
        'RECCE':      [0.623, 0.603, 0.984],
        'RFM':        [0.644, 0.666, 0.981],
    },
}
T4_SBI = [0.679, 0.609, 0.723]

T5_BASELINES = {
    'fs': {
        'Xception':   [0.882, 0.394, 0.384, 0.535, 0.577, 0.616, 0.426, 0.553, 0.546],
        'CLIP-large': [0.930, 0.539, 0.540, 0.439, 0.896, 0.746, 0.730, 0.738, 0.674],
        'RECCE':      [0.899, 0.537, 0.293, 0.509, 0.580, 0.599, 0.399, 0.520, 0.492],
        'RFM':        [0.918, 0.719, 0.286, 0.496, 0.652, 0.570, 0.705, 0.689, 0.798],
    },
    'fr': {
        'Xception':   [0.705, 0.473, 0.459, 0.323, 0.492, 0.456, 0.006, 0.175, 0.050],
        'CLIP-large': [0.845, 0.614, 0.632, 0.466, 0.762, 0.436, 0.298, 0.631, 0.611],
        'RECCE':      [0.724, 0.576, 0.314, 0.278, 0.529, 0.374, 0.005, 0.177, 0.060],
        'RFM':        [0.739, 0.588, 0.511, 0.325, 0.407, 0.423, 0.009, 0.201, 0.030],
    },
    'efs': {
        'Xception':   [0.497, 0.325, 0.472, 0.772, 0.777, 0.677, 0.984, 0.611, 0.997],
        'CLIP-large': [0.745, 0.506, 0.534, 0.828, 0.946, 0.823, 0.929, 0.923, 0.983],
        'RECCE':      [0.583, 0.505, 0.442, 0.753, 0.769, 0.724, 0.964, 0.643, 0.979],
        'RFM':        [0.619, 0.349, 0.551, 0.623, 0.730, 0.636, 0.966, 0.665, 0.979],
    },
}
T5_SBI = [0.764, 0.402, 0.342, 0.426, 0.591, 0.586, 0.564, 0.379, 0.570]

T6_BASELINES = {
    'Xception':   [0.752, 0.831, 0.681, 0.851, 0.704, 0.269, 0.632, 0.721, 0.569, 0.495, 0.675, 0.542],
    'CLIP-base':  [0.915, 0.926, 0.843, 0.907, 0.671, 0.548, 0.684, 0.913, 0.782, 0.813, 0.948, 0.823],
    'CLIP-large': [0.942, 0.896, 0.858, 0.948, 0.784, 0.746, 0.849, 0.974, 0.909, 0.929, 0.977, 0.967],
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def trunc3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return math.floor(float(x) * 1000) / 1000


def cell(x, bold=False, diag=False):
    """Return a string of exactly COL_W characters."""
    if x is None:
        return 'N/A'.center(COL_W)
    t = trunc3(x)
    s = f'{t:.3f}'
    if diag:
        s = f'[{s}]'
    if bold:
        s = f'**{s}**'
    return s.center(COL_W)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_ours(mode, table_num):
    path = os.path.join(OUTPUTS, f'rf_moe_{mode}', f'table{table_num}_{mode}_results.json')
    data = load_json(path)
    if data is None:
        return None
    return data.get('results', {})


def avg_valid(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def beats_all(our_val, baseline_dict, col_idx):
    if our_val is None:
        return False
    for vals in baseline_dict.values():
        if our_val <= vals[col_idx]:
            return False
    return True


def beats_all_avg(our_avg, baseline_dict):
    if our_avg is None:
        return False
    for vals in baseline_dict.values():
        if our_avg <= sum(vals) / len(vals):
            return False
    return True


# ─── Table builders ───────────────────────────────────────────────────────────

def build_table3(lines):
    ds_keys  = ['FSAll_ff', 'FRAll_ff', 'EFSAll_ff']
    diag_idx = {'fs': 0, 'fr': 1, 'efs': 2}
    col_hdrs = ['FS (FF)', 'FR (FF)', 'EFS (FF)', 'Avg']

    W = COL_W
    lw = 14   # label col width
    mw = 14   # model col width
    sep = '=' * (lw + mw + W * 4 + 2)

    lines.append(sep)
    lines.append('TABLE 3  Cross-Forgery Evaluation (Protocol-1)')
    lines.append('Train: FF domain   Test: FF domain   Metric: AUC')
    lines.append('[x.xxx] = within-forgery diagonal (gray cells in paper)')
    lines.append('**x.xxx** = beats ALL baselines in that column')
    lines.append(sep)
    hdr = f"{'Train':<{lw}} {'Model':<{mw}}"
    for h in col_hdrs:
        hdr += h.center(W)
    lines.append(hdr)
    lines.append('-' * (lw + mw + W * 4 + 2))

    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T3_BASELINES[mode]
        ours_raw  = get_ours(mode, 3)
        di        = diag_idx[mode]

        first = True
        for model, vals in baselines.items():
            avg    = sum(vals) / 3
            prefix = f'{label:<{lw}}' if first else f'{"":>{lw}}'
            first  = False
            row = prefix + f' {model:<{mw-1}}'
            for i, v in enumerate(vals):
                row += cell(v, diag=(i == di))
            row += cell(avg)
            lines.append(row)

        # SBI row (no diagonal, separate BI section)
        row = f'{"":>{lw}} {"SBI":<{mw-1}}'
        for v in T3_SBI:
            row += cell(v)
        row += cell(sum(T3_SBI) / 3)
        lines.append(row)

        # Ours
        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid(our_vals)
            row = f'{"":>{lw}} {"Ours":<{mw-1}}'
            for i, v in enumerate(our_vals):
                bold = beats_all(v, baselines, i)
                row += cell(v, bold=bold, diag=(i == di))
            avg_bold = beats_all_avg(our_avg, baselines)
            row += cell(our_avg, bold=avg_bold)
            lines.append(row)
        else:
            lines.append(f'{"":>{lw}} {"Ours":<{mw-1}} {"N/A — not evaluated yet":<{W*4}}')

        lines.append('-' * (lw + mw + W * 4 + 2))

    lines.append('')


def build_table4(lines):
    ds_keys  = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf']
    col_hdrs = ['FS (CDF)', 'FR (CDF)', 'EFS (CDF)', 'Avg']

    W  = COL_W
    lw = 14
    mw = 14
    sep = '=' * (lw + mw + W * 4 + 2)

    lines.append(sep)
    lines.append('TABLE 4  Cross-Domain Evaluation (Protocol-2)')
    lines.append('Train: FF domain   Test: CDF domain   Metric: AUC')
    lines.append('**x.xxx** = beats ALL baselines in that column')
    lines.append(sep)
    hdr = f"{'Train':<{lw}} {'Model':<{mw}}"
    for h in col_hdrs:
        hdr += h.center(W)
    lines.append(hdr)
    lines.append('-' * (lw + mw + W * 4 + 2))

    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T4_BASELINES[mode]
        ours_raw  = get_ours(mode, 4)

        first = True
        for model, vals in baselines.items():
            avg    = sum(vals) / 3
            prefix = f'{label:<{lw}}' if first else f'{"":>{lw}}'
            first  = False
            row = prefix + f' {model:<{mw-1}}'
            for v in vals:
                row += cell(v)
            row += cell(avg)
            lines.append(row)

        row = f'{"":>{lw}} {"SBI":<{mw-1}}'
        for v in T4_SBI:
            row += cell(v)
        row += cell(sum(T4_SBI) / 3)
        lines.append(row)

        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid(our_vals)
            row = f'{"":>{lw}} {"Ours":<{mw-1}}'
            for i, v in enumerate(our_vals):
                bold = beats_all(v, baselines, i)
                row += cell(v, bold=bold)
            avg_bold = beats_all_avg(our_avg, baselines)
            row += cell(our_avg, bold=avg_bold)
            lines.append(row)
        else:
            lines.append(f'{"":>{lw}} {"Ours":<{mw-1}} {"N/A — not evaluated yet":<{W*4}}')

        lines.append('-' * (lw + mw + W * 4 + 2))

    lines.append('')


def build_table5(lines):
    ds_keys  = ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'CollabDiff']
    col_hdrs = ['DFL', 'HeyGen', 'MidJ', 'WiR', 'SG', 'SG2', 'SCLIP', 'e4e', 'CDiff', 'Avg']

    W  = COL_W
    lw = 14
    mw = 14
    sep = '=' * (lw + mw + W * 10 + 2)

    lines.append(sep)
    lines.append('TABLE 5  Unknown Domain Evaluation (Protocol-3)')
    lines.append('Train: FF domain   Test: Unseen forgeries + domains   Metric: AUC')
    lines.append('DFL=DeepFaceLab  WiR=Whichisreal  SG=StarGAN  SG2=StarGAN2  SCLIP=StyleCLIP  CDiff=CollabDiff')
    lines.append('**x.xxx** = beats ALL baselines in that column')
    lines.append(sep)
    hdr = f"{'Train':<{lw}} {'Model':<{mw}}"
    for h in col_hdrs:
        hdr += h.center(W)
    lines.append(hdr)
    lines.append('-' * (lw + mw + W * 10 + 2))

    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T5_BASELINES[mode]
        ours_raw  = get_ours(mode, 5)

        first = True
        for model, vals in baselines.items():
            avg    = sum(vals) / len(vals)
            prefix = f'{label:<{lw}}' if first else f'{"":>{lw}}'
            first  = False
            row = prefix + f' {model:<{mw-1}}'
            for v in vals:
                row += cell(v)
            row += cell(avg)
            lines.append(row)

        row = f'{"":>{lw}} {"SBI":<{mw-1}}'
        for v in T5_SBI:
            row += cell(v)
        row += cell(sum(T5_SBI) / len(T5_SBI))
        lines.append(row)

        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid([v for v in our_vals if v is not None])
            row = f'{"":>{lw}} {"Ours":<{mw-1}}'
            for i, v in enumerate(our_vals):
                bold = beats_all(v, baselines, i)
                row += cell(v, bold=bold)
            avg_bold = beats_all_avg(our_avg, baselines)
            row += cell(our_avg, bold=avg_bold)
            lines.append(row)
        else:
            lines.append(f'{"":>{lw}} {"Ours":<{mw-1}} {"N/A — not evaluated yet":<{W*10}}')

        lines.append('-' * (lw + mw + W * 10 + 2))

    lines.append('')


def build_table6(lines):
    ds_keys  = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf',
                'deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                'stargan', 'starganv2', 'styleclip', 'CollabDiff', 'e4e_ff']
    col_hdrs = ['FS(CDF)', 'FR(CDF)', 'EFS(CDF)',
                'DFL', 'HeyGen', 'MidJ', 'WiR', 'SG', 'SG2', 'SCLIP', 'CDiff', 'e4e', 'Avg']

    W  = COL_W
    mw = 14
    sep = '=' * (mw + W * 13 + 1)

    lines.append(sep)
    lines.append('TABLE 6  Joint Training Evaluation (Protocol-4)')
    lines.append('Train: DF40(FF) = FS+FR+EFS   Test: CDF domain + Unknown domain   Metric: AUC')
    lines.append('**x.xxx** = beats ALL baselines in that column')
    lines.append(sep)
    hdr = f"{'Model':<{mw}}"
    for h in col_hdrs:
        hdr += h.center(W)
    lines.append(hdr)
    lines.append('-' * (mw + W * 13 + 1))

    for model, vals in T6_BASELINES.items():
        avg = sum(vals) / len(vals)
        row = f'{model:<{mw}}'
        for v in vals:
            row += cell(v)
        row += cell(avg)
        lines.append(row)

    ours_raw = get_ours('joint', 6)
    if ours_raw:
        our_vals = [ours_raw.get(k) for k in ds_keys]
        our_avg  = avg_valid([v for v in our_vals if v is not None])
        row = f'{"Ours":<{mw}}'
        for i, v in enumerate(our_vals):
            bold = beats_all(v, T6_BASELINES, i)
            row += cell(v, bold=bold)
        avg_bold = beats_all_avg(our_avg, T6_BASELINES)
        row += cell(our_avg, bold=avg_bold)
        lines.append(row)
    else:
        lines.append(f'{"Ours":<{mw}} {"N/A — not evaluated yet"}')

    lines.append('-' * (mw + W * 13 + 1))
    lines.append('')


def parse_args():
    p = argparse.ArgumentParser(description='Generate final RF-MoE result tables')
    p.add_argument('--out_dir', default=None,
                   help='Directory to save final_results_tables.txt '
                        '(default: same directory as this script)')
    p.add_argument('--out_file', default=None,
                   help='Full output file path (overrides --out_dir)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.out_file:
        out_file = args.out_file
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_file = os.path.join(args.out_dir, 'final_results_tables.txt')
    else:
        out_file = DEFAULT_OUT

    lines = []
    lines.append('RF-MoE — Final Evaluation Results')
    lines.append('Architecture: Model1 SL-CLIP-2bl-val-auc  |  CLIP-Large, blocks 22-23 unfrozen, Sobel+Laplacian spectral branch')
    lines.append('Checkpoint selection: Best Val AUC on held-out 10% split')
    lines.append('')

    build_table3(lines)
    build_table4(lines)
    build_table5(lines)
    build_table6(lines)

    out = '\n'.join(lines)
    with open(out_file, 'w') as f:
        f.write(out)
    print(out)
    print(f'\nSaved to: {out_file}')


if __name__ == '__main__':
    main()
