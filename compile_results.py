#!/usr/bin/env python3
"""
compile_results.py — Load all RF-MoE evaluation results and print/save
formatted comparison tables matching the DF40 paper (NeurIPS 2024).

Saves tables as .txt and .csv to:
  /home/ibubu/ketupati/outputs/results_tables/

Usage:
  python compile_results.py                      # print + save all tables
  python compile_results.py --show 3 4           # print + save tables 3 & 4
  python compile_results.py --results_dir PATH   # scan specific directory
"""

import os
import csv
import json
import pickle
import argparse
import glob
from datetime import datetime
from typing import Dict, Optional, List

# ============================================================
# PATHS
# ============================================================
BASE       = '/home/ibubu/ketupati'
OUTPUTS    = f'{BASE}/outputs/rf_moe'
TABLES_DIR = f'{BASE}/outputs/results_tables'

# ============================================================
# Complete paper baselines — DF40 NeurIPS 2024
# ============================================================

# ── Table 3: Cross-forgery, FF domain ───────────────────────
# Cols: FSAll_ff, FRAll_ff, EFSAll_ff
PAPER_T3 = {
    'FS': {
        'Xception':   {'FSAll_ff': 0.991, 'FRAll_ff': 0.892, 'EFSAll_ff': 0.810},
        'CLIP-large': {'FSAll_ff': 0.996, 'FRAll_ff': 0.908, 'EFSAll_ff': 0.837},
        'RECCE':      {'FSAll_ff': 0.991, 'FRAll_ff': 0.855, 'EFSAll_ff': 0.758},
        'RFM':        {'FSAll_ff': 0.992, 'FRAll_ff': 0.884, 'EFSAll_ff': 0.821},
    },
    'FR': {
        'Xception':   {'FSAll_ff': 0.838, 'FRAll_ff': 0.996, 'EFSAll_ff': 0.670},
        'CLIP-large': {'FSAll_ff': 0.932, 'FRAll_ff': 0.999, 'EFSAll_ff': 0.798},
        'RECCE':      {'FSAll_ff': 0.865, 'FRAll_ff': 0.997, 'EFSAll_ff': 0.716},
        'RFM':        {'FSAll_ff': 0.892, 'FRAll_ff': 0.999, 'EFSAll_ff': 0.776},
    },
    'EFS': {
        'Xception':   {'FSAll_ff': 0.665, 'FRAll_ff': 0.807, 'EFSAll_ff': 0.999},
        'CLIP-large': {'FSAll_ff': 0.688, 'FRAll_ff': 0.889, 'EFSAll_ff': 0.999},
        'RECCE':      {'FSAll_ff': 0.691, 'FRAll_ff': 0.801, 'EFSAll_ff': 0.999},
        'RFM':        {'FSAll_ff': 0.653, 'FRAll_ff': 0.795, 'EFSAll_ff': 0.999},
    },
    'BI': {
        'SBI':        {'FSAll_ff': 0.810, 'FRAll_ff': 0.714, 'EFSAll_ff': 0.678},
    },
}

# ── Table 4: Cross-domain, CDF ───────────────────────────────
# Cols: FSAll_cdf, FRAll_cdf, EFSAll_cdf
PAPER_T4 = {
    'FS': {
        'Xception':   {'FSAll_cdf': 0.922, 'FRAll_cdf': 0.657, 'EFSAll_cdf': 0.642},
        'CLIP-large': {'FSAll_cdf': 0.967, 'FRAll_cdf': 0.744, 'EFSAll_cdf': 0.730},
        'RECCE':      {'FSAll_cdf': 0.926, 'FRAll_cdf': 0.632, 'EFSAll_cdf': 0.610},
        'RFM':        {'FSAll_cdf': 0.939, 'FRAll_cdf': 0.637, 'EFSAll_cdf': 0.628},
    },
    'FR': {
        'Xception':   {'FSAll_cdf': 0.481, 'FRAll_cdf': 0.857, 'EFSAll_cdf': 0.369},
        'CLIP-large': {'FSAll_cdf': 0.638, 'FRAll_cdf': 0.933, 'EFSAll_cdf': 0.209},
        'RECCE':      {'FSAll_cdf': 0.452, 'FRAll_cdf': 0.881, 'EFSAll_cdf': 0.332},
        'RFM':        {'FSAll_cdf': 0.492, 'FRAll_cdf': 0.882, 'EFSAll_cdf': 0.359},
    },
    'EFS': {
        'Xception':   {'FSAll_cdf': 0.586, 'FRAll_cdf': 0.594, 'EFSAll_cdf': 0.983},
        'CLIP-large': {'FSAll_cdf': 0.617, 'FRAll_cdf': 0.735, 'EFSAll_cdf': 0.988},
        'RECCE':      {'FSAll_cdf': 0.623, 'FRAll_cdf': 0.603, 'EFSAll_cdf': 0.984},
        'RFM':        {'FSAll_cdf': 0.644, 'FRAll_cdf': 0.666, 'EFSAll_cdf': 0.981},
    },
    'BI': {
        'SBI':        {'FSAll_cdf': 0.679, 'FRAll_cdf': 0.609, 'EFSAll_cdf': 0.723},
    },
}

# ── Table 5: Unknown domain ──────────────────────────────────
# Cols: deepfacelab, heygen, MidJourney, whichisreal,
#       stargan, starganv2, styleclip, e4e_ff, CollabDiff
T5_COLS = ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
           'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'CollabDiff']
T5_LABELS = ['DeepFaceLab❶', 'HeyGen❷', 'MidJourney❸', 'Whichisreal❸',
             'StarGAN❹', 'StarGAN2❹', 'StyleCLIP❹', 'e4e❹', 'CollabDiff❹']

PAPER_T5 = {
    'FS': {
        'Xception':   [0.882, 0.394, 0.384, 0.535, 0.577, 0.616, 0.426, 0.553, 0.546],
        'CLIP-large': [0.930, 0.539, 0.540, 0.439, 0.896, 0.746, 0.730, 0.738, 0.674],
        'RECCE':      [0.899, 0.537, 0.293, 0.509, 0.580, 0.599, 0.399, 0.520, 0.492],
        'RFM':        [0.918, 0.719, 0.286, 0.496, 0.652, 0.570, 0.705, 0.689, 0.798],
    },
    'FR': {
        'Xception':   [0.705, 0.473, 0.459, 0.323, 0.492, 0.456, 0.006, 0.175, 0.050],
        'CLIP-large': [0.845, 0.614, 0.632, 0.466, 0.762, 0.436, 0.298, 0.631, 0.611],
        'RECCE':      [0.724, 0.576, 0.314, 0.278, 0.529, 0.374, 0.005, 0.177, 0.060],
        'RFM':        [0.739, 0.588, 0.511, 0.325, 0.407, 0.423, 0.009, 0.201, 0.030],
    },
    'EFS': {
        'Xception':   [0.497, 0.325, 0.472, 0.772, 0.777, 0.677, 0.984, 0.611, 0.997],
        'CLIP-large': [0.745, 0.506, 0.534, 0.828, 0.946, 0.823, 0.929, 0.923, 0.983],
        'RECCE':      [0.583, 0.505, 0.442, 0.753, 0.769, 0.724, 0.964, 0.643, 0.979],
        'RFM':        [0.619, 0.349, 0.551, 0.623, 0.730, 0.636, 0.966, 0.665, 0.979],
    },
    'BI': {
        'SBI':        [0.764, 0.402, 0.342, 0.426, 0.591, 0.586, 0.564, 0.379, 0.570],
    },
}

# ── Table 6: Joint model ─────────────────────────────────────
# Cols: FSAll_cdf, FRAll_cdf, EFSAll_cdf,
#       deepfacelab, heygen, MidJourney, whichisreal,
#       stargan, starganv2, styleclip, CollabDiff, e4e_ff
T6_COLS = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf',
           'deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
           'stargan', 'starganv2', 'styleclip', 'CollabDiff', 'e4e_ff']
T6_LABELS = ['FS(CDF)', 'FR(CDF)', 'EFS(CDF)',
             'DeepFaceLab❶', 'HeyGen❷', 'MidJourney❸', 'Whichisreal❸',
             'StarGAN❹', 'StarGAN2❹', 'StyleCLIP❹', 'CollabDiff❹', 'e4e❹']

PAPER_T6 = {
    'Xception':   [0.752, 0.831, 0.681, 0.851, 0.704, 0.269, 0.632, 0.721, 0.569, 0.495, 0.675, 0.542],
    'CLIP-base':  [0.915, 0.926, 0.843, 0.907, 0.671, 0.548, 0.684, 0.913, 0.782, 0.813, 0.948, 0.823],
    'CLIP-large': [0.942, 0.896, 0.858, 0.948, 0.784, 0.746, 0.849, 0.974, 0.909, 0.929, 0.977, 0.967],
}


# ============================================================
# Helpers
# ============================================================
def avg(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None and x >= 0]
    return sum(v) / len(v) if v else None


def fv(v: Optional[float], highlight: bool = False) -> str:
    """Format a value for display — truncates to 3 decimal places (no rounding)."""
    if v is None or v < 0:
        return '  N/A '
    s = f'{int(v * 1000) / 1000:.3f}'
    return f'[{s}]' if highlight else f' {s} '


def dict_avg(d: dict, keys: list) -> Optional[float]:
    return avg([d.get(k) for k in keys])


def list_to_dict(lst: list, keys: list) -> dict:
    return {k: v for k, v in zip(keys, lst)}


# ============================================================
# Result file scanning
# ============================================================
def scan_result_files(results_dir: str) -> Dict:
    found = {}
    pattern = os.path.join(results_dir, '**', 'table*_*_results.json')
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(fpath) as f:
                data = json.load(f)
            table_num  = data.get('table')
            trained_on = data.get('trained_on')
            results    = data.get('results', {})
            if table_num and trained_on:
                key = (table_num, trained_on)
                if key not in found:
                    found[key] = results
                    print(f"  Loaded: table{table_num} trained_on={trained_on}")
        except Exception as e:
            print(f"  [WARN] {fpath}: {e}")
    return found


# ============================================================
# Table builders — return (lines_txt, rows_csv)
# ============================================================
def build_table3(rfmoe_results: Dict) -> tuple:
    """Table 3: Cross-forgery, FF domain."""
    T3_COLS   = ['FSAll_ff', 'FRAll_ff', 'EFSAll_ff']
    T3_LABELS = ['FS (FF)', 'FR (FF)', 'EFS (FF)', 'Avg (FF)']

    # Within-forgery diagonal: trained on X → test on X(ff)
    DIAG = {'FS': 'FSAll_ff', 'FR': 'FRAll_ff', 'EFS': 'EFSAll_ff'}

    W = 80
    SEP = '-' * W

    lines = []
    lines.append('=' * W)
    lines.append('TABLE 3 — Cross-Forgery Evaluation (Protocol-1)')
    lines.append('Same Data Domain (FF), Different Forgery Types')
    lines.append('Metric: AUC  |  [x.xxx] = within-forgery (same as paper gray cells)')
    lines.append('Paper baselines: Xception, CLIP-large, RECCE, RFM  |  Ours: RF-MoE')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * W)

    hdr = f"{'Training Set':<13}  {'Model':<16}  " + \
          ''.join(f'{l:>9}' for l in T3_LABELS)
    lines.append(hdr)

    csv_rows = [['Training Set', 'Model'] + T3_LABELS]

    for train_type in ['FS', 'FR', 'EFS']:
        lines.append(SEP)
        diag_col = DIAG[train_type]
        paper_models = PAPER_T3.get(train_type, {})

        for i, (model, vals) in enumerate(paper_models.items()):
            ts = f'{train_type} (FF)' if i == 0 else ''
            row_vals = [vals.get(c) for c in T3_COLS]
            row_avg  = avg(row_vals)
            cells = [fv(v, highlight=(c == diag_col)) for v, c in zip(row_vals, T3_COLS)]
            cells.append(fv(row_avg))
            lines.append(f'{ts:<13}  {model:<16}  {"".join(cells)}')
            csv_rows.append([f'{train_type} (FF)', model] +
                            [f'{v:.3f}' if v else 'N/A' for v in row_vals + [row_avg]])

        # RF-MoE row
        t_lower = train_type.lower()
        res = rfmoe_results.get((3, t_lower), {})
        if res:
            row_vals = [res.get(c) for c in T3_COLS]
            row_avg  = avg([v for v in row_vals if v])
            cells    = [fv(v, highlight=(c == diag_col)) for v, c in zip(row_vals, T3_COLS)]
            cells.append(fv(row_avg))
            marker = ' ** OURS **'
            lines.append(f'{"" :<13}  {"RF-MoE (ours)":<16}  {"".join(cells)}{marker}')
            csv_rows.append([f'{train_type} (FF)', 'RF-MoE (ours)'] +
                            [f'{v:.3f}' if v else 'N/A' for v in row_vals + [row_avg]])
        else:
            lines.append(f'{"" :<13}  {"RF-MoE (ours)":<16}  {"N/A — not evaluated yet":}')

    lines.append(SEP)
    lines.append('BI (FF)       SBI             ' +
                 ''.join(fv(PAPER_T3['BI']['SBI'].get(c)) for c in T3_COLS) +
                 fv(dict_avg(PAPER_T3['BI']['SBI'], T3_COLS)))
    lines.append('=' * W)
    lines.append('[x.xxx] = within-forgery (diagonal) — same as paper gray cells')

    return lines, csv_rows


def build_table4(rfmoe_results: Dict) -> tuple:
    """Table 4: Cross-domain, CDF."""
    T4_COLS   = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf']
    T4_LABELS = ['FS (CDF)', 'FR (CDF)', 'EFS (CDF)', 'Avg (CDF)']

    W = 80
    SEP = '-' * W

    lines = []
    lines.append('=' * W)
    lines.append('TABLE 4 — Cross-Domain Evaluation (Protocol-2)')
    lines.append('Same Forgery Types, Different Data Domain (CDF)')
    lines.append('Metric: AUC  |  Training: FF domain  |  Testing: CDF domain')
    lines.append('Paper baselines: Xception, CLIP-large, RECCE, RFM  |  Ours: RF-MoE')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * W)

    hdr = f"{'Training Set':<13}  {'Model':<16}  " + \
          ''.join(f'{l:>10}' for l in T4_LABELS)
    lines.append(hdr)

    csv_rows = [['Training Set', 'Model'] + T4_LABELS]

    for train_type in ['FS', 'FR', 'EFS']:
        lines.append(SEP)
        paper_models = PAPER_T4.get(train_type, {})

        for i, (model, vals) in enumerate(paper_models.items()):
            ts = f'{train_type} (FF)' if i == 0 else ''
            row_vals = [vals.get(c) for c in T4_COLS]
            row_avg  = avg(row_vals)
            cells    = [fv(v) for v in row_vals] + [fv(row_avg)]
            lines.append(f'{ts:<13}  {model:<16}  {"".join(cells)}')
            csv_rows.append([f'{train_type} (FF)', model] +
                            [f'{v:.3f}' if v else 'N/A' for v in row_vals + [row_avg]])

        t_lower = train_type.lower()
        res = rfmoe_results.get((4, t_lower), {})
        if res:
            row_vals = [res.get(c) for c in T4_COLS]
            row_avg  = avg([v for v in row_vals if v])
            cells    = [fv(v) for v in row_vals] + [fv(row_avg)]
            lines.append(f'{"" :<13}  {"RF-MoE (ours)":<16}  {"".join(cells)} ** OURS **')
            csv_rows.append([f'{train_type} (FF)', 'RF-MoE (ours)'] +
                            [f'{v:.3f}' if v else 'N/A' for v in row_vals + [row_avg]])
        else:
            lines.append(f'{"" :<13}  {"RF-MoE (ours)":<16}  N/A — not evaluated yet')

    lines.append(SEP)
    sbi = PAPER_T4['BI']['SBI']
    lines.append('BI (FF)       SBI             ' +
                 ''.join(fv(sbi.get(c)) for c in T4_COLS) +
                 fv(dict_avg(sbi, T4_COLS)))
    lines.append('=' * W)

    return lines, csv_rows


def build_table5(rfmoe_results: Dict) -> tuple:
    """Table 5: Unknown domain (Protocol-3)."""
    W   = 120
    SEP = '-' * W

    col_w = 13
    hdr_labels = ['DFL❶', 'HeyGen❷', 'MidJ❸', 'WiR❸', 'SG❹', 'SG2❹', 'SCLIP❹', 'e4e❹', 'CDiff❹', 'Avg']

    lines = []
    lines.append('=' * W)
    lines.append('TABLE 5 — Toward Real-World Open-Set Evaluation (Protocol-3)')
    lines.append('Different Forgery Types, Different Data Domains (Unknown Domain)')
    lines.append('Metric: AUC  |  Training: FF domain  |  Testing: unseen forgeries + domains')
    lines.append('  ❶=FS method  ❷=FR method  ❸=EFS method  ❹=FE method')
    lines.append('  DFL=DeepFaceLab, WiR=Whichisreal, SG=StarGAN, SG2=StarGAN2,')
    lines.append('  SCLIP=StyleCLIP, CDiff=CollabDiff')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * W)

    hdr = f"{'Training Set':<13}  {'Model':<16}  " + \
          ''.join(f'{l:>{col_w}}' for l in hdr_labels)
    lines.append(hdr)

    csv_rows = [['Training Set', 'Model'] + T5_LABELS + ['Avg']]

    for train_type in ['FS', 'FR', 'EFS']:
        lines.append(SEP)
        paper_models = PAPER_T5.get(train_type, {})

        for i, (model, vals_list) in enumerate(paper_models.items()):
            ts = f'{train_type} (FF)' if i == 0 else ''
            row_avg = avg(vals_list)
            cells   = [fv(v) for v in vals_list] + [fv(row_avg)]
            line    = f'{ts:<13}  {model:<16}  ' + \
                      ''.join(f'{c:>{col_w}}' for c in cells)
            lines.append(line)
            csv_rows.append([f'{train_type} (FF)', model] +
                            [f'{v:.3f}' for v in vals_list] +
                            [f'{row_avg:.3f}' if row_avg else 'N/A'])

        t_lower = train_type.lower()
        res = rfmoe_results.get((5, t_lower), {})
        if res:
            row_vals = [res.get(c) for c in T5_COLS]
            row_avg  = avg([v for v in row_vals if v])
            cells    = [fv(v) for v in row_vals] + [fv(row_avg)]
            line     = f'{"" :<13}  {"RF-MoE (ours)":<16}  ' + \
                       ''.join(f'{c:>{col_w}}' for c in cells) + ' ** OURS **'
            lines.append(line)
            csv_rows.append([f'{train_type} (FF)', 'RF-MoE (ours)'] +
                            [f'{v:.3f}' if v else 'N/A' for v in row_vals] +
                            [f'{row_avg:.3f}' if row_avg else 'N/A'])
        else:
            lines.append(f'{"" :<13}  {"RF-MoE (ours)":<16}  N/A — not evaluated yet')

    lines.append(SEP)
    sbi = PAPER_T5['BI']['SBI']
    sbi_avg = avg(sbi)
    cells = [fv(v) for v in sbi] + [fv(sbi_avg)]
    lines.append('BI (FF)       SBI             ' +
                 ''.join(f'{c:>{col_w}}' for c in cells))
    lines.append('=' * W)
    lines.append('N/A entries may be due to RecursionError in dataloader for stargan/WiR.')

    return lines, csv_rows


def build_table6(rfmoe_results: Dict) -> tuple:
    """Table 6: Joint training — cross-domain + unknown."""
    W   = 130
    SEP = '-' * W

    col_w  = 11
    hdr_labels = ['FS(CDF)', 'FR(CDF)', 'EFS(CDF)',
                  'DFL❶', 'HeyGen❷', 'MidJ❸', 'WiR❸',
                  'SG❹', 'SG2❹', 'SCLIP❹', 'CDiff❹', 'e4e❹', 'Avg']

    lines = []
    lines.append('=' * W)
    lines.append('TABLE 6 — Comparison of Models Trained on DF40 (FF) Joint')
    lines.append('Training Set: DF40(FF) = FS(FF) + FR(FF) + EFS(FF)')
    lines.append('Testing: CDF domain + Unknown domain (open-set evaluation)')
    lines.append('Metric: AUC  |  ❶=FS  ❷=FR  ❸=EFS  ❹=FE')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * W)

    hdr = f"{'Model':<16}  " + ''.join(f'{l:>{col_w}}' for l in hdr_labels)
    lines.append(hdr)
    lines.append(SEP)

    csv_rows = [['Model'] + hdr_labels]

    for model, vals_list in PAPER_T6.items():
        row_avg = avg(vals_list)
        cells   = [fv(v) for v in vals_list] + [fv(row_avg)]
        line    = f'{model:<16}  ' + ''.join(f'{c:>{col_w}}' for c in cells)
        lines.append(line)
        csv_rows.append([model] + [f'{v:.3f}' for v in vals_list] +
                        [f'{row_avg:.3f}' if row_avg else 'N/A'])

    lines.append(SEP)

    res = rfmoe_results.get((6, 'joint'), {})
    if res:
        row_vals = [res.get(c) for c in T6_COLS]
        row_avg  = avg([v for v in row_vals if v])
        cells    = [fv(v) for v in row_vals] + [fv(row_avg)]
        line     = f'{"RF-MoE (ours)":<16}  ' + \
                   ''.join(f'{c:>{col_w}}' for c in cells) + ' ** OURS **'
        lines.append(line)
        csv_rows.append(['RF-MoE (ours)'] +
                        [f'{v:.3f}' if v else 'N/A' for v in row_vals] +
                        [f'{row_avg:.3f}' if row_avg else 'N/A'])
    else:
        lines.append(f'{"RF-MoE (ours)":<16}  N/A — not evaluated yet')

    lines.append('=' * W)

    return lines, csv_rows


# ============================================================
# Save to disk
# ============================================================
def save_tables_to_disk(rfmoe_results: Dict, tables_to_show: List[int], save_dir: str):
    """Save formatted tables as .txt and .csv files."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    builders = {
        3: ('table3_cross_forgery',     build_table3),
        4: ('table4_cross_domain',      build_table4),
        5: ('table5_unknown_domain',    build_table5),
        6: ('table6_joint_model',       build_table6),
    }

    saved = []
    all_lines = []

    for tnum in tables_to_show:
        if tnum not in builders:
            continue
        fname_base, builder = builders[tnum]
        lines, csv_rows = builder(rfmoe_results)

        # .txt
        txt_path = os.path.join(save_dir, f'{fname_base}_{timestamp}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        saved.append(txt_path)

        # .csv
        csv_path = os.path.join(save_dir, f'{fname_base}_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        saved.append(csv_path)

        all_lines.extend(lines)
        all_lines.append('')

    # Combined file
    if len(tables_to_show) > 1:
        combined_path = os.path.join(save_dir, f'all_tables_{timestamp}.txt')
        with open(combined_path, 'w') as f:
            f.write('\n'.join(all_lines) + '\n')
        saved.append(combined_path)

    print(f"\n  Saved {len(saved)} table files to: {save_dir}")
    for p in saved:
        print(f"    {os.path.basename(p)}")

    return saved


# ============================================================
# Print to terminal
# ============================================================
def print_table(lines: list):
    print('\n' + '\n'.join(lines))


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='Compile and save RF-MoE result tables')
    p.add_argument('--results_dir', default=OUTPUTS,
                   help=f'Scan directory (default: {OUTPUTS})')
    p.add_argument('--save_dir', default=TABLES_DIR,
                   help=f'Where to save table files (default: {TABLES_DIR})')
    p.add_argument('--show', nargs='+', default=['all'],
                   help='Tables to process: 3 4 5 6 all (default: all)')
    return p.parse_args()


def main():
    args = parse_args()

    if 'all' in args.show:
        tables = [3, 4, 5, 6]
    else:
        tables = [int(x) for x in args.show]

    print('=' * 70)
    print('RF-MoE Results Compiler')
    print(f'  Scanning:  {args.results_dir}')
    print(f'  Saving to: {args.save_dir}')
    print('=' * 70)

    print('\nLoading result files:')
    rfmoe_results = scan_result_files(args.results_dir)

    builders = {3: build_table3, 4: build_table4,
                5: build_table5, 6: build_table6}

    for tnum in tables:
        if tnum in builders:
            lines, _ = builders[tnum](rfmoe_results)
            print_table(lines)

    # Always save
    save_tables_to_disk(rfmoe_results, tables, args.save_dir)


if __name__ == '__main__':
    main()
