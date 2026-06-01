#!/usr/bin/env python3
"""
generate_html_tables.py — Generate HTML result tables for Model1 (SL-CLIP-2bl-val-auc).
Reads from the same result JSONs as make_final_tables.py.
Output: final_results_tables.html
"""

import os
import json
import math

BASE    = '/home/ibubu/ketupati'
OUTPUTS = f'{BASE}/outputs'
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_results_tables.html')

# ─── Same baselines as make_final_tables.py ───────────────────────────────────

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

def fmt(x, bold=False, diag=False):
    if x is None:
        return '<td>N/A</td>'
    t = trunc3(x)
    s = f'{t:.3f}'
    classes = []
    if diag:
        classes.append('diag')
    if bold:
        classes.append('bold')
    cls = f' class="{" ".join(classes)}"' if classes else ''
    return f'<td{cls}>{s}</td>'

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


# ─── HTML builders ────────────────────────────────────────────────────────────

def table3_html():
    ds_keys  = ['FSAll_ff', 'FRAll_ff', 'EFSAll_ff']
    diag_idx = {'fs': 0, 'fr': 1, 'efs': 2}
    col_hdrs = ['FS (FF)', 'FR (FF)', 'EFS (FF)', 'Avg']

    rows = ''
    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T3_BASELINES[mode]
        ours_raw  = get_ours(mode, 3)
        di        = diag_idx[mode]
        first     = True

        for model, vals in baselines.items():
            avg    = sum(vals) / 3
            train  = f'<td rowspan="{len(baselines)+2}" class="train-label">{label}</td>' if first else ''
            first  = False
            rows  += f'<tr>{train}<td>{model}</td>'
            for i, v in enumerate(vals):
                rows += fmt(v, diag=(i == di))
            rows += fmt(avg) + '</tr>\n'

        # SBI row
        rows += f'<tr><td class="sbi">SBI</td>'
        for v in T3_SBI:
            rows += fmt(v)
        rows += fmt(sum(T3_SBI)/3) + '</tr>\n'

        # Ours row
        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid(our_vals)
            rows += f'<tr><td class="ours">Ours</td>'
            for i, v in enumerate(our_vals):
                rows += fmt(v, bold=beats_all(v, baselines, i), diag=(i == di))
            rows += fmt(our_avg, bold=beats_all_avg(our_avg, baselines)) + '</tr>\n'
        else:
            rows += f'<tr><td class="ours">Ours</td><td colspan="4">N/A</td></tr>\n'

    headers = ''.join(f'<th>{h}</th>' for h in col_hdrs)
    return f'''
    <h2>Table 3 — Cross-Forgery Evaluation (Protocol-1)</h2>
    <p>Train: FF domain &nbsp;|&nbsp; Test: FF domain &nbsp;|&nbsp; Metric: AUC<br>
    <span class="diag-note">Gray = within-forgery diagonal</span> &nbsp;|&nbsp;
    <span class="bold-note"><b>Bold</b> = beats ALL 4 baselines</span></p>
    <table>
      <thead><tr><th>Train</th><th>Model</th>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def table4_html():
    ds_keys  = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf']
    col_hdrs = ['FS (CDF)', 'FR (CDF)', 'EFS (CDF)', 'Avg']

    rows = ''
    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T4_BASELINES[mode]
        ours_raw  = get_ours(mode, 4)
        first     = True

        for model, vals in baselines.items():
            avg   = sum(vals) / 3
            train = f'<td rowspan="{len(baselines)+2}" class="train-label">{label}</td>' if first else ''
            first = False
            rows += f'<tr>{train}<td>{model}</td>'
            for v in vals:
                rows += fmt(v)
            rows += fmt(avg) + '</tr>\n'

        rows += f'<tr><td class="sbi">SBI</td>'
        for v in T4_SBI:
            rows += fmt(v)
        rows += fmt(sum(T4_SBI)/3) + '</tr>\n'

        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid(our_vals)
            rows += f'<tr><td class="ours">Ours</td>'
            for i, v in enumerate(our_vals):
                rows += fmt(v, bold=beats_all(v, baselines, i))
            rows += fmt(our_avg, bold=beats_all_avg(our_avg, baselines)) + '</tr>\n'
        else:
            rows += f'<tr><td class="ours">Ours</td><td colspan="4">N/A</td></tr>\n'

    headers = ''.join(f'<th>{h}</th>' for h in col_hdrs)
    return f'''
    <h2>Table 4 — Cross-Domain Evaluation (Protocol-2)</h2>
    <p>Train: FF domain &nbsp;|&nbsp; Test: CDF domain &nbsp;|&nbsp; Metric: AUC<br>
    <span class="bold-note"><b>Bold</b> = beats ALL 4 baselines</span></p>
    <table>
      <thead><tr><th>Train</th><th>Model</th>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def table5_html():
    ds_keys  = ['deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                'stargan', 'starganv2', 'styleclip', 'e4e_ff', 'CollabDiff']
    col_hdrs = ['DFL', 'HeyGen', 'MidJ', 'WiR', 'SG', 'SG2', 'SCLIP', 'e4e', 'CDiff', 'Avg']

    rows = ''
    for mode in ['fs', 'fr', 'efs']:
        label     = mode.upper() + ' (FF)'
        baselines = T5_BASELINES[mode]
        ours_raw  = get_ours(mode, 5)
        first     = True

        for model, vals in baselines.items():
            avg   = sum(vals) / len(vals)
            train = f'<td rowspan="{len(baselines)+2}" class="train-label">{label}</td>' if first else ''
            first = False
            rows += f'<tr>{train}<td>{model}</td>'
            for v in vals:
                rows += fmt(v)
            rows += fmt(avg) + '</tr>\n'

        rows += f'<tr><td class="sbi">SBI</td>'
        for v in T5_SBI:
            rows += fmt(v)
        rows += fmt(sum(T5_SBI)/len(T5_SBI)) + '</tr>\n'

        if ours_raw:
            our_vals = [ours_raw.get(k) for k in ds_keys]
            our_avg  = avg_valid([v for v in our_vals if v is not None])
            rows += f'<tr><td class="ours">Ours</td>'
            for i, v in enumerate(our_vals):
                rows += fmt(v, bold=beats_all(v, baselines, i))
            rows += fmt(our_avg, bold=beats_all_avg(our_avg, baselines)) + '</tr>\n'
        else:
            rows += f'<tr><td class="ours">Ours</td><td colspan="10">N/A</td></tr>\n'

    headers = ''.join(f'<th>{h}</th>' for h in col_hdrs)
    return f'''
    <h2>Table 5 — Unknown Domain Evaluation (Protocol-3)</h2>
    <p>Train: FF domain &nbsp;|&nbsp; Test: Unseen forgeries + domains &nbsp;|&nbsp; Metric: AUC<br>
    DFL=DeepFaceLab &nbsp; WiR=Whichisreal &nbsp; SG=StarGAN &nbsp; SG2=StarGAN2 &nbsp; SCLIP=StyleCLIP &nbsp; CDiff=CollabDiff<br>
    <span class="bold-note"><b>Bold</b> = beats ALL 4 baselines</span></p>
    <table>
      <thead><tr><th>Train</th><th>Model</th>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def table6_html():
    ds_keys  = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf',
                'deepfacelab', 'heygen', 'MidJourney', 'whichisreal',
                'stargan', 'starganv2', 'styleclip', 'CollabDiff', 'e4e_ff']
    col_hdrs = ['FS(CDF)', 'FR(CDF)', 'EFS(CDF)',
                'DFL', 'HeyGen', 'MidJ', 'WiR', 'SG', 'SG2', 'SCLIP', 'CDiff', 'e4e', 'Avg']

    rows = ''
    for model, vals in T6_BASELINES.items():
        avg   = sum(vals) / len(vals)
        rows += f'<tr><td>{model}</td>'
        for v in vals:
            rows += fmt(v)
        rows += fmt(avg) + '</tr>\n'

    ours_raw = get_ours('joint', 6)
    if ours_raw:
        our_vals = [ours_raw.get(k) for k in ds_keys]
        our_avg  = avg_valid([v for v in our_vals if v is not None])
        rows += f'<tr><td class="ours">Ours</td>'
        for i, v in enumerate(our_vals):
            rows += fmt(v, bold=beats_all(v, T6_BASELINES, i))
        rows += fmt(our_avg, bold=beats_all_avg(our_avg, T6_BASELINES)) + '</tr>\n'
    else:
        rows += f'<tr><td class="ours">Ours</td><td colspan="13">N/A</td></tr>\n'

    headers = ''.join(f'<th>{h}</th>' for h in col_hdrs)
    return f'''
    <h2>Table 6 — Joint Training Evaluation (Protocol-4)</h2>
    <p>Train: DF40(FF) = FS+FR+EFS &nbsp;|&nbsp; Test: CDF domain + Unknown domain &nbsp;|&nbsp; Metric: AUC<br>
    <span class="bold-note"><b>Bold</b> = beats ALL baselines</span></p>
    <table>
      <thead><tr><th>Model</th>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>'''


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model1 SL-CLIP-2bl-val-auc — Results</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; color: #222; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; }}
  h2   {{ color: #16213e; margin-top: 40px; }}
  p    {{ color: #555; font-size: 0.9em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 0.88em; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  th   {{ background: #1a1a2e; color: white; padding: 8px 10px; text-align: center; white-space: nowrap; }}
  td   {{ padding: 6px 10px; text-align: center; border: 1px solid #ddd; white-space: nowrap; }}
  tr:hover td {{ background: #f0f4ff; }}
  td.train-label {{ background: #e8eaf6; font-weight: bold; vertical-align: middle; }}
  td.diag  {{ background: #e0e0e0; color: #444; font-style: italic; }}
  td.bold  {{ font-weight: bold; color: #c0392b; }}
  td.bold.diag {{ font-weight: bold; color: #c0392b; background: #e0e0e0; }}
  td.ours  {{ font-weight: bold; background: #fff9e6; }}
  td.sbi   {{ color: #888; }}
  .diag-note {{ color: #888; }}
  .bold-note {{ color: #c0392b; }}
  .meta {{ background: #e8f4fd; border-left: 4px solid #2196f3; padding: 12px 16px; margin: 20px 0; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Model 1: SL-CLIP-2bl-val-auc</h1>
<div class="meta">
  <b>Architecture:</b> CLIP-Large (blocks 22–23 unfrozen) + Sobel/Laplacian spectral branch + Region-Aware MoE<br>
  <b>Checkpoint selection:</b> Best Val AUC on held-out 10% split<br>
  <b>Best epochs:</b> FS=2 &nbsp; FR=6 &nbsp; EFS=3 &nbsp; Joint=3<br>
  <b>Generated:</b> {__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>

{table3_html()}
{table4_html()}
{table5_html()}
{table6_html()}

</body>
</html>'''

    with open(OUT_FILE, 'w') as f:
        f.write(html)
    print(f'Saved to: {OUT_FILE}')

if __name__ == '__main__':
    main()
