#!/usr/bin/env python3
"""
run_evaluation.py  —  Evaluate RF-MoE checkpoints for Tables 3, 4, 5, and 6.

Protocol summary:
  Table 3  Cross-forgery, FF domain:
             FS-trained   → test on FSAll_ff, FRAll_ff, EFSAll_ff
             FR-trained   → test on FSAll_ff, FRAll_ff, EFSAll_ff
             EFS-trained  → test on FSAll_ff, FRAll_ff, EFSAll_ff

  Table 4  Cross-domain, CDF:
             (same 3 models as Table 3)
             Each → test on FSAll_cdf, FRAll_cdf, EFSAll_cdf

  Table 5  Unknown domain:
             (same 3 models)
             Each → test on deepfacelab, heygen, MidJourney,
                            styleclip, e4e_ff, CollabDiff
             (stargan / starganv2 / whichisreal skipped — RecursionError)

  Table 6  Joint model, cross-domain + unknown:
             joint-trained → all of Table 4 + Table 5 datasets

Usage:
  # Evaluate FS-trained model on all tables (3, 4, 5):
  python run_evaluation.py --trained_on fs --checkpoint PATH

  # Evaluate joint model (Table 6):
  python run_evaluation.py --trained_on joint --checkpoint PATH

  # Auto-find newest checkpoint for a trained_on mode:
  python run_evaluation.py --trained_on fr --auto_ckpt

  # Only run specific table:
  python run_evaluation.py --trained_on efs --checkpoint PATH --tables 3 4

Finding the checkpoint path:
  Checkpoints live at:
    {OUTPUTS}/rf_moe/{timestamp}/test/{first_test_dataset}/ckpt_best.pth
  For fs/fr/efs: first_test_dataset = FSAll_ff
  For joint:     first_test_dataset = simswap_ff
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
from datetime import datetime

# ============================================================
# PATHS  (must match setup.py)
# ============================================================
BASE     = '/home/ibubu/ketupati'
DATA     = f'{BASE}/data'
REPO     = f'{BASE}/DeepfakeBench_DF40'
OUTPUTS  = f'{BASE}/outputs/rf_moe7'   # default; overridden per-mode in main()
JSON_DIR = f'{DATA}/dataset_json'
DETECTOR_YAML = os.path.join(REPO, 'training', 'config', 'detector', 'rfmoe.yaml')

# ============================================================
# Evaluation dataset definitions per table
# ============================================================
TABLE3_DATASETS = ['FSAll_ff', 'FRAll_ff', 'EFSAll_ff']

TABLE4_DATASETS = ['FSAll_cdf', 'FRAll_cdf', 'EFSAll_cdf']

TABLE5_DATASETS = [
    'deepfacelab', 'heygen', 'MidJourney',
    'styleclip', 'e4e_ff', 'CollabDiff',
    # These 3 may fail with RecursionError — attempted last
    'stargan', 'starganv2', 'whichisreal',
]

# For Table 6 (joint model), test on CDF + unknown domain
TABLE6_DATASETS = TABLE4_DATASETS + TABLE5_DATASETS

# Which first test dataset to look for when auto-finding checkpoint
FIRST_TEST_DS = {
    'fs':    'FSAll_ff',
    'fr':    'FSAll_ff',
    'efs':   'FSAll_ff',
    'joint': 'simswap_ff',
}


# ============================================================
# Utilities
# ============================================================
def find_latest_checkpoint(trained_on: str) -> str:
    """
    Auto-discover the most recently modified ckpt_best.pth under OUTPUTS.
    For fs/fr/efs runs: looks in test/FSAll_ff/ckpt_best.pth
    For joint runs:     looks in test/simswap_ff/ckpt_best.pth
    """
    first_ds = FIRST_TEST_DS.get(trained_on, 'FSAll_ff')
    candidates = []

    if not os.path.exists(OUTPUTS):
        return None

    for run_dir in sorted(os.listdir(OUTPUTS)):
        run_path = os.path.join(OUTPUTS, run_dir)
        if not os.path.isdir(run_path):
            continue
        ckpt = os.path.join(run_path, 'test', first_ds, 'ckpt_best.pth')
        if os.path.exists(ckpt):
            candidates.append((os.path.getmtime(ckpt), ckpt))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def check_json_exists(ds_name: str) -> bool:
    return os.path.exists(os.path.join(JSON_DIR, f'{ds_name}.json'))


def parse_auc_from_output(output: str) -> float:
    """Parse AUC value from test.py stdout. Looks for 'auc: 0.xxx' line."""
    import re
    # Match lines like: "auc: 0.9969999048115956"
    match = re.search(r'^auc:\s*([0-9.]+)', output, re.MULTILINE)
    if match:
        return float(match.group(1))
    return None


def run_single_eval(ds_name: str, checkpoint: str, results_dir: str, dry_run: bool = False):
    """
    Run test.py for one dataset.  Returns AUC (float) or None on failure.
    Parses AUC directly from stdout since test.py does not save pickle files.
    """
    if not check_json_exists(ds_name):
        print(f"    [SKIP] JSON missing: {ds_name}.json")
        return None

    cmd = [
        sys.executable, 'training/test.py',
        '--detector_path', DETECTOR_YAML,
        '--test_dataset',  ds_name,
        '--weights_path',  checkpoint,
    ]

    print(f"    CMD: {' '.join(cmd[-6:])}", flush=True)

    if dry_run:
        print("    [DRY RUN]")
        return -1.0

    # Stream output in real-time AND capture for AUC parsing
    import io
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    captured = io.StringIO()
    for line in proc.stdout:
        print(line, end='', flush=True)
        captured.write(line)
    proc.wait()
    output = captured.getvalue()

    if proc.returncode != 0:
        print(f"    [FAILED] returncode={proc.returncode}")
        return None

    # Parse AUC directly from stdout output
    auc = parse_auc_from_output(output)
    if auc is None:
        print(f"    [WARN] Could not parse AUC from output for {ds_name}")
    else:
        print(f"    [AUC] {ds_name}: {auc:.4f}")
    return auc


def save_results(results: dict, trained_on: str, table_num: int, checkpoint: str):
    """Persist evaluation results as JSON and trigger visual table save."""
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_path  = os.path.join(OUTPUTS, f'table{table_num}_{trained_on}_results.json')
    payload   = {
        'trained_on':  trained_on,
        'table':       table_num,
        'checkpoint':  checkpoint,
        'timestamp':   timestamp,
        'results':     results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved JSON: {out_path}")

    # Save visual table immediately after this evaluation
    print(f"\n  Saving visual table {table_num}...")
    compile_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compile_results.py')
    if not os.path.exists(compile_script):
        # fallback: look in cwd or script dir
        compile_script = 'compile_results.py'
    cmd = [
        sys.executable, compile_script,
        '--results_dir', OUTPUTS,
        '--show', str(table_num),
    ]
    subprocess.run(cmd, check=False)


# ============================================================
# Per-table evaluation runners
# ============================================================
def eval_table3(checkpoint, trained_on, dry_run=False):
    print(f"\n{'=' * 60}")
    print(f"TABLE 3 — Cross-forgery FF domain  (trained_on={trained_on})")
    print('=' * 60)
    os.chdir(REPO)

    results = {}
    for ds in TABLE3_DATASETS:
        print(f"\n  [{ds}]")
        auc = run_single_eval(ds, checkpoint, OUTPUTS, dry_run)
        results[ds] = auc
        if auc is not None:
            print(f"  → AUC: {auc:.4f}")

    save_results(results, trained_on, 3, checkpoint)

    print(f"\n  Summary:")
    valid = [v for v in results.values() if v and v > 0]
    for ds, auc in results.items():
        tag = f'{auc:.4f}' if (auc and auc > 0) else 'N/A'
        print(f"    {ds}: {tag}")
    if valid:
        print(f"    Avg: {sum(valid)/len(valid):.4f}")
    return results


def eval_table4(checkpoint, trained_on, dry_run=False):
    print(f"\n{'=' * 60}")
    print(f"TABLE 4 — Cross-domain CDF  (trained_on={trained_on})")
    print('=' * 60)
    os.chdir(REPO)

    results = {}
    for ds in TABLE4_DATASETS:
        print(f"\n  [{ds}]")
        auc = run_single_eval(ds, checkpoint, OUTPUTS, dry_run)
        results[ds] = auc
        if auc is not None:
            print(f"  → AUC: {auc:.4f}")

    save_results(results, trained_on, 4, checkpoint)

    print(f"\n  Summary:")
    valid = [v for v in results.values() if v and v > 0]
    for ds, auc in results.items():
        tag = f'{auc:.4f}' if (auc and auc > 0) else 'N/A'
        print(f"    {ds}: {tag}")
    if valid:
        print(f"    Avg: {sum(valid)/len(valid):.4f}")
    return results


def eval_table5(checkpoint, trained_on, dry_run=False):
    print(f"\n{'=' * 60}")
    print(f"TABLE 5 — Unknown domain  (trained_on={trained_on})")
    print('=' * 60)
    os.chdir(REPO)

    # Known-problematic datasets (RecursionError)
    recursion_risk = {'stargan', 'starganv2', 'whichisreal'}

    results = {}
    for ds in TABLE5_DATASETS:
        print(f"\n  [{ds}]{'  [RecursionError risk]' if ds in recursion_risk else ''}")
        auc = run_single_eval(ds, checkpoint, OUTPUTS, dry_run)
        results[ds] = auc
        if auc is not None and auc > 0:
            print(f"  → AUC: {auc:.4f}")

    save_results(results, trained_on, 5, checkpoint)

    print(f"\n  Summary:")
    valid = [v for v in results.values() if v and v > 0]
    for ds, auc in results.items():
        tag = f'{auc:.4f}' if (auc and auc > 0) else 'N/A (skipped or failed)'
        print(f"    {ds}: {tag}")
    if valid:
        print(f"    Avg (available): {sum(valid)/len(valid):.4f}")
    return results


def eval_table6(checkpoint, dry_run=False):
    """Table 6 uses the joint-trained model on CDF + unknown domain."""
    print(f"\n{'=' * 60}")
    print(f"TABLE 6 — Joint model: CDF + unknown domain")
    print('=' * 60)
    os.chdir(REPO)

    recursion_risk = {'stargan', 'starganv2', 'whichisreal'}
    results = {}

    for ds in TABLE6_DATASETS:
        print(f"\n  [{ds}]{'  [RecursionError risk]' if ds in recursion_risk else ''}")
        auc = run_single_eval(ds, checkpoint, OUTPUTS, dry_run)
        results[ds] = auc
        if auc is not None and auc > 0:
            print(f"  → AUC: {auc:.4f}")

    save_results(results, 'joint', 6, checkpoint)

    print(f"\n  Summary:")
    valid = [v for v in results.values() if v and v > 0]
    for ds, auc in results.items():
        tag = f'{auc:.4f}' if (auc and auc > 0) else 'N/A'
        print(f"    {ds}: {tag}")
    if valid:
        print(f"    Avg ({len(valid)}/{len(TABLE6_DATASETS)}): {sum(valid)/len(valid):.4f}")
    return results


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='RF-MoE evaluation launcher')
    p.add_argument('--trained_on', required=True, choices=['fs', 'fr', 'efs', 'joint'],
                   help='Which model to evaluate')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Explicit path to ckpt_best.pth')
    p.add_argument('--auto_ckpt', action='store_true',
                   help='Auto-discover newest checkpoint for --trained_on mode')
    p.add_argument('--tables', nargs='+', type=int, default=None,
                   choices=[3, 4, 5, 6],
                   help='Which tables to evaluate (default: 3 4 5 for single-type, 6 for joint)')
    p.add_argument('--dry_run', action='store_true',
                   help='Print commands without running them')
    return p.parse_args()


def main():
    global OUTPUTS
    args = parse_args()

    # Use mode-specific outputs dir (e.g. outputs/rf_moe_fr).
    # Falls back to outputs/rf_moe for FS backward-compatibility
    # (FS training ran before mode-specific dirs were added).
    mode_dir = f'{BASE}/outputs/rf_moe7_{args.trained_on}'
    if os.path.isdir(mode_dir):
        OUTPUTS = mode_dir

    # Resolve checkpoint
    if args.auto_ckpt:
        ckpt = find_latest_checkpoint(args.trained_on)
        if ckpt is None:
            print(f"ERROR: No checkpoint found for trained_on={args.trained_on}")
            print(f"  Searched under: {OUTPUTS}")
            sys.exit(1)
        print(f"Auto-found checkpoint: {ckpt}")
    elif args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
    else:
        print("ERROR: Provide --checkpoint PATH or --auto_ckpt")
        sys.exit(1)

    if not os.path.exists(ckpt) and not args.dry_run:
        print(f"ERROR: Checkpoint not found: {ckpt}")
        sys.exit(1)

    # Default table selection
    if args.tables:
        tables = args.tables
    elif args.trained_on == 'joint':
        tables = [6]
    else:
        tables = [3, 4, 5]

    print("=" * 70)
    print("RF-MoE Evaluation")
    print(f"  Trained on:  {args.trained_on}")
    print(f"  Checkpoint:  {ckpt}")
    print(f"  Tables:      {tables}")
    print("=" * 70)

    all_results = {}

    if 3 in tables and args.trained_on != 'joint':
        all_results['table3'] = eval_table3(ckpt, args.trained_on, args.dry_run)

    if 4 in tables and args.trained_on != 'joint':
        all_results['table4'] = eval_table4(ckpt, args.trained_on, args.dry_run)

    if 5 in tables and args.trained_on != 'joint':
        all_results['table5'] = eval_table5(ckpt, args.trained_on, args.dry_run)

    if 6 in tables:
        if args.trained_on != 'joint':
            print("WARNING: Table 6 should use the joint-trained model.")
        all_results['table6'] = eval_table6(ckpt, args.dry_run)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("Run compile_results.py to print formatted comparison tables.")
    print("=" * 70)


if __name__ == '__main__':
    main()
