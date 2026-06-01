#!/usr/bin/env python3
"""
find_best_epoch.py — Find the best epoch checkpoint using minimum val or train loss.

Reads TensorBoard event files written by the trainer. When a validation split
was used (create_val_split.py + patched train.py), reads val_loss by default.
Falls back to train loss if val_loss not found.

Usage:
  python find_best_epoch.py --mode fs
  python find_best_epoch.py --mode fr
  python find_best_epoch.py --mode efs
  python find_best_epoch.py --mode joint
  python find_best_epoch.py --mode fs --no_val_loss   # force train loss
  python find_best_epoch.py --mode fs --log_dir /path/to/custom/logdir

After running, pass the printed checkpoint path to run_evaluation.py:
  python run_evaluation.py --trained_on fs --checkpoint <path>
"""

import os
import argparse
import glob

BASE    = '/home/ibubu/ketupati'
OUTPUTS = f'{BASE}/outputs/rf_moe'

TRAIN_DATASET_MAP = {
    'fs':    'FSAll_ff_train',
    'fr':    'FRAll_ff_train',
    'efs':   'EFSAll_ff_train',
    'joint': 'FSAll_ff_train,FRAll_ff_train,EFSAll_ff_train',
}


def find_log_dir(mode):
    if os.path.isdir(os.path.join(OUTPUTS, 'epoch_ckpts')):
        return OUTPUTS
    pattern = os.path.join(OUTPUTS, '*rfmoe*')
    candidates = sorted(glob.glob(pattern), reverse=True)
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'epoch_ckpts')):
            return c
    return OUTPUTS


def read_tb_scalars(log_dir, tag_substring='train_loss'):
    """
    Read TensorBoard event files and return {step: value} for a given tag.
    Returns scalars for the first matching tag containing tag_substring.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard")
        return {}

    event_dirs = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_dirs.append(root)
                break

    all_scalars = {}
    for ed in event_dirs:
        ea = EventAccumulator(ed)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        for tag in tags:
            if tag_substring in tag:
                for event in ea.Scalars(tag):
                    step = event.step
                    val  = event.value
                    if step not in all_scalars:
                        all_scalars[step] = []
                    all_scalars[step].append(val)

    return {step: sum(vals)/len(vals) for step, vals in all_scalars.items()}


def read_val_loss_by_epoch(log_dir):
    """
    Read val_loss from TensorBoard or from training log files.
    Returns {epoch: val_loss} dict.
    """
    import math, glob

    # Try TensorBoard first
    raw = read_tb_scalars(log_dir, tag_substring='val_loss')
    if raw:
        clean = {int(step): v for step, v in raw.items() if not math.isnan(v)}
        if clean:
            return clean

    # Fall back: parse [VAL] lines from log files
    # Prefer mode-specific log file (e.g. efs_p2v2_train.log) over other modes
    val_losses = {}
    cwd = os.getcwd()
    all_logs = glob.glob(os.path.join(os.path.dirname(log_dir), '*.log'))
    all_logs += glob.glob(os.path.join(cwd, '*.log'))

    # Detect mode from log_dir path (e.g. rf_moe_efs → mode hint = 'efs')
    mode_hint = ''
    for part in log_dir.replace('\\', '/').split('/'):
        if part.startswith('rf_moe_'):
            mode_hint = part[len('rf_moe_'):]
            break
    if mode_hint:
        priority = sorted([l for l in all_logs if mode_hint in os.path.basename(l).lower()], key=os.path.getmtime, reverse=True)
        rest     = sorted([l for l in all_logs if l not in priority], key=os.path.getmtime, reverse=True)
        log_candidates = priority + rest
    else:
        log_candidates = sorted(all_logs, key=os.path.getmtime, reverse=True)
    for log_file in log_candidates:
        try:
            with open(log_file) as f:
                for line in f:
                    if '[VAL] Epoch' in line and 'val_loss:' in line:
                        parts = line.strip().split()
                        ep  = int(parts[parts.index('Epoch') + 1])
                        val = float(parts[parts.index('val_loss:') + 1])
                        if not math.isnan(val):
                            val_losses[ep] = val
        except Exception:
            continue
        if val_losses:
            print(f"  Found val_loss in log file: {os.path.basename(log_file)}")
            break

    return val_losses


def read_val_auc_by_epoch(log_dir):
    """
    Read val_auc from TensorBoard or from training log files.
    Returns {epoch: val_auc} dict.
    """
    import math, glob

    # Try TensorBoard first
    raw = read_tb_scalars(log_dir, tag_substring='val_auc')
    if raw:
        clean = {int(step): v for step, v in raw.items() if not math.isnan(v)}
        if clean:
            return clean

    # Fall back: parse [VAL] lines from log files
    val_aucs = {}
    cwd = os.getcwd()
    all_logs = glob.glob(os.path.join(os.path.dirname(log_dir), '*.log'))
    all_logs += glob.glob(os.path.join(cwd, '*.log'))

    mode_hint = ''
    for part in log_dir.replace('\\', '/').split('/'):
        if part.startswith('rf_moe_'):
            mode_hint = part[len('rf_moe_'):]
            break
    if mode_hint:
        priority = sorted([l for l in all_logs if mode_hint in os.path.basename(l).lower()], key=os.path.getmtime, reverse=True)
        rest     = sorted([l for l in all_logs if l not in priority], key=os.path.getmtime, reverse=True)
        log_candidates = priority + rest
    else:
        log_candidates = sorted(all_logs, key=os.path.getmtime, reverse=True)

    for log_file in log_candidates:
        try:
            with open(log_file) as f:
                for line in f:
                    if '[VAL] Epoch' in line and 'val_auc:' in line:
                        parts = line.strip().split()
                        ep  = int(parts[parts.index('Epoch') + 1])
                        val = float(parts[parts.index('val_auc:') + 1])
                        if not math.isnan(val):
                            val_aucs[ep] = val
        except Exception:
            continue
        if val_aucs:
            print(f"  Found val_auc in log file: {os.path.basename(log_file)}")
            break

    return val_aucs


def estimate_steps_per_epoch(scalars, n_epochs):
    if not scalars:
        return None
    max_step = max(scalars.keys())
    return max_step // n_epochs


def get_epoch_avg_loss(scalars, steps_per_epoch, n_epochs):
    epoch_losses = {}
    for epoch in range(1, n_epochs + 1):
        step_start = (epoch - 1) * steps_per_epoch
        step_end   = epoch * steps_per_epoch
        epoch_steps = {s: v for s, v in scalars.items()
                       if step_start <= s < step_end}
        if epoch_steps:
            epoch_losses[epoch] = sum(epoch_steps.values()) / len(epoch_steps)
    return epoch_losses


def find_epoch_ckpt(log_dir, epoch):
    path = os.path.join(log_dir, 'epoch_ckpts', f'ckpt_epoch_{epoch:02d}.pth')
    if os.path.exists(path):
        return path
    path2 = os.path.join(log_dir, 'epoch_ckpts', f'ckpt_epoch_{epoch}.pth')
    return path2 if os.path.exists(path2) else None


def main():
    global OUTPUTS
    p = argparse.ArgumentParser(description='Find best epoch by max val AUC (or min val/train loss)')
    p.add_argument('--mode', required=True, choices=['fs', 'fr', 'efs', 'joint'])
    p.add_argument('--log_dir', default=None,
                   help='Override log directory (auto-detected if not set)')
    p.add_argument('--epochs', type=int, default=30,
                   help='Number of training epochs (default: 30)')
    p.add_argument('--no_val_loss', action='store_true',
                   help='Force use of train loss even if val metrics are available')
    args = p.parse_args()

    mode_dir = f'{BASE}/outputs/rf_moe_{args.mode}'
    if os.path.isdir(mode_dir):
        OUTPUTS = mode_dir

    log_dir = args.log_dir or find_log_dir(args.mode)
    if not log_dir:
        print(f"ERROR: No training output found under {OUTPUTS}")
        return

    print(f"Log directory: {log_dir}")

    epoch_ckpt_dir = os.path.join(log_dir, 'epoch_ckpts')
    if not os.path.isdir(epoch_ckpt_dir):
        print(f"ERROR: No epoch_ckpts/ folder found in {log_dir}")
        return

    ckpts = sorted(glob.glob(os.path.join(epoch_ckpt_dir, 'ckpt_epoch_*.pth')))
    print(f"\nFound {len(ckpts)} epoch checkpoints:")
    for c in ckpts:
        print(f"  {os.path.basename(c)}")

    import math

    epoch_metrics = {}
    metric_source = None
    higher_is_better = False

    if not args.no_val_loss:
        # ── Try val AUC first (highest = best) ────────────────────────────
        print(f"\nLooking for val_auc in logs...")
        val_aucs = read_val_auc_by_epoch(log_dir)
        if val_aucs:
            epoch_metrics    = val_aucs
            metric_source    = 'val_auc'
            higher_is_better = True
            print(f"  Found val_auc for {len(val_aucs)} epochs.")
        else:
            # ── Fall back to val loss (lowest = best) ──────────────────────
            print(f"  val_auc not found — trying val_loss...")
            val_losses = read_val_loss_by_epoch(log_dir)
            if val_losses:
                epoch_metrics    = val_losses
                metric_source    = 'val_loss'
                higher_is_better = False
                print(f"  Found val_loss for {len(val_losses)} epochs.")
            else:
                print(f"  val_loss not found — falling back to train loss.")

    # ── Fall back to train loss ────────────────────────────────────────────
    if not epoch_metrics:
        print(f"\nReading train loss from TensorBoard scalars...")
        scalars = {}
        for tag in ['cls_loss', 'train_loss', 'overall']:
            raw = read_tb_scalars(log_dir, tag_substring=tag)
            if raw:
                clean = {s: v for s, v in raw.items() if not math.isnan(v)}
                if clean:
                    print(f"  Found scalars using tag: '{tag}'  ({len(clean)} valid steps)")
                    scalars = clean
                    break
                else:
                    print(f"  Tag '{tag}' found but all values are NaN — skipping")

        if not scalars:
            print("\nWARNING: Could not read any metric data.")
            print("Falling back to last epoch checkpoint.")
            best_ckpt = ckpts[-1] if ckpts else None
            if best_ckpt:
                print(f"Best checkpoint: {best_ckpt}")
            return

        steps_per_epoch = estimate_steps_per_epoch(scalars, args.epochs)
        print(f"  Total steps: {max(scalars.keys())}  |  ~{steps_per_epoch} steps/epoch")
        epoch_metrics    = get_epoch_avg_loss(scalars, steps_per_epoch, args.epochs)
        metric_source    = 'train_loss'
        higher_is_better = False

    # ── Print table ────────────────────────────────────────────────────────
    best_epoch = (max if higher_is_better else min)(epoch_metrics, key=epoch_metrics.get)
    col_labels = {'val_auc': 'Val AUC', 'val_loss': 'Val Loss', 'train_loss': 'Avg Train Loss'}
    col_label  = col_labels.get(metric_source, metric_source)

    print(f"\n{'Epoch':>6}  {col_label:>15}  {'Checkpoint':>10}")
    print('-' * 50)
    for ep in sorted(epoch_metrics):
        ckpt   = find_epoch_ckpt(log_dir, ep)
        exists = 'OK' if ckpt else 'MISSING'
        marker = ' ←' if ep == best_epoch else ''
        print(f"  {ep:>4}   {epoch_metrics[ep]:>15.6f}  {exists}{marker}")

    best_ckpt = find_epoch_ckpt(log_dir, best_epoch)
    direction = 'max' if higher_is_better else 'min'
    print(f"\nBest epoch: {best_epoch}  ({metric_source}={epoch_metrics[best_epoch]:.6f}, {direction})")

    if best_ckpt and os.path.exists(best_ckpt):
        print(f"Best checkpoint: {best_ckpt}")
        print(f"\nNext step:")
        print(f"  python run_evaluation.py --trained_on {args.mode} --checkpoint {best_ckpt} --tables 3 4 5")
    else:
        print(f"ERROR: Checkpoint file not found for best epoch.")


if __name__ == '__main__':
    main()
