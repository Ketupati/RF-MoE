#!/usr/bin/env python3
"""
run_training.py  —  Launch RF-MoE training for any protocol.

Modes:
  fs     Train on FS methods only   (Tables 3/4/5 FS row)
  fr     Train on FR methods only   (Tables 3/4/5 FR row)
  efs    Train on EFS methods only  (Tables 3/4/5 EFS row)
  joint  Train on all methods       (Table 6)

Usage:
  python run_training.py --mode fs
  python run_training.py --mode fr
  python run_training.py --mode efs
  python run_training.py --mode joint
  python run_training.py --mode fs --epochs 10 --batch_train 32 --batch_test 16

NOTE: Run from the REPO directory or let this script cd into it.
"""

import os
import sys
import argparse
import subprocess
import shutil

# ============================================================
# PATHS  (must match setup.py)
# ============================================================
BASE    = '/home/ibubu/ketupati'
REPO    = f'{BASE}/DeepfakeBench_DF40'
OUTPUTS = f'{BASE}/outputs'
DETECTOR_YAML = './training/config/detector/rfmoe.yaml'

# ============================================================
# Protocol definitions
# ============================================================
#  For Tables 3/4/5:  train on ONE type, test on ALL three groups
#    (test datasets during training give you Table 3 in-domain numbers)
#  For Table 6:       train on ALL types, test on 6 representative methods
PROTOCOLS = {
    'fs': {
        'description': 'FS-only training  (Tables 3/4/5, FS row)',
        'train':     ['FSAll_ff_train'],
        'train_val': ['FSAll_ff_train90'],   # 90% split (if val split exists)
        'val':       'FSAll_ff_val10',
        'test':      ['simswap_ff', 'facevid2vid_ff'],
    },
    'fr': {
        'description': 'FR-only training  (Tables 3/4/5, FR row)',
        'train':     ['FRAll_ff_train'],
        'train_val': ['FRAll_ff_train90'],
        'val':       'FRAll_ff_val10',
        'test':      ['facevid2vid_ff', 'simswap_ff'],
    },
    'efs': {
        'description': 'EFS-only training  (Tables 3/4/5, EFS row)',
        'train':     ['EFSAll_ff_train'],
        'train_val': ['EFSAll_ff_train90'],
        'val':       'EFSAll_ff_val10',
        'test':      ['StyleGAN3_ff', 'simswap_ff'],
    },
    'joint': {
        'description': 'Joint training on all types  (Table 6)',
        'train':     ['FSAll_ff_train', 'FRAll_ff_train', 'EFSAll_ff_train'],
        'train_val': ['FSAll_ff_train90', 'FRAll_ff_train90', 'EFSAll_ff_train90'],
        'val':       'joint_val10',
        'test':      ['simswap_ff', 'facevid2vid_ff', 'StyleGAN3_ff'],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='RF-MoE training launcher')
    p.add_argument('--mode', required=True, choices=list(PROTOCOLS.keys()),
                   help='Training mode / protocol')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override nEpochs (default: value in rfmoe.yaml = 10)')
    p.add_argument('--batch_train', type=int, default=None,
                   help='Override train_batchSize (default: 64 for A100)')
    p.add_argument('--batch_test', type=int, default=None,
                   help='Override test_batchSize (default: 32)')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--no_test', action='store_true',
                   help='Skip per-epoch testing; select best checkpoint via min val/train loss after training')
    p.add_argument('--no_val', action='store_true',
                   help='Disable validation split — use full training set and train loss for checkpoint selection')
    p.add_argument('--dry_run', action='store_true',
                   help='Print command without running it')
    return p.parse_args()


def patch_trainer_add_method():
    """
    Patch trainer.py to add the save_epoch_ckpt() method.
    Uses self.model (not self.detector) — correct attribute name.
    Idempotent — re-applies if old broken version (self.detector) exists.
    """
    trainer_path = os.path.join(REPO, 'training', 'trainer', 'trainer.py')
    if not os.path.exists(trainer_path):
        print("  [WARN] trainer.py not found — skipping")
        return

    with open(trainer_path) as f:
        content = f.read()

    # Remove old broken version if present (used self.detector)
    if 'self.detector' in content and 'save_epoch_ckpt' in content:
        print("  [FIX] Removing old broken save_epoch_ckpt (used self.detector)...")
        # Remove the entire method block
        import re
        content = re.sub(
            r'\n    def save_epoch_ckpt\(self.*?(?=\n    def |\Z)',
            '', content, flags=re.DOTALL
        )

    if 'save_epoch_ckpt' in content and 'self.model' in content:
        print("  [SKIP] trainer.py already has correct save_epoch_ckpt")
        return

    epoch_save_code = '''
    def save_epoch_ckpt(self, epoch):
        """Save checkpoint after every epoch regardless of AUC."""
        try:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            save_dir = os.path.join(self.config['log_dir'], 'epoch_ckpts')
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'ckpt_epoch_{epoch:02d}.pth')
            torch.save(model.state_dict(), path)
            self.logger.info(f'[EPOCH CKPT] Saved epoch {epoch} to {path}')
        except Exception as e:
            self.logger.warning(f'[EPOCH CKPT] Failed to save epoch {epoch}: {e}')
'''

    if 'class Trainer' not in content:
        print("  [WARN] Trainer class not found in trainer.py")
        return

    content = content + epoch_save_code
    with open(trainer_path, 'w') as f:
        f.write(content)
    print("  [OK] Added save_epoch_ckpt method to trainer.py")


def patch_train_add_epoch_call():
    """
    Patch train.py to CALL trainer.save_epoch_ckpt(epoch) after each epoch.
    The anchor is the line right after train_epoch() completes.
    Idempotent — skips if already patched.
    """
    train_path = os.path.join(REPO, 'training', 'train.py')
    if not os.path.exists(train_path):
        print("  [WARN] train.py not found — skipping epoch call patch")
        return

    with open(train_path) as f:
        content = f.read()

    if 'save_epoch_ckpt' in content:
        print("  [SKIP] train.py already calls save_epoch_ckpt")
        return

    # Anchor: the line after train_epoch() that checks best_metric
    anchor = '        if best_metric is not None:'
    if anchor not in content:
        print("  [WARN] Could not find anchor in train.py for epoch call")
        return

    content = content.replace(anchor,
        '        trainer.save_epoch_ckpt(epoch)  # save every epoch\n' + anchor)
    with open(train_path, 'w') as f:
        f.write(content)
    print("  [OK] Patched train.py — trainer.save_epoch_ckpt(epoch) called after each epoch")


def patch_train_no_test():
    """
    Patch train.py to handle missing test_dataset gracefully.
    When --no_test is used, test_dataset is not passed so config has no key.
    Idempotent — skips if already patched.
    """
    train_path = os.path.join(REPO, 'training', 'train.py')
    if not os.path.exists(train_path):
        print("  [WARN] train.py not found — skipping no-test patch")
        return
    with open(train_path) as f:
        content = f.read()
    if 'test_dataset_default_patch' in content:
        print("  [SKIP] train.py already patched for missing test_dataset")
        return
    anchor = 'test_data_loaders = prepare_testing_data(config)'
    if anchor not in content:
        print("  [WARN] Could not find prepare_testing_data anchor in train.py")
        return
    patch = (
        "config.setdefault('test_dataset', [])  # test_dataset_default_patch\n"
        "    "
    )
    content = content.replace(anchor, patch + anchor)
    with open(train_path, 'w') as f:
        f.write(content)
    print("  [OK] Patched train.py — missing test_dataset defaults to []")


def patch_train_val_loss(val_json_path):
    """
    Patch train.py to compute and log val AUC (and val loss) after each epoch.
    Uses a simple custom dataset to bypass DeepfakeBench registry checks.
    Logs val_auc and val_loss to TensorBoard and prints to stdout.
    Idempotent — skips if already patched (marker: val_auc_patch_v3).
    """
    train_path = os.path.join(REPO, 'training', 'train.py')
    if not os.path.exists(train_path):
        print("  [WARN] train.py not found — skipping val patch")
        return

    with open(train_path) as f:
        content = f.read()

    # Remove old patches (v1, v2) if present
    for _old_marker, _old_start in [
        ('val_loss_patch_v2', '        # val_loss_patch_v2'),
        ('val_loss_patch',    '        # val_loss_patch'),
    ]:
        if _old_marker in content and 'val_auc_patch_v3' not in content:
            _end_guard = '        if best_metric is not None:'
            if _old_start in content and _end_guard in content:
                _idx_s  = content.index(_old_start)
                _idx_e  = content.index(_end_guard, _idx_s)
                content = content[:_idx_s] + content[_idx_e:]
                print(f"  [FIX] Removed old patch ({_old_marker})")

    if 'val_auc_patch_v3' in content:
        print("  [SKIP] train.py already has val AUC computation (v3)")
        return

    # Anchor: right after save_epoch_ckpt call
    anchor = '        trainer.save_epoch_ckpt(epoch)  # save every epoch'
    if anchor not in content:
        print("  [WARN] Could not find save_epoch_ckpt anchor — skipping val patch")
        return

    val_code = f'''
        # val_auc_patch_v3 — compute val AUC and val loss after each epoch
        try:
            import json as _json
            import torch as _torch, torchvision.transforms as _T
            from PIL import Image as _Image
            from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
            from sklearn.metrics import roc_auc_score as _roc_auc_score
            import numpy as _np

            _val_json_path = "{val_json_path}"

            class _ValDataset(_Dataset):
                def __init__(self, json_path, frame_num, resolution, mean, std):
                    with open(json_path) as _f:
                        _d = _json.load(_f)
                    self.tf = _T.Compose([
                        _T.Resize((resolution, resolution)),
                        _T.ToTensor(),
                        _T.Normalize(mean=mean, std=std),
                    ])
                    self.frame_num = frame_num
                    self.samples = []
                    for _ds, _ld in _d.items():
                        for _lname, _sd in _ld.items():
                            _label = 0 if 'Real' in _lname else 1
                            for _sk, _videos in _sd.items():
                                for _vname, _vinfo in _videos.items():
                                    _frames = _vinfo.get('frames', [])
                                    if _frames:
                                        self.samples.append((_frames, _label))
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx):
                    _frames, _label = self.samples[idx]
                    _sel = _frames[:self.frame_num] if len(_frames) >= self.frame_num else _frames
                    _imgs = []
                    for _fp in _sel:
                        try: _imgs.append(self.tf(_Image.open(_fp).convert('RGB')))
                        except: pass
                    if not _imgs: return None
                    return {{'image': _imgs[0], 'label': _torch.tensor(_label, dtype=_torch.long)}}

            def _collate(_batch):
                _batch = [b for b in _batch if b is not None]
                if not _batch: return None
                return {{
                    'image': _torch.stack([b['image'] for b in _batch]),
                    'label': _torch.stack([b['label'] for b in _batch]),
                }}

            if os.path.exists(_val_json_path):
                _fn = config.get('frame_num', {{}})
                _val_ds = _ValDataset(
                    _val_json_path,
                    frame_num=_fn.get('test', 8),
                    resolution=config.get('resolution', 224),
                    mean=config.get('mean', [0.485, 0.456, 0.406]),
                    std=config.get('std', [0.229, 0.224, 0.225]),
                )
                _val_loader = _DataLoader(
                    _val_ds,
                    batch_size=config.get('test_batchSize', 16),
                    shuffle=False,
                    num_workers=config.get('workers', 2),
                    collate_fn=_collate,
                    pin_memory=True,
                )
                _model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                _model.eval()
                _val_losses, _all_probs, _all_labels = [], [], []
                with _torch.no_grad():
                    for _batch in _val_loader:
                        if _batch is None: continue
                        _imgs = _batch['image'].cuda()
                        _lbls = _batch['label'].cuda()
                        _pred = _model({{'image': _imgs, 'label': _lbls}})
                        _losses = _model.get_losses({{'image': _imgs, 'label': _lbls}}, _pred)
                        _val_losses.append(_losses['overall'].item())
                        _probs = _torch.softmax(_pred['cls'], dim=1)[:, 1]
                        _all_probs.extend(_probs.cpu().numpy().tolist())
                        _all_labels.extend(_lbls.cpu().numpy().tolist())
                _model.train()
                if _val_losses:
                    _val_loss_avg = sum(_val_losses) / len(_val_losses)
                    trainer.logger.info(f'[VAL] Epoch {{epoch}} val_loss: {{_val_loss_avg:.6f}}')
                    print(f'[VAL] Epoch {{epoch}} val_loss: {{_val_loss_avg:.6f}}')
                    if hasattr(trainer, 'tb_writer') and trainer.tb_writer is not None:
                        trainer.tb_writer.add_scalar('val_loss', _val_loss_avg, epoch)
                if len(set(_all_labels)) == 2:
                    _val_auc = _roc_auc_score(_all_labels, _all_probs)
                    trainer.logger.info(f'[VAL] Epoch {{epoch}} val_auc: {{_val_auc:.6f}}')
                    print(f'[VAL] Epoch {{epoch}} val_auc: {{_val_auc:.6f}}')
                    if hasattr(trainer, 'tb_writer') and trainer.tb_writer is not None:
                        trainer.tb_writer.add_scalar('val_auc', _val_auc, epoch)
        except Exception as _e:
            trainer.logger.warning(f'[VAL] Val metric computation failed: {{_e}}')
'''

    content = content.replace(anchor, anchor + val_code)
    with open(train_path, 'w') as f:
        f.write(content)
    print(f"  [OK] Patched train.py — val AUC+loss computed each epoch from {os.path.basename(val_json_path)}")


def patch_yaml_log_dir(mode):
    """
    Set log_dir in rfmoe.yaml to outputs/rf_moe_{mode} so each training
    run gets its own directory and doesn't overwrite prior checkpoints.
    """
    yaml_path = os.path.join(REPO, 'training', 'config', 'detector', 'rfmoe.yaml')
    if not os.path.exists(yaml_path):
        print("  [WARN] rfmoe.yaml not found — skipping log_dir patch")
        return
    with open(yaml_path) as f:
        lines = f.readlines()
    mode_log_dir = f'{BASE}/outputs/rf_moe7_{mode}'
    patched = []
    found = False
    for line in lines:
        if line.startswith('log_dir:'):
            line = f'log_dir: {mode_log_dir}\n'
            found = True
        patched.append(line)
    if not found:
        patched.append(f'log_dir: {mode_log_dir}\n')
    with open(yaml_path, 'w') as f:
        f.writelines(patched)
    print(f"  [PATCHED] log_dir → {mode_log_dir}")


def patch_yaml_overrides(epochs=None, batch_train=None, batch_test=None):
    """Temporarily patch rfmoe.yaml with CLI overrides (in-place)."""
    if not any([epochs, batch_train, batch_test]):
        return

    yaml_path = os.path.join(REPO, 'training', 'config', 'detector', 'rfmoe.yaml')
    with open(yaml_path) as f:
        lines = f.readlines()

    patched = []
    for line in lines:
        if epochs is not None and line.startswith('nEpochs:'):
            line = f'nEpochs: {epochs}\n'
        if batch_train is not None and line.startswith('train_batchSize:'):
            line = f'train_batchSize: {batch_train}\n'
        if batch_test is not None and line.startswith('test_batchSize:'):
            line = f'test_batchSize: {batch_test}\n'
        patched.append(line)

    with open(yaml_path, 'w') as f:
        f.writelines(patched)

    if epochs:      print(f"  [PATCHED] nEpochs → {epochs}")
    if batch_train: print(f"  [PATCHED] train_batchSize → {batch_train}")
    if batch_test:  print(f"  [PATCHED] test_batchSize → {batch_test}")


def main():
    args = parse_args()
    proto = PROTOCOLS[args.mode]

    print("=" * 70)
    print(f"RF-MoE Training: {proto['description']}")
    print(f"  REPO:    {REPO}")
    print(f"  OUTPUTS: {OUTPUTS}")
    print(f"  Mode:    {args.mode}")
    print("=" * 70)

    # Verify pre-conditions
    if not os.path.exists(f'{REPO}/training/test.py'):
        print("ERROR: Repo not found. Run setup.py first.")
        sys.exit(1)

    json_dir = f'{BASE}/data/dataset_json'

    # Decide whether to use val split
    use_val = not args.no_val
    val_jsons_exist = all(
        os.path.exists(os.path.join(json_dir, f'{ds}.json'))
        for ds in proto['train_val']
    )
    if use_val and not val_jsons_exist:
        print("  [INFO] Val split JSONs not found — run create_val_split.py first.")
        print("  [INFO] Falling back to full training set (no val loss).")
        use_val = False

    train_datasets = proto['train_val'] if use_val else proto['train']

    for ds in train_datasets:
        jpath = os.path.join(json_dir, f'{ds}.json')
        if not os.path.exists(jpath):
            print(f"ERROR: Training JSON not found: {jpath}")
            print("Run generate_jsons.py (and create_val_split.py for val splits) first.")
            sys.exit(1)

    os.chdir(REPO)
    print(f"Working directory: {os.getcwd()}")

    # Set mode-specific log_dir so runs don't overwrite each other
    print("\nPatching rfmoe.yaml log_dir for this mode:")
    patch_yaml_log_dir(args.mode)

    # Patch trainer.py — add save_epoch_ckpt method
    print("\nPatching trainer.py for epoch-wise checkpoints:")
    patch_trainer_add_method()

    # Patch train.py — call save_epoch_ckpt after each epoch
    print("\nPatching train.py to call save_epoch_ckpt:")
    patch_train_add_epoch_call()

    # Patch train.py to handle missing test_dataset when --no_test is used
    if args.no_test:
        print("\nPatching train.py for no-test mode:")
        patch_train_no_test()

    # Patch train.py to compute val loss each epoch
    if use_val:
        val_json_path = os.path.join(json_dir, f'{proto["val"]}.json')
        print(f"\nPatching train.py for val loss (using {proto['val']}.json):")
        patch_train_val_loss(val_json_path)
    else:
        print("\n  [INFO] Skipping val loss patch (no val split).")

    # Apply YAML overrides
    if any([args.epochs, args.batch_train, args.batch_test]):
        print("\nPatching rfmoe.yaml with CLI overrides:")
        patch_yaml_overrides(args.epochs, args.batch_train, args.batch_test)

    # Build command
    cmd = [
        sys.executable, 'training/train.py',
        '--detector_path', DETECTOR_YAML,
        '--train_dataset', *train_datasets,
    ]
    if not args.no_test:
        cmd += ['--test_dataset', *proto['test']]
    else:
        print("  [no_test] Skipping per-epoch evaluation.")
        if use_val:
            print("  [val]     Val loss logged each epoch — find_best_epoch.py will use it.")
        print("  After training, run: python find_best_epoch.py --mode", args.mode)
    if args.resume:
        cmd += ['--resume', args.resume]

    print("\nCommand:")
    print('  ' + ' \\\n    '.join(cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Command printed above — not executed.")
        return

    # Run training
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Training exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print(f"Training complete for mode: {args.mode}")
    print(f"All epoch checkpoints: {OUTPUTS}/rf_moe7_{args.mode}/epoch_ckpts/")
    print("=" * 70)

    # Auto-find best checkpoint and copy to checkpoints/{mode}/
    print("\nFinding best checkpoint...")
    find_best_script = os.path.join(BASE, "find_best_epoch.py")
    fb_result = subprocess.run(
        [sys.executable, find_best_script, "--mode", args.mode],
        capture_output=True, text=True, cwd=BASE
    )
    print(fb_result.stdout)
    best_ckpt = None
    for line in fb_result.stdout.split("\n"):
        if line.startswith("Best checkpoint:"):
            best_ckpt = line.split("Best checkpoint:")[1].strip()
            break
    if best_ckpt and os.path.exists(best_ckpt):
        dest_dir = os.path.join(BASE, "checkpoints", args.mode)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "ckpt_best.pth")
        shutil.copy2(best_ckpt, dest)
        print(f"  [OK] Best checkpoint -> {dest}")

        import re, datetime
        best_epoch, best_metric, metric_name = None, None, 'val_auc'
        for line in fb_result.stdout.split("\n"):
            m = re.search(r'Best epoch:\s*(\d+)\s*\((\w+)=([\d.]+)', line)
            if m:
                best_epoch  = int(m.group(1))
                metric_name = m.group(2)
                best_metric = float(m.group(3))
        info_path = os.path.join(dest_dir, "best_checkpoint_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"mode:           {args.mode}\n")
            f.write(f"best_epoch:     {best_epoch}\n")
            f.write(f"metric:         {metric_name}\n")
            f.write(f"best_val_auc:   {best_metric}\n")
            f.write(f"checkpoint:     {dest}\n")
            f.write(f"source:         {best_ckpt}\n")
            f.write(f"saved_at:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  [OK] Info saved  -> {info_path}")
    else:
        print("  [WARN] Best checkpoint not found — run find_best_epoch.py manually.")

    # Save latest epoch checkpoint for resuming training
    epoch_ckpt_dir = os.path.join(OUTPUTS, f"rf_moe7_{args.mode}", "epoch_ckpts")
    if os.path.isdir(epoch_ckpt_dir):
        epoch_files = sorted(
            [f for f in os.listdir(epoch_ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
        )
        if epoch_files:
            import re, datetime
            latest_file = epoch_files[-1]
            latest_src  = os.path.join(epoch_ckpt_dir, latest_file)
            dest_dir    = os.path.join(BASE, "checkpoints", args.mode)
            os.makedirs(dest_dir, exist_ok=True)
            latest_dest = os.path.join(dest_dir, "ckpt_latest.pth")
            shutil.copy2(latest_src, latest_dest)
            m = re.search(r'ckpt_epoch_(\d+)\.pth', latest_file)
            latest_epoch = int(m.group(1)) if m else None
            info_path = os.path.join(dest_dir, "latest_checkpoint_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"mode:           {args.mode}\n")
                f.write(f"latest_epoch:   {latest_epoch}\n")
                f.write(f"total_epochs:   {args.epochs or 'yaml default'}\n")
                f.write(f"checkpoint:     {latest_dest}\n")
                f.write(f"source:         {latest_src}\n")
                f.write(f"saved_at:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"  [OK] Latest checkpoint (epoch {latest_epoch}) -> {latest_dest}")
            print(f"  [OK] Info saved  -> {info_path}")
        else:
            print("  [WARN] No epoch checkpoints found in epoch_ckpts/")

    print("\nNext step: python run_evaluation.py --trained_on", args.mode, "--auto_ckpt")
    print("=" * 70)


if __name__ == '__main__':
    main()
