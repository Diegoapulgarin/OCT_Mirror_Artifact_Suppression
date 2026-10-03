"""Train one PC-CGAN configuration (M2-M8) from a YAML file.

Example:
    python scripts/train_pccgan.py --config configs/pccgan/m3.yaml \
        --data-root /path/to/Data --out-dir runs/pccgan
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

import torch
import yaml
from torch.utils.data import DataLoader

from octmirror.datasets import build_pccgan_datasets
from octmirror.models.pccgan import PatchGANDiscriminator, UNetGenerator
from octmirror.training import set_seed, train_loop


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg['training']['adversarial_loss'] != 'hinge':
        raise ValueError("Only the hinge adversarial loss is supported.")
    return cfg


def update_ablation_summary(path: str, entry: dict) -> None:
    """Insert or replace the entry of this configuration in ``ablation_summary.json``."""
    summary = []
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
    summary = [s for s in summary if s.get('name') != entry['name']] + [entry]
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True, help='YAML configuration (configs/pccgan/m*.yaml).')
    parser.add_argument('--data-root', required=True, help='Directory containing the data folders.')
    parser.add_argument('--out-dir', default='runs/pccgan',
                        help='Runs are written to <out-dir>/<config name>.')
    parser.add_argument('--device', default=None, help='Torch device (default: cuda:0 if available, else cpu).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg['training']
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'is on: {device}')

    savePath = os.path.join(args.out_dir, cfg['name'])
    os.makedirs(savePath, exist_ok=True)
    with open(os.path.join(savePath, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print('starting data preparation...')
    train_ds, val_ds = build_pccgan_datasets(
        args.data_root, cfg['data']['folders'], cfg['data']['files'],
        val_fraction=cfg['val_fraction'], split_seed=cfg['split_seed'],
    )
    train_dl = DataLoader(train_ds, batch_size=tr['batch_size'], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tr['batch_size'], shuffle=False, drop_last=False)
    print(f'Train batches: {len(train_dl)}, Val batches: {len(val_dl)} with batch size of {tr["batch_size"]}')
    in_channels = train_ds[0][0].shape[0]
    out_channels = train_ds[0][1].shape[0]

    summary_path = os.path.join(args.out_dir, 'ablation_summary.json')
    try:
        set_seed(cfg['seed'])
        G = UNetGenerator(in_ch=in_channels, out_ch=out_channels)
        D = PatchGANDiscriminator(in_ch=in_channels)

        _, _, best_amp_mae, best_epoch = train_loop(
            train_dl, val_dl, G, D, tr['epochs'],
            savePath=savePath,
            device=device,
            lr=tr['lr_g'],
            dlr=tr['lr_d'],
            betas=tuple(tr['betas']),
            lambda_l1=tr['lambda_l1'],
            use_phase=True,
            lambda_phase=tr['lambda_phase'],
            lambda_amp=tr['lambda_amp'],
            phase_weight_mode=tr['phase_weight_mode'],
            use_phase_grad=tr['use_phase_grad'],
            lambda_phase_grad=tr['lambda_phase_grad'],
            use_ema=tr['use_ema'],
            ema_decay=tr['ema_decay'],
            grad_clip=tr['grad_clip'],
            checkpoint_every=tr['checkpoint_every'],
            scheduler_patience=tr['scheduler']['patience'],
            scheduler_factor=tr['scheduler']['factor'],
            scheduler_min_lr=tr['scheduler']['min_lr'],
            image_log_every=tr['image_log_every'],
        )
        update_ablation_summary(summary_path, {
            'name': cfg['name'],
            'best_amp_mae': best_amp_mae,
            'best_epoch': best_epoch,
            'best_weight_path': os.path.join(savePath, 'weight', 'G_best.pth'),
        })
    except Exception as ex:
        update_ablation_summary(summary_path, {
            'name': cfg['name'],
            'best_amp_mae': None,
            'best_epoch': None,
            'best_weight_path': os.path.join(savePath, 'weight', 'G_best.pth'),
            'error': str(ex),
            'traceback': traceback.format_exc(limit=10),
        })
        raise


if __name__ == '__main__':
    main()
