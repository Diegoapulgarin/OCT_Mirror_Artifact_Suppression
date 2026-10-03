"""Train one conditional diffusion configuration (D1-D4) from a YAML file.

Example:
    python scripts/train_diffusion.py --config configs/diffusion/d1.yaml \
        --data-root /path/to/Data --out-dir runs/diffusion
"""
from __future__ import annotations

import argparse
import datetime
import os
import time

import torch
import yaml

from octmirror.datasets import build_diffusion_dataset
from octmirror.models.diffusion import DDPMScheduler
from octmirror.training import (
    check_diffusion_overwrite,
    set_seed,
    train_diffusion,
    write_config,
    write_run_metadata,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tr = cfg['training']
    sched = DDPMScheduler(num_time_steps=tr['num_time_steps'], device='cpu')
    if (abs(float(sched.beta[0]) - tr['beta_start']) > 1e-9
            or abs(float(sched.beta[-1]) - tr['beta_end']) > 1e-9):
        raise ValueError("beta_start/beta_end must match the linear schedule of DDPMScheduler (1e-4, 0.02).")
    if cfg['data']['roi_mode'] != 'center':
        raise ValueError("Only roi_mode 'center' is supported.")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True, help='YAML configuration (configs/diffusion/d*.yaml).')
    parser.add_argument('--data-root', required=True, help='Directory containing the data folders.')
    parser.add_argument('--out-dir', default='runs/diffusion',
                        help='Runs are written to <out-dir>/<config name>.')
    parser.add_argument('--device', default=None, help='Torch device (default: cuda if available, else cpu).')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers.')
    parser.add_argument('--data-parallel', action=argparse.BooleanOptionalAction, default=True,
                        help='Use nn.DataParallel when several GPUs are visible (default: yes).')
    parser.add_argument('--allow-overwrite', action='store_true',
                        help='Allow writing into a directory that already holds a run.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr, data = cfg['training'], cfg['data']
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    savePath = os.path.join(args.out_dir, cfg['name'])
    check_diffusion_overwrite(savePath, args.allow_overwrite)
    os.makedirs(savePath, exist_ok=True)

    run_start_time = time.time()
    write_run_metadata(savePath, {'experiment_id': cfg['experiment_id'], 'name': cfg['name'],
                                  'status': 'running', 'start_time': run_start_time})
    try:
        print(f"\nActive configuration: {cfg['name']} "
              f"| batch_size={tr['batch_size']} | lr={tr['lr']} "
              f"| lambda_phase_diff={tr['lambda_phase_diff']} | use_phase_loss={tr['use_phase_loss']} "
              f"| num_epochs={tr['epochs']} | seed={cfg['seed']}")

        train_ds = build_diffusion_dataset(args.data_root, data['folders'],
                                           roi_size=data['roi_size'],
                                           n_bscans_per_volume=data['n_bscans_per_volume'])
        input_tensor, output_tensor = train_ds[0]
        print(f"Input tensor shape: {tuple(input_tensor.shape)}")
        print(f"Target tensor shape: {tuple(output_tensor.shape)}")
        print(f'Primary device: {device}')

        set_seed(cfg['seed'])

        write_config(savePath, {
            'experiment_id': cfg['experiment_id'],
            'name': cfg['name'],
            'use_phase_loss': tr['use_phase_loss'],
            'lambda_phase_diff': tr['lambda_phase_diff'],
            'batch_size': tr['batch_size'],
            'lr': tr['lr'],
            'num_epochs': tr['epochs'],
            'num_time_steps': tr['num_time_steps'],
            'seed': cfg['seed'],
            'roi_size': data['roi_size'],
            'roi_mode': data['roi_mode'],
            'n_bscans_per_volume': data['n_bscans_per_volume'],
            'data_path': args.data_root,
            'folders': data['folders'],
            'save_path': savePath,
            'native_roi_no_interpolation': True,
            'timestamp': datetime.datetime.now().isoformat(),
        })

        train_diffusion(train_ds,
                        savePath=savePath,
                        device=device,
                        batch_size=tr['batch_size'],
                        num_time_steps=tr['num_time_steps'],
                        num_epochs=tr['epochs'],
                        seed=cfg['seed'],
                        ema_decay=tr['ema_decay'],
                        lr=tr['lr'],
                        use_phase_loss=tr['use_phase_loss'],
                        lambda_phase_diff=tr['lambda_phase_diff'],
                        experiment_id=cfg['experiment_id'],
                        roi_size=data['roi_size'],
                        num_workers=args.num_workers,
                        data_parallel=args.data_parallel)

        write_run_metadata(savePath, {'experiment_id': cfg['experiment_id'], 'name': cfg['name'],
                                      'status': 'ok', 'start_time': run_start_time,
                                      'end_time': time.time()})
    except Exception as e:
        write_run_metadata(savePath, {'experiment_id': cfg['experiment_id'], 'name': cfg['name'],
                                      'status': 'failed', 'error': str(e),
                                      'start_time': run_start_time, 'end_time': time.time()})
        raise
    print('done!')


if __name__ == '__main__':
    main()
