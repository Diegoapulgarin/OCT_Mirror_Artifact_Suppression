#%%
"""Final condition-only reverse-DDPM evaluation for D1c and D2c."""
import csv
import datetime
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import wilcoxon
from timm.utils import ModelEmaV3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffComplexField_ROI256_D1c_D4c import (
    UNET, DDPM_Scheduler, extract_center_roi, encode_amplitude_phase,
    decode_amplitude_phase,
)
from complex_field_utils import mirrorArtifact
from evaluate_ablation_unified import (
    calculate_ssim, calculate_psnr, calculate_mse, histogram_difference,
    despeckle_nlm, bootstrap_ci, holm_bonferroni,
    weighted_phase_coherence, masked_complex_coherence, phase_gradient_ssim,
)

DIFFUSION_ROOT = '/home/rsg-dapulgaris/Models/diffCxModels/ROI256_AmpPhase_X0Prediction_Ablation'
VALIDATION_DATA_DIR = '/home/rsg-dapulgaris/Data/validationCxPaper'
EVALUATE_CONFIG_IDS = ['D1', 'D2', 'D3', 'D4']
EVAL_SEED = 20240601
EVAL_N_BSCANS_PER_VOLUME = 40
INFERENCE_SEED = 20260828
OUT_DIR = os.path.join(DIFFUSION_ROOT, 'evaluation_D1c_D2c_D3c_D4c_reverse_ddpm')
EVAL_BSCAN_IDS_PATH = os.path.join(OUT_DIR, 'eval_bscan_ids.json')

AMP_PHASE_CONFIGS = [
    dict(id='D1', name='D1c_ROI256_AmpPhase_X0_NoPhaseLoss',
         folder='D1c_ROI256_AmpPhase_X0_NoPhaseLoss', prediction_type='x0',
         representation='log_amplitude_phase_normalized', phase_lambda=0.0,
         checkpoint_preference=['best_by_val_intensity_mse_pre_nlm.pth',
                                'best_by_val_amplitude_mse_pre_nlm.pth',
                                'best_by_val_amplitude_ssim_pre_nlm.pth',
                                'checkpoint_last.pth']),
    dict(id='D2', name='D2c_ROI256_AmpPhase_X0_PhaseLossLow',
         folder='D2c_ROI256_AmpPhase_X0_PhaseLossLow', prediction_type='x0',
         representation='log_amplitude_phase_normalized', phase_lambda=0.1,
         checkpoint_preference=['best_by_val_intensity_mse_pre_nlm.pth',
                                'best_by_val_amplitude_mse_pre_nlm.pth',
                                'best_by_val_amplitude_ssim_pre_nlm.pth',
                                'checkpoint_last.pth']),
    dict(id='D3', name='D3c_ROI256_AmpPhase_X0_PhaseLossLow_B16',
         folder='D3c_ROI256_AmpPhase_X0_PhaseLossLow_B16', prediction_type='x0',
         representation='log_amplitude_phase_normalized', phase_lambda=0.1,
         checkpoint_preference=['best_by_val_intensity_mse_pre_nlm.pth',
                                'best_by_val_amplitude_mse_pre_nlm.pth',
                                'best_by_val_amplitude_ssim_pre_nlm.pth',
                                'checkpoint_last.pth']),
    dict(id='D4', name='D4c_ROI256_AmpPhase_X0_PhaseLossModerate',
         folder='D4c_ROI256_AmpPhase_X0_PhaseLossModerate', prediction_type='x0',
         representation='log_amplitude_phase_normalized', phase_lambda=0.5,
         checkpoint_preference=['best_by_val_intensity_mse_pre_nlm.pth',
                                'best_by_val_amplitude_mse_pre_nlm.pth',
                                'best_by_val_amplitude_ssim_pre_nlm.pth',
                                'checkpoint_last.pth']),
]
# Historical identifiers remain documented but are intentionally not selected here.
HISTORICAL_CONFIG_IDS = ('D1', 'D2', 'D3', 'D4', 'D1b', 'D2b', 'D3b', 'D4b')
ALLOWED_TISSUE_SPECS = [
    dict(label='Optic nerve', filename='OpticNerve.npy'),
    dict(label='Chicken thigh', filename='[ComplexConjugateRemoval][ChickenThigh][08-07-2026_20-54-54]_z=(512)_x=(512)_y=(512).npy'),
    dict(label='Fingertip', filename='[ComplexConjugateRemoval][Fingertip][08-07-2026_19-57-30]_z=(512)_x=(512)_y=(256).npy'),
    dict(label='Dorsal hand', filename='[ComplexConjugateRemoval][HandBack][08-07-2026_20-07-13]_z=(512)_x=(512)_y=(256).npy'),
    dict(label='Nail bed', filename='[ComplexConjugateRemoval][NailBed][08-07-2026_19-50-10]_z=(512)_x=(512)_y=(256).npy'),
    dict(label='Salmon', filename='[ComplexConjugateRemoval][Salmon][08-07-2026_20-43-58]_z=(512)_x=(512)_y=(512).npy'),
]
FORBIDDEN_TERMS = ('cadaver', 'heart', 'hearth', 'nailbedocta', 'octa')
METRIC_COLUMNS = ['ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc', 'pgssim']
CSV_COLUMNS = ['bscan_id', 'tissue', 'source_filename', 'bscan_index', 'config_id',
               'config_name', 'checkpoint_path', 'used_ema', 'inference_seed',
               'sampler_type', 'num_time_steps'] + METRIC_COLUMNS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def select_configs(config_ids):
    if not config_ids:
        raise ValueError('EVALUATE_CONFIG_IDS must contain at least one ID (D1, D2, D3, or D4).')
    valid = {cfg['id']: cfg for cfg in AMP_PHASE_CONFIGS}
    unknown = [str(value) for value in config_ids if str(value).upper() not in valid]
    if unknown:
        raise ValueError(f'Unknown EVALUATE_CONFIG_IDS: {unknown}; allowed IDs are D1, D2, D3, and D4.')
    selected_ids = [str(value).upper() for value in config_ids]
    return [cfg for cfg in AMP_PHASE_CONFIGS if cfg['id'] in selected_ids]


CONFIGS = select_configs(EVALUATE_CONFIG_IDS)
if [cfg['id'] for cfg in CONFIGS] != ['D1', 'D2', 'D3', 'D4']:
    raise ValueError('The final evaluation must select D1, D2, D3, and D4 in that order.')


def validate_allowlist():
    if len(ALLOWED_TISSUE_SPECS) != 6:
        raise ValueError('ALLOWED_TISSUE_SPECS must contain exactly six entries.')
    for spec in ALLOWED_TISSUE_SPECS:
        lowered = spec['filename'].lower().replace('-', '')
        if any(term in lowered for term in FORBIDDEN_TERMS):
            raise ValueError(f"Forbidden filename in allowlist: {spec['filename']}")
        path = os.path.join(VALIDATION_DATA_DIR, spec['filename'])
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Allowlisted validation file does not exist: {path}')


def load_tomogram(path):
    raw = np.load(path)
    return raw[..., 0] + 1j * raw[..., 1] if raw.ndim == 4 else raw


def build_or_load_eval_bscan_ids():
    validate_allowlist()
    if os.path.exists(EVAL_BSCAN_IDS_PATH):
        with open(EVAL_BSCAN_IDS_PATH) as handle:
            result = json.load(handle)
        if len(result) != 240:
            raise ValueError(f'Existing eval split must contain 240 B-scans, got {len(result)}.')
        return result
    rng = np.random.default_rng(EVAL_SEED)
    result = []
    for spec in ALLOWED_TISSUE_SPECS:
        tomogram = load_tomogram(os.path.join(VALIDATION_DATA_DIR, spec['filename']))
        if tomogram.ndim != 3 or tomogram.shape[2] < EVAL_N_BSCANS_PER_VOLUME:
            raise ValueError(f"{spec['filename']} does not contain 40 usable B-scans.")
        indices = rng.choice(tomogram.shape[2], size=EVAL_N_BSCANS_PER_VOLUME, replace=False)
        for index in sorted(int(value) for value in indices):
            result.append(dict(filename=spec['filename'], tissue=spec['label'], bscan_index=index))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(EVAL_BSCAN_IDS_PATH, 'w') as handle:
        json.dump(result, handle, indent=2)
    return result


def load_amp_phase_x0_model(cfg, model_device):
    folder = os.path.join(DIFFUSION_ROOT, cfg['folder'])
    candidates = [os.path.join(folder, name) for name in cfg['checkpoint_preference']]
    checkpoint_path = next((path for path in candidates if os.path.isfile(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"[{cfg['id']}] no checkpoint in {folder}; candidates={cfg['checkpoint_preference']}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=model_device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=model_device)
    metadata = checkpoint.get('metadata', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ('prediction_type', 'representation'):
        declared = checkpoint.get(key, metadata.get(key))
        if declared is not None and declared != cfg[key]:
            raise RuntimeError(f"[{cfg['id']}] incompatible checkpoint {key}={declared!r}.")
    model = UNET().to(model_device)
    weights = checkpoint['weights']
    if weights and all(key.startswith('module.') for key in weights):
        weights = {key[len('module.'):]: value for key, value in weights.items()}
    model.load_state_dict(weights)
    used_ema = False
    if checkpoint.get('ema') is not None:
        ema = ModelEmaV3(model, decay=0.9999)
        ema.load_state_dict(checkpoint['ema'])
        model = ema.module.to(model_device)
        used_ema = True
    model.eval()
    return model, used_ema, checkpoint_path


def sample_reverse_ddpm_x0_amp_phase(model, scheduler, condition_tensor, *, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_t = torch.randn(1, 2, 256, 256, device=device)
    with torch.no_grad():
        for t in range(999, 0, -1):
            t_tensor = torch.tensor([t], device=device)
            beta_t = scheduler.beta[t_tensor].view(1, 1, 1, 1)
            alpha_t = 1.0 - beta_t
            alpha_bar_t = scheduler.alpha[t_tensor].view(1, 1, 1, 1)
            alpha_bar_prev = scheduler.alpha[t_tensor - 1].view(1, 1, 1, 1)
            x0_pred = model(x_t, t_tensor, condition_tensor)
            if not torch.isfinite(x0_pred).all():
                raise RuntimeError(f'Non-finite x0 prediction at timestep {t}.')
            x0_pred = torch.cat((torch.clamp(x0_pred[:, 0:1], 0.0, 1.0),
                                 torch.clamp(x0_pred[:, 1:2], -1.0, 1.0)), dim=1)
            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            posterior_mean = ((torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)) * x0_pred
                              + (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) /
                                 (1.0 - alpha_bar_t)) * x_t)
            x_t = posterior_mean + torch.sqrt(torch.clamp(posterior_variance, min=1e-20)) * torch.randn_like(x_t)
            if not torch.isfinite(x_t).all():
                raise RuntimeError(f'Non-finite reverse-DDPM state at timestep {t}.')
        t_zero = torch.tensor([0], device=device)
        x0_final = model(x_t, t_zero, condition_tensor)
        if not torch.isfinite(x0_final).all():
            raise RuntimeError('Non-finite x0 prediction at timestep 0.')
        return torch.cat((torch.clamp(x0_final[:, 0:1], 0.0, 1.0),
                          torch.clamp(x0_final[:, 1:2], -1.0, 1.0)), dim=1)


def audit_figure(config_id, bscan_id, images):
    safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', bscan_id)
    output = os.path.join(OUT_DIR, 'audit_samples', config_id)
    os.makedirs(output, exist_ok=True)
    title = 'Reverse-DDPM sample from Gaussian noise; condition-only generation'
    input_db, gt_db, pred_db, input_phase, gt_phase, pred_phase = images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmin, vmax = gt_db.min(), gt_db.max()
    for axis, image, label in zip(axes, [input_db, gt_db, pred_db, abs(pred_db - gt_db)],
                                  ['Input amplitude dB', 'Ground-truth amplitude dB',
                                   'Reverse-DDPM predicted amplitude dB', 'Absolute amplitude-dB error']):
        axis.imshow(image, cmap='gray', vmin=vmin if 'error' not in label else None,
                    vmax=vmax if 'error' not in label else None)
        axis.set_title(label); axis.axis('off')
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(output, f'{safe_id}_amplitude.png'), dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    circular = np.angle(np.exp(1j * (pred_phase - gt_phase)))
    for axis, image, label in zip(axes, [input_phase, gt_phase, pred_phase, circular],
                                  ['Input phase', 'Ground-truth phase', 'Reverse-DDPM predicted phase', 'Circular phase error']):
        axis.imshow(image, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axis.set_title(label); axis.axis('off')
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(output, f'{safe_id}_phase.png'), dpi=120); plt.close(fig)


def evaluate_bscan(model, scheduler, target_roi, seed):
    input_roi = mirrorArtifact(target_roi)
    target_encoded, tgt_max, tgt_min, _, _ = encode_amplitude_phase(target_roi[np.newaxis, ...])
    input_encoded, in_max, in_min, _, _ = encode_amplitude_phase(input_roi[np.newaxis, ...])
    condition = np.stack([input_encoded[0][..., 0], input_encoded[0][..., 1]], axis=0)
    condition = torch.from_numpy(condition).unsqueeze(0).float().to(device)
    final = sample_reverse_ddpm_x0_amp_phase(model, scheduler, condition, seed=seed, device=device)
    channels = final.detach().cpu().numpy()
    pred_encoded = np.stack([channels[0, 0], channels[0, 1]], axis=-1)[np.newaxis, ...]
    pred_amp, pred_db, pred_phase, pred_intensity = decode_amplitude_phase(pred_encoded, tgt_max, tgt_min)
    gt_amp, gt_db, gt_phase, gt_intensity = decode_amplitude_phase(target_encoded, tgt_max, tgt_min)
    _, input_db, input_phase, _ = decode_amplitude_phase(input_encoded, in_max, in_min)
    pred_amp, pred_db, pred_phase, pred_intensity = pred_amp[0], pred_db[0], pred_phase[0], pred_intensity[0]
    gt_amp, gt_db, gt_phase, gt_intensity = gt_amp[0], gt_db[0], gt_phase[0], gt_intensity[0]
    input_db, input_phase = input_db[0], input_phase[0]

    # Complex phasor recomposition, used only for the PC-CGAN-equivalent phase metrics below.
    pred_cx = pred_amp * np.exp(1j * pred_phase)
    gt_cx = gt_amp * np.exp(1j * gt_phase)
    wpc_val = weighted_phase_coherence(pred_cx, gt_cx)
    ccc_val = masked_complex_coherence(pred_cx, gt_cx, percentile=50)
    pgssim_val, _, _ = phase_gradient_ssim(pred_cx, gt_cx, percentile=50)

    gt_min, gt_max = gt_db.min(), gt_db.max()
    pred_norm = np.clip((pred_db - gt_min) / (gt_max - gt_min + 1e-12), 0, 1)
    gt_norm = np.clip((gt_db - gt_min) / (gt_max - gt_min + 1e-12), 0, 1)
    pred_post, gt_post = despeckle_nlm(pred_norm), despeckle_nlm(gt_norm)
    scale = max(float(np.percentile(gt_intensity, 99.5)), 1e-12)
    pred_i = np.clip(pred_intensity / scale, 0, 1)
    gt_i = np.clip(gt_intensity / scale, 0, 1)

    pred_i_filtered = despeckle_nlm(pred_i, h=0.20, patch_size=5, patch_distance=6)
    gt_i_filtered = despeckle_nlm(gt_i, h=0.20, patch_size=5, patch_distance=6)

    ssim_post = calculate_ssim(pred_i_filtered, gt_i_filtered)
    psnr_post = calculate_psnr(pred_i_filtered, gt_i_filtered, max_val=1.0)
    mse_post = calculate_mse(pred_i_filtered, gt_i_filtered)
    hist_post = histogram_difference(pred_i_filtered, gt_i_filtered, method='cosine-similarity')

    delta = np.angle(np.exp(1j * (pred_phase - gt_phase)))
    mask = gt_amp > np.percentile(gt_amp, 50)
    if mask.sum() < 10:
        phase = dict.fromkeys(['phase_circular_mae_masked_rad', 'phase_circular_mae_masked_deg',
                               'phase_circular_rmse_masked_rad', 'phase_cosine_similarity_masked'], np.nan)
        print('Phase mask has fewer than 10 pixels.')
    else:
        selected = delta[mask]; mae = float(np.mean(np.abs(selected)))
        phase = dict(phase_circular_mae_masked_rad=mae, phase_circular_mae_masked_deg=float(np.degrees(mae)),
                     phase_circular_rmse_masked_rad=float(np.sqrt(np.mean(selected ** 2))),
                     phase_cosine_similarity_masked=float(np.mean(np.cos(selected))))
    metrics = dict(
        ssim=ssim_post, psnr=psnr_post, mse=mse_post, hist_cosine_similarity=hist_post,
        wpc=wpc_val, ccc=ccc_val, pgssim=pgssim_val,
        # Secondary diagnostics, not part of the PC-CGAN-equivalent primary tables.
        amplitude_ssim_pre_nlm=calculate_ssim(pred_norm, gt_norm), amplitude_psnr_pre_nlm=calculate_psnr(pred_norm, gt_norm),
        amplitude_mse_pre_nlm=calculate_mse(pred_norm, gt_norm), amplitude_hist_cosine_similarity_pre_nlm=histogram_difference(pred_norm, gt_norm, method='cosine-similarity'),
        amplitude_ssim_post_nlm=calculate_ssim(pred_post, gt_post), amplitude_psnr_post_nlm=calculate_psnr(pred_post, gt_post),
        amplitude_mse_post_nlm=calculate_mse(pred_post, gt_post), amplitude_hist_cosine_similarity_post_nlm=histogram_difference(pred_post, gt_post, method='cosine-similarity'),
        intensity_linear_ssim_pre_nlm=calculate_ssim(pred_i, gt_i), intensity_linear_psnr_pre_nlm=calculate_psnr(pred_i, gt_i),
        intensity_linear_mse_pre_nlm=calculate_mse(pred_i, gt_i), intensity_energy_ratio=float(pred_intensity.sum() / (gt_intensity.sum() + 1e-12)),
        phase_mask_fraction=float(mask.mean()), final_amplitude_fraction_at_0=float(np.mean(np.isclose(channels[0, 0], 0))),
        final_amplitude_fraction_at_1=float(np.mean(np.isclose(channels[0, 0], 1))), final_amplitude_fraction_saturated=float(np.mean((channels[0, 0] <= 0) | (channels[0, 0] >= 1))),
        final_phase_fraction_at_neg1=float(np.mean(np.isclose(channels[0, 1], -1))), final_phase_fraction_at_pos1=float(np.mean(np.isclose(channels[0, 1], 1))),
        final_phase_fraction_saturated=float(np.mean((channels[0, 1] <= -1) | (channels[0, 1] >= 1))),
    )
    metrics.update(phase); metrics['intensity_energy_relative_error'] = abs(metrics['intensity_energy_ratio'] - 1.0)
    return metrics, (input_db, gt_db, pred_db, input_phase, gt_phase, pred_phase)


def build_summary(df):
    rows = []
    for cfg in CONFIGS:
        group = df[df.config_id == cfg['id']]; row = dict(config_id=cfg['id'], config_name=cfg['name'])
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors='coerce').to_numpy(); values = values[np.isfinite(values)]
            row[f'{metric}_mean'] = float(np.mean(values)) if len(values) else np.nan
            row[f'{metric}_std'] = float(np.std(values)) if len(values) else np.nan
            row[f'{metric}_ci95_lower'], row[f'{metric}_ci95_upper'] = bootstrap_ci(values) if len(values) else (np.nan, np.nan)
        rows.append(row)
    result = pd.DataFrame(rows); result.to_csv(os.path.join(OUT_DIR, 'summary_metrics.csv'), index=False); return result


def build_wilcoxon(df):
    reference_id = 'D1'
    comparison_ids = ['D2', 'D3', 'D4']
    rows, pvalues = [], []
    for other_id in comparison_ids:
        for metric in METRIC_COLUMNS:
            a = df[df.config_id == other_id][['bscan_id', metric]].rename(columns={metric: 'a'})
            b = df[df.config_id == reference_id][['bscan_id', metric]].rename(columns={metric: 'b'})
            paired = a.merge(b, on='bscan_id'); valid = np.isfinite(paired.a) & np.isfinite(paired.b)
            x, y = paired.loc[valid, 'a'].to_numpy(), paired.loc[valid, 'b'].to_numpy(); status = 'ok'
            if len(x) < 10:
                statistic = pvalue = np.nan; status = 'insufficient_paired_samples'
            elif np.all(x - y == 0):
                statistic = pvalue = np.nan; status = 'all_paired_differences_zero'
            else:
                try: statistic, pvalue = wilcoxon(x, y, alternative='two-sided')
                except Exception: statistic = pvalue = np.nan; status = 'wilcoxon_failed'
            rows.append(dict(metric=metric, config_a=other_id, config_b=reference_id, n_pairs=len(x), statistic=statistic, p_value=pvalue, test_status=status))
            pvalues.append(pvalue if np.isfinite(pvalue) else np.nan)
    finite = [i for i, value in enumerate(pvalues) if np.isfinite(value)]
    adjusted = holm_bonferroni([pvalues[i] for i in finite]) if finite else []
    for index, row in enumerate(rows):
        row['p_value_holm_corrected'] = adjusted[finite.index(index)] if index in finite else np.nan
        row['significant_0.05'] = bool(np.isfinite(row['p_value_holm_corrected']) and row['p_value_holm_corrected'] < 0.05)
    columns = ['metric', 'config_a', 'config_b', 'n_pairs', 'statistic', 'p_value', 'p_value_holm_corrected', 'significant_0.05', 'test_status']
    result = pd.DataFrame(rows, columns=columns); result.to_csv(os.path.join(OUT_DIR, 'wilcoxon_vs_D1.csv'), index=False); return result


def write_manifest(eval_ids, loaded):
    manifest = dict(configurations=[dict(id=c['id'], name=c['name'], checkpoint_path=loaded[c['id']]['path'], used_ema=loaded[c['id']]['ema']) for c in CONFIGS],
                    dataset_allowlist=ALLOWED_TISSUE_SPECS, exclusions=['cadaver heart', 'pairCadaverhearth.npy', 'nail-bed OCTA', 'NailBedOCTA', 'OpticNerveRene.npy', 'opticNerveReneNoMc.npy'],
                    bscans_fixed=eval_ids, inference_seed=INFERENCE_SEED, sampler='ddpm_x0_posterior', num_time_steps=1000,
                    representation='log_amplitude_phase_normalized', channel_limits={'amplitude': [0, 1], 'phase': [-1, 1]},
                    target_used_inside_reverse_loop=False, teacher_forcing_used=False, phasor_recomposition_used=True, nlm_primary_metrics=True,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with open(os.path.join(OUT_DIR, 'evaluation_manifest.json'), 'w') as handle: json.dump(manifest, handle, indent=2)


def write_markdown(summary, tests, eval_ids, loaded, df):
    lines = ['# PC-CGAN-equivalent (M3) intensity and phase summary', '', '## Configuration', '']
    lines += [f"- {c['id']}: {c['name']}; checkpoint `{loaded[c['id']]['path']}`; EMA={loaded[c['id']]['ema']}" for c in CONFIGS]
    lines += ['', f'Exactly {len(eval_ids)} B-scans were evaluated: 40 from each of six authorized tissues.', 'Authorized tissues: ' + ', '.join(s['label'] for s in ALLOWED_TISSUE_SPECS) + '.', '', '## Mean metrics with CI95', '', '| config | metric | mean | CI95 |', '|---|---|---:|---:|']
    for _, row in summary.iterrows():
        for metric in METRIC_COLUMNS: lines.append(f"| {row.config_id} | {metric} | {row[f'{metric}_mean']:.6g} | [{row[f'{metric}_ci95_lower']:.6g}, {row[f'{metric}_ci95_upper']:.6g}] |")
    lines += ['', '## Wilcoxon vs D1 (Holm-Bonferroni corrected across all 7 metrics x 3 comparisons)', '', '| config_a | config_b | metric | n_pairs | statistic | p_value | Holm p | status |', '|---|---|---:|---:|---:|---:|---:|---|']
    for _, row in tests.iterrows(): lines.append(f'| {row.config_a} | {row.config_b} | {row.metric} | {row.n_pairs} | {row.statistic} | {row.p_value} | {row.p_value_holm_corrected} | {row.test_status} |')
    lines += ['', '## Interpretation limits', '', '1. Results are reverse-DDPM samples generated from Gaussian noise and conditioned only on the mirror-corrupted ROI.', '2. Ground truth is used only after sampling for matched decoding and evaluation.', '3. Intensity metrics (SSIM/PSNR/MSE/HistSim) are computed post-NLM despeckling, matching PC-CGAN (M3).', '4. Phase metrics (WPC/CCC/PG-SSIM) are computed on the recomposed complex phasor without filtering, matching PC-CGAN (M3).', '5. Statistical results describe paired B-scan comparisons and do not establish subject-level independence.']
    with open(os.path.join(OUT_DIR, 'amplitude_phase_intensity_summary.md'), 'w') as handle: handle.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_DIR, exist_ok=True); eval_ids = build_or_load_eval_bscan_ids()
    print(f'Evaluating exactly {len(eval_ids)} B-scans per configuration.')
    loaded, rows = {}, []
    for cfg in CONFIGS:
        model, ema, checkpoint_path = load_amp_phase_x0_model(cfg, device)
        loaded[cfg['id']] = dict(path=checkpoint_path, ema=ema); scheduler = DDPM_Scheduler(num_time_steps=1000, device=str(device))
        for global_index, entry in enumerate(eval_ids):
            tomogram = load_tomogram(os.path.join(VALIDATION_DATA_DIR, entry['filename']))
            target_roi = extract_center_roi(tomogram[:, :, entry['bscan_index']], roi_size=256)
            bscan_id = f"{entry['filename']}::{entry['bscan_index']}"; seed = INFERENCE_SEED + global_index
            metrics, images = evaluate_bscan(model, scheduler, target_roi, seed)
            rows.append(dict(bscan_id=bscan_id, tissue=entry['tissue'], source_filename=entry['filename'], bscan_index=entry['bscan_index'], config_id=cfg['id'], config_name=cfg['name'], checkpoint_path=checkpoint_path, used_ema=ema, inference_seed=seed, sampler_type='ddpm_x0_posterior', num_time_steps=1000, **metrics))
            if global_index < 2: audit_figure(cfg['id'], bscan_id, images)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS); df.to_csv(os.path.join(OUT_DIR, 'per_bscan_metrics.csv'), index=False, quoting=csv.QUOTE_MINIMAL)
    summary = build_summary(df); tests = build_wilcoxon(df); write_manifest(eval_ids, loaded); write_markdown(summary, tests, eval_ids, loaded, df)


if __name__ == '__main__':
    main()