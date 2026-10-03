"""Statistics for the evaluation: bootstrap CIs, Holm-Bonferroni, paired Wilcoxon tests."""
from __future__ import annotations

import traceback
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from octmirror.metrics import METRIC_COLUMNS

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42
REFERENCE_CONFIG = "M3_PCCGAN_Reference"


def bootstrap_ci(data, n_boot: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean (non-finite values are dropped)."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan, np.nan
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return float(lower), float(upper)


def holm_bonferroni(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down correction.

    P-values are sorted ascending, each is multiplied by (n - rank), capped at 1, and
    monotonicity is enforced with a running maximum.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvalues[idx] * (n - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def wilcoxon_vs_reference(per_bscan_df: pd.DataFrame,
                          config_names: Sequence[str],
                          reference_config: str = REFERENCE_CONFIG,
                          checkpoint_type: str = 'G_best',
                          metric_columns: Sequence[str] = METRIC_COLUMNS,
                          log_error: Callable[[str], None] = print) -> pd.DataFrame:
    """Paired Wilcoxon tests of every configuration against the reference, per metric.

    B-scans are paired on ``(tomogram_file, bscan_index)``. Holm-Bonferroni correction
    is applied across all resulting p-values as a single family (7 metrics x 6
    comparisons = 42 tests for M2-M8 vs M3). Failed tests get NaN statistics and enter
    the correction with p = 1.

    Returns:
        DataFrame with columns ``metric, config_comparado, statistic, p_value,
        p_value_holm_corrected, significant_0.05``.
    """
    metric_columns = list(metric_columns)
    ref_df = per_bscan_df[
        (per_bscan_df['config'] == reference_config) &
        (per_bscan_df['checkpoint_type'] == checkpoint_type)
    ][['tomogram_file', 'bscan_index'] + metric_columns]

    rows = []
    raw_pvalues = []
    row_refs = []

    for config_name in config_names:
        if config_name == reference_config:
            continue
        try:
            other_df = per_bscan_df[
                (per_bscan_df['config'] == config_name) &
                (per_bscan_df['checkpoint_type'] == checkpoint_type)
            ][['tomogram_file', 'bscan_index'] + metric_columns]

            merged = pd.merge(
                ref_df, other_df, on=['tomogram_file', 'bscan_index'],
                suffixes=('_ref', '_other')
            )

            for metric in metric_columns:
                x = merged[f'{metric}_ref'].values
                y = merged[f'{metric}_other'].values
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                try:
                    statistic, p_value = wilcoxon(x, y)
                except Exception as ex_test:
                    statistic, p_value = np.nan, np.nan
                    log_error(f"Wilcoxon failed for metric={metric}, config={config_name}: {ex_test}")

                row = dict(metric=metric, config_comparado=config_name,
                           statistic=statistic, p_value=p_value)
                rows.append(row)
                raw_pvalues.append(p_value if np.isfinite(p_value) else 1.0)
                row_refs.append(row)
        except Exception as ex:
            log_error(f"[{config_name}] wilcoxon comparison failed: {ex}\n{traceback.format_exc(limit=5)}")

    adjusted = holm_bonferroni(raw_pvalues) if raw_pvalues else []
    for row, adj_p in zip(row_refs, adjusted):
        row['p_value_holm_corrected'] = adj_p
        row['significant_0.05'] = bool(adj_p < 0.05)

    return pd.DataFrame(rows, columns=[
        'metric', 'config_comparado', 'statistic', 'p_value',
        'p_value_holm_corrected', 'significant_0.05'
    ])
