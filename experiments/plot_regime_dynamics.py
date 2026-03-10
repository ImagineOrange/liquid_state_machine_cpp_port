#!/usr/bin/env python3
"""
Six-panel small-multiples figure: dynamics–accuracy correlation by adaptation regime.

Rows = Δa terciles (Low / Mid / High), Columns = ISI CV / Participation Ratio.
Each panel shows a single regime's per-bin Pearson r curve with the pooled
curve as a faint dashed reference. This layout avoids the confusion of
overlapping regime curves on shared axes.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'regime_dynamics.png'

BIN_MS = 20.0


def _compute_correlations(entries, n_bins):
    """Compute per-bin Pearson r (ISI CV and PR vs bin accuracy) with Fisher z CIs."""
    isi_cvs = np.array([e['isi_cv_mean'] for e in entries])
    prs = np.array([e['participation_ratio_mean'] for e in entries])
    n = len(entries)

    r_isi = np.full(n_bins, np.nan)
    r_pr = np.full(n_bins, np.nan)
    ci_isi_lo = np.full(n_bins, np.nan)
    ci_isi_hi = np.full(n_bins, np.nan)
    ci_pr_lo = np.full(n_bins, np.nan)
    ci_pr_hi = np.full(n_bins, np.nan)

    if n < 5:
        return r_isi, r_pr, ci_isi_lo, ci_isi_hi, ci_pr_lo, ci_pr_hi

    se = 1.0 / np.sqrt(max(n - 3, 1))
    for b in range(n_bins):
        bin_accs = np.array([e['per_bin_accuracy'][b] for e in entries])
        if np.std(bin_accs) < 1e-10 or np.std(isi_cvs) < 1e-10:
            continue

        r, _ = stats.pearsonr(isi_cvs, bin_accs)
        r_isi[b] = r
        z = np.arctanh(np.clip(r, -0.9999, 0.9999))
        ci_isi_lo[b] = np.tanh(z - 1.96 * se)
        ci_isi_hi[b] = np.tanh(z + 1.96 * se)

        if np.std(prs) < 1e-10:
            continue
        r, _ = stats.pearsonr(prs, bin_accs)
        r_pr[b] = r
        z = np.arctanh(np.clip(r, -0.9999, 0.9999))
        ci_pr_lo[b] = np.tanh(z - 1.96 * se)
        ci_pr_hi[b] = np.tanh(z + 1.96 * se)

    return r_isi, r_pr, ci_isi_lo, ci_isi_hi, ci_pr_lo, ci_pr_hi


def _find_crossover(r_vals, bin_times, min_bin=3):
    """Find first positive→negative zero-crossing after min_bin, via linear interpolation.

    Skips early bins to avoid spurious crossings from noise at trial onset.
    """
    for i in range(max(1, min_bin), len(r_vals)):
        if np.isnan(r_vals[i-1]) or np.isnan(r_vals[i]):
            continue
        if r_vals[i-1] > 0 and r_vals[i] <= 0:
            t0, t1 = bin_times[i-1], bin_times[i]
            r0, r1 = r_vals[i-1], r_vals[i]
            if abs(r1 - r0) < 1e-10:
                return t0
            return t0 + (0 - r0) / (r1 - r0) * (t1 - t0)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default=str(DEFAULT_INPUT))
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--branch', '-b', type=str, default='B_matched',
                        choices=['A_unmatched', 'B_matched'])
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    n_bins = len(data['grid_results'][0]['per_bin_accuracy'])
    bin_times = np.arange(n_bins) * BIN_MS + BIN_MS / 2

    inc_vals = sorted(set(e['adapt_inc'] for e in data['grid_results'] if e['inc_idx'] > 0))

    # Split into terciles by adapt_inc
    inc_arr = np.array(inc_vals)
    t1, t2 = np.percentile(inc_arr, [33.3, 66.7])

    regime_defs = [
        ('Low Δa', lambda e: e['adapt_inc'] <= t1),
        ('Mid Δa', lambda e: t1 < e['adapt_inc'] <= t2),
        ('High Δa', lambda e: e['adapt_inc'] > t2),
    ]

    # Also include pooled for reference
    all_entries = [e for e in data['grid_results']
                   if e['branch'] == args.branch and e['inc_idx'] > 0]

    regime_colors = {
        'Low Δa': '#2ca02c',
        'Mid Δa': '#ff7f0e',
        'High Δa': '#d62728',
        'Pooled': '#888888',
    }

    # Compute correlations per regime
    regime_results = {}
    for name, filt in regime_defs:
        entries = [e for e in all_entries if filt(e)]
        r_isi, r_pr, ci_isi_lo, ci_isi_hi, ci_pr_lo, ci_pr_hi = \
            _compute_correlations(entries, n_bins)
        regime_results[name] = {
            'n': len(entries),
            'r_isi': r_isi, 'r_pr': r_pr,
            'ci_isi_lo': ci_isi_lo, 'ci_isi_hi': ci_isi_hi,
            'ci_pr_lo': ci_pr_lo, 'ci_pr_hi': ci_pr_hi,
            'inc_range': (min(e['adapt_inc'] for e in entries),
                          max(e['adapt_inc'] for e in entries)),
        }

    # Pooled
    r_isi, r_pr, ci_isi_lo, ci_isi_hi, ci_pr_lo, ci_pr_hi = \
        _compute_correlations(all_entries, n_bins)
    regime_results['Pooled'] = {
        'n': len(all_entries),
        'r_isi': r_isi, 'r_pr': r_pr,
        'ci_isi_lo': ci_isi_lo, 'ci_isi_hi': ci_isi_hi,
        'ci_pr_lo': ci_pr_lo, 'ci_pr_hi': ci_pr_hi,
    }

    # --- Small-multiples figure: 3 rows (regimes) × 2 cols (metrics) ---
    metrics = [
        ('ISI CV', 'r_isi', 'ci_isi_lo', 'ci_isi_hi'),
        ('Participation Ratio', 'r_pr', 'ci_pr_lo', 'ci_pr_hi'),
    ]
    regime_order = ['Low Δa', 'Mid Δa', 'High Δa']

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.25, wspace=0.18,
                  left=0.08, right=0.96, bottom=0.07, top=0.91)

    panel_idx = 0
    for row, regime_name in enumerate(regime_order):
        res = regime_results[regime_name]
        color = regime_colors[regime_name]
        lo, hi = res.get('inc_range', (0, 0))
        is_bottom = (row == 2)

        for col, (metric_name, r_key, ci_lo_key, ci_hi_key) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('white')

            # Pooled reference (faint dashed)
            pooled = regime_results['Pooled']
            ax.plot(bin_times, pooled[r_key], color='#bbbbbb',
                    linewidth=1.2, linestyle='--', alpha=0.6, zorder=3,
                    label='Pooled' if row == 0 else None)

            # This regime's curve
            ax.fill_between(bin_times, res[ci_lo_key], res[ci_hi_key],
                            alpha=0.15, color=color, linewidth=0)
            ax.plot(bin_times, res[r_key], color=color, linewidth=2.2,
                    zorder=5)

            # Crossover marker
            t_cross = _find_crossover(res[r_key], bin_times)
            if t_cross is not None:
                ax.plot(t_cross, 0, marker='D', markersize=7, color=color,
                        zorder=7, markeredgecolor='white', markeredgewidth=0.8)
                ax.annotate(f'{t_cross:.0f} ms',
                            xy=(t_cross, 0),
                            xytext=(t_cross + 60, 0.35 if res[r_key][0] > 0 else -0.35),
                            fontsize=8, color=color, fontweight='bold', ha='left',
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                            zorder=8)

            # Zero line
            ax.axhline(y=0, color='#888888', linewidth=0.8, linestyle='-', zorder=2)

            # Stimulus shading
            ax.axvspan(0, 300, alpha=0.06, color='#2ca02c', zorder=1)
            ax.axvspan(300, 960, alpha=0.04, color='#ff7f0e', zorder=1)

            # Region labels only on top row
            if row == 0:
                ax.text(150, 0.95, 'Stimulus', ha='center', va='top', fontsize=8,
                        color='#2ca02c', fontstyle='italic', alpha=0.8,
                        transform=ax.get_xaxis_transform())
                ax.text(630, 0.95, 'Post-stimulus', ha='center', va='top', fontsize=8,
                        color='#cc6600', fontstyle='italic', alpha=0.8,
                        transform=ax.get_xaxis_transform())

            # Row label on left column
            if col == 0:
                ax.set_ylabel(f'Pearson r', fontsize=10, labelpad=6)

            # Column title on top row
            if row == 0:
                ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)

            # Regime info box (top-left of each panel)
            ax.text(0.03, 0.95,
                    f'{regime_name}\nn = {res["n"]}\n'
                    f'Δa ∈ [{lo:.3f}, {hi:.3f}]',
                    transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.9))

            # X-axis label only on bottom row
            if is_bottom:
                ax.set_xlabel('Time (ms)', fontsize=10, labelpad=6)
            else:
                ax.set_xticklabels([])

            ax.set_xlim(0, n_bins * BIN_MS)
            ax.set_ylim(-1.0, 1.0)
            ax.set_yticks(np.arange(-1.0, 1.1, 0.5))
            ax.tick_params(labelsize=8)

            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(0.8)
            ax.grid(axis='y', alpha=0.2, linewidth=0.5)

            panel_idx += 1

    branch_label = 'B: Rate-matched' if args.branch == 'B_matched' else 'A: Unmatched'
    fig.suptitle(
        f'Dynamics–accuracy correlation by adaptation regime ({branch_label})\n'
        f'Each row = one Δa tercile; dashed gray = pooled reference',
        fontsize=14, fontweight='bold', y=0.97)

    fig.text(0.5, 0.02,
             'Per-bin Pearson r within Δa terciles; shaded bands = 95% Fisher z CI',
             ha='center', fontsize=9, color='#888888')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
