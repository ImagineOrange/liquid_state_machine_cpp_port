#!/usr/bin/env python3
"""
Two-panel figure: per-bin Pearson correlation of dynamics with bin accuracy.

Panel A: ISI CV × per-bin accuracy correlation (r) across 48 bins
Panel B: Participation ratio × per-bin accuracy correlation (r) across 48 bins

Shows that both dynamics variables flip sign at stimulus offset (~320 ms),
revealing the encoding–memory tradeoff at the single-bin level.
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
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'dynamics_sign_flip.png'

BIN_MS = 20.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default=str(DEFAULT_INPUT))
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    n_bins = len(data['grid_results'][0]['per_bin_accuracy'])
    bin_times = np.arange(n_bins) * BIN_MS + BIN_MS / 2  # bin centers

    # Compute per-bin correlations for each branch
    results = {}
    for branch_name, label in [('A_unmatched', 'A: Unmatched'),
                                ('B_matched', 'B: Rate-matched')]:
        entries = [e for e in data['grid_results']
                   if e['branch'] == branch_name and e['inc_idx'] > 0]

        isi_cvs = np.array([e['isi_cv_mean'] for e in entries])
        prs = np.array([e['participation_ratio_mean'] for e in entries])

        r_isi = np.full(n_bins, np.nan)
        r_pr = np.full(n_bins, np.nan)
        ci_isi_lo = np.full(n_bins, np.nan)
        ci_isi_hi = np.full(n_bins, np.nan)
        ci_pr_lo = np.full(n_bins, np.nan)
        ci_pr_hi = np.full(n_bins, np.nan)

        n = len(entries)
        for b in range(n_bins):
            bin_accs = np.array([e['per_bin_accuracy'][b] for e in entries])

            r, _ = stats.pearsonr(isi_cvs, bin_accs)
            r_isi[b] = r
            # Fisher z CI
            z = np.arctanh(r)
            se = 1.0 / np.sqrt(n - 3)
            ci_isi_lo[b] = np.tanh(z - 1.96 * se)
            ci_isi_hi[b] = np.tanh(z + 1.96 * se)

            r, _ = stats.pearsonr(prs, bin_accs)
            r_pr[b] = r
            z = np.arctanh(r)
            ci_pr_lo[b] = np.tanh(z - 1.96 * se)
            ci_pr_hi[b] = np.tanh(z + 1.96 * se)

        results[branch_name] = {
            'label': label, 'r_isi': r_isi, 'r_pr': r_pr,
            'ci_isi_lo': ci_isi_lo, 'ci_isi_hi': ci_isi_hi,
            'ci_pr_lo': ci_pr_lo, 'ci_pr_hi': ci_pr_hi,
        }

    # --- Figure ---
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 2, figure=fig, wspace=0.22, left=0.07, right=0.97,
                  bottom=0.12, top=0.88)

    colors = {'A_unmatched': '#2c7bb6', 'B_matched': '#d7191c'}
    alphas = {'A_unmatched': 0.15, 'B_matched': 0.15}

    panels = [
        ('ISI CV', 'r_isi', 'ci_isi_lo', 'ci_isi_hi'),
        ('Participation Ratio', 'r_pr', 'ci_pr_lo', 'ci_pr_hi'),
    ]

    for col, (metric_name, r_key, ci_lo_key, ci_hi_key) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('white')

        for branch_name in ['A_unmatched', 'B_matched']:
            res = results[branch_name]
            color = colors[branch_name]

            ax.fill_between(bin_times, res[ci_lo_key], res[ci_hi_key],
                            alpha=alphas[branch_name], color=color, linewidth=0)
            ax.plot(bin_times, res[r_key], color=color, linewidth=2.0,
                    label=res['label'], zorder=5)

        # Zero line
        ax.axhline(y=0, color='#888888', linewidth=0.8, linestyle='-', zorder=2)

        # Stimulus region shading
        ax.axvspan(0, 300, alpha=0.06, color='#2ca02c', zorder=1)
        ax.axvspan(300, 960, alpha=0.04, color='#ff7f0e', zorder=1)

        # Region labels at top
        ax.text(150, 0.93, 'Stimulus', ha='center', va='top', fontsize=9,
                color='#2ca02c', fontstyle='italic', alpha=0.8,
                transform=ax.get_xaxis_transform())
        ax.text(630, 0.93, 'Post-stimulus', ha='center', va='top', fontsize=9,
                color='#cc6600', fontstyle='italic', alpha=0.8,
                transform=ax.get_xaxis_transform())

        # Crossover annotation
        r_b = results['B_matched'][r_key]
        cross_idx = None
        for i in range(1, n_bins):
            if r_b[i - 1] > 0 and r_b[i] <= 0:
                cross_idx = i
                break
            if r_b[i - 1] < 0 and r_b[i] >= 0:
                cross_idx = i
                break
        if cross_idx is not None:
            # Linear interpolation for crossover time
            t0 = bin_times[cross_idx - 1]
            t1 = bin_times[cross_idx]
            r0 = r_b[cross_idx - 1]
            r1 = r_b[cross_idx]
            t_cross = t0 + (0 - r0) / (r1 - r0) * (t1 - t0)
            ax.axvline(x=t_cross, color='#444444', linewidth=1.0,
                       linestyle='--', alpha=0.6, zorder=3)
            y_offset = 0.25 if col == 0 else -0.25
            ax.annotate(f'crossover ~{t_cross:.0f} ms',
                        xy=(t_cross, 0), xytext=(t_cross + 80, y_offset),
                        fontsize=8, color='#444444', ha='left',
                        arrowprops=dict(arrowstyle='->', color='#444444',
                                        lw=0.8))

        ax.set_xlabel('Time (ms)', fontsize=11, labelpad=6)
        ax.set_ylabel(f'Pearson r (bin accuracy ~ {metric_name})', fontsize=11,
                      labelpad=6)
        ax.set_title(f'{chr(65 + col)}: {metric_name}', fontsize=13,
                     fontweight='bold', pad=10)
        ax.set_xlim(0, n_bins * BIN_MS)
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks(np.arange(-1.0, 1.1, 0.25))
        ax.tick_params(labelsize=9)
        ax.legend(loc='lower left' if col == 0 else 'upper left',
                  fontsize=9, framealpha=0.9, edgecolor='#cccccc')

        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.8)

        ax.grid(axis='y', alpha=0.2, linewidth=0.5)

    fig.suptitle(
        'Dynamics–accuracy coupling flips at stimulus offset\n'
        'ISI CV and participation ratio have opposite effects on encoding vs memory',
        fontsize=14, fontweight='bold', y=0.97)

    fig.text(0.5, 0.02,
             'Per-bin Pearson r across 285 non-baseline grid cells per branch; '
             'shaded bands = 95% Fisher z CI',
             ha='center', fontsize=9, color='#888888')

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
