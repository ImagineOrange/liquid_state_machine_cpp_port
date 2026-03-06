#!/usr/bin/env python3
"""
Generate figures from classification adaptation sweep results (C++ or Python).

Usage: python3 plot_results.py <results_json> <output_dir>
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_results.py <results_json> <output_dir>")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2]

    with open(results_path) as f:
        data = json.load(f)

    grid = data['grid']
    UNIFIED_INC = np.array(grid['unified_inc'])
    UNIFIED_TAU = np.array(grid['unified_tau'])
    n_inc = len(UNIFIED_INC)
    n_tau = len(UNIFIED_TAU)

    grid_results = data['grid_results']
    bsa_baseline = data.get('bsa_baseline', {})
    baseline_result = data.get('lhs021_baseline', {})

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'savefig.facecolor': '#1a1a2e',
    })

    # Build 2D matrices
    acc_mat = np.full((n_inc, n_tau), np.nan)
    gap_mat = np.full((n_inc, n_tau), np.nan)
    rate_mat = np.full((n_inc, n_tau), np.nan)
    stim_mat = np.full((n_inc, n_tau), np.nan)
    isi_cv_mat = np.full((n_inc, n_tau), np.nan)
    adapt_mat = np.full((n_inc, n_tau), np.nan)
    pr_mat = np.full((n_inc, n_tau), np.nan)

    for r in grid_results:
        i = r.get('inc_idx')
        t = r.get('tau_idx')
        if i is None or t is None or i < 0 or t < 0:
            continue
        acc_mat[i, t] = r['classification_accuracy']
        gap_mat[i, t] = r.get('classification_gap_pp', np.nan)
        rate_mat[i, t] = r.get('firing_rate_hz', np.nan)
        stim_mat[i, t] = r.get('matched_stimulus_current', np.nan)
        isi_cv_mean = r.get('isi_cv_mean')
        isi_cv_mat[i, t] = isi_cv_mean if isi_cv_mean is not None else np.nan
        adapt_mat[i, t] = r.get('adapt_at_stim_end_mean', np.nan)
        pr_mat[i, t] = r.get('participation_ratio_mean', np.nan)

    bsa_acc = bsa_baseline.get('accuracy', 0.2) * 100
    chance = 100.0 / 5

    def _format_inc(v):
        return f'{v:.3f}' if v < 0.1 else f'{v:.2f}'

    def _format_tau(v):
        return f'{v/1000:.1f}k' if v >= 1000 else f'{v:.0f}'

    # --- Figure 1: Accuracy heatmap ---
    fig, ax = plt.subplots(figsize=(14, 8))
    measured_mask = ~np.isnan(acc_mat)
    im = ax.imshow(acc_mat * 100, origin='lower', aspect='auto',
                    cmap='magma', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Accuracy (%)', fontsize=10)

    measured_pct = np.where(measured_mask, acc_mat * 100, np.nan)
    if not np.all(np.isnan(measured_pct)):
        best_flat = np.nanargmax(measured_pct)
        bi, bt = np.unravel_index(best_flat, measured_pct.shape)
        ax.plot(bt, bi, marker='*', color='cyan', markersize=18,
                markeredgecolor='white', markeredgewidth=1.0, zorder=5)
        best_val = measured_pct[bi, bt]
        best_inc = UNIFIED_INC[bi]
        best_tau = UNIFIED_TAU[bt]
        ax.set_title(f'Classification Accuracy\n'
                     f'best={best_val:.1f}% @ inc={best_inc:.3f}, tau={best_tau:.0f}  |  '
                     f'BSA={bsa_acc:.1f}%, chance={chance:.0f}%  |  '
                     f'{int(np.sum(measured_mask))} points')

    for i in range(n_inc):
        for j in range(n_tau):
            if np.isnan(acc_mat[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    hatch='///', edgecolor='gray', alpha=0.3,
                    linewidth=0, zorder=2))

    ax.set_xticks(range(n_tau))
    ax.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_inc))
    ax.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=8)
    ax.set_xlabel(r'$\tau_{adapt}$ (ms)')
    ax.set_ylabel('adapt_inc')
    fig.tight_layout()
    p = f'{output_dir}/accuracy_heatmap.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Figure 2: Neural dynamics landscape (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax_dyn, mat, cmap, clabel, title in [
        (axes[0, 0], isi_cv_mat, 'viridis', 'ISI CV',
         'ISI Coefficient of Variation (stimulus)'),
        (axes[0, 1], adapt_mat, 'inferno', 'Mean g_adapt',
         'Adaptation Conductance at Stimulus End'),
        (axes[1, 0], pr_mat, 'cividis', 'Participation Ratio',
         'Population Dimensionality (stimulus)'),
        (axes[1, 1], stim_mat, 'plasma', 'stimulus_current',
         'Drive Cost (stimulus_current to rate-match)'),
    ]:
        im = ax_dyn.imshow(mat, origin='lower', aspect='auto',
                           cmap=cmap, interpolation='nearest')
        div = make_axes_locatable(ax_dyn)
        cx = div.append_axes('right', size='5%', pad=0.08)
        fig.colorbar(im, cax=cx).set_label(clabel, fontsize=10)
        ax_dyn.set_xticks(range(n_tau))
        ax_dyn.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                               rotation=45, ha='right', fontsize=7)
        ax_dyn.set_yticks(range(n_inc))
        ax_dyn.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=7)
        ax_dyn.set_xlabel(r'$\tau_{adapt}$ (ms)')
        ax_dyn.set_ylabel('adapt_inc')
        ax_dyn.set_title(title)

    fig.suptitle('Neural Dynamics Landscape (Classification)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p = f'{output_dir}/dynamics_landscape.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Figure 3: Rate-matching verification ---
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(rate_mat, origin='lower', aspect='auto',
                    cmap='viridis', interpolation='nearest')
    div = make_axes_locatable(ax)
    cx = div.append_axes('right', size='5%', pad=0.08)
    fig.colorbar(im, cax=cx).set_label('Mean Firing Rate (Hz)')
    ax.set_xticks(range(n_tau))
    ax.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_inc))
    ax.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=8)
    ax.set_xlabel(r'$\tau_{adapt}$ (ms)')
    ax.set_ylabel('adapt_inc')
    ax.set_title('Rate Verification (should be uniform)')
    fig.tight_layout()
    p = f'{output_dir}/rate_matching.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


if __name__ == '__main__':
    main()
