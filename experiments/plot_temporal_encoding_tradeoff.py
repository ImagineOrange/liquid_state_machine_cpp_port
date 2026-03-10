#!/usr/bin/env python3
"""
Six-panel heatmap: per-bin classification accuracy across SFA parameter space.

Shows how adaptation trades off instantaneous encoding fidelity vs memory trace
persistence. Three time windows × two branches:

Row 1: Onset encoding (bins 0-6, 0–140 ms) — instantaneous discrimination
Row 2: Peak encoding (bins 7-14, 140–300 ms) — mid-stimulus
Row 3: Post-stimulus trace (bins 30-47, 600–960 ms) — memory persistence

Each cell shows the change in bin-window accuracy relative to the inc=0 baseline.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'temporal_encoding_tradeoff.png'

# Time windows (bin indices, each bin = 20 ms)
WINDOWS = [
    ('Onset encoding (0–140 ms)', range(0, 7)),
    ('Peak encoding (140–300 ms)', range(7, 15)),
    ('Post-stimulus trace (600–960 ms)', range(30, 48)),
]


def _format_tau_labels(tau_vals):
    return [f'{t/1000:.1f}k' if t >= 1000 else f'{t:.0f}' for t in tau_vals]


def _format_inc_labels(inc_vals):
    labels = []
    for v in inc_vals:
        if v == 0:
            labels.append('0')
        elif v < 0.01:
            labels.append(f'{v:.4f}')
        elif v < 0.1:
            labels.append(f'{v:.3f}')
        elif v < 1:
            labels.append(f'{v:.2f}')
        else:
            labels.append(f'{v:.1f}')
    return labels


def _draw_panel(ax, gap_pp, inc_vals, tau_vals, norm, cmap, title,
                show_xlabel=True, show_ylabel=True):
    im = ax.imshow(gap_pp, aspect='auto', origin='lower',
                   cmap=cmap, norm=norm, interpolation='nearest')

    n_inc, n_tau = gap_pp.shape

    ax.set_xticks(np.arange(len(tau_vals)))
    if show_xlabel:
        ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=7, rotation=45,
                           ha='right')
        ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=9, labelpad=4)
    else:
        ax.set_xticklabels([])

    ax.set_yticks(np.arange(len(inc_vals)))
    if show_ylabel:
        ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=7)
        ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=9, labelpad=4)
    else:
        ax.set_yticklabels([])

    # Cell annotations
    for i in range(n_inc):
        for j in range(n_tau):
            val = gap_pp[i, j]
            if np.isnan(val):
                continue
            norm_val = norm(val)
            text_color = 'white' if norm_val < 0.35 or norm_val > 0.85 else 'black'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=5.5, color=text_color, fontweight='medium')

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default=str(DEFAULT_INPUT))
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    # Extract per-bin accuracy into (n_inc, n_tau, n_bins) arrays per branch
    n_bins = len(data['grid_results'][0]['per_bin_accuracy'])

    bins_a = np.full((n_inc, n_tau, n_bins), np.nan)
    bins_b = np.full((n_inc, n_tau, n_bins), np.nan)

    for entry in data['grid_results']:
        ii, ti = entry['inc_idx'], entry['tau_idx']
        branch = entry.get('branch', 'A_unmatched')
        pba = np.array(entry['per_bin_accuracy'])
        if branch == 'A_unmatched':
            bins_a[ii, ti, :] = pba
        elif branch == 'B_matched':
            bins_b[ii, ti, :] = pba

    # Compute window-averaged accuracy for each branch
    # Then compute gap from inc=0 baseline (averaged over all tau at inc=0)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='#e0e0e0')

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('white')

    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.30, wspace=0.08,
                  left=0.06, right=0.92, bottom=0.06, top=0.93)

    for row_idx, (window_name, bin_range) in enumerate(WINDOWS):
        is_bottom = (row_idx == 2)
        ax_a = fig.add_subplot(gs[row_idx, 0])
        ax_b = fig.add_subplot(gs[row_idx, 1])
        ax_cb = fig.add_subplot(gs[row_idx, 2])
        ax_a.set_facecolor('white')
        ax_b.set_facecolor('white')

        bin_indices = list(bin_range)

        # Window-averaged accuracy (n_inc, n_tau)
        win_a = np.nanmean(bins_a[:, :, bin_indices], axis=2) * 100
        win_b = np.nanmean(bins_b[:, :, bin_indices], axis=2) * 100

        # Baseline: inc=0 row, averaged over all tau
        baseline_a = np.nanmean(win_a[0, :])
        baseline_b = np.nanmean(win_b[0, :])

        gap_a = win_a - baseline_a
        gap_b = win_b - baseline_b

        # Per-row color norm
        all_vals = np.concatenate([gap_a.ravel(), gap_b.ravel()])
        all_vals = all_vals[~np.isnan(all_vals)]
        abs_max = np.max(np.abs(all_vals)) if len(all_vals) > 0 else 1
        pad = max(abs_max * 0.15, 0.3)
        norm = TwoSlopeNorm(vmin=-(abs_max + pad), vcenter=0, vmax=abs_max + pad)

        im_a = _draw_panel(ax_a, gap_a, inc_vals, tau_vals, norm, cmap,
                           f'{window_name} — A: Unmatched',
                           show_xlabel=is_bottom, show_ylabel=True)
        im_b = _draw_panel(ax_b, gap_b, inc_vals, tau_vals, norm, cmap,
                           f'{window_name} — B: Rate-matched',
                           show_xlabel=is_bottom, show_ylabel=False)

        # Baseline annotation
        ax_a.annotate(
            f'inc=0 baseline: {baseline_a:.1f}%',
            xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=7, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))
        ax_b.annotate(
            f'inc=0 baseline: {baseline_b:.1f}%',
            xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=7, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

        cbar = fig.colorbar(im_b, cax=ax_cb)
        cbar.set_label(r'$\Delta$ acc. from inc=0 (pp)', fontsize=9, labelpad=8)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    # Legend
    better = mpatches.Patch(facecolor='#4a90d9', edgecolor='none',
                            label='SFA improves encoding in this window')
    worse = mpatches.Patch(facecolor='#d9534f', edgecolor='none',
                           label='SFA degrades encoding in this window')
    fig.legend(handles=[better, worse],
               loc='lower center', bbox_to_anchor=(0.48, 0.005), ncol=2,
               fontsize=10, framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        'Encoding–memory tradeoff: per-bin classification accuracy across SFA parameter space\n'
        'Onset encoding degrades with SFA; post-stimulus memory trace strengthens',
        fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
