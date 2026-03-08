#!/usr/bin/env python3
"""
Publication-quality two-panel heatmap of classification accuracy across
adaptation increment x adaptation time-constant parameter space.

Panel A: Branch A (unmatched / natural firing rate)
Panel B: Branch B (tonic-conductance-matched to 20 Hz)

Usage:
  python experiments/plot_adaptation_heatmap.py [--input <path>] [--output <path>]

Default input:  results/classification_adaptation_sweep/classification_adaptation_sweep.json
Default output: results/classification_adaptation_sweep/adaptation_heatmap.pdf
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'adaptation_heatmap.pdf'


def load_sweep(path):
    with open(path) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc = len(inc_vals)
    n_tau = len(tau_vals)

    acc_a = np.full((n_inc, n_tau), np.nan)
    acc_b = np.full((n_inc, n_tau), np.nan)

    for entry in data['grid_results']:
        ii = entry['inc_idx']
        ti = entry['tau_idx']
        branch = entry.get('branch', 'A_unmatched')
        if branch == 'A_unmatched':
            acc_a[ii, ti] = entry['classification_accuracy']
        elif branch == 'B_matched':
            acc_b[ii, ti] = entry['classification_accuracy']

    return inc_vals, tau_vals, acc_a, acc_b, data


def _format_tau_labels(tau_vals):
    labels = []
    for t in tau_vals:
        if t >= 1000:
            labels.append(f'{t/1000:.1f}k')
        else:
            labels.append(f'{t:.0f}')
    return labels


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


def _draw_panel(ax, acc_pct, inc_vals, tau_vals, vmin, vmax, cmap, title):
    """Draw a single heatmap panel."""
    im = ax.imshow(acc_pct, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # X-axis: tau
    ax.set_xticks(np.arange(len(tau_vals)))
    ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=8, rotation=45, ha='right')
    ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=11, labelpad=6)

    # Y-axis: increment
    ax.set_yticks(np.arange(len(inc_vals)))
    ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=8)
    ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=11, labelpad=6)

    # Cell annotations
    n_inc, n_tau = acc_pct.shape
    for i in range(n_inc):
        for j in range(n_tau):
            val = acc_pct[i, j]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = 'white' if norm_val < 0.45 else 'black'
            fontsize = 6.5 if n_inc * n_tau > 200 else 7.5
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='medium')

    # Title
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Peak marker
    if not np.all(np.isnan(acc_pct)):
        peak_idx = np.unravel_index(np.nanargmax(acc_pct), acc_pct.shape)
        peak_val = acc_pct[peak_idx]
        peak_inc = inc_vals[peak_idx[0]]
        peak_tau = tau_vals[peak_idx[1]]
        ax.plot(peak_idx[1], peak_idx[0], marker='*', markersize=14,
                markeredgecolor='white', markerfacecolor='gold',
                markeredgewidth=1.0, zorder=5)
        ax.annotate(f'Peak: {peak_val:.1f}%\n'
                    rf'$\Delta_a$={peak_inc:.3f}, $\tau_a$={peak_tau:.0f}',
                    xy=(peak_idx[1], peak_idx[0]),
                    xytext=(12, 12), textcoords='offset points',
                    fontsize=8, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#999999', alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color='#666666',
                                   connectionstyle='arc3,rad=0.2'))

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', length=0)

    # Spines
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    return im


def make_heatmap(inc_vals, tau_vals, acc_a, acc_b, data, output_path):
    acc_a_pct = acc_a * 100
    acc_b_pct = acc_b * 100

    # Shared colorscale across both panels
    all_vals = np.concatenate([acc_a_pct.ravel(), acc_b_pct.ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        print('No data to plot.')
        return
    vmin = max(np.min(all_vals) - 2, 15)
    vmax = min(np.max(all_vals) + 2, 100)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#e0e0e0')

    # Two-panel figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    ax_a.set_facecolor('white')
    ax_b.set_facecolor('white')

    im_a = _draw_panel(ax_a, acc_a_pct, inc_vals, tau_vals, vmin, vmax, cmap,
                       'A: Natural firing rate (unmatched)')
    im_b = _draw_panel(ax_b, acc_b_pct, inc_vals, tau_vals, vmin, vmax, cmap,
                       'B: Rate-matched (20 Hz, tonic conductance)')

    # Shared colorbar
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.025, pad=0.06,
                        location='right')
    cbar.set_label('Classification accuracy (%)', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Chance level annotation (bottom-right of panel B)
    n_digits = len(data.get('digits', [0, 1, 2, 3, 4]))
    chance = 100.0 / n_digits
    ax_b.annotate(f'Chance = {chance:.0f}%', xy=(0.98, 0.02),
                  xycoords='axes fraction', ha='right', va='bottom',
                  fontsize=9, color='#666666',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#cccccc', alpha=0.9))

    # Completion stats (bottom-left of panel A)
    n_a = np.count_nonzero(~np.isnan(acc_a_pct))
    n_b = np.count_nonzero(~np.isnan(acc_b_pct))
    n_total = acc_a_pct.size
    ax_a.annotate(f'A: {n_a}/{n_total}  B: {n_b}/{n_total}',
                  xy=(0.02, 0.02), xycoords='axes fraction',
                  ha='left', va='bottom', fontsize=8, color='#999999',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#cccccc', alpha=0.9))

    # Suptitle
    fig.suptitle('Classification accuracy across adaptation parameter space',
                 fontsize=15, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.06, right=0.85, bottom=0.10, top=0.91, wspace=0.25)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {png_path}')

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Adaptation sweep heatmap')
    parser.add_argument('--input', '-i', type=str, default=str(DEFAULT_INPUT),
                        help='Path to sweep JSON')
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output path (PDF)')
    args = parser.parse_args()

    inc_vals, tau_vals, acc_a, acc_b, data = load_sweep(args.input)

    n_a = np.count_nonzero(~np.isnan(acc_a))
    n_b = np.count_nonzero(~np.isnan(acc_b))
    n_total = acc_a.size
    print(f'Grid: {len(inc_vals)} inc x {len(tau_vals)} tau = {n_total} points')
    print(f'Branch A (unmatched): {n_a}/{n_total}')
    print(f'Branch B (matched):   {n_b}/{n_total}')
    if n_a > 0:
        print(f'Branch A range: {np.nanmin(acc_a)*100:.1f}% - {np.nanmax(acc_a)*100:.1f}%')
    if n_b > 0:
        print(f'Branch B range: {np.nanmin(acc_b)*100:.1f}% - {np.nanmax(acc_b)*100:.1f}%')

    make_heatmap(inc_vals, tau_vals, acc_a, acc_b, data, args.output)


if __name__ == '__main__':
    main()
