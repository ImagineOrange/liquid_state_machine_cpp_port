#!/usr/bin/env python3
"""
Two-panel figure showing top-10% parameter regions for CLS, WM, and XOR tasks.

Left panel:  Branch A (natural rate)
Right panel: Branch B (rate-matched)

Color-coded by task, with overlaps shown. Non-top-10% cells grey/hatched.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WM  = CPP_DIR / 'results' / 'wm_adaptation_sweep' / 'wm_adaptation_sweep.json'
DEFAULT_CLS = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'wm_xor_top_regions.png'

# Task colors (chosen for perceptual distinctness)
COL_CLS  = '#4daf4a'   # green  — classification
COL_WM   = '#2166ac'   # blue   — working memory
COL_XOR  = '#b2182b'   # red    — temporal XOR
COL_GREY = '#e8e8e8'   # non-top cells
COL_GRID = '#cccccc'

# Precompute all 8 combination colors (3-bit: CLS=bit0, WM=bit1, XOR=bit2)
def _blend_colors(*hex_colors):
    """Average RGB of given colors."""
    rgbs = [to_rgb(c) for c in hex_colors]
    return tuple(np.mean(rgbs, axis=0))

_COMBO_COLORS = {
    0:             COL_GREY,                              # none
    1:             COL_CLS,                               # CLS only
    2:             COL_WM,                                # WM only
    3:             _blend_colors(COL_CLS, COL_WM),        # CLS + WM
    4:             COL_XOR,                               # XOR only
    5:             _blend_colors(COL_CLS, COL_XOR),       # CLS + XOR
    6:             _blend_colors(COL_WM, COL_XOR),        # WM + XOR
    7:             _blend_colors(COL_CLS, COL_WM, COL_XOR),  # all three
}


def _format_tau(tau_vals):
    return [f'{t/1000:.1f}k' if t >= 1000 else f'{t:.0f}' for t in tau_vals]


def _format_inc(inc_vals):
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


def _top_mask(acc_grid, percentile=90):
    """Boolean mask: True for cells >= the percentile threshold (top 10%)."""
    valid = acc_grid[~np.isnan(acc_grid)]
    if len(valid) == 0:
        return np.full_like(acc_grid, False, dtype=bool)
    threshold = np.percentile(valid, percentile)
    return acc_grid >= threshold


def _draw_panel(ax, cls_acc, wm_acc, xor_acc, cls_top, wm_top, xor_top,
                inc_vals, tau_vals, title, show_ylabel=True):
    n_inc, n_tau = wm_acc.shape

    # Build RGB image from combination codes
    img = np.zeros((n_inc, n_tau, 3))
    cat = np.zeros((n_inc, n_tau), dtype=int)
    for i in range(n_inc):
        for j in range(n_tau):
            code = 0
            if cls_top[i, j]: code |= 1
            if wm_top[i, j]:  code |= 2
            if xor_top[i, j]: code |= 4
            cat[i, j] = code
            c = _COMBO_COLORS[code]
            img[i, j] = to_rgb(c) if isinstance(c, str) else c

    ax.imshow(img, aspect='auto', origin='lower', interpolation='nearest')

    # Hatch non-top cells
    for i in range(n_inc):
        for j in range(n_tau):
            if cat[i, j] == 0:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, hatch='xxx',
                                     edgecolor='#bbbbbb', linewidth=0.3,
                                     zorder=3)
                ax.add_patch(rect)

    # Annotate top cells with accuracy values (compact to avoid overlap)
    for i in range(n_inc):
        for j in range(n_tau):
            if cat[i, j] == 0:
                continue
            lines = []
            if cls_top[i, j]:
                lines.append(f'C{cls_acc[i,j]*100:.0f}')
            if wm_top[i, j]:
                lines.append(f'W{wm_acc[i,j]*100:.0f}')
            if xor_top[i, j]:
                lines.append(f'X{xor_acc[i,j]*100:.0f}')
            n_lines = len(lines)
            fs = {1: 5.5, 2: 4.5, 3: 3.8}[n_lines]
            txt = '\n'.join(lines)
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=fs, color='white', fontweight='bold',
                    linespacing=0.9)

    # Draw borders around contiguous top-10% clusters per task
    for mask, color in [(cls_top, COL_CLS), (wm_top, COL_WM), (xor_top, COL_XOR)]:
        for i in range(n_inc):
            for j in range(n_tau):
                if not mask[i, j]:
                    continue
                x0, y0 = j - 0.5, i - 0.5
                lw = 1.2
                if j == 0 or not mask[i, j - 1]:
                    ax.plot([x0, x0], [y0, y0 + 1], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=6)
                if j == n_tau - 1 or not mask[i, j + 1]:
                    ax.plot([x0 + 1, x0 + 1], [y0, y0 + 1], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=6)
                if i == 0 or not mask[i - 1, j]:
                    ax.plot([x0, x0 + 1], [y0, y0], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=6)
                if i == n_inc - 1 or not mask[i + 1, j]:
                    ax.plot([x0, x0 + 1], [y0 + 1, y0 + 1], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=6)

    # Axes
    ax.set_xticks(np.arange(len(tau_vals)))
    ax.set_xticklabels(_format_tau(tau_vals), fontsize=7, rotation=45, ha='right')
    ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=9, labelpad=4)

    ax.set_yticks(np.arange(len(inc_vals)))
    if show_ylabel:
        ax.set_yticklabels(_format_inc(inc_vals), fontsize=7)
        ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=9, labelpad=4)
    else:
        ax.set_yticklabels([])

    # Grid
    ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.5)
    ax.tick_params(which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_color(COL_GRID)
        spine.set_linewidth(0.8)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)


def main():
    parser = argparse.ArgumentParser(description='CLS/WM/XOR top-10% region map')
    parser.add_argument('--wm-input', type=str, default=str(DEFAULT_WM))
    parser.add_argument('--cls-input', type=str, default=str(DEFAULT_CLS))
    parser.add_argument('-o', '--output', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--percentile', type=float, default=90,
                        help='Percentile threshold for "top" region (default: 90 = top 10%%)')
    args = parser.parse_args()

    # Load WM/XOR data
    with open(args.wm_input) as f:
        wm_data = json.load(f)

    inc_vals = np.array(wm_data['grid']['unified_inc'])
    tau_vals = np.array(wm_data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    wm_acc_a  = np.full((n_inc, n_tau), np.nan)
    wm_acc_b  = np.full((n_inc, n_tau), np.nan)
    xor_acc_a = np.full((n_inc, n_tau), np.nan)
    xor_acc_b = np.full((n_inc, n_tau), np.nan)

    for entry in wm_data['grid_results']:
        ii, ti = entry['inc_idx'], entry['tau_idx']
        if entry['branch'] == 'A_unmatched':
            wm_acc_a[ii, ti]  = entry['wm_accuracy']
            xor_acc_a[ii, ti] = entry['xor_accuracy']
        elif entry['branch'] == 'B_matched':
            wm_acc_b[ii, ti]  = entry['wm_accuracy']
            xor_acc_b[ii, ti] = entry['xor_accuracy']

    # Load classification data
    with open(args.cls_input) as f:
        cls_data = json.load(f)

    cls_acc_a = np.full((n_inc, n_tau), np.nan)
    cls_acc_b = np.full((n_inc, n_tau), np.nan)

    for entry in cls_data['grid_results']:
        ii, ti = entry['inc_idx'], entry['tau_idx']
        if entry.get('branch', 'A_unmatched') == 'A_unmatched':
            cls_acc_a[ii, ti] = entry['classification_accuracy']
        elif entry['branch'] == 'B_matched':
            cls_acc_b[ii, ti] = entry['classification_accuracy']

    # Top-10% masks
    cls_top_a = _top_mask(cls_acc_a, args.percentile)
    cls_top_b = _top_mask(cls_acc_b, args.percentile)
    wm_top_a  = _top_mask(wm_acc_a,  args.percentile)
    wm_top_b  = _top_mask(wm_acc_b,  args.percentile)
    xor_top_a = _top_mask(xor_acc_a, args.percentile)
    xor_top_b = _top_mask(xor_acc_b, args.percentile)

    # Stats
    pct_label = f'top {100 - args.percentile:.0f}%'
    for name, ct, wt, xt in [('A', cls_top_a, wm_top_a, xor_top_a),
                              ('B', cls_top_b, wm_top_b, xor_top_b)]:
        all3 = np.sum(ct & wt & xt)
        print(f'Branch {name}: CLS {pct_label}={np.sum(ct)}, '
              f'WM {pct_label}={np.sum(wt)}, XOR {pct_label}={np.sum(xt)}, '
              f'all-3 overlap={all3}')

    # Figure
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 2, figure=fig, wspace=0.08, left=0.06, right=0.94,
                  bottom=0.15, top=0.88)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.set_facecolor('white')
    ax_b.set_facecolor('white')

    _draw_panel(ax_a, cls_acc_a, wm_acc_a, xor_acc_a,
                cls_top_a, wm_top_a, xor_top_a,
                inc_vals, tau_vals,
                f'A: Natural firing rate — {pct_label} regions',
                show_ylabel=True)
    _draw_panel(ax_b, cls_acc_b, wm_acc_b, xor_acc_b,
                cls_top_b, wm_top_b, xor_top_b,
                inc_vals, tau_vals,
                f'B: Rate-matched (20 Hz) — {pct_label} regions',
                show_ylabel=False)

    # Baselines as subtitle
    bsa_wm  = wm_data['bsa_wm_baseline']['accuracy']
    bsa_xor = wm_data['bsa_xor_baseline']['accuracy']
    bsa_cls = cls_data['bsa_baseline']['accuracy']

    baseline_text = (
        f'BSA baselines — CLS: {bsa_cls*100:.1f}%,  WM: {bsa_wm*100:.1f}%,  '
        f'XOR: {bsa_xor*100:.1f}%'
    )
    fig.text(0.5, 0.91, baseline_text, ha='center', va='bottom',
             fontsize=9, color='#888888')

    # Legend — single tasks + pairwise + triple
    cls_patch  = mpatches.Patch(facecolor=COL_CLS,  edgecolor='none',
                                label=f'Classification {pct_label}')
    wm_patch   = mpatches.Patch(facecolor=COL_WM,   edgecolor='none',
                                label=f'Working memory {pct_label}')
    xor_patch  = mpatches.Patch(facecolor=COL_XOR,  edgecolor='none',
                                label=f'Temporal XOR {pct_label}')
    cw_patch   = mpatches.Patch(facecolor=_COMBO_COLORS[3], edgecolor='none',
                                label='CLS + WM')
    cx_patch   = mpatches.Patch(facecolor=_COMBO_COLORS[5], edgecolor='none',
                                label='CLS + XOR')
    wx_patch   = mpatches.Patch(facecolor=_COMBO_COLORS[6], edgecolor='none',
                                label='WM + XOR')
    all_patch  = mpatches.Patch(facecolor=_COMBO_COLORS[7], edgecolor='none',
                                label='All three')
    grey_patch = mpatches.Patch(facecolor=COL_GREY, edgecolor=COL_GRID,
                                hatch='xxx', label='Below threshold')
    fig.legend(handles=[cls_patch, wm_patch, xor_patch,
                        cw_patch, cx_patch, wx_patch, all_patch, grey_patch],
               loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4,
               fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        f'Triple dissociation: optimal adaptation regimes ({pct_label} of parameter space)',
        fontsize=14, fontweight='bold', y=0.95)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
