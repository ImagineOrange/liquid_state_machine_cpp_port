#!/usr/bin/env python3
"""
Two-panel heatmap: classification accuracy gap from BSA baseline.

Panel A: Branch A (unmatched / natural firing rate)
Panel B: Branch B (tonic-conductance-matched to 20 Hz)

Cells significantly better than BSA (one-sided Welch t, BH-FDR q=0.05)
are shown without hatching; non-significant cells are hatched.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'bsa_comparison.png'


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


def _welch_t_one_sided(cell_reps, bsa_mean, bsa_sample_std, bsa_n):
    """One-sided Welch t-test: H1 = cell > BSA."""
    cell_mean = np.mean(cell_reps)
    cell_std = np.std(cell_reps, ddof=1)
    cell_n = len(cell_reps)
    se = np.sqrt(bsa_sample_std**2 / bsa_n + cell_std**2 / cell_n)
    if se == 0:
        return np.nan, np.nan
    t_stat = (cell_mean - bsa_mean) / se
    num = (bsa_sample_std**2 / bsa_n + cell_std**2 / cell_n)**2
    denom = ((bsa_sample_std**2 / bsa_n)**2 / (bsa_n - 1) +
             (cell_std**2 / cell_n)**2 / (cell_n - 1))
    df = num / denom if denom > 0 else 1.0
    p_one = stats.t.sf(t_stat, df)
    return t_stat, p_one


def _benjamini_hochberg(p_values):
    p = np.array(p_values)
    n = len(p)
    sorted_idx = np.argsort(p)
    adjusted = np.empty(n)
    cummin = np.inf
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = p[sorted_idx[i]] * n / rank
        cummin = min(cummin, adj)
        adjusted[sorted_idx[i]] = cummin
    return np.clip(adjusted, 0, 1)


def _draw_panel(ax, gap_pp, sig_mask, inc_vals, tau_vals, norm, cmap, title):
    im = ax.imshow(gap_pp, aspect='auto', origin='lower',
                   cmap=cmap, norm=norm, interpolation='nearest')

    n_inc, n_tau = gap_pp.shape

    # Hatch non-significant cells
    for i in range(n_inc):
        for j in range(n_tau):
            if np.isnan(gap_pp[i, j]):
                continue
            if not sig_mask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, hatch='///',
                                     edgecolor='#666666', linewidth=0.5,
                                     zorder=3)
                ax.add_patch(rect)

    # X-axis
    ax.set_xticks(np.arange(len(tau_vals)))
    ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=8, rotation=45,
                       ha='right')
    ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=11,
                  labelpad=6)

    # Y-axis
    ax.set_yticks(np.arange(len(inc_vals)))
    ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=8)
    ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=11,
                  labelpad=6)

    # Cell annotations
    for i in range(n_inc):
        for j in range(n_tau):
            val = gap_pp[i, j]
            if np.isnan(val):
                continue
            norm_val = norm(val)
            text_color = 'white' if norm_val < 0.35 or norm_val > 0.85 else 'black'
            fontsize = 6.0 if n_inc * n_tau > 200 else 7.0
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='medium')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Cluster boundary around significant cells — draw border segments only
    # where a significant cell is adjacent to a non-significant cell or the
    # grid edge.  This produces a single contour around each connected cluster.
    border_color = '#1a1a1a'
    border_lw = 1.1
    for i in range(n_inc):
        for j in range(n_tau):
            if not sig_mask[i, j]:
                continue
            x0, y0 = j - 0.5, i - 0.5  # bottom-left of cell in imshow coords
            # left edge
            if j == 0 or not sig_mask[i, j - 1]:
                ax.plot([x0, x0], [y0, y0 + 1], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
            # right edge
            if j == n_tau - 1 or not sig_mask[i, j + 1]:
                ax.plot([x0 + 1, x0 + 1], [y0, y0 + 1], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
            # bottom edge
            if i == 0 or not sig_mask[i - 1, j]:
                ax.plot([x0, x0 + 1], [y0, y0], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
            # top edge
            if i == n_inc - 1 or not sig_mask[i + 1, j]:
                ax.plot([x0, x0 + 1], [y0 + 1, y0 + 1], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)

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
    parser = argparse.ArgumentParser(description='BSA comparison heatmap')
    parser.add_argument('--input', '-i', type=str, default=str(DEFAULT_INPUT))
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    bsa_mean = data['bsa_baseline']['accuracy']
    bsa_pop_std = data['bsa_baseline']['accuracy_std']
    bsa_n = 5
    bsa_sample_std = bsa_pop_std * np.sqrt(bsa_n / (bsa_n - 1))

    # Load per-repeat data
    reps_a = np.empty((n_inc, n_tau), dtype=object)
    reps_b = np.empty((n_inc, n_tau), dtype=object)
    acc_a = np.full((n_inc, n_tau), np.nan)
    acc_b = np.full((n_inc, n_tau), np.nan)

    for entry in data['grid_results']:
        ii, ti = entry['inc_idx'], entry['tau_idx']
        branch = entry.get('branch', 'A_unmatched')
        if branch == 'A_unmatched':
            reps_a[ii, ti] = entry.get('classification_per_repeat_accuracy')
            acc_a[ii, ti] = entry['classification_accuracy']
        elif branch == 'B_matched':
            reps_b[ii, ti] = entry.get('classification_per_repeat_accuracy')
            acc_b[ii, ti] = entry['classification_accuracy']

    # Compute gaps from BSA and one-sided p-values
    gap_a = (acc_a - bsa_mean) * 100
    gap_b = (acc_b - bsa_mean) * 100

    # Collect all p-values jointly for BH-FDR
    pvals_all = []
    coords_all = []

    for br_name, reps_grid in [('A', reps_a), ('B', reps_b)]:
        for i in range(n_inc):
            for j in range(n_tau):
                if reps_grid[i, j] is not None:
                    _, p = _welch_t_one_sided(reps_grid[i, j], bsa_mean,
                                              bsa_sample_std, bsa_n)
                    pvals_all.append(p)
                    coords_all.append((br_name, i, j))

    adj_pvals = _benjamini_hochberg(pvals_all)

    # Build significance masks (one-sided: cell > BSA)
    sig_a = np.full((n_inc, n_tau), False)
    sig_b = np.full((n_inc, n_tau), False)
    for idx, (branch, i, j) in enumerate(coords_all):
        gap = gap_a[i, j] if branch == 'A' else gap_b[i, j]
        if adj_pvals[idx] < 0.05 and gap > 0:
            if branch == 'A':
                sig_a[i, j] = True
            else:
                sig_b[i, j] = True

    n_sig_a = np.sum(sig_a)
    n_sig_b = np.sum(sig_b)
    n_total = n_inc * n_tau
    n_tests = len(pvals_all)
    print(f'BSA baseline: {bsa_mean*100:.2f}%')
    print(f'Branch A: {n_sig_a}/{n_total} cells significantly > BSA')
    print(f'Branch B: {n_sig_b}/{n_total} cells significantly > BSA')
    print(f'BH-FDR q=0.05, {n_tests} one-sided tests')

    # Shared diverging color scale centered on 0
    all_gaps = np.concatenate([gap_a.ravel(), gap_b.ravel()])
    all_gaps = all_gaps[~np.isnan(all_gaps)]
    gap_abs_max = np.max(np.abs(all_gaps))
    pad = max(gap_abs_max * 0.15, 0.3)
    vmin = -(gap_abs_max + pad)
    vmax = gap_abs_max + pad
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='#e0e0e0')

    # Figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    ax_a.set_facecolor('white')
    ax_b.set_facecolor('white')

    im_a = _draw_panel(ax_a, gap_a, sig_a, inc_vals, tau_vals, norm, cmap,
                       'A: Natural firing rate (unmatched)')
    im_b = _draw_panel(ax_b, gap_b, sig_b, inc_vals, tau_vals, norm, cmap,
                       'B: Rate-matched (20 Hz, tonic conductance)')

    # Shared colorbar
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.025, pad=0.06,
                        location='right')
    cbar.set_label(r'$\Delta$ accuracy from BSA baseline (pp)', fontsize=12,
                   labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    # BSA reference annotations
    ax_a.annotate(
        f'BSA baseline: {bsa_mean*100:.2f}%\n'
        f'Sig. better: {n_sig_a}/{n_total}',
        xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=8, color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))
    ax_b.annotate(
        f'BSA baseline: {bsa_mean*100:.2f}%\n'
        f'Sig. better: {n_sig_b}/{n_total}',
        xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=8, color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

    # Legend
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='#666666',
                                 hatch='///',
                                 label='n.s. vs BSA (one-sided, BH-FDR $q$=0.05)')
    solid_better = mpatches.Patch(facecolor='#4a90d9', edgecolor='none',
                                  label='$p_{adj}$ < 0.05 (better than BSA)')
    solid_worse = mpatches.Patch(facecolor='#d9534f', edgecolor='none',
                                 label='Below BSA')
    ax_a.legend(handles=[solid_better, solid_worse, hatch_patch],
                loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
                fontsize=7.5, framealpha=0.9, edgecolor='#cccccc')
    ax_b.legend(handles=[solid_better, solid_worse, hatch_patch],
                loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
                fontsize=7.5, framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        'Classification accuracy relative to BSA baseline (no adaptation)',
        fontsize=15, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.06, right=0.85, bottom=0.14, top=0.91,
                        wspace=0.25)

    fig.text(0.5, 0.02,
             f'One-sided Welch t-test (cell n=5 vs BSA n=5), '
             f'BH-FDR correction across {n_tests} tests',
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
