#!/usr/bin/env python3
"""
Six-panel heatmap: accuracy gap from BSA baseline for CLS, WM, and XOR.

Row 1: Classification (Branch A, Branch B)
Row 2: Working Memory  (Branch A, Branch B)
Row 3: XOR             (Branch A, Branch B)

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
DEFAULT_CLS = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_WM  = CPP_DIR / 'results' / 'wm_adaptation_sweep' / 'wm_adaptation_sweep_checkpoint.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'triplet_bsa_comparison.png'


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


def _welch_t_two_sided(cell_reps, bsa_mean, bsa_sample_std, bsa_n):
    """Two-sided Welch t-test: H1 = cell ≠ BSA."""
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
    p_two = 2 * stats.t.sf(abs(t_stat), df)
    return t_stat, p_two


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


def _draw_panel(ax, gap_pp, sig_mask, inc_vals, tau_vals, norm, cmap, title,
                show_xlabel=True, show_ylabel=True):
    im = ax.imshow(gap_pp, aspect='auto', origin='lower',
                   cmap=cmap, norm=norm, interpolation='nearest')

    n_inc, n_tau = gap_pp.shape

    # Hatch non-significant cells
    for i in range(n_inc):
        for j in range(n_tau):
            if np.isnan(gap_pp[i, j]):
                # Missing data — grey fill
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=True, facecolor='#e0e0e0',
                                     edgecolor='#cccccc', linewidth=0.5,
                                     zorder=2)
                ax.add_patch(rect)
                continue
            if not sig_mask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, hatch='///',
                                     edgecolor='#666666', linewidth=0.5,
                                     zorder=3)
                ax.add_patch(rect)

    # X-axis
    ax.set_xticks(np.arange(len(tau_vals)))
    if show_xlabel:
        ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=7, rotation=45,
                           ha='right')
        ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=9,
                      labelpad=4)
    else:
        ax.set_xticklabels([])

    # Y-axis
    ax.set_yticks(np.arange(len(inc_vals)))
    if show_ylabel:
        ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=7)
        ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=9,
                      labelpad=4)
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
            fontsize = 5.5
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='medium')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    # Cluster boundary around significant cells
    border_color = '#1a1a1a'
    border_lw = 1.0
    for i in range(n_inc):
        for j in range(n_tau):
            if not sig_mask[i, j]:
                continue
            x0, y0 = j - 0.5, i - 0.5
            if j == 0 or not sig_mask[i, j - 1]:
                ax.plot([x0, x0], [y0, y0 + 1], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
            if j == n_tau - 1 or not sig_mask[i, j + 1]:
                ax.plot([x0 + 1, x0 + 1], [y0, y0 + 1], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
            if i == 0 or not sig_mask[i - 1, j]:
                ax.plot([x0, x0 + 1], [y0, y0], color=border_color,
                        linewidth=border_lw, solid_capstyle='round', zorder=6)
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


def _load_cls(path):
    """Load classification sweep data. Returns (inc_vals, tau_vals, acc_a, acc_b, reps_a, reps_b, bsa_mean, bsa_sample_std)."""
    with open(path) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    bsa_mean = data['bsa_baseline']['accuracy']
    bsa_pop_std = data['bsa_baseline']['accuracy_std']
    bsa_n = 5
    bsa_sample_std = bsa_pop_std * np.sqrt(bsa_n / (bsa_n - 1))

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

    return inc_vals, tau_vals, acc_a, acc_b, reps_a, reps_b, bsa_mean, bsa_sample_std


def _load_wm(path):
    """Load WM sweep data. Returns (inc_vals, tau_vals, wm_acc_a/b, wm_reps_a/b, xor_acc_a/b, xor_reps_a/b, bsa_wm_mean/std, bsa_xor_mean/std)."""
    with open(path) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    bsa_wm_mean = data['bsa_wm_baseline']['accuracy']
    bsa_wm_pop_std = data['bsa_wm_baseline']['accuracy_std']
    bsa_xor_mean = data['bsa_xor_baseline']['accuracy']
    bsa_xor_pop_std = data['bsa_xor_baseline']['accuracy_std']
    bsa_n = 5
    bsa_wm_sample_std = bsa_wm_pop_std * np.sqrt(bsa_n / (bsa_n - 1))
    bsa_xor_sample_std = bsa_xor_pop_std * np.sqrt(bsa_n / (bsa_n - 1))

    wm_acc_a = np.full((n_inc, n_tau), np.nan)
    wm_acc_b = np.full((n_inc, n_tau), np.nan)
    xor_acc_a = np.full((n_inc, n_tau), np.nan)
    xor_acc_b = np.full((n_inc, n_tau), np.nan)
    wm_reps_a = np.empty((n_inc, n_tau), dtype=object)
    wm_reps_b = np.empty((n_inc, n_tau), dtype=object)
    xor_reps_a = np.empty((n_inc, n_tau), dtype=object)
    xor_reps_b = np.empty((n_inc, n_tau), dtype=object)

    for entry in data['grid_results']:
        ii, ti = entry['inc_idx'], entry['tau_idx']
        branch = entry['branch']
        if branch == 'A_unmatched':
            wm_acc_a[ii, ti] = entry['wm_accuracy']
            xor_acc_a[ii, ti] = entry['xor_accuracy']
            wm_reps_a[ii, ti] = entry.get('wm_per_repeat_accuracy')
            xor_reps_a[ii, ti] = entry.get('xor_per_repeat_accuracy')
        elif branch == 'B_matched':
            wm_acc_b[ii, ti] = entry['wm_accuracy']
            xor_acc_b[ii, ti] = entry['xor_accuracy']
            wm_reps_b[ii, ti] = entry.get('wm_per_repeat_accuracy')
            xor_reps_b[ii, ti] = entry.get('xor_per_repeat_accuracy')

    return (inc_vals, tau_vals,
            wm_acc_a, wm_acc_b, wm_reps_a, wm_reps_b, bsa_wm_mean, bsa_wm_sample_std,
            xor_acc_a, xor_acc_b, xor_reps_a, xor_reps_b, bsa_xor_mean, bsa_xor_sample_std)


def _compute_sig(reps_a, reps_b, acc_a, acc_b, bsa_mean, bsa_sample_std, bsa_n=5):
    """Compute gap matrices and significance masks with joint BH-FDR."""
    n_inc, n_tau = acc_a.shape
    gap_a = (acc_a - bsa_mean) * 100
    gap_b = (acc_b - bsa_mean) * 100

    pvals_all = []
    coords_all = []

    for br_name, reps_grid in [('A', reps_a), ('B', reps_b)]:
        for i in range(n_inc):
            for j in range(n_tau):
                if reps_grid[i, j] is not None:
                    _, p = _welch_t_two_sided(reps_grid[i, j], bsa_mean,
                                              bsa_sample_std, bsa_n)
                    pvals_all.append(p)
                    coords_all.append((br_name, i, j))

    if len(pvals_all) == 0:
        sig_a = np.full((n_inc, n_tau), False)
        sig_b = np.full((n_inc, n_tau), False)
        return gap_a, gap_b, sig_a, sig_b, 0, 0

    adj_pvals = _benjamini_hochberg(pvals_all)

    sig_a = np.full((n_inc, n_tau), False)
    sig_b = np.full((n_inc, n_tau), False)
    for idx, (branch, i, j) in enumerate(coords_all):
        gap = gap_a[i, j] if branch == 'A' else gap_b[i, j]
        if adj_pvals[idx] < 0.05:
            if branch == 'A':
                sig_a[i, j] = True
            else:
                sig_b[i, j] = True

    return gap_a, gap_b, sig_a, sig_b, np.sum(sig_a), np.sum(sig_b)


def main():
    parser = argparse.ArgumentParser(description='Triple BSA comparison heatmap')
    parser.add_argument('--cls-input', type=str, default=str(DEFAULT_CLS))
    parser.add_argument('--wm-input', type=str, default=str(DEFAULT_WM))
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    # Load data
    cls_inc, cls_tau, cls_acc_a, cls_acc_b, cls_reps_a, cls_reps_b, cls_bsa, cls_bsa_std = _load_cls(args.cls_input)
    (wm_inc, wm_tau,
     wm_acc_a, wm_acc_b, wm_reps_a, wm_reps_b, wm_bsa, wm_bsa_std,
     xor_acc_a, xor_acc_b, xor_reps_a, xor_reps_b, xor_bsa, xor_bsa_std) = _load_wm(args.wm_input)

    # Compute gaps and significance for each task
    cls_gap_a, cls_gap_b, cls_sig_a, cls_sig_b, cls_nsig_a, cls_nsig_b = \
        _compute_sig(cls_reps_a, cls_reps_b, cls_acc_a, cls_acc_b, cls_bsa, cls_bsa_std)
    wm_gap_a, wm_gap_b, wm_sig_a, wm_sig_b, wm_nsig_a, wm_nsig_b = \
        _compute_sig(wm_reps_a, wm_reps_b, wm_acc_a, wm_acc_b, wm_bsa, wm_bsa_std)
    xor_gap_a, xor_gap_b, xor_sig_a, xor_sig_b, xor_nsig_a, xor_nsig_b = \
        _compute_sig(xor_reps_a, xor_reps_b, xor_acc_a, xor_acc_b, xor_bsa, xor_bsa_std)

    n_inc, n_tau = len(cls_inc), len(cls_tau)

    print(f'CLS BSA: {cls_bsa*100:.2f}% | A sig: {cls_nsig_a}/{n_inc*n_tau} | B sig: {cls_nsig_b}/{n_inc*n_tau}')
    print(f'WM  BSA: {wm_bsa*100:.2f}% | A sig: {wm_nsig_a}/{n_inc*n_tau} | B sig: {wm_nsig_b}/{n_inc*n_tau}')
    print(f'XOR BSA: {xor_bsa*100:.2f}% | A sig: {xor_nsig_a}/{n_inc*n_tau} | B sig: {xor_nsig_b}/{n_inc*n_tau}')

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='#e0e0e0')

    def _make_row_norm(gap_a, gap_b):
        """Per-row diverging color norm centered on 0."""
        vals = np.concatenate([gap_a.ravel(), gap_b.ravel()])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        abs_max = np.max(np.abs(vals))
        pad = max(abs_max * 0.15, 0.3)
        return TwoSlopeNorm(vmin=-(abs_max + pad), vcenter=0, vmax=abs_max + pad)

    # Figure: 3 rows x 3 cols (col 2 = colorbar space)
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('white')

    # Use GridSpec: 3 rows, 3 cols (two heatmap cols + one narrow colorbar col per row)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.30, wspace=0.08,
                  left=0.06, right=0.92, bottom=0.06, top=0.93)

    row_configs = [
        ('Classification', cls_gap_a, cls_gap_b, cls_sig_a, cls_sig_b,
         cls_bsa, cls_nsig_a, cls_nsig_b, cls_inc, cls_tau),
        ('Working Memory', wm_gap_a, wm_gap_b, wm_sig_a, wm_sig_b,
         wm_bsa, wm_nsig_a, wm_nsig_b, wm_inc, wm_tau),
        ('XOR (same/different)', xor_gap_a, xor_gap_b, xor_sig_a, xor_sig_b,
         xor_bsa, xor_nsig_a, xor_nsig_b, wm_inc, wm_tau),
    ]

    for row_idx, (task_name, gap_a, gap_b, sig_a, sig_b, bsa_val,
                  nsig_a, nsig_b, inc_vals, tau_vals) in enumerate(row_configs):
        is_bottom = (row_idx == 2)
        ax_a = fig.add_subplot(gs[row_idx, 0])
        ax_b = fig.add_subplot(gs[row_idx, 1])
        ax_cb = fig.add_subplot(gs[row_idx, 2])
        ax_a.set_facecolor('white')
        ax_b.set_facecolor('white')

        row_norm = _make_row_norm(gap_a, gap_b)

        im_a = _draw_panel(ax_a, gap_a, sig_a, inc_vals, tau_vals, row_norm, cmap,
                           f'{task_name} — A: Natural firing rate (unmatched)',
                           show_xlabel=is_bottom, show_ylabel=True)
        im_b = _draw_panel(ax_b, gap_b, sig_b, inc_vals, tau_vals, row_norm, cmap,
                           f'{task_name} — B: Rate-matched (20 Hz, tonic conductance)',
                           show_xlabel=is_bottom, show_ylabel=False)

        # Per-row colorbar
        cbar = fig.colorbar(im_b, cax=ax_cb)
        cbar.set_label(r'$\Delta$ acc. from BSA (pp)', fontsize=9, labelpad=8)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

        # BSA annotation
        for ax, nsig in [(ax_a, nsig_a), (ax_b, nsig_b)]:
            ax.annotate(
                f'BSA: {bsa_val*100:.1f}% | Sig: {nsig}/{n_inc*n_tau}',
                xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=7, color='#666666',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

    # Legend at bottom
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='#666666',
                                 hatch='///',
                                 label='n.s. vs BSA (one-sided Welch t, BH-FDR $q$=0.05)')
    solid_better = mpatches.Patch(facecolor='#4a90d9', edgecolor='none',
                                  label='$p_{adj}$ < 0.05 (better than BSA)')
    solid_worse = mpatches.Patch(facecolor='#d9534f', edgecolor='none',
                                 label='Below BSA')
    grey_patch = mpatches.Patch(facecolor='#e0e0e0', edgecolor='#cccccc',
                                label='Not yet computed')
    fig.legend(handles=[solid_better, solid_worse, hatch_patch, grey_patch],
               loc='lower center', bbox_to_anchor=(0.48, 0.005), ncol=4,
               fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        'Triple dissociation: accuracy relative to BSA baseline across SFA parameter space',
        fontsize=16, fontweight='bold', y=0.97)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
