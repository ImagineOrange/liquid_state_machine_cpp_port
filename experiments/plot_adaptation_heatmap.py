#!/usr/bin/env python3
"""
Publication-quality heatmaps of classification accuracy across
adaptation increment x adaptation time-constant parameter space.

Figure 1 (adaptation_heatmap): Raw accuracy for both branches.
Figure 2 (adaptation_vs_lhs021): Gap from LHS-021 baseline with
         Welch t-test significance (BH-FDR corrected).

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
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats

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

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {png_path}')

    plt.close(fig)


def load_sweep_full(path):
    """Load sweep data including per-repeat accuracies, std, and dynamics grids."""
    with open(path) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc = len(inc_vals)
    n_tau = len(tau_vals)

    acc_a = np.full((n_inc, n_tau), np.nan)
    acc_b = np.full((n_inc, n_tau), np.nan)
    std_a = np.full((n_inc, n_tau), np.nan)
    std_b = np.full((n_inc, n_tau), np.nan)
    tonic_g_b = np.full((n_inc, n_tau), np.nan)
    # Store per-repeat data as object arrays of lists
    reps_a = np.empty((n_inc, n_tau), dtype=object)
    reps_b = np.empty((n_inc, n_tau), dtype=object)

    for entry in data['grid_results']:
        ii = entry['inc_idx']
        ti = entry['tau_idx']
        branch = entry.get('branch', 'A_unmatched')
        if branch == 'A_unmatched':
            acc_a[ii, ti] = entry['classification_accuracy']
            std_a[ii, ti] = entry['classification_accuracy_std']
            reps_a[ii, ti] = entry.get('classification_per_repeat_accuracy')
        elif branch == 'B_matched':
            acc_b[ii, ti] = entry['classification_accuracy']
            std_b[ii, ti] = entry['classification_accuracy_std']
            reps_b[ii, ti] = entry.get('classification_per_repeat_accuracy')
            tonic_g_b[ii, ti] = entry.get('tonic_conductance', np.nan)

    extras = {'tonic_g_b': tonic_g_b}
    return inc_vals, tau_vals, acc_a, acc_b, std_a, std_b, reps_a, reps_b, data, extras


def _welch_t_from_summary(mean1, std_pop1, n1, mean2, std_pop2, n2):
    """Welch t-test from summary statistics. Stds are population (ddof=0)."""
    # Convert population std to sample std (ddof=1)
    s1 = std_pop1 * np.sqrt(n1 / (n1 - 1))
    s2 = std_pop2 * np.sqrt(n2 / (n2 - 1))
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return np.nan, np.nan
    t_stat = (mean1 - mean2) / se
    # Welch-Satterthwaite degrees of freedom
    num = (s1**2 / n1 + s2**2 / n2)**2
    denom = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
    df = num / denom if denom > 0 else 1.0
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return t_stat, p_value


def _benjamini_hochberg(p_values, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.array(p_values)
    n = len(p)
    sorted_idx = np.argsort(p)
    adjusted = np.empty(n)
    # Walk backwards through sorted p-values
    cummin = np.inf
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = p[sorted_idx[i]] * n / rank
        cummin = min(cummin, adj)
        adjusted[sorted_idx[i]] = cummin
    return np.clip(adjusted, 0, 1)


def _draw_gap_panel(ax, gap_pp, sig_mask, inc_vals, tau_vals, vmin, vmax,
                    norm, cmap, title):
    """Draw a gap-from-baseline heatmap panel with significance hatching."""
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

    # X-axis: tau
    ax.set_xticks(np.arange(len(tau_vals)))
    ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=8, rotation=45,
                       ha='right')
    ax.set_xlabel(r'Adaptation time constant $\tau_a$ (ms)', fontsize=11,
                  labelpad=6)

    # Y-axis: increment
    ax.set_yticks(np.arange(len(inc_vals)))
    ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=8)
    ax.set_ylabel(r'Adaptation increment $\Delta_a$ (nS)', fontsize=11,
                  labelpad=6)

    # Cell annotations: show gap in pp
    for i in range(n_inc):
        for j in range(n_tau):
            val = gap_pp[i, j]
            if np.isnan(val):
                continue
            # Text colour based on background
            norm_val = norm(val)
            text_color = 'white' if norm_val < 0.35 or norm_val > 0.85 else 'black'
            fontsize = 6.0 if n_inc * n_tau > 200 else 7.0
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='medium')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    return im


def _pool_inc0_baseline(reps_grid):
    """Pool all per-repeat accuracies from inc_idx=0 row into one array.

    Since adaptation increment = 0 makes tau irrelevant, all inc=0 cells
    are replicates of the same condition.  Pooling gives a robust baseline
    with n = n_tau * n_repeats_per_cell observations.
    """
    n_inc, n_tau = reps_grid.shape
    pooled = []
    for j in range(n_tau):
        if reps_grid[0, j] is not None:
            pooled.extend(reps_grid[0, j])
    return np.array(pooled)


def _draw_tonic_g_boundary(ax, tonic_g_grid):
    """Draw a boundary line on Branch B where tonic conductance changes.

    The rate-matcher uses discrete tonic conductance steps; when adaptation
    pushes rate below tolerance the calibrator steps down, changing the
    network's inhibitory operating point.  This boundary marks where that
    regime change happens so the reader can assess the confound.
    """
    n_inc, n_tau = tonic_g_grid.shape
    # Reference: tonic_g at inc=0 (no adaptation)
    ref_g = np.nanmedian(tonic_g_grid[0, :])
    if np.isnan(ref_g):
        return

    # For each row, find the boundary between ref_g and a different value
    for i in range(n_inc):
        for j in range(n_tau - 1):
            g_left = tonic_g_grid[i, j]
            g_right = tonic_g_grid[i, j + 1]
            if np.isnan(g_left) or np.isnan(g_right):
                continue
            if g_left != g_right:
                # Vertical boundary between columns j and j+1
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5],
                        color='#ff8c00', linewidth=2.0, linestyle='-',
                        zorder=4)
        # Also check row boundaries (between inc rows at same tau)
        if i < n_inc - 1:
            for j in range(n_tau):
                g_here = tonic_g_grid[i, j]
                g_above = tonic_g_grid[i + 1, j]
                if np.isnan(g_here) or np.isnan(g_above):
                    continue
                if g_here != g_above:
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5],
                            color='#ff8c00', linewidth=2.0, linestyle='-',
                            zorder=4)


def make_marginal_effect(inc_vals, tau_vals, acc_a, acc_b, reps_a, reps_b,
                         data, output_path, extras=None):
    """Second figure: marginal effect of adaptation relative to inc=0 baseline.

    Each cell shows accuracy change (pp) from the pooled no-adaptation
    baseline (inc=0 row).  Welch t-test (cell n=5 vs pooled baseline n=75)
    with BH-FDR correction identifies cells where adaptation significantly
    helps or hurts.
    """
    n_inc, n_tau = acc_a.shape

    # --- Pool inc=0 baselines ---
    baseline_a = _pool_inc0_baseline(reps_a)
    baseline_b = _pool_inc0_baseline(reps_b)

    if len(baseline_a) == 0 or len(baseline_b) == 0:
        print('No inc=0 baseline data available.')
        return

    bl_mean_a = np.mean(baseline_a)
    bl_mean_b = np.mean(baseline_b)
    print(f'\n--- Marginal effect of adaptation (inc=0 baseline) ---')
    print(f'Branch A baseline: {bl_mean_a*100:.3f}% (n={len(baseline_a)})')
    print(f'Branch B baseline: {bl_mean_b*100:.3f}% (n={len(baseline_b)})')

    # --- Compute gaps and Welch t-tests ---
    gap_a = np.full((n_inc, n_tau), np.nan)
    gap_b = np.full((n_inc, n_tau), np.nan)
    pvals_flat = []
    coords = []

    for i in range(n_inc):
        for j in range(n_tau):
            if reps_a[i, j] is not None:
                cell_reps = np.array(reps_a[i, j])
                gap_a[i, j] = (np.mean(cell_reps) - bl_mean_a) * 100
                _, p = stats.ttest_ind(cell_reps, baseline_a,
                                       equal_var=False)
                pvals_flat.append(p)
                coords.append(('A', i, j))

            if reps_b[i, j] is not None:
                cell_reps = np.array(reps_b[i, j])
                gap_b[i, j] = (np.mean(cell_reps) - bl_mean_b) * 100
                _, p = stats.ttest_ind(cell_reps, baseline_b,
                                       equal_var=False)
                pvals_flat.append(p)
                coords.append(('B', i, j))

    # --- BH-FDR correction ---
    n_tests = len(pvals_flat)
    if n_tests == 0:
        print('No data for marginal effect figure.')
        return

    adj_pvals = _benjamini_hochberg(pvals_flat, q=0.05)

    sig_a = np.full((n_inc, n_tau), False)
    sig_b = np.full((n_inc, n_tau), False)
    for idx, (branch, i, j) in enumerate(coords):
        if branch == 'A':
            sig_a[i, j] = adj_pvals[idx] < 0.05
        else:
            sig_b[i, j] = adj_pvals[idx] < 0.05

    n_sig_a = np.sum(sig_a & ~np.isnan(gap_a))
    n_pop_a = np.count_nonzero(~np.isnan(gap_a))
    n_sig_b = np.sum(sig_b & ~np.isnan(gap_b))
    n_pop_b = np.count_nonzero(~np.isnan(gap_b))
    print(f'BH-FDR q=0.05, {n_tests} tests')
    print(f'Branch A: {n_sig_a}/{n_pop_a} significantly different from inc=0')
    print(f'Branch B: {n_sig_b}/{n_pop_b} significantly different from inc=0')

    # --- Shared color scale (diverging around 0) ---
    all_gaps = np.concatenate([gap_a.ravel(), gap_b.ravel()])
    all_gaps = all_gaps[~np.isnan(all_gaps)]
    gap_abs_max = np.max(np.abs(all_gaps))
    pad = max(gap_abs_max * 0.15, 0.3)
    vmin = -(gap_abs_max + pad)
    vmax = gap_abs_max + pad
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='#e0e0e0')

    # --- Figure ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    ax_a.set_facecolor('white')
    ax_b.set_facecolor('white')

    im_a = _draw_gap_panel(ax_a, gap_a, sig_a, inc_vals, tau_vals,
                           vmin, vmax, norm, cmap,
                           'A: Natural firing rate (unmatched)')
    im_b = _draw_gap_panel(ax_b, gap_b, sig_b, inc_vals, tau_vals,
                           vmin, vmax, norm, cmap,
                           'B: Rate-matched (20 Hz, tonic conductance)')

    # Tonic conductance regime boundary on Branch B
    if extras and 'tonic_g_b' in extras:
        _draw_tonic_g_boundary(ax_b, extras['tonic_g_b'])

    # Shared colorbar
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.025, pad=0.06,
                        location='right')
    cbar.set_label(r'$\Delta$ accuracy from no-adaptation baseline (pp)',
                   fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Zero-line reference on colorbar
    cbar.ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    # Baseline annotations
    ax_a.annotate(
        f'No-adapt baseline: {bl_mean_a*100:.2f}%\n(pooled inc=0, n={len(baseline_a)})',
        xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=8, color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))
    ax_b.annotate(
        f'No-adapt baseline: {bl_mean_b*100:.2f}%\n(pooled inc=0, n={len(baseline_b)})',
        xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=8, color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

    # Legend for hatching (Panel A)
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='#666666',
                                 hatch='///',
                                 label='n.s. vs no-adapt (BH-FDR $q$=0.05)')
    solid_patch = mpatches.Patch(facecolor='#d9534f', edgecolor='none',
                                 label='$p_{adj}$ < 0.05 (worse)')
    solid_patch2 = mpatches.Patch(facecolor='#4a90d9', edgecolor='none',
                                  label='$p_{adj}$ < 0.05 (better)')
    ax_a.legend(handles=[solid_patch2, solid_patch, hatch_patch],
                loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
                fontsize=7.5, framealpha=0.9, edgecolor='#cccccc')

    # Legend for Panel B (includes tonic conductance boundary)
    boundary_line = plt.Line2D([], [], color='#ff8c00', linewidth=2.0,
                               label=r'$g_{tonic}$ regime boundary')
    handles_b = [solid_patch2, solid_patch, hatch_patch]
    if extras and 'tonic_g_b' in extras:
        handles_b.append(boundary_line)
    ax_b.legend(handles=handles_b, loc='upper center',
                bbox_to_anchor=(0.5, -0.18), ncol=len(handles_b),
                fontsize=7.5, framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        'Marginal effect of adaptation on classification accuracy',
        fontsize=15, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.06, right=0.85, bottom=0.14, top=0.91,
                        wspace=0.25)

    # Method note (placed below axes with enough clearance)
    fig.text(0.5, 0.02,
             f'Welch t-test (cell n=5 vs pooled inc=0 n='
             f'{len(baseline_a)}), '
             f'BH-FDR correction across {n_tests} tests',
             ha='center', fontsize=9, color='#888888')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {png_path}')

    plt.close(fig)


def _draw_dynamics_panel(ax, grid, inc_vals, tau_vals, vmin, vmax, cmap,
                         title, fmt='{:.1f}'):
    """Draw a single heatmap panel for dynamics comparison figure."""
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    ax.set_xticks(np.arange(len(tau_vals)))
    ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=7, rotation=45,
                       ha='right')
    ax.set_xlabel(r'$\tau_a$ (ms)', fontsize=10, labelpad=6)

    ax.set_yticks(np.arange(len(inc_vals)))
    ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=7)
    ax.set_ylabel(r'$\Delta_a$ (nS)', fontsize=10, labelpad=6)

    n_inc, n_tau = grid.shape
    for i in range(n_inc):
        for j in range(n_tau):
            val = grid[i, j]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = 'white' if norm_val < 0.45 else 'black'
            ax.text(j, i, fmt.format(val), ha='center', va='center',
                    fontsize=5.5, color=text_color, fontweight='medium')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    return im


def make_rate_vs_accuracy(input_path, output_path):
    """Figure 3: three-panel Branch A heatmap — accuracy, ISI_CV, firing rate.

    Demonstrates that the accuracy surface tracks ISI_CV topology
    (tau-dominated) rather than firing rate topology (inc-dominated).
    """
    with open(input_path) as f:
        data = json.load(f)

    inc_vals = np.array(data['grid']['unified_inc'])
    tau_vals = np.array(data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    acc = np.full((n_inc, n_tau), np.nan)
    rate = np.full((n_inc, n_tau), np.nan)
    isicv = np.full((n_inc, n_tau), np.nan)

    for entry in data['grid_results']:
        if entry.get('branch', 'A_unmatched') != 'A_unmatched':
            continue
        ii, ti = entry['inc_idx'], entry['tau_idx']
        acc[ii, ti] = entry['classification_accuracy']
        rate[ii, ti] = entry['firing_rate_hz']
        isicv[ii, ti] = entry['isi_cv_mean']

    acc_pct = acc * 100

    # Compute inc=0 baselines (mean across all tau at inc_idx=0)
    bl_acc = np.nanmean(acc_pct[0, :])
    bl_isicv = np.nanmean(isicv[0, :])
    bl_rate = np.nanmean(rate[0, :])

    # Delta grids relative to baseline
    dacc = acc_pct - bl_acc
    disicv = isicv - bl_isicv
    drate = rate - bl_rate

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor('white')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('white')

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='#e0e0e0')

    def _diverging_panel(ax, grid, title, cbar_label, fmt='{:+.1f}'):
        vals = grid[~np.isnan(grid)]
        absmax = np.max(np.abs(vals))
        pad = max(absmax * 0.15, 0.1)
        norm = TwoSlopeNorm(vmin=-(absmax + pad), vcenter=0,
                            vmax=absmax + pad)
        im = ax.imshow(grid, aspect='auto', origin='lower',
                       cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_xticks(np.arange(len(tau_vals)))
        ax.set_xticklabels(_format_tau_labels(tau_vals), fontsize=7,
                           rotation=45, ha='right')
        ax.set_xlabel(r'$\tau_a$ (ms)', fontsize=10, labelpad=6)
        ax.set_yticks(np.arange(len(inc_vals)))
        ax.set_yticklabels(_format_inc_labels(inc_vals), fontsize=7)
        ax.set_ylabel(r'$\Delta_a$ (nS)', fontsize=10, labelpad=6)
        ni, nt = grid.shape
        for i in range(ni):
            for j in range(nt):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                nv = norm(val)
                tc = 'white' if nv < 0.35 or nv > 0.85 else 'black'
                ax.text(j, i, fmt.format(val), ha='center', va='center',
                        fontsize=5.5, color=tc, fontweight='medium')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks(np.arange(-0.5, len(tau_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(inc_vals), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-')

    _diverging_panel(ax1, dacc,
                     r'$\Delta$ Accuracy (pp)',
                     f'Change from baseline ({bl_acc:.1f}%)')
    _diverging_panel(ax2, disicv,
                     r'$\Delta$ ISI CV',
                     f'Change from baseline ({bl_isicv:.2f})',
                     fmt='{:+.2f}')
    _diverging_panel(ax3, drate,
                     r'$\Delta$ Firing rate (Hz)',
                     f'Change from baseline ({bl_rate:.1f} Hz)')

    # Correlations (on raw values — deltas give same r)
    mask = (~np.isnan(acc.ravel()) & ~np.isnan(isicv.ravel())
            & ~np.isnan(rate.ravel()))
    acc_f = acc.ravel()[mask]
    isicv_f = isicv.ravel()[mask]
    rate_f = rate.ravel()[mask]
    r_cv, p_cv = stats.pearsonr(isicv_f, acc_f)
    r_rate, p_rate = stats.pearsonr(rate_f, acc_f)

    fig.text(0.5, 0.02,
             f'Branch A correlations with accuracy:  '
             f'ISI_CV: r={r_cv:+.2f}, p={p_cv:.1e}  |  '
             f'Firing rate: r={r_rate:+.2f}, p={p_rate:.1e}  '
             f'(n={int(mask.sum())} cells)',
             ha='center', fontsize=9, color='#888888')

    fig.suptitle(
        'Branch A: Change from no-adaptation baseline',
        fontsize=14, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.90,
                        wspace=0.35)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    # --- Second figure: marginal effect of adaptation ---
    inc2, tau2, acc_a2, acc_b2, std_a2, std_b2, reps_a2, reps_b2, data2, extras = \
        load_sweep_full(args.input)
    marginal_output = Path(args.output).parent / 'adaptation_marginal_effect.pdf'
    make_marginal_effect(inc2, tau2, acc_a2, acc_b2, reps_a2, reps_b2,
                         data2, marginal_output, extras=extras)

    # --- Third figure: rate vs accuracy (Branch A) ---
    rate_output = Path(args.output).parent / 'branch_a_rate_vs_accuracy.pdf'
    make_rate_vs_accuracy(args.input, rate_output)


if __name__ == '__main__':
    main()
