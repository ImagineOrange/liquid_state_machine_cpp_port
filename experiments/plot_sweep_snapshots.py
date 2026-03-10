import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ── Load data ──
CPP_DIR = Path(__file__).resolve().parent.parent
INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'

with open(INPUT) as f:
    raw = json.load(f)

grid = raw['grid']
inc_vals = np.array(grid['unified_inc'])
tau_vals = np.array(grid['unified_tau'])
n_inc = grid['n_inc']
n_tau = grid['n_tau']
n_bins = 48
bin_centers = np.array([10 + i * 20 for i in range(n_bins)])
bsa_baseline = raw['bsa_baseline']['accuracy']

# Build per-branch grids: shape (n_inc, n_tau, n_bins)
grids = {}
for r in raw['grid_results']:
    b = r['branch']
    if b not in grids:
        grids[b] = np.full((n_inc, n_tau, n_bins), np.nan)
    grids[b][r['inc_idx'], r['tau_idx'], :] = r['per_bin_accuracy']

# ── Select 5 snapshot timepoints ──
# Strategy: capture early rise, peak region, mid-decay, late plateau
# Look at mean accuracy over all grid points to find peak
mean_acc_B = np.nanmean(grids['B_matched'], axis=(0, 1))
peak_bin = np.argmax(mean_acc_B)
peak_ms = bin_centers[peak_bin]
print(f"Branch B mean peak at bin {peak_bin} = {peak_ms} ms, acc={mean_acc_B[peak_bin]:.4f}")

mean_acc_A = np.nanmean(grids['A_unmatched'], axis=(0, 1))
peak_bin_A = np.argmax(mean_acc_A)
print(f"Branch A mean peak at bin {peak_bin_A} = {bin_centers[peak_bin_A]} ms, acc={mean_acc_A[peak_bin_A]:.4f}")

# Pick 5 snapshots: early, pre-peak, peak, post-peak, late
snapshot_ms = [50, 190, 310, 530, 870]
snapshot_bins = [np.argmin(np.abs(bin_centers - ms)) for ms in snapshot_ms]
snapshot_ms_actual = [bin_centers[b] for b in snapshot_bins]
print(f"Snapshot bins: {snapshot_bins}, ms: {snapshot_ms_actual}")

# ── Algorithmically select 4 diverse grid points ──
# Maximize pairwise distance of temporal profiles using greedy farthest-point sampling
def select_diverse_points(grid_3d, n_select=4):
    """Greedy farthest-point sampling in temporal-profile space."""
    profiles = grid_3d.reshape(-1, n_bins)  # (n_inc*n_tau, n_bins)
    n_pts = profiles.shape[0]

    # Normalize profiles
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1

    # Start with the point that has highest variance (most interesting profile)
    variances = np.var(profiles, axis=1)
    selected = [np.argmax(variances)]

    for _ in range(n_select - 1):
        # Distance from each point to nearest selected point
        min_dists = np.full(n_pts, np.inf)
        for s in selected:
            dists = np.sum((profiles - profiles[s:s+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
        # Zero out already selected
        for s in selected:
            min_dists[s] = -1
        selected.append(np.argmax(min_dists))

    # Convert flat indices to (inc_idx, tau_idx)
    coords = [(idx // n_tau, idx % n_tau) for idx in selected]
    return coords

sel_A = select_diverse_points(grids['A_unmatched'], 4)
sel_B = select_diverse_points(grids['B_matched'], 4)

print("Selected points Branch A:", [(inc_vals[ii], tau_vals[ti]) for ii, ti in sel_A])
print("Selected points Branch B:", [(inc_vals[ii], tau_vals[ti]) for ii, ti in sel_B])

# ── Styling ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.4,
    'pdf.fonttype': 42,  # TrueType for editability
    'ps.fonttype': 42,
})

TRACE_COLORS = ['#2166ac', '#d6604d', '#4dac26', '#b2abd2']
TRACE_LABELS_FMT = lambda ii, ti, idx=0: f'$\\Delta_a$={inc_vals[ii]:.3g}, $\\tau_a$={tau_vals[ti]:.0f}'

# Acc range for heatmaps
vmin, vmax = 0.19, 0.78

# Y-axis labels for heatmap
inc_labels = []
for v in inc_vals:
    if v == 0:
        inc_labels.append('0')
    elif v < 0.01:
        inc_labels.append(f'{v:.3f}')
    elif v < 0.1:
        inc_labels.append(f'{v:.3f}')
    elif v < 1:
        inc_labels.append(f'{v:.2f}')
    else:
        inc_labels.append(f'{v:.1f}')

tau_labels = []
for v in tau_vals:
    if v < 100:
        tau_labels.append(f'{v:.0f}')
    elif v < 1000:
        tau_labels.append(f'{v:.0f}')
    else:
        tau_labels.append(f'{v/1000:.1f}k')


def make_figure(branch_key, branch_label, selected_pts, filename):
    grid_3d = grids[branch_key]

    fig = plt.figure(figsize=(7.2, 5.8))

    # Layout: Row A = 5 heatmaps, Row B = temporal traces
    # Use gridspec with height ratios
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.38,
                              left=0.08, right=0.95, top=0.93, bottom=0.08)

    # Row A: 5 heatmaps
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0],
                                               wspace=0.08, width_ratios=[1,1,1,1,1,0.06])

    hm_axes = []
    for pi in range(5):
        ax = fig.add_subplot(gs_top[0, pi])
        hm_axes.append(ax)

        b = snapshot_bins[pi]
        data_slice = grid_3d[:, :, b]

        im = ax.imshow(data_slice, origin='lower', aspect='auto',
                        cmap='viridis', vmin=vmin, vmax=vmax,
                        interpolation='nearest')

        ax.set_title(f'{bin_centers[b]} ms', fontsize=8, fontweight='semibold', pad=4)

        # X ticks (tau) - sparse
        ax.set_xticks([0, 4, 7, 10, 14])
        if pi == 2:
            ax.set_xticklabels([tau_labels[i] for i in [0, 4, 7, 10, 14]], fontsize=6)
            ax.set_xlabel('$\\tau_a$ (ms)', fontsize=8, labelpad=2)
        else:
            ax.set_xticklabels([tau_labels[i] for i in [0, 4, 7, 10, 14]], fontsize=6)

        # Y ticks (inc) - sparse
        ytick_idx = [0, 5, 10, 15, 19]
        ax.set_yticks(ytick_idx)
        if pi == 0:
            ax.set_yticklabels([inc_labels[i] for i in ytick_idx], fontsize=6)
            ax.set_ylabel('$\\Delta_a$ (adapt. increment)', fontsize=8, labelpad=2)
        else:
            ax.set_yticklabels([])

        # Mark selected points — filled dot with thin white edge
        for idx, (ii, ti) in enumerate(selected_pts):
            ax.plot(ti, ii, marker='o', markersize=5.5,
                    markerfacecolor=TRACE_COLORS[idx],
                    markeredgecolor='white', markeredgewidth=0.8, zorder=7)

    # Colorbar
    cax = fig.add_subplot(gs_top[0, 5])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Accuracy', fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6)

    # Row B: Temporal traces
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
    ax_trace = fig.add_subplot(gs_bot[0, 0])

    for idx, (ii, ti) in enumerate(selected_pts):
        profile = grid_3d[ii, ti, :]
        label = TRACE_LABELS_FMT(ii, ti, idx)
        ax_trace.plot(bin_centers, profile, color=TRACE_COLORS[idx],
                      linewidth=1.8, label=label, zorder=3)
        # Small marker at peak
        peak_b = np.argmax(profile)
        ax_trace.plot(bin_centers[peak_b], profile[peak_b], marker='o',
                      markersize=4.5, markerfacecolor=TRACE_COLORS[idx],
                      markeredgecolor='white', markeredgewidth=0.6, zorder=5)

    # Snapshot time markers
    for b in snapshot_bins:
        ax_trace.axvline(bin_centers[b], color='#cccccc', linewidth=0.6,
                         linestyle=':', zorder=1, alpha=0.7)

    # Baseline
    ax_trace.axhline(bsa_baseline, color='#999999', linewidth=0.7,
                     linestyle='--', zorder=1)
    ax_trace.text(bin_centers[-1] + 8, bsa_baseline, 'BSA',
                  fontsize=6, color='#999999', va='center')

    # Chance
    ax_trace.axhline(0.2, color='#bbbbbb', linewidth=0.5,
                     linestyle=':', zorder=1, alpha=0.5)
    ax_trace.text(bin_centers[-1] + 8, 0.2, 'chance',
                  fontsize=5.5, color='#aaaaaa', va='center')

    ax_trace.set_xlim(bin_centers[0] - 5, bin_centers[-1] + 5)
    ax_trace.set_ylim(0.17, 0.82)
    ax_trace.set_xlabel('Time (ms)', fontsize=9)
    ax_trace.set_ylabel('Classification Accuracy', fontsize=9)
    ax_trace.xaxis.set_major_locator(MultipleLocator(100))
    ax_trace.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_trace.spines['top'].set_visible(False)
    ax_trace.spines['right'].set_visible(False)

    leg = ax_trace.legend(loc='upper right', frameon=True, framealpha=0.9,
                          edgecolor='#dddddd', fontsize=7, handlelength=2,
                          borderpad=0.4, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.4)

    # Panel labels
    fig.text(0.02, 0.96, 'A', fontsize=14, fontweight='bold', va='top')
    fig.text(0.02, 0.52, 'B', fontsize=14, fontweight='bold', va='top')

    # Suptitle
    fig.suptitle(f'{branch_label}', fontsize=11, fontweight='bold', y=0.99)

    fig.savefig(filename, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Generate both figures ──
OUT_DIR = CPP_DIR / 'results' / 'classification_adaptation_sweep'
make_figure('A_unmatched', 'Branch A (unmatched)', sel_A,
            str(OUT_DIR / 'fig_sweep_branchA.png'))
make_figure('B_matched', 'Branch B (rate-matched)', sel_B,
            str(OUT_DIR / 'fig_sweep_branchB.png'))

print("\nDone!")
