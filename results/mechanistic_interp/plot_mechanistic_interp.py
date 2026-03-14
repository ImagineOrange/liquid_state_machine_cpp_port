#!/usr/bin/env python3
"""
Mechanistic Interpretability — Figure Generation
Produces 7 publication-quality figures from mechanistic_interp.json,
network_snapshot.npz, and example_rasters.json.

Usage:
    python3 results/mechanistic_interp/plot_mechanistic_interp.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
JSON_PATH = os.path.join(SCRIPT_DIR, 'mechanistic_interp.json')
RASTER_PATH = os.path.join(SCRIPT_DIR, 'example_rasters.json')
SNAP_PATH = os.path.join(PROJECT_ROOT, 'results', 'network_snapshot', 'network_snapshot.npz')
OUT_DIR = SCRIPT_DIR

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

# ── Colors ──
C_GREEN  = '#2ca02c'
C_RED    = '#d62728'
C_BLUE   = '#1f77b4'
C_ORANGE = '#ff7f0e'
C_DKBLUE = '#1a3a5c'
C_GRAY   = '#7f7f7f'
C_PURPLE = '#9467bd'

POINT_COLORS = [C_DKBLUE, C_GREEN, C_RED, C_BLUE]
POINT_LABELS = [
    r'No adapt ($\Delta_a$=0)',
    r'CLS opt ($\Delta_a$=0.071)',
    r'Per-bin opt ($\Delta_a$=0.127)',
    r'WM opt ($\Delta_a$=0.171)',
]
Q_COLORS = [C_GREEN, C_ORANGE, C_RED, C_BLUE]
Q_LABELS = ['Q1 (low input)', 'Q2', 'Q3', 'Q4 (high input)']
DIGIT_COLORS = [C_GREEN, C_BLUE, C_RED, C_ORANGE, C_PURPLE]


def load_data():
    with open(JSON_PATH) as f:
        data = json.load(f)
    snap = np.load(SNAP_PATH, allow_pickle=True)
    return data, snap


def compute_input_weights(snap):
    """Total absolute input weight from input neurons to each reservoir neuron."""
    W = snap['weights']
    input_idx = snap['input_neuron_indices']
    res_idx = snap['reservoir_zone_indices']
    n_res = len(res_idx)
    input_weight = np.zeros(n_res)
    for src in input_idx:
        for p, gid in enumerate(res_idx):
            w = W[src, gid]
            if w != 0:
                input_weight[p] += abs(w)
    return input_weight


def quartile_masks(input_weight):
    q25, q50, q75 = np.percentile(input_weight, [25, 50, 75])
    return [
        input_weight <= q25,
        (input_weight > q25) & (input_weight <= q50),
        (input_weight > q50) & (input_weight <= q75),
        input_weight > q75,
    ]


# ═════════════════════════════════════════════════════════════
# FIGURE 1: Headline — adaptation conductance IS the memory
# ═════════════════════════════════════════════════════════════
def fig1_headline(results, inc_vals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Adaptation conductance is the memory',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93,
             'Ridge classification of digit A identity, 5-fold x 5-repeat CV, Branch B rate-matched at 20 Hz',
             ha='center', fontsize=10, color=C_GRAY)

    # ── Panel A: bar chart ──
    ax = axes[0]
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')

    wm_accs   = [r['wm_accuracy'] * 100 for r in results]
    wm_stds   = [r['wm_accuracy_std'] * 100 for r in results]
    adapt_accs = [r['adapt_state_wm_accuracy'] * 100 for r in results]
    adapt_stds = [r['adapt_state_wm_accuracy_std'] * 100 for r in results]

    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w/2, adapt_accs, w, yerr=adapt_stds, color=C_RED, alpha=0.85,
           label='Adaptation state (604-d)', capsize=3, error_kw={'linewidth': 1.2})
    ax.bar(x + w/2, wm_accs, w, yerr=wm_stds, color=C_BLUE, alpha=0.85,
           label='B+Post spike counts (4228-d)', capsize=3, error_kw={'linewidth': 1.2})
    ax.axhline(20, color=C_GRAY, ls=':', lw=1, label='Chance (20%)')

    ax.set_xticks(x)
    ax.set_xticklabels([r['label'].replace('_', '\n') for r in results], fontsize=9)
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Adaptation state vs. spike-count readout', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    for i in range(1, len(results)):
        gap = adapt_accs[i] - wm_accs[i]
        mid_y = (adapt_accs[i] + wm_accs[i]) / 2
        ax.annotate(f'{gap:.0f} pp\ngap', xy=(x[i], mid_y), fontsize=8,
                    ha='center', color=C_GRAY, fontweight='bold')

    # ── Panel B: stim-adapt correlation ──
    ax = axes[1]
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold')

    for di in range(5):
        vals = [results[pi]['stim_adapt_r_per_digit'][di] for pi in range(len(results))]
        ax.plot(inc_vals, vals, 'o-', color=DIGIT_COLORS[di],
                label=f'Digit {di}', markersize=6, linewidth=1.5)

    mean_vals = [r['stim_adapt_r_mean'] for r in results]
    ax.plot(inc_vals, mean_vals, 's--', color='black', label='Mean',
            markersize=7, linewidth=2, zorder=5)

    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Pearson r (A spikes vs adapt at B onset)')
    ax.set_title('Link 1: Stimulus drive to adaptation trace', fontweight='bold')
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc='center left', framealpha=0.9, ncol=2)
    ax.axhline(0, color=C_GRAY, ls=':', lw=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, 'fig1_headline.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 2: Ridge weight structure
# ═════════════════════════════════════════════════════════════
def fig2_weight_structure(results, inc_vals, qmasks):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ridge weight structure: how the classifier reads adaptation through spikes',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93,
             'Absolute ridge weight energy decomposed by temporal bin and input-weight quartile',
             ha='center', fontsize=10, color=C_GRAY)

    bin_labels = ['B1', 'B2', 'B3', 'B4', 'B5', 'Post1', 'Post2']

    # ── Panel A: temporal weight distribution ──
    ax = axes[0]
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for pi, r in enumerate(results):
        rw = np.array(r['ridge_weights_wm'])
        n_bins = rw.shape[0] // r['n_reservoir']
        abs_w = np.abs(rw).sum(axis=1).reshape(n_bins, r['n_reservoir'])
        bin_frac = abs_w.sum(axis=1)
        bin_frac = bin_frac / bin_frac.sum() * 100
        ax.plot(range(n_bins), bin_frac, 'o-', color=POINT_COLORS[pi],
                label=POINT_LABELS[pi], linewidth=1.5, markersize=5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel('Weight energy share (%)')
    ax.set_title('Temporal weight concentration', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)

    # ── Panel B: B vs Post stacked bar ──
    ax = axes[1]
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold')
    b_fracs, post_fracs = [], []
    for r in results:
        rw = np.array(r['ridge_weights_wm'])
        n_bins = rw.shape[0] // r['n_reservoir']
        abs_w = np.abs(rw).sum(axis=1).reshape(n_bins, r['n_reservoir'])
        be = abs_w.sum(axis=1)
        total = be.sum()
        b_fracs.append(be[:5].sum() / total * 100)
        post_fracs.append(be[5:].sum() / total * 100)

    x = np.arange(len(results))
    ax.bar(x, b_fracs, color=C_RED, alpha=0.85, label='B-epoch (B1-B5)')
    ax.bar(x, post_fracs, bottom=b_fracs, color=C_BLUE, alpha=0.85, label='Post-stimulus')
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'].replace('_', '\n') for r in results], fontsize=9)
    ax.set_ylabel('Weight energy share (%)')
    ax.set_ylim(0, 105)
    ax.set_title('B-epoch dominates readout', fontweight='bold')
    ax.legend(framealpha=0.9)
    for i in range(len(results)):
        ax.text(x[i], b_fracs[i] / 2, f'{b_fracs[i]:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(x[i], b_fracs[i] + post_fracs[i] / 2, f'{post_fracs[i]:.1f}%',
                ha='center', va='center', fontsize=8, color='white')

    # ── Panel C: weight share by quartile ──
    ax = axes[2]
    ax.text(-0.12, 1.05, 'C', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for qi in range(4):
        fracs = []
        for r in results:
            rw = np.array(r['ridge_weights_wm'])
            awpn = np.abs(rw).sum(axis=1).reshape(-1, r['n_reservoir']).sum(axis=0)
            total = awpn.sum()
            fracs.append(awpn[qmasks[qi]].sum() / total * 100)
        ax.plot(inc_vals, fracs, 'o-', color=Q_COLORS[qi], label=Q_LABELS[qi],
                linewidth=1.8, markersize=6)
    ax.axhline(25, color=C_GRAY, ls=':', lw=1, label='Uniform (25%)')
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Ridge weight share (%)')
    ax.set_title('Adaptation equalizes weight across quartiles', fontweight='bold')
    ax.set_ylim(10, 40)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, 'fig2_weight_structure.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 3: Population equalization
# ═════════════════════════════════════════════════════════════
def fig3_population_equalization(results, inc_vals, qmasks):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Adaptation redistributes discriminative information across the population',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93,
             'Neurons binned by total input connection weight from input layer (quartiles)',
             ha='center', fontsize=10, color=C_GRAY)

    # ── Panel A: firing rate by quartile ──
    ax = axes[0]
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for qi in range(4):
        rates = []
        for r in results:
            sel = np.array(r['digit_selectivity_vectors'])
            rates.append(sel.mean(axis=0)[qmasks[qi]].mean())
        ax.plot(inc_vals, rates, 'o-', color=Q_COLORS[qi], label=Q_LABELS[qi],
                linewidth=1.8, markersize=6)
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Mean firing rate (spk/ms)')
    ax.set_title('Firing rate by input-weight quartile', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)

    # ── Panel B: discrimination index by quartile ──
    ax = axes[1]
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for qi in range(4):
        disc_vals = []
        for r in results:
            sel = np.array(r['digit_selectivity_vectors'])
            grand_mean = sel.mean(axis=0)
            std_across = np.sqrt(np.var(sel, axis=0)[qmasks[qi]])
            safe_mean = np.where(grand_mean[qmasks[qi]] > 1e-9,
                                 grand_mean[qmasks[qi]], 1e-9)
            disc_vals.append((std_across / safe_mean).mean())
        ax.plot(inc_vals, disc_vals, 'o-', color=Q_COLORS[qi], label=Q_LABELS[qi],
                linewidth=1.8, markersize=6)
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Discrimination index (CV across digits)')
    ax.set_title('Adaptation compresses discrimination', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)

    # ── Panel C: between-class correlation by quartile ──
    ax = axes[2]
    ax.text(-0.12, 1.05, 'C', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for qi in range(4):
        corr_vals = []
        for r in results:
            sel = np.array(r['digit_selectivity_vectors'])
            corr_mat = np.corrcoef(sel[:, qmasks[qi]])
            offdiag = ~np.eye(5, dtype=bool)
            corr_vals.append(corr_mat[offdiag].mean())
        ax.plot(inc_vals, corr_vals, 'o-', color=Q_COLORS[qi], label=Q_LABELS[qi],
                linewidth=1.8, markersize=6)
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Mean between-class correlation')
    ax.set_title('Selectivity structure is uniform', fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, 'fig3_population_equalization.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 4: PCA + selectivity structure
# ═════════════════════════════════════════════════════════════
def fig4_representation_structure(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Adaptation state is a low-dimensional, high-fidelity representation of digit identity',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93,
             'PCA on mean adaptation conductance vectors at B onset (5 digit classes x 604 neurons)',
             ha='center', fontsize=10, color=C_GRAY)

    # ── Panel A: PCA variance explained ──
    ax = axes[0]
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')
    for pi in range(1, len(results)):  # skip no_adapt
        adapt_vecs = np.array(results[pi]['mean_adapt_at_b_onset_by_digit'])
        pca = PCA()
        pca.fit(adapt_vecs)
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color=POINT_COLORS[pi],
                label=POINT_LABELS[pi], linewidth=1.8, markersize=7)
    ax.axhline(100, color=C_GRAY, ls=':', lw=0.8)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('PCA on adaptation vectors', fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(40, 105)
    ax.legend(fontsize=8, framealpha=0.9)

    # ── Panel B: selectivity correlation matrix (WM optimum) ──
    ax = axes[1]
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold')
    corr = np.array(results[3]['selectivity_corr_matrix'])
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=0.85, vmax=1.0, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'Digit {i}' for i in range(5)])
    ax.set_yticklabels([f'Digit {i}' for i in range(5)])
    ax.set_title('Digit selectivity correlation (WM optimum)', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{corr[i, j]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if corr[i, j] > 0.95 else 'black')

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, 'fig4_representation_structure.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 5: The mode switch — concentrated to distributed code
# ═════════════════════════════════════════════════════════════
def _get_neuron_weights(r, n_res):
    rw = np.array(r['ridge_weights_wm'])
    n_bins = rw.shape[0] // n_res
    return np.abs(rw).sum(axis=1).reshape(n_bins, n_res).sum(axis=0)


def _gini(arr):
    s = np.sort(arr)
    n = len(s)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n


def fig5_mode_switch(results, inc_vals, input_weight):
    from matplotlib.gridspec import GridSpec

    n_res = len(input_weight)
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.97, top=0.88, bottom=0.08)

    fig.suptitle('The adaptation mode switch: from concentrated rate code '
                 'to distributed conductance code',
                 fontsize=16, fontweight='bold', y=0.96)
    fig.text(0.5, 0.92,
             '604-neuron reservoir, Branch B rate-matched at 20 Hz',
             ha='center', fontsize=10, color=C_GRAY)

    # ── Panel A: Lorenz curves ──
    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.14, 1.05, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax.plot([0, 1], [0, 1], '--', color='#d0d0d0', lw=1.5, label='Perfect equality')

    ginis = []
    for pi, r in enumerate(results):
        awpn = _get_neuron_weights(r, n_res)
        sort_idx = np.argsort(awpn)
        cum = np.concatenate([[0], np.cumsum(awpn[sort_idx]) / awpn.sum()])
        xf = np.linspace(0, 1, len(cum))
        g = _gini(awpn)
        ginis.append(g)
        ax.plot(xf, cum, '-', color=POINT_COLORS[pi], lw=2.5,
                label=f'{r["label"].replace("_", " ")} (Gini={g:.2f})')
        if pi == 0:
            ax.fill_between(xf, xf, cum, alpha=0.10, color=C_DKBLUE)

    ax.set_xlabel('Fraction of neurons (sorted by ridge weight)')
    ax.set_ylabel('Cumulative ridge weight share')
    ax.set_title('Classifier weight inequality', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # ── Panel B: Readout bottleneck ──
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.14, 1.05, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold')

    adapt_accs = [r['adapt_state_wm_accuracy'] * 100 for r in results]
    spike_accs = [r['wm_accuracy'] * 100 for r in results]
    x = np.arange(4)
    width = 0.55

    for i in range(4):
        ax.bar(x[i], adapt_accs[i], width, color=POINT_COLORS[i], alpha=0.20,
               edgecolor=POINT_COLORS[i], linewidth=2)
        ax.bar(x[i], spike_accs[i], width * 0.75, color=POINT_COLORS[i], alpha=0.85)

    ax.axhline(20, color=C_GRAY, ls=':', lw=1)
    for i in range(1, 4):
        ybot = spike_accs[i] + 1
        ytop = adapt_accs[i] - 1
        ymid = (ybot + ytop) / 2
        loss = adapt_accs[i] - spike_accs[i]
        ax.annotate('', xy=(x[i] + width / 2 + 0.02, ybot),
                    xytext=(x[i] + width / 2 + 0.02, ytop),
                    arrowprops=dict(arrowstyle='<->', color=C_GRAY, lw=1.5))
        ax.text(x[i] + width / 2 + 0.08, ymid, f'-{loss:.0f}pp\nloss',
                fontsize=7.5, color=C_GRAY, ha='left', va='center', fontweight='bold')

    ax.text(0.98, 0.98, 'Outer: stored in conductance\nInner: recovered from spikes',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#d0d0d0', alpha=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels(['No\nadapt', 'CLS\nopt', 'Per-bin\nopt', 'WM\nopt'], fontsize=9)
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_title('The readout bottleneck', fontweight='bold')
    ax.set_ylim(0, 110)

    # ── Panel C: Ridge weight share by input-weight decile ──
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.14, 1.05, 'C', transform=ax.transAxes, fontsize=16, fontweight='bold')

    n_deciles = 10
    decile_edges = np.percentile(input_weight, np.linspace(0, 100, n_deciles + 1))

    for pi, idx in enumerate([0, 1, 3]):
        r = results[idx]
        awpn = _get_neuron_weights(r, n_res)
        total = awpn.sum()
        shares = []
        for di in range(n_deciles):
            if di < n_deciles - 1:
                mask = (input_weight >= decile_edges[di]) & (input_weight < decile_edges[di + 1])
            else:
                mask = input_weight >= decile_edges[di]
            shares.append(awpn[mask].sum() / total * 100)
        color = [C_DKBLUE, C_GREEN, C_BLUE][pi]
        label = ['No adapt', 'CLS opt', 'WM opt'][pi]
        marker = ['o', 's', 'D'][pi]
        ax.plot(range(1, n_deciles + 1), shares, f'{marker}-', color=color,
                lw=2.2, markersize=7, label=label)

    ax.axhline(10, color='#d0d0d0', ls='--', lw=1, label='Uniform (10%)')
    ax.set_xlabel('Input weight decile (1=weakest, 10=strongest)')
    ax.set_ylabel('Share of total ridge weight (%)')
    ax.set_title('Where the classifier looks', fontweight='bold')
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(4, 22)

    # ── Panel D: Discrimination index by decile ──
    ax = fig.add_subplot(gs[1, 0])
    ax.text(-0.14, 1.05, 'D', transform=ax.transAxes, fontsize=16, fontweight='bold')

    for pi, idx in enumerate([0, 1, 3]):
        r = results[idx]
        sel = np.array(r['digit_selectivity_vectors'])
        grand_mean = sel.mean(axis=0)
        std_across = np.sqrt(np.var(sel, axis=0))
        safe_mean = np.where(grand_mean > 1e-9, grand_mean, 1e-9)
        disc = std_across / safe_mean
        dec_disc = []
        for di in range(n_deciles):
            if di < n_deciles - 1:
                mask = (input_weight >= decile_edges[di]) & (input_weight < decile_edges[di + 1])
            else:
                mask = input_weight >= decile_edges[di]
            dec_disc.append(disc[mask].mean())
        color = [C_DKBLUE, C_GREEN, C_BLUE][pi]
        label = ['No adapt', 'CLS opt', 'WM opt'][pi]
        marker = ['o', 's', 'D'][pi]
        ax.plot(range(1, n_deciles + 1), dec_disc, f'{marker}-', color=color,
                lw=2.2, markersize=7, label=label)

    ax.set_xlabel('Input weight decile (1=weakest, 10=strongest)')
    ax.set_ylabel('Discrimination index\n(CV across digit-specific rates)')
    ax.set_title('Digit selectivity by input drive', fontweight='bold')
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xticks(range(1, 11))

    # ── Panel E: Combined mode switch metric ──
    ax = fig.add_subplot(gs[1, 1])
    ax.text(-0.14, 1.05, 'E', transform=ax.transAxes, fontsize=16, fontweight='bold')

    iw_corrs = [np.corrcoef(input_weight, _get_neuron_weights(r, n_res))[0, 1]
                for r in results]

    ax2 = ax.twinx()
    l1, = ax.plot(inc_vals, ginis, 'o-', color=C_RED, lw=2.5, markersize=9,
                  label='Weight Gini coefficient')
    l2, = ax.plot(inc_vals, iw_corrs, 's-', color=C_ORANGE, lw=2.5, markersize=9,
                  label='Input wt / ridge wt correlation')
    l3, = ax2.plot(inc_vals, spike_accs, 'D-', color=C_BLUE, lw=2.5, markersize=9,
                   label='WM spike accuracy (%)')

    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Concentration metrics')
    ax2.set_ylabel('WM accuracy (%)', color=C_BLUE)
    ax2.tick_params(axis='y', labelcolor=C_BLUE)
    ax.axhline(0, color='#d0d0d0', ls=':', lw=0.8)
    ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
              loc='center right', fontsize=8.5, framealpha=0.9)
    ax.set_title('Concentration drops, accuracy rises', fontweight='bold')

    out = os.path.join(OUT_DIR, 'fig5_mode_switch.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 6: Raster plots — mode switch from actual activity
# ═════════════════════════════════════════════════════════════
def fig6_rasters(results):
    """Side-by-side rasters: no_adapt vs WM optimum, shared neuron sort."""
    from matplotlib.gridspec import GridSpec

    if not os.path.exists(RASTER_PATH):
        print(f'Skipping fig6: {RASTER_PATH} not found '
              '(generate with ./cls_sweep --mech-raster)')
        return

    with open(RASTER_PATH) as f:
        rdata = json.load(f)

    n_res = rdata['rasters'][0]['n_reservoir']

    # Epoch boundaries (ms)
    a_end = rdata['stim_a_end_ms']
    b_start = rdata['stim_b_offset_ms']
    b_end = rdata['stim_b_end_ms']
    trial_end = b_end + rdata['post_stim_ms']

    # Parse spike data
    rasters = {}
    for entry in rdata['rasters']:
        times = np.array(entry['spike_times_ms'])
        pos = np.array(entry['spike_neuron_pos'])
        rasters[entry['label']] = (times, pos)

    # Shared sort: average spike count across both conditions
    counts_na = np.zeros(n_res)
    counts_wm = np.zeros(n_res)
    for nid in rasters['no_adapt'][1]:
        if 0 <= nid < n_res:
            counts_na[nid] += 1
    for nid in rasters['wm_optimum'][1]:
        if 0 <= nid < n_res:
            counts_wm[nid] += 1
    avg_counts = (counts_na + counts_wm) / 2.0
    shared_order = np.argsort(avg_counts)
    pos_to_row = np.zeros(n_res, dtype=int)
    for row, nid in enumerate(shared_order):
        pos_to_row[nid] = row

    # Ridge weight sidebar (from main results: no_adapt=0, wm_optimum=3)
    awpn_na = _get_neuron_weights(results[0], n_res)
    awpn_wm = _get_neuron_weights(results[3], n_res)

    # Epoch shading config
    epoch_spans = [
        (0, a_end, '#2ca02c20', 'A'),
        (a_end, b_start, '#f0f0f0', 'Gap'),
        (b_start, b_end, '#1f77b420', 'B'),
        (b_end, trial_end, '#ff7f0e20', 'Post'),
    ]

    # Compute smoothed PSTH for both conditions
    from scipy.ndimage import gaussian_filter1d
    psth_bin_ms = 1.0  # 1 ms bins
    psth_n_bins = int(np.ceil(trial_end / psth_bin_ms))
    psth_sigma = 10.0  # 10 ms Gaussian kernel
    psths = {}
    for lbl in ['no_adapt', 'wm_optimum']:
        times_all = rasters[lbl][0]
        hist, _ = np.histogram(times_all, bins=psth_n_bins,
                               range=(0, trial_end))
        # Convert to Hz: spikes per bin / (bin_width_s * n_neurons)
        rate_hz = hist / (psth_bin_ms / 1000.0) / n_res
        psths[lbl] = gaussian_filter1d(rate_hz.astype(float), psth_sigma)
    psth_t = np.linspace(psth_bin_ms / 2, trial_end - psth_bin_ms / 2,
                         psth_n_bins)
    psth_ymax = max(psths['no_adapt'].max(), psths['wm_optimum'].max()) * 1.15

    # Ridge weights sorted in shared order, normalized to global max
    awpn_na_sorted = awpn_na[shared_order]
    awpn_wm_sorted = awpn_wm[shared_order]
    ridge_global_max = max(awpn_na_sorted.max(), awpn_wm_sorted.max())

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 2, figure=fig, wspace=0.08, hspace=0.12,
                  height_ratios=[5, 0.8, 0.8], left=0.05, right=0.95,
                  top=0.90, bottom=0.05)
    fig.suptitle('Spike-frequency adaptation redistributes classifier-relevant '
                 'activity across the reservoir',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93,
             f'Single trial, digit A={rdata["digit_a"]}, digit B={rdata["digit_b"]}  |  '
             f'604 reservoir neurons sorted by mean spike count (shared order)',
             ha='center', fontsize=10, color=C_GRAY)

    C_DKBLUE_DARK = '#0f2640'
    C_BLUE_DARK = '#0d4a8a'
    panels = [
        ('no_adapt', r'No adaptation ($\Delta_a$=0)', C_DKBLUE_DARK,
         awpn_na_sorted),
        ('wm_optimum', r'WM optimum ($\Delta_a$=0.171)', C_BLUE_DARK,
         awpn_wm_sorted),
    ]

    for col, (label, title, color, sorted_awpn) in enumerate(panels):
        # Row 0: raster
        ax_r = fig.add_subplot(gs[0, col])
        # Row 1: PSTH
        ax_p = fig.add_subplot(gs[1, col])
        # Row 2: ridge weights
        ax_w = fig.add_subplot(gs[2, col])

        times, pos = rasters[label]

        # Epoch shading on raster and PSTH
        for t0, t1, fc, ep_label in epoch_spans:
            ax_r.axvspan(t0, t1, facecolor=fc, edgecolor='none')
            ax_p.axvspan(t0, t1, facecolor=fc, edgecolor='none')
            mid = (t0 + t1) / 2
            ax_r.text(mid, n_res * 0.97, ep_label, ha='center', va='top',
                      fontsize=9, fontweight='bold', color=C_GRAY,
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='none', alpha=0.85), clip_on=True)

        # Build spike trains per row for eventplot
        trains = [[] for _ in range(n_res)]
        for t, p in zip(times, pos):
            if 0 <= p < n_res:
                trains[pos_to_row[p]].append(t)

        ax_r.eventplot(trains, colors=color, lineoffsets=np.arange(n_res),
                       linelengths=0.9, linewidths=1.0)

        ax_r.set_xlim(0, trial_end)
        ax_r.set_ylim(-1, n_res)
        ax_r.set_xticklabels([])
        if col == 0:
            ax_r.set_ylabel('Neuron (sorted by activity)')
        else:
            ax_r.set_yticklabels([])
        ax_r.set_title(title, fontweight='bold', fontsize=12)

        rate = rdata['rasters'][col]['rate_hz']
        n_spikes = rdata['rasters'][col]['n_spikes']
        ax_r.text(0.98, 0.02, f'{n_spikes:,} spikes | {rate:.1f} Hz',
                  transform=ax_r.transAxes, ha='right', va='bottom',
                  fontsize=8, color=C_GRAY,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#d0d0d0', alpha=0.9))

        # PSTH subplot
        ax_p.fill_between(psth_t, psths[label], color=color, alpha=0.3)
        ax_p.plot(psth_t, psths[label], color=color, lw=1.2)
        ax_p.set_xlim(0, trial_end)
        ax_p.set_ylim(0, psth_ymax)
        ax_p.set_xticklabels([])
        if col == 0:
            ax_p.set_ylabel('Rate (Hz)', fontsize=9)
        else:
            ax_p.set_yticklabels([])

        # Ridge weight binned bar chart (flipped: high-activity on left)
        n_bins_ridge = 30
        flipped = sorted_awpn[::-1]
        bin_edges = np.linspace(0, n_res, n_bins_ridge + 1, dtype=int)
        binned = np.array([flipped[bin_edges[i]:bin_edges[i+1]].mean()
                           for i in range(n_bins_ridge)])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = n_res / n_bins_ridge
        ax_w.bar(bin_centers, binned / ridge_global_max,
                 width=bin_width * 0.9, color=color, alpha=0.7,
                 edgecolor='none')
        ax_w.set_xlim(-1, n_res)
        ax_w.set_ylim(0, 1.05)
        ax_w.set_xlabel('Neuron (high activity \u2192 low)')
        if col == 0:
            ax_w.set_ylabel('Ridge wt', fontsize=9)
        else:
            ax_w.set_yticklabels([])
        ax_w.set_yticks([0, 0.5, 1.0])

    out = os.path.join(OUT_DIR, 'fig6_rasters.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# FIGURE 7: Redistribution is adaptation-driven, not a calibration artifact
# ═════════════════════════════════════════════════════════════
def fig7_calibration_control(results, inc_vals, qmasks):
    """Show that rate equalization is driven by adaptation, not tonic calibration."""
    from matplotlib.gridspec import GridSpec

    n_res = results[0]['n_reservoir']

    fig = plt.figure(figsize=(18, 10))
    gs_top = GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.30,
                      left=0.06, right=0.97, top=0.88, bottom=0.55)
    gs_bot = GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.30,
                      left=0.12, right=0.88, top=0.45, bottom=0.08)
    fig.suptitle('Rate redistribution is a robust consequence of adaptation, '
                 'not a calibration artifact',
                 fontsize=15, fontweight='bold', y=0.96)
    fig.text(0.5, 0.92,
             'Tonic conductance calibration is uniform across neurons; '
             'adaptation is activity-dependent',
             ha='center', fontsize=10, color=C_GRAY)

    # ── Panel A: g_tonic vs adapt_inc ──
    ax = fig.add_subplot(gs_top[0, 0])
    ax.text(-0.14, 1.05, 'A', transform=ax.transAxes, fontsize=16,
            fontweight='bold')
    g_tonics = [r['tonic_conductance'] for r in results]
    ax.plot(inc_vals, g_tonics, 'o-', color=C_PURPLE, lw=2.5, markersize=9)
    for i, r in enumerate(results):
        ax.annotate(f'{g_tonics[i]:.2f}', (inc_vals[i], g_tonics[i]),
                    textcoords='offset points', xytext=(8, 5), fontsize=9,
                    color=C_PURPLE, fontweight='bold')
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Tonic conductance (nS)')
    ax.set_title('Calibration effort decreases with adaptation',
                 fontweight='bold')
    ax.text(0.97, 0.97,
            'More adaptation\n= less tonic drive needed\n= calibration does LESS',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            color=C_GRAY, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#d0d0d0', alpha=0.9))

    # ── Panel B: Rate Gini drops with adaptation ──
    ax = fig.add_subplot(gs_top[0, 1])
    ax.text(-0.14, 1.05, 'B', transform=ax.transAxes, fontsize=16,
            fontweight='bold')
    ginis = []
    cvs = []
    for r in results:
        sel = np.array(r['digit_selectivity_vectors'])
        gm = sel.mean(axis=0)
        s = np.sort(gm)
        n = len(s)
        idx = np.arange(1, n + 1)
        ginis.append((2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n)
        cvs.append(np.std(gm) / np.mean(gm))

    ax.plot(inc_vals, ginis, 'o-', color=C_RED, lw=2.5, markersize=9,
            label='Firing rate Gini')
    ax.plot(inc_vals, cvs, 's-', color=C_ORANGE, lw=2.5, markersize=9,
            label='Firing rate CV')
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Inequality metric')
    ax.set_title('Adaptation compresses the rate distribution',
                 fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    for i in range(len(results)):
        ax.annotate(f'{ginis[i]:.2f}', (inc_vals[i], ginis[i]),
                    textcoords='offset points', xytext=(-5, -18), fontsize=8,
                    color=C_RED, ha='center')

    # ── Panel C: Top-10% share of total spikes ──
    ax = fig.add_subplot(gs_top[0, 2])
    ax.text(-0.14, 1.05, 'C', transform=ax.transAxes, fontsize=16,
            fontweight='bold')
    top10_shares = []
    for r in results:
        sel = np.array(r['digit_selectivity_vectors'])
        gm = sel.mean(axis=0)
        top10_shares.append(
            np.sort(gm)[-int(n_res * 0.1):].sum() / gm.sum() * 100)

    ax.bar(range(len(results)), top10_shares,
           color=POINT_COLORS, alpha=0.85, edgecolor='none')
    ax.axhline(10, color=C_GRAY, ls=':', lw=1, label='Uniform (10%)')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(['No\nadapt', 'CLS\nopt', 'Per-bin\nopt', 'WM\nopt'],
                       fontsize=9)
    ax.set_ylabel('Share of total spikes (%)')
    ax.set_title('Top 10% of neurons: spike share', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    for i, v in enumerate(top10_shares):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10,
                fontweight='bold', color=POINT_COLORS[i])

    # ── Panel D: Quartile firing rates ──
    ax = fig.add_subplot(gs_bot[0, 0])
    ax.text(-0.14, 1.05, 'D', transform=ax.transAxes, fontsize=16,
            fontweight='bold')
    for qi in range(4):
        rates = []
        for r in results:
            sel = np.array(r['digit_selectivity_vectors'])
            gm = sel.mean(axis=0)
            rates.append(gm[qmasks[qi]].mean())
        ax.plot(inc_vals, rates, 'o-', color=Q_COLORS[qi],
                label=Q_LABELS[qi], lw=2, markersize=7)
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Mean firing rate (spk/ms)')
    ax.set_title('Quartile rates converge with adaptation',
                 fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.text(0.97, 0.03,
            'Q4 suppressed, Q1-Q3 recruited\n= activity-dependent compression',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            color=C_GRAY, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#d0d0d0', alpha=0.9))

    # ── Panel E: Q4/Q1 ratio ──
    ax = fig.add_subplot(gs_bot[0, 1])
    ax.text(-0.14, 1.05, 'E', transform=ax.transAxes, fontsize=16,
            fontweight='bold')
    ratios = []
    for r in results:
        sel = np.array(r['digit_selectivity_vectors'])
        gm = sel.mean(axis=0)
        q1_mean = max(gm[qmasks[0]].mean(), 1e-9)
        q4_mean = gm[qmasks[3]].mean()
        ratios.append(q4_mean / q1_mean)
    ax.plot(inc_vals, ratios, 'D-', color=C_DKBLUE, lw=2.5, markersize=10)
    for i in range(len(results)):
        ax.annotate(f'{ratios[i]:.1f}x', (inc_vals[i], ratios[i]),
                    textcoords='offset points', xytext=(8, 5), fontsize=10,
                    color=C_DKBLUE, fontweight='bold')
    ax.set_xlabel(r'Adaptation increment ($\Delta_a$)')
    ax.set_ylabel('Q4 / Q1 rate ratio')
    ax.set_title('Rate dominance ratio collapses', fontweight='bold')
    ax.axhline(1, color=C_GRAY, ls=':', lw=1, label='Perfect equality')
    ax.legend(fontsize=9, framealpha=0.9)

    out = os.path.join(OUT_DIR, 'fig7_calibration_control.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    data, snap = load_data()
    results = data['results']
    inc_vals = [r['adapt_inc'] for r in results]

    iw = compute_input_weights(snap)
    qmasks = quartile_masks(iw)

    print(f"Input weight: min={iw.min():.4f}, max={iw.max():.4f}, "
          f"mean={iw.mean():.4f}, median={np.median(iw):.4f}")
    print(f"Nonzero: {np.count_nonzero(iw)}/{len(iw)}")

    fig1_headline(results, inc_vals)
    fig2_weight_structure(results, inc_vals, qmasks)
    fig3_population_equalization(results, inc_vals, qmasks)
    fig4_representation_structure(results)
    fig5_mode_switch(results, inc_vals, iw)
    fig6_rasters(results)
    fig7_calibration_control(results, inc_vals, qmasks)

    print("\nAll figures saved to", OUT_DIR)


if __name__ == '__main__':
    main()
