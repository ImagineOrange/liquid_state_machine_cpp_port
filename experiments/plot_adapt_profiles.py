#!/usr/bin/env python3
"""
Adaptation profile snapshots: sorted adaptation conductance curves at key
timepoints, showing that the neuron-by-neuron pattern persists through
the silent gap.

Usage:
    python experiments/plot_adapt_profiles.py

Reads:  results/mechanistic_interp/example_rasters.json
Writes: results/adapt_profile_snapshots.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RASTER_PATH = 'results/mechanistic_interp/example_rasters.json'
OUT_PATH = 'results/adapt_profile_snapshots.png'
C_GRAY = '#888888'


def main():
    if not os.path.exists(RASTER_PATH):
        print(f'{RASTER_PATH} not found.')
        sys.exit(1)

    print(f'Loading {RASTER_PATH}...')
    with open(RASTER_PATH) as f:
        rdata = json.load(f)

    for entry in rdata['rasters']:
        if 'adapt_snapshots' not in entry or not entry['adapt_snapshots']:
            print(f"No adapt_snapshots in '{entry['label']}'")
            sys.exit(1)

    n_res = rdata['rasters'][0]['n_reservoir']
    a_end = rdata['stim_a_end_ms']
    b_start = rdata['stim_b_offset_ms']
    b_end = rdata['stim_b_end_ms']
    trial_end = b_end + rdata['post_stim_ms']
    gap_mid = (a_end + b_start) / 2.0

    # Parse adaptation matrices
    adapt_matrices = {}
    for entry in rdata['rasters']:
        n_snaps = entry['n_adapt_snapshots']
        flat = np.array(entry['adapt_snapshots'])
        adapt_matrices[entry['label']] = flat.reshape(n_snaps, n_res)

    # Define snapshot timepoints
    snapshots = [
        (int(a_end * 0.5),        'Mid-A',        '#2ca02c'),
        (int(a_end) - 1,          'End of A',     '#1a7a1a'),
        (int(gap_mid),            'Mid-Gap',      '#999999'),
        (int(b_start),            'B onset',      '#1f77b4'),
        (int((b_start+b_end)/2),  'Mid-B',        '#0d4a8a'),
        (int(b_end) - 1,          'End of B',     '#6a3d9a'),
    ]

    # Sort order: by adaptation at end of A in WM condition (most informative)
    wm_mat = adapt_matrices['wm_optimum']
    sort_idx = np.argsort(wm_mat[int(a_end) - 1, :])[::-1]

    # --- Figure ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.12, hspace=0.25,
                  left=0.07, right=0.95, top=0.90, bottom=0.08)

    fig.suptitle('Adaptation profile is preserved through the silent gap',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93,
             f'Single trial, digit A={rdata["digit_a"]}, digit B={rdata["digit_b"]}  |  '
             f'604 reservoir neurons sorted by end-of-A adaptation (WM condition)',
             ha='center', fontsize=10, color=C_GRAY)

    # --- Panel A: WM optimum sorted profiles ---
    ax_wm = fig.add_subplot(gs[0, 0])
    for t_ms, label, color in snapshots:
        t_ms = min(t_ms, wm_mat.shape[0] - 1)
        vals = wm_mat[t_ms, sort_idx]
        ax_wm.plot(np.arange(n_res), vals, color=color, lw=1.5,
                   label=f'{label} (t={t_ms} ms)', alpha=0.85)
    ax_wm.set_xlabel('Neuron (sorted by end-of-A adaptation)')
    ax_wm.set_ylabel('Adaptation conductance ($g_{adapt}$)')
    ax_wm.set_title(r'WM optimum ($\Delta_a$=0.171)', fontweight='bold')
    ax_wm.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_wm.set_xlim(0, n_res)

    # --- Panel B: No adapt sorted profiles (should be ~0) ---
    na_mat = adapt_matrices['no_adapt']
    ax_na = fig.add_subplot(gs[0, 1])
    for t_ms, label, color in snapshots:
        t_ms = min(t_ms, na_mat.shape[0] - 1)
        vals = na_mat[t_ms, sort_idx]
        ax_na.plot(np.arange(n_res), vals, color=color, lw=1.5,
                   label=f'{label} (t={t_ms} ms)', alpha=0.85)
    ax_na.set_xlabel('Neuron (sorted by end-of-A adaptation)')
    ax_na.set_ylabel('Adaptation conductance ($g_{adapt}$)')
    ax_na.set_title(r'No adaptation ($\Delta_a$=0)', fontweight='bold')
    ax_na.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_na.set_xlim(0, n_res)
    ax_na.set_ylim(ax_wm.get_ylim())  # match y-axis

    # --- Panel C: Correlation of adaptation profiles across timepoints ---
    # Shows how well the pattern at each timepoint matches end-of-A
    ax_corr = fig.add_subplot(gs[1, 0])
    ref_profile = wm_mat[int(a_end) - 1, :]  # end of A
    t_axis = np.arange(wm_mat.shape[0])
    correlations = np.array([
        np.corrcoef(ref_profile, wm_mat[t, :])[0, 1]
        if np.std(wm_mat[t, :]) > 1e-10 else 0.0
        for t in range(wm_mat.shape[0])
    ])

    ax_corr.plot(t_axis, correlations, color='#d62728', lw=1.5)
    ax_corr.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)

    # Epoch shading
    epoch_spans = [
        (0, a_end, '#2ca02c20', 'A'),
        (a_end, b_start, '#e0e0e0', 'Gap'),
        (b_start, b_end, '#1f77b420', 'B'),
        (b_end, trial_end, '#ff7f0e20', 'Post'),
    ]
    for t0, t1, fc, ep in epoch_spans:
        ax_corr.axvspan(t0, t1, facecolor=fc, edgecolor='none')
        ax_corr.text((t0+t1)/2, 1.02, ep, ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=C_GRAY,
                     transform=ax_corr.get_xaxis_transform())

    ax_corr.set_xlabel('Time (ms)')
    ax_corr.set_ylabel('Correlation with end-of-A profile')
    ax_corr.set_title('Pattern persistence (WM optimum)', fontweight='bold')
    ax_corr.set_xlim(0, trial_end)
    ax_corr.set_ylim(-0.1, 1.1)

    # --- Panel D: Euclidean distance from end-of-A ---
    ax_dist = fig.add_subplot(gs[1, 1])
    distances = np.array([
        np.linalg.norm(wm_mat[t, :] - ref_profile)
        for t in range(wm_mat.shape[0])
    ])
    # Normalize to max for readability
    ax_dist.plot(t_axis, distances, color='#9467bd', lw=1.5)
    for t0, t1, fc, ep in epoch_spans:
        ax_dist.axvspan(t0, t1, facecolor=fc, edgecolor='none')
        ax_dist.text((t0+t1)/2, 1.02, ep, ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=C_GRAY,
                     transform=ax_dist.get_xaxis_transform())
    ax_dist.set_xlabel('Time (ms)')
    ax_dist.set_ylabel('Euclidean distance from end-of-A')
    ax_dist.set_title('State drift (WM optimum)', fontweight='bold')
    ax_dist.set_xlim(0, trial_end)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
