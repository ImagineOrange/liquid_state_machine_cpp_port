#!/usr/bin/env python3
"""
Adaptation-level raster figure: neuron × time heatmap of adaptation
conductance, mirroring fig6_rasters layout (no_adapt vs WM optimum).

Usage:
    python experiments/plot_adapt_raster.py

Reads:  results/mechanistic_interp/example_rasters.json
        (must contain adapt_snapshots — regenerate with ./cls_sweep --mech-raster)
Writes: results/adapt_raster_heatmap.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d

RASTER_PATH = 'results/mechanistic_interp/example_rasters.json'
OUT_PATH = 'results/adapt_raster_heatmap.png'
C_GRAY = '#888888'


def main():
    if not os.path.exists(RASTER_PATH):
        print(f'{RASTER_PATH} not found. Regenerate with: ./cls_sweep --mech-raster')
        sys.exit(1)

    print(f'Loading {RASTER_PATH}...')
    with open(RASTER_PATH) as f:
        rdata = json.load(f)

    # Check that adapt_snapshots are present
    for entry in rdata['rasters']:
        if 'adapt_snapshots' not in entry or not entry['adapt_snapshots']:
            print(f"No adapt_snapshots in '{entry['label']}' — "
                  "regenerate with updated ./cls_sweep --mech-raster")
            sys.exit(1)

    n_res = rdata['rasters'][0]['n_reservoir']

    # Epoch boundaries (ms)
    a_end = rdata['stim_a_end_ms']
    b_start = rdata['stim_b_offset_ms']
    b_end = rdata['stim_b_end_ms']
    trial_end = b_end + rdata['post_stim_ms']

    # Parse adaptation data into time × neuron matrices
    adapt_matrices = {}
    for entry in rdata['rasters']:
        n_snaps = entry['n_adapt_snapshots']
        flat = np.array(entry['adapt_snapshots'])
        mat = flat.reshape(n_snaps, n_res)
        # Trim to trial_end
        n_ms = min(n_snaps, int(np.ceil(trial_end)))
        adapt_matrices[entry['label']] = mat[:n_ms, :]

    # Parse spike data for sorting
    spike_counts = {}
    for entry in rdata['rasters']:
        counts = np.zeros(n_res)
        for nid in entry['spike_neuron_pos']:
            if 0 <= nid < n_res:
                counts[nid] += 1
        spike_counts[entry['label']] = counts

    # Shared sort by average spike count (same as fig6)
    avg_counts = (spike_counts['no_adapt'] + spike_counts['wm_optimum']) / 2.0
    shared_order = np.argsort(avg_counts)[::-1]  # high activity at top

    # Compute mean adaptation per neuron over time for population plot
    mean_adapt = {}
    for lbl, mat in adapt_matrices.items():
        mean_adapt[lbl] = mat.mean(axis=1)  # mean across neurons at each ms

    # Global color scale across both panels (skip no_adapt if it's all zeros)
    wm_mat = adapt_matrices['wm_optimum'][::1, shared_order].T  # neuron × time
    na_mat = adapt_matrices['no_adapt'][::1, shared_order].T

    # Use WM panel to set color range (no_adapt will be ~0 everywhere)
    vmax = np.percentile(wm_mat, 99.5)
    vmin = 0.0

    # Epoch shading config
    epoch_spans = [
        (0, a_end, '#2ca02c20', 'A'),
        (a_end, b_start, '#f0f0f0', 'Gap'),
        (b_start, b_end, '#1f77b420', 'B'),
        (b_end, trial_end, '#ff7f0e20', 'Post'),
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.08, hspace=0.15,
                  height_ratios=[5, 1], left=0.05, right=0.92,
                  top=0.90, bottom=0.06)
    fig.suptitle('Adaptation conductance builds stimulus-specific persistent '
                 'state across the reservoir',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93,
             f'Single trial, digit A={rdata["digit_a"]}, digit B={rdata["digit_b"]}  |  '
             f'604 reservoir neurons sorted by mean spike count (shared order)',
             ha='center', fontsize=10, color=C_GRAY)

    cmaps = ['viridis', 'viridis']
    panels = [
        ('no_adapt', r'No adaptation ($\Delta_a$=0)'),
        ('wm_optimum', r'WM optimum ($\Delta_a$=0.171)'),
    ]

    for col, (label, title) in enumerate(panels):
        mat = adapt_matrices[label][:, shared_order].T  # neuron × time
        n_t = mat.shape[1]
        t_extent = n_t  # 1 ms per snapshot

        ax_h = fig.add_subplot(gs[0, col])
        ax_m = fig.add_subplot(gs[1, col])

        # Epoch shading on mean plot
        for t0, t1, fc, ep_label in epoch_spans:
            ax_m.axvspan(t0, t1, facecolor=fc, edgecolor='none')

        # Heatmap
        im = ax_h.imshow(mat, aspect='auto', origin='lower',
                         extent=[0, t_extent, 0, n_res],
                         cmap='inferno', vmin=vmin, vmax=vmax,
                         interpolation='bilinear')

        # Epoch boundary lines on heatmap
        for t0, t1, fc, ep_label in epoch_spans:
            for boundary in [t0, t1]:
                if 0 < boundary < t_extent:
                    ax_h.axvline(boundary, color='white', lw=0.8, alpha=0.7)
            mid = (t0 + t1) / 2
            ax_h.text(mid, n_res * 0.97, ep_label, ha='center', va='top',
                      fontsize=9, fontweight='bold', color='white',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                                edgecolor='none', alpha=0.5), clip_on=True)

        ax_h.set_xlim(0, trial_end)
        ax_h.set_ylim(0, n_res)
        ax_h.set_xticklabels([])
        if col == 0:
            ax_h.set_ylabel('Neuron (sorted by activity)')
        else:
            ax_h.set_yticklabels([])
        ax_h.set_title(title, fontweight='bold', fontsize=12)

        # Mean adaptation over time
        t_axis = np.arange(n_t)
        smoothed = gaussian_filter1d(mean_adapt[label][:n_t], sigma=5)
        ax_m.fill_between(t_axis, smoothed, color='#d62728', alpha=0.3)
        ax_m.plot(t_axis, smoothed, color='#d62728', lw=1.5)
        ax_m.set_xlim(0, trial_end)
        ax_m.set_xlabel('Time (ms)')
        if col == 0:
            ax_m.set_ylabel('Mean $g_{adapt}$', fontsize=9)
        else:
            ax_m.set_yticklabels([])

        # Shared y-axis for mean plots
        if col == 0:
            ymax_mean = max(
                gaussian_filter1d(mean_adapt['wm_optimum'][:int(trial_end)], 5).max(),
                gaussian_filter1d(mean_adapt['no_adapt'][:int(trial_end)], 5).max()
            ) * 1.15
        ax_m.set_ylim(0, ymax_mean)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Adaptation conductance ($g_{adapt}$)', fontsize=10)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
