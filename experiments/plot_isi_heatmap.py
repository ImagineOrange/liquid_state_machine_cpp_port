#!/usr/bin/env python3
"""
ISI heatmap: neuron × time heatmap of inter-spike intervals at the WM optimum,
showing the transition from burst-pause to adapted steady-state within a trial.

Usage:
    python experiments/plot_isi_heatmap.py

Reads:  results/mechanistic_interp/example_rasters.json
Writes: results/isi_heatmap_wm.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

RASTER_PATH = 'results/mechanistic_interp/example_rasters.json'
OUT_PATH = 'results/isi_heatmap_wm.png'
C_GRAY = '#888888'


def build_isi_matrix(times, positions, n_res, trial_end, bin_ms=10.0):
    """Compute median ISI per neuron per time bin."""
    n_bins = int(np.ceil(trial_end / bin_ms))

    # Collect spike times per neuron
    neuron_spikes = [[] for _ in range(n_res)]
    for t, p in zip(times, positions):
        if 0 <= p < n_res:
            neuron_spikes[p].append(t)

    # Sort each neuron's spikes
    for n in range(n_res):
        neuron_spikes[n].sort()

    # Compute ISI assigned to each time bin (using midpoint of spike pair)
    isi_lists = [[[] for _ in range(n_bins)] for _ in range(n_res)]
    for n in range(n_res):
        spks = neuron_spikes[n]
        for i in range(1, len(spks)):
            isi = spks[i] - spks[i - 1]
            midpoint = (spks[i] + spks[i - 1]) / 2.0
            b = min(int(midpoint / bin_ms), n_bins - 1)
            isi_lists[n][b].append(isi)

    # Median ISI per cell per bin (NaN where no ISIs)
    isi_mat = np.full((n_res, n_bins), np.nan)
    for n in range(n_res):
        for b in range(n_bins):
            if isi_lists[n][b]:
                isi_mat[n, b] = np.median(isi_lists[n][b])

    return isi_mat, neuron_spikes


def main():
    if not os.path.exists(RASTER_PATH):
        print(f'{RASTER_PATH} not found.')
        sys.exit(1)

    print(f'Loading {RASTER_PATH}...')
    with open(RASTER_PATH) as f:
        rdata = json.load(f)

    n_res = rdata['rasters'][0]['n_reservoir']
    a_end = rdata['stim_a_end_ms']
    b_start = rdata['stim_b_offset_ms']
    b_end = rdata['stim_b_end_ms']
    trial_end = b_end + rdata['post_stim_ms']

    # Get WM optimum entry
    wm_entry = next(e for e in rdata['rasters'] if e['label'] == 'wm_optimum')
    na_entry = next(e for e in rdata['rasters'] if e['label'] == 'no_adapt')

    bin_ms = 10.0

    # Build ISI matrices
    wm_times = np.array(wm_entry['spike_times_ms'])
    wm_pos = np.array(wm_entry['spike_neuron_pos'])
    wm_isi, wm_spikes = build_isi_matrix(wm_times, wm_pos, n_res, trial_end, bin_ms)

    na_times = np.array(na_entry['spike_times_ms'])
    na_pos = np.array(na_entry['spike_neuron_pos'])
    na_isi, na_spikes = build_isi_matrix(na_times, na_pos, n_res, trial_end, bin_ms)

    # Sort neurons by mean spike count (shared order, same as fig6)
    wm_counts = np.array([len(s) for s in wm_spikes])
    na_counts = np.array([len(s) for s in na_spikes])
    avg_counts = (wm_counts + na_counts) / 2.0
    sort_order = np.argsort(avg_counts)[::-1]  # high activity at top

    wm_isi_sorted = wm_isi[sort_order, :]
    na_isi_sorted = na_isi[sort_order, :]

    n_bins = wm_isi_sorted.shape[1]

    # Compute population median ISI over time (ignoring NaN)
    wm_pop_median = np.nanmedian(wm_isi_sorted, axis=0)
    na_pop_median = np.nanmedian(na_isi_sorted, axis=0)
    t_bin_centers = (np.arange(n_bins) + 0.5) * bin_ms

    # Compute ISI CV per time bin (across neurons that have ISIs)
    def isi_cv_per_bin(isi_mat):
        cv = np.full(isi_mat.shape[1], np.nan)
        for b in range(isi_mat.shape[1]):
            col = isi_mat[:, b]
            valid = col[~np.isnan(col)]
            if len(valid) >= 5:
                cv[b] = np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 0
        return cv

    wm_cv = isi_cv_per_bin(wm_isi_sorted)
    na_cv = isi_cv_per_bin(na_isi_sorted)

    # Shared color scale
    vmin, vmax = 1.0, 200.0  # ms

    epoch_spans = [
        (0, a_end, 'A'),
        (a_end, b_start, 'Gap'),
        (b_start, b_end, 'B'),
        (b_end, trial_end, 'Post'),
    ]

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 2, figure=fig, wspace=0.08, hspace=0.18,
                  height_ratios=[5, 1, 1], left=0.06, right=0.91,
                  top=0.89, bottom=0.06)
    fig.suptitle('Inter-spike interval structure across the trial',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93,
             f'Single trial, digit A={rdata["digit_a"]}, digit B={rdata["digit_b"]}  |  '
             f'Median ISI per neuron per {bin_ms:.0f} ms bin  |  '
             f'604 neurons sorted by spike count',
             ha='center', fontsize=10, color=C_GRAY)

    panels = [
        ('no_adapt', r'No adaptation ($\Delta_a$=0)',
         na_isi_sorted, na_pop_median, na_cv),
        ('wm_optimum', r'WM optimum ($\Delta_a$=0.171)',
         wm_isi_sorted, wm_pop_median, wm_cv),
    ]

    # Pre-compute shared y-limits for bottom panels
    valid_na_med = na_pop_median[~np.isnan(na_pop_median)]
    valid_wm_med = wm_pop_median[~np.isnan(wm_pop_median)]
    ymax_med = max(valid_na_med.max() if len(valid_na_med) else 1,
                   valid_wm_med.max() if len(valid_wm_med) else 1) * 1.15

    valid_na_cv = na_cv[~np.isnan(na_cv)]
    valid_wm_cv = wm_cv[~np.isnan(wm_cv)]
    ymax_cv = max(valid_na_cv.max() if len(valid_na_cv) else 1,
                  valid_wm_cv.max() if len(valid_wm_cv) else 1) * 1.15

    ax_h_list, ax_m_list, ax_cv_list = [], [], []

    for col, (label, title, isi_sorted, pop_median, cv) in enumerate(panels):
        ax_h = fig.add_subplot(gs[0, col])
        ax_m = fig.add_subplot(gs[1, col])
        ax_cv = fig.add_subplot(gs[2, col])
        ax_h_list.append(ax_h)
        ax_m_list.append(ax_m)
        ax_cv_list.append(ax_cv)

        # Heatmap (log scale for ISI)
        im = ax_h.imshow(isi_sorted, aspect='auto', origin='upper',
                         extent=[0, trial_end, n_res, 0],
                         cmap='RdYlBu', norm=LogNorm(vmin=vmin, vmax=vmax),
                         interpolation='nearest')

        # Set NaN pixels to dark gray
        ax_h.set_facecolor('#2a2a2a')

        # Epoch boundaries
        for t0, t1, ep in epoch_spans:
            for boundary in [t0, t1]:
                if 0 < boundary < trial_end:
                    ax_h.axvline(boundary, color='black', lw=0.8, alpha=0.6)
                    ax_m.axvline(boundary, color='black', lw=0.5, alpha=0.4)
                    ax_cv.axvline(boundary, color='black', lw=0.5, alpha=0.4)
            mid = (t0 + t1) / 2
            ax_h.text(mid, n_res * 0.03, ep, ha='center', va='top',
                      fontsize=9, fontweight='bold', color='black',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='none', alpha=0.7), clip_on=True)

        ax_h.set_xlim(0, trial_end)
        ax_h.set_xticklabels([])
        if col == 0:
            ax_h.set_ylabel('Neuron (high → low activity)')
        else:
            ax_h.set_yticklabels([])
        ax_h.set_title(title, fontweight='bold', fontsize=12)

        # Population median ISI
        valid = ~np.isnan(pop_median)
        ax_m.plot(t_bin_centers[valid], pop_median[valid],
                  color='#d62728', lw=1.5)
        ax_m.set_xlim(0, trial_end)
        ax_m.set_ylim(0, ymax_med)
        ax_m.set_xticklabels([])
        if col == 0:
            ax_m.set_ylabel('Median ISI\n(ms)', fontsize=9)
        else:
            ax_m.set_yticklabels([])

        # ISI CV across neurons
        valid_cv = ~np.isnan(cv)
        ax_cv.plot(t_bin_centers[valid_cv], cv[valid_cv],
                   color='#9467bd', lw=1.5)
        ax_cv.set_xlim(0, trial_end)
        ax_cv.set_ylim(0, ymax_cv)
        ax_cv.set_xlabel('Time (ms)')
        if col == 0:
            ax_cv.set_ylabel('ISI CV\n(across neurons)', fontsize=9)
        else:
            ax_cv.set_yticklabels([])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.45])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Median ISI (ms)', fontsize=10)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
