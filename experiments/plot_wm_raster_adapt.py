#!/usr/bin/env python3
"""
WM compound trial figure: side-by-side no_adapt vs WM optimum.
  Row 0: Spike raster (neuron x time), per column
  Row 1: Adaptation conductance heatmap (neuron x time), per column
         with readout bracket underneath
  Row 2: Superimposed PSTH + mean adaptation trace (full width)

Usage:
    python experiments/plot_wm_raster_adapt.py

Reads:  results/mechanistic_interp/example_rasters.json
Writes: results/wm_raster_adapt_figure.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

RASTER_PATH = 'results/mechanistic_interp/example_rasters.json'
OUT_PATH = 'results/wm_raster_adapt_figure.png'
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

    # Epoch boundaries
    a_end = rdata['stim_a_end_ms']
    b_start = rdata['stim_b_offset_ms']
    b_end = rdata['stim_b_end_ms']
    trial_end = b_end + rdata['post_stim_ms']

    # Parse spike and adaptation data
    rasters = {}
    adapt_matrices = {}
    for entry in rdata['rasters']:
        times = np.array(entry['spike_times_ms'])
        pos = np.array(entry['spike_neuron_pos'])
        rasters[entry['label']] = (times, pos)

        n_snaps = entry['n_adapt_snapshots']
        flat = np.array(entry['adapt_snapshots'])
        mat = flat.reshape(n_snaps, n_res)
        n_ms = min(n_snaps, int(np.ceil(trial_end)))
        adapt_matrices[entry['label']] = mat[:n_ms, :]

    # Shared sort by average spike count
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

    # Epoch shading
    epoch_spans = [
        (0, a_end, '#2ca02c20', 'A'),
        (a_end, b_start, '#f0f0f0', 'Gap'),
        (b_start, b_end, '#1f77b420', 'B'),
        (b_end, trial_end, '#ff7f0e20', 'Post'),
    ]

    # PSTH
    psth_bin_ms = 1.0
    psth_n_bins = int(np.ceil(trial_end / psth_bin_ms))
    psth_sigma = 10.0
    psths = {}
    for lbl in ['no_adapt', 'wm_optimum']:
        times_all = rasters[lbl][0]
        hist, _ = np.histogram(times_all, bins=psth_n_bins, range=(0, trial_end))
        rate_hz = hist / (psth_bin_ms / 1000.0) / n_res
        psths[lbl] = gaussian_filter1d(rate_hz.astype(float), psth_sigma)
    psth_t = np.linspace(psth_bin_ms / 2, trial_end - psth_bin_ms / 2, psth_n_bins)
    psth_ymax = max(psths['no_adapt'].max(), psths['wm_optimum'].max()) * 1.15

    # Mean adaptation traces
    mean_adapt_smooth = {}
    for lbl in ['no_adapt', 'wm_optimum']:
        mean_adapt_smooth[lbl] = gaussian_filter1d(
            adapt_matrices[lbl].mean(axis=1), sigma=5)
    adapt_ymax = max(mean_adapt_smooth['wm_optimum'].max(),
                     mean_adapt_smooth['no_adapt'].max()) * 1.15

    # Adaptation heatmap color scale (99.5th percentile of WM)
    wm_adapt = adapt_matrices['wm_optimum']
    vmax_adapt = np.percentile(wm_adapt[wm_adapt > 0], 99.5)

    # White-to-blue colormap for adaptation
    adapt_cmap = LinearSegmentedColormap.from_list('white_blue', [
        '#ffffff', '#d4e6f1', '#7fb3d8', '#2980b9', '#1a5276', '#0b2545'
    ])

    # --- Figure layout ---
    # Rows 0-1: 2-column (raster, adapt heatmap)
    # Row 2: full-width superimposed traces
    fig = plt.figure(figsize=(18, 11))
    gs_top = GridSpec(2, 2, figure=fig, wspace=0.08, hspace=0.08,
                      height_ratios=[3, 3],
                      left=0.05, right=0.92, top=0.91, bottom=0.28)
    gs_bot = GridSpec(1, 2, figure=fig, wspace=0.08,
                      left=0.05, right=0.92, top=0.22, bottom=0.05)

    fig.suptitle('Working memory trial: spike raster and adaptation conductance',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.94,
             f'Single compound trial: digit A={rdata["digit_a"]}, '
             f'digit B={rdata["digit_b"]}  |  '
             f'604 reservoir neurons sorted by mean spike count',
             ha='center', fontsize=10, color=C_GRAY)

    C_DKBLUE_DARK = '#0f2640'
    C_BLUE_DARK = '#0d4a8a'
    panels = [
        ('no_adapt', r'No adaptation ($\Delta_a$=0)', C_DKBLUE_DARK),
        ('wm_optimum', r'WM optimum ($\Delta_a$=0.171)', C_BLUE_DARK),
    ]

    im = None
    for col, (label, title, spike_color) in enumerate(panels):
        ax_r = fig.add_subplot(gs_top[0, col])
        ax_a = fig.add_subplot(gs_top[1, col])

        times, pos = rasters[label]

        # --- Row 0: Spike raster ---
        for t0, t1, fc, ep_label in epoch_spans:
            ax_r.axvspan(t0, t1, facecolor=fc, edgecolor='none')
            mid = (t0 + t1) / 2
            ax_r.text(mid, n_res * 0.97, ep_label, ha='center', va='top',
                      fontsize=9, fontweight='bold', color=C_GRAY,
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='none', alpha=0.85), clip_on=True)

        trains = [[] for _ in range(n_res)]
        for t, p in zip(times, pos):
            if 0 <= p < n_res:
                trains[pos_to_row[p]].append(t)

        ax_r.eventplot(trains, colors=spike_color,
                       lineoffsets=np.arange(n_res),
                       linelengths=0.9, linewidths=0.8)
        ax_r.set_xlim(0, trial_end)
        ax_r.set_ylim(-1, n_res)
        ax_r.set_xticklabels([])
        if col == 0:
            ax_r.set_ylabel('Neuron (sorted)')
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

        # --- Row 1: Adaptation heatmap ---
        mat = adapt_matrices[label][:, shared_order].T
        n_t = mat.shape[1]

        for t0, t1, fc, ep_label in epoch_spans:
            ax_a.axvspan(t0, t1, facecolor=fc, edgecolor='none', alpha=0.15)

        im = ax_a.imshow(mat, aspect='auto', origin='lower',
                         extent=[0, n_t, 0, n_res],
                         cmap=adapt_cmap, vmin=0, vmax=vmax_adapt,
                         interpolation='bilinear')

        for t0, t1, fc, ep_label in epoch_spans:
            for boundary in [t0, t1]:
                if 0 < boundary < n_t:
                    ax_a.axvline(boundary, color='#666666', lw=0.6, alpha=0.5)

        ax_a.set_xlim(0, trial_end)
        ax_a.set_ylim(0, n_res)
        ax_a.set_xticklabels([])
        if col == 0:
            ax_a.set_ylabel('Neuron (sorted)')
        else:
            ax_a.set_yticklabels([])

        # Readout bracket beneath adaptation heatmap
        bracket_y = -n_res * 0.06
        tick_h = n_res * 0.02
        readout_color = '#c0392b'
        ax_a.plot([b_start, b_start], [bracket_y, bracket_y + tick_h],
                  color=readout_color, lw=1.5, clip_on=False)
        ax_a.plot([b_start, trial_end], [bracket_y, bracket_y],
                  color=readout_color, lw=2.0, clip_on=False)
        ax_a.plot([trial_end, trial_end], [bracket_y, bracket_y + tick_h],
                  color=readout_color, lw=1.5, clip_on=False)
        mid_readout = (b_start + trial_end) / 2
        ax_a.text(mid_readout, bracket_y - n_res * 0.025,
                  'Readout (B+Post)', ha='center', va='top', fontsize=8,
                  color=readout_color, fontweight='bold', clip_on=False)

    # Colorbar for adaptation heatmap
    cbar_ax = fig.add_axes([0.93, 0.42, 0.012, 0.28])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('$g_{adapt}$', fontsize=10)

    # --- Bottom row ---
    # Left: PSTH only (both conditions, spikes)
    # Right: WM optimum spikes + adaptation overlaid on twin axes
    ax_psth = fig.add_subplot(gs_bot[0, 0])
    ax_dual = fig.add_subplot(gs_bot[0, 1])

    # Epoch shading
    for ax in [ax_psth, ax_dual]:
        for t0, t1, fc, ep_label in epoch_spans:
            ax.axvspan(t0, t1, facecolor=fc, edgecolor='none')

    # Left panel: both PSTHs overlaid (spikes only)
    ax_psth.fill_between(psth_t, psths['no_adapt'],
                         color=C_DKBLUE_DARK, alpha=0.15)
    ax_psth.plot(psth_t, psths['no_adapt'],
                 color=C_DKBLUE_DARK, lw=1.5, label='No adapt')
    ax_psth.fill_between(psth_t, psths['wm_optimum'],
                         color=C_BLUE_DARK, alpha=0.15)
    ax_psth.plot(psth_t, psths['wm_optimum'],
                 color=C_BLUE_DARK, lw=1.5, ls='--', label='WM opt')
    ax_psth.set_xlim(0, trial_end)
    ax_psth.set_ylim(0, psth_ymax)
    ax_psth.set_xlabel('Time (ms)')
    ax_psth.set_ylabel('Rate (Hz)', fontsize=9)
    ax_psth.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_psth.set_title('Population firing rate', fontsize=10,
                      fontweight='bold')

    # Right panel: WM optimum PSTH + mean adaptation on twin y-axes
    n_adapt_t = len(mean_adapt_smooth['wm_optimum'])
    at = np.arange(n_adapt_t)

    # Spikes (left y-axis)
    ax_dual.fill_between(psth_t, psths['wm_optimum'],
                         color=C_BLUE_DARK, alpha=0.12)
    l1, = ax_dual.plot(psth_t, psths['wm_optimum'],
                       color=C_BLUE_DARK, lw=1.5, label='Firing rate')
    ax_dual.set_xlim(0, trial_end)
    ax_dual.set_ylim(0, psth_ymax)
    ax_dual.set_xlabel('Time (ms)')
    ax_dual.set_ylabel('Rate (Hz)', fontsize=9, color=C_BLUE_DARK)
    ax_dual.tick_params(axis='y', labelcolor=C_BLUE_DARK)

    # Adaptation (right y-axis)
    ax_adapt = ax_dual.twinx()
    ax_adapt.fill_between(at, mean_adapt_smooth['wm_optimum'],
                          color='#c0392b', alpha=0.12)
    l2, = ax_adapt.plot(at, mean_adapt_smooth['wm_optimum'],
                        color='#c0392b', lw=1.5, ls='--',
                        label='Mean $g_{adapt}$')
    ax_adapt.set_ylim(0, adapt_ymax)
    ax_adapt.set_ylabel('Mean $g_{adapt}$', fontsize=9, color='#c0392b')
    ax_adapt.tick_params(axis='y', labelcolor='#c0392b')

    ax_dual.legend([l1, l2], ['Firing rate', 'Mean $g_{adapt}$'],
                   fontsize=8, loc='upper left', framealpha=0.9)
    ax_dual.set_title('WM optimum: spikes vs. adaptation', fontsize=10,
                      fontweight='bold')

    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
