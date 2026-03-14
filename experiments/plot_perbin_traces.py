#!/usr/bin/env python3
"""
Per-bin classification accuracy traces across the adaptation parameter space.

Panel A: All 300 Branch B grid points as rainbow-colored traces (by adapt_inc),
         with the global-best trace highlighted.
Panel B: Mean per-bin accuracy across all grid points, with shaded ±1 SD.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def smooth(y, window=3):
    """Simple moving average smoothing with edge-value padding."""
    pad = window // 2
    padded = np.pad(y, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLS = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
DEFAULT_OUTPUT = CPP_DIR / 'results' / 'perbin_accuracy_traces.png'

BIN_MS = 20
N_BINS = 48

# Match dissociation_analysis.png style
COL_CLS = '#4daf4a'
COL_WM  = '#2166ac'
COL_XOR = '#b2182b'


def load_perbin_data(cls_path):
    with open(cls_path) as f:
        cls_data = json.load(f)

    inc_vals = np.array(cls_data['grid']['unified_inc'])
    tau_vals = np.array(cls_data['grid']['unified_tau'])
    n_inc, n_tau = len(inc_vals), len(tau_vals)

    # Collect all Branch B per-bin traces
    traces = []       # list of (inc_idx, tau_idx, per_bin_array, overall_acc)
    grid_perbin = {}  # (inc_idx, tau_idx) -> per_bin array

    for entry in cls_data['grid_results']:
        if entry.get('branch', 'A_unmatched') != 'B_matched':
            continue
        pba = entry.get('per_bin_accuracy')
        if pba is None or len(pba) != N_BINS:
            continue
        ii, ti = entry['inc_idx'], entry['tau_idx']
        overall = entry['classification_accuracy']
        traces.append((ii, ti, np.array(pba), overall))
        grid_perbin[(ii, ti)] = np.array(pba)

    return {
        'inc_vals': inc_vals, 'tau_vals': tau_vals,
        'traces': traces, 'grid_perbin': grid_perbin,
        'n_inc': n_inc, 'n_tau': n_tau,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls-input', type=str, default=str(DEFAULT_CLS))
    parser.add_argument('-o', '--output', type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data = load_perbin_data(args.cls_input)
    traces = data['traces']
    inc_vals = data['inc_vals']

    time_ms = np.arange(N_BINS) * BIN_MS + BIN_MS / 2  # bin centers

    # Sort traces by inc for rainbow coloring
    # Use log(inc) for color mapping (skip inc=0)
    inc_for_color = []
    for ii, ti, pba, oacc in traces:
        inc_for_color.append(inc_vals[ii] if inc_vals[ii] > 0 else inc_vals[1] * 0.5)
    inc_for_color = np.array(inc_for_color)
    log_inc = np.log10(inc_for_color)
    norm = Normalize(vmin=log_inc.min(), vmax=log_inc.max())
    cmap = plt.cm.RdYlBu_r  # red=high inc, blue=low inc

    # Find global best trace (highest overall accuracy)
    best_idx = np.argmax([t[3] for t in traces])
    best_trace = traces[best_idx]

    # Find zero-adaptation traces (inc=0)
    zero_traces = [(ii, ti, pba, oacc) for ii, ti, pba, oacc in traces if inc_vals[ii] == 0]

    # Stack all traces for mean/std
    all_pba = np.array([t[2] for t in traces])
    mean_pba = np.mean(all_pba, axis=0) * 100
    std_pba = np.std(all_pba, axis=0) * 100

    # Also compute mean by inc group for Panel B
    # Group: low inc (< 0.05), mid (0.05-0.15), high (> 0.15)
    low_mask = np.array([inc_vals[t[0]] < 0.05 for t in traces])
    mid_mask = np.array([(0.05 <= inc_vals[t[0]] < 0.15) for t in traces])
    high_mask = np.array([inc_vals[t[0]] >= 0.15 for t in traces])

    mean_low = np.mean(all_pba[low_mask], axis=0) * 100 if low_mask.sum() > 0 else None
    mean_mid = np.mean(all_pba[mid_mask], axis=0) * 100 if mid_mask.sum() > 0 else None
    mean_high = np.mean(all_pba[high_mask], axis=0) * 100 if high_mask.sum() > 0 else None

    # ================================================================
    # FIGURE
    # ================================================================
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.82, wspace=0.28)

    # --- Panel A: All traces, rainbow colored by inc ---
    # Sort by inc so high-inc traces draw on top
    sorted_idx = np.argsort(inc_for_color)
    for si in sorted_idx:
        ii, ti, pba, oacc = traces[si]
        color = cmap(norm(log_inc[si]))
        ax_a.plot(time_ms, smooth(pba * 100), color=color, alpha=0.25, linewidth=0.6, zorder=2)

    # Per-bin envelope: at each bin, the max accuracy across all traces
    envelope_raw = np.max(all_pba, axis=0) * 100
    # Find which config achieves the peak of the envelope
    peak_bin = np.argmax(envelope_raw)
    peak_trace_idx = np.argmax(all_pba[:, peak_bin])
    eii, eti = traces[peak_trace_idx][0], traces[peak_trace_idx][1]
    env_inc_color = inc_vals[eii] if inc_vals[eii] > 0 else inc_vals[1] * 0.5
    env_color = cmap(norm(np.log10(env_inc_color)))
    ax_a.plot(time_ms, envelope_raw, color='white', linewidth=3.5, zorder=4)
    ax_a.plot(time_ms, envelope_raw, color=env_color, linewidth=2.2, linestyle='-', zorder=5,
              label=f'Per-bin max envelope ({np.mean(envelope_raw):.1f}% mean, '
                    f'peak: inc={inc_vals[eii]:.3f}, $\\tau$={data["tau_vals"][eti]:.0f})')

    # Highlight global best overall accuracy (colored by its inc value)
    bii, bti, bpba, boacc = best_trace
    best_inc_color = inc_vals[bii] if inc_vals[bii] > 0 else inc_vals[1] * 0.5
    best_color = cmap(norm(np.log10(best_inc_color)))
    ax_a.plot(time_ms, bpba * 100, color='white', linewidth=4.0, zorder=5)
    ax_a.plot(time_ms, bpba * 100, color=best_color, linewidth=2.5, zorder=6,
              label=f'Best overall ({boacc*100:.1f}%, '
                    f'inc={inc_vals[bii]:.3f}, $\\tau$={data["tau_vals"][bti]:.0f})')

    # Zero-adaptation baseline (inc=0): mean across all tau values
    if zero_traces:
        zero_pba = np.mean([t[2] for t in zero_traces], axis=0) * 100
        zero_acc = np.mean(zero_pba)
        zero_color = cmap(norm(log_inc.min()))
        ax_a.plot(time_ms, zero_pba, color='white', linewidth=4.0, zorder=4)
        ax_a.plot(time_ms, zero_pba, color=zero_color, linewidth=2.2, linestyle='--', zorder=5,
                  label=f'No adaptation (inc=0, {zero_acc:.1f}% mean, n={len(zero_traces)})')

    # Chance line
    ax_a.axhline(20, color='grey', linestyle=':', alpha=0.5, linewidth=1, label='Chance (20%)')

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_a, pad=0.02, aspect=30, shrink=0.85)
    cbar.set_label(r'$\log_{10}$($\Delta_a$)', fontsize=9)
    # Set tick labels to actual inc values
    tick_locs = cbar.get_ticks()
    cbar.set_ticklabels([f'{10**v:.3f}' if 10**v < 0.01 else f'{10**v:.2f}' if 10**v < 1 else f'{10**v:.1f}'
                         for v in tick_locs])

    ax_a.set_xlabel('Time from stimulus onset (ms)', fontsize=9)
    ax_a.set_ylabel('Per-bin classification accuracy (%)', fontsize=9)
    ax_a.set_title('Per-bin accuracy: all adaptation configurations (rate-matched)',
                   fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=7, loc='upper right')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, N_BINS * BIN_MS)
    ax_a.set_ylim(15, 85)

    # --- Panel B: Mean traces by inc group ---
    # Overall mean ± SD
    ax_b.fill_between(time_ms, mean_pba - std_pba, mean_pba + std_pba,
                      alpha=0.15, color='#888888')
    ax_b.plot(time_ms, mean_pba, color='#888888', linewidth=1.5, linestyle='--',
              label=f'Overall mean (n={len(traces)})', zorder=3)

    if mean_low is not None:
        ax_b.plot(time_ms, mean_low, color=COL_CLS, linewidth=2,
                  label=f'Low inc (<0.05, n={low_mask.sum()})', zorder=4)
    if mean_mid is not None:
        ax_b.plot(time_ms, mean_mid, color=COL_XOR, linewidth=2,
                  label=f'Mid inc (0.05–0.15, n={mid_mask.sum()})', zorder=4)
    if mean_high is not None:
        ax_b.plot(time_ms, mean_high, color=COL_WM, linewidth=2,
                  label=f'High inc (>0.15, n={high_mask.sum()})', zorder=4)

    ax_b.axhline(20, color='grey', linestyle=':', alpha=0.5, linewidth=1, label='Chance (20%)')

    ax_b.set_xlabel('Time from stimulus onset (ms)', fontsize=9)
    ax_b.set_ylabel('Per-bin classification accuracy (%)', fontsize=9)
    ax_b.set_title('Mean per-bin accuracy by adaptation strength (rate-matched)',
                   fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=7, loc='upper right')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(0, N_BINS * BIN_MS)
    ax_b.set_ylim(15, 85)

    # Panel labels
    for ax, label in [(ax_a, 'A'), (ax_b, 'B')]:
        ax.text(-0.06, 1.04, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom')

    fig.suptitle('Temporal profile of classification accuracy across adaptation parameter space',
                 fontsize=13, fontweight='bold', y=0.96)
    fig.text(0.5, 0.90, 'Branch B (rate-matched), all 300 grid points, 5-fold × 5-repeat CV',
             ha='center', va='bottom', fontsize=8, color='#666666')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
