#!/usr/bin/env python3
"""
3-panel diagnostic: BSA power → conductance → spikes
Shows the information transformation chain for one input neuron.
"""
import subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
CPP_BIN = CPP_DIR / 'cls_sweep'
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
DATA_DIR = CPP_DIR / 'data'
BSA_DIR = DATA_DIR / 'spike_trains_bsa'

# Optimal params (MI-primary grid search)
STIM_CURRENT = 0.0518
INPUT_TAU_E = 1.05
INPUT_ADAPT_INC = 0.0

# Neuron 493 receives from freq bins 17-22
NEURON = 493
FREQ_BINS = [17, 18, 19, 20, 21, 22]
SAMPLE = 'spike_train_0_george_10.npz'  # sparser sample

OUT_DIR = CPP_DIR / 'results' / 'neuron_diagnostics'


def main():
    # 1. Run C++ trace
    trace_csv = OUT_DIR / 'trace_3panel.csv'
    trace_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CPP_BIN), '--snapshot', str(SNAPSHOT),
        '--trace-neuron', str(NEURON),
        '--trace-file', str(BSA_DIR / SAMPLE),
        '--trace-output', str(trace_csv),
        '--data-dir', str(DATA_DIR),
        '--samples-per-digit', '1', '--n-workers', '1',
        '--no-noise', '--no-input-nmda',
        '--stim-current', str(STIM_CURRENT),
        '--input-tau-e', str(INPUT_TAU_E),
        '--input-adapt-inc', str(INPUT_ADAPT_INC),
    ]
    print(f"Running trace...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout[-200:] if result.stdout else "")
    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return

    df = pd.read_csv(trace_csv)
    t_trace = df['t_ms'].values
    dt = t_trace[1] - t_trace[0]
    total_ms = t_trace[-1]

    # 2. Load BSA and compute per-bin power for this neuron's freq bins
    bsa = np.load(BSA_DIR / SAMPLE)
    spike_times = bsa['spike_times_ms']
    freq_bins = bsa['freq_bin_indices']

    # Filter to this neuron's freq bins
    mask = np.isin(freq_bins, FREQ_BINS)
    neuron_bsa_times = spike_times[mask]

    # Bin BSA spikes into 1ms bins for high-res, then smooth
    bin_width = 1.0  # ms
    n_bins_bsa = int(total_ms / bin_width) + 1
    bsa_hist, bsa_edges = np.histogram(neuron_bsa_times, bins=n_bins_bsa,
                                        range=(0, total_ms))
    bsa_t = bsa_edges[:-1] + bin_width / 2

    # Smooth with a Gaussian kernel (~5ms)
    from scipy.ndimage import gaussian_filter1d
    bsa_smooth = gaussian_filter1d(bsa_hist.astype(float), sigma=5.0 / bin_width)

    # Also compute per-bin breakdown for the stacked view
    bin_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(FREQ_BINS)))
    per_bin_smooth = {}
    for fb in FREQ_BINS:
        fb_mask = freq_bins == fb
        fb_times = spike_times[fb_mask]
        fb_hist, _ = np.histogram(fb_times, bins=n_bins_bsa, range=(0, total_ms))
        per_bin_smooth[fb] = gaussian_filter1d(fb_hist.astype(float), sigma=5.0 / bin_width)

    # 3. Get spike times and g_e from trace
    spike_mask = df['spiked'].values == 1
    spike_times_out = t_trace[spike_mask]
    g_e = df['g_e'].values

    # 4. Compute correlation between BSA rate and spike rate at 20ms
    win = 20.0
    n_corr_bins = int(total_ms / win)
    bsa_binned = np.zeros(n_corr_bins)
    spk_binned = np.zeros(n_corr_bins)
    for t in neuron_bsa_times:
        b = min(int(t / win), n_corr_bins - 1)
        if b >= 0:
            bsa_binned[b] += 1
    for t in spike_times_out:
        b = min(int(t / win), n_corr_bins - 1)
        if b >= 0:
            spk_binned[b] += 1
    from scipy.stats import pearsonr
    r20, _ = pearsonr(bsa_binned, spk_binned) if len(bsa_binned) > 3 else (0, 1)

    # Also g_e-BSA correlation
    # Resample g_e to 1ms bins
    ge_1ms = np.interp(bsa_t, t_trace, g_e)
    r_ge, _ = pearsonr(bsa_smooth, ge_1ms) if len(bsa_smooth) > 3 else (0, 1)

    # 5. Plot
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 12,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 0.6]})

    # ── Panel 1: BSA input power (stacked by freq bin) ──
    ax = axes[0]
    bottom = np.zeros_like(bsa_t)
    for j, fb in enumerate(FREQ_BINS):
        ax.fill_between(bsa_t, bottom, bottom + per_bin_smooth[fb],
                         alpha=0.7, color=bin_colors[j], label=f'bin {fb}')
        bottom += per_bin_smooth[fb]
    ax.set_ylabel('BSA spike rate\n(spk/ms, smoothed)', fontsize=10)
    ax.set_title(f'BSA Input Power — Freq Bins {FREQ_BINS[0]}–{FREQ_BINS[-1]} '
                 f'(neuron {NEURON})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=len(FREQ_BINS), loc='upper right')
    ax.set_xlim(0, total_ms)

    # ── Panel 2: Conductance ──
    ax = axes[1]
    ax.plot(t_trace, g_e, color='#2ca02c', linewidth=0.5, alpha=0.9)
    ax.set_ylabel('g_e (conductance)', fontsize=10)
    ax.set_title(f'Excitatory Conductance (tau_e={INPUT_TAU_E}ms) — '
                 f'r(g_e, BSA) = {r_ge:.3f}', fontsize=12, fontweight='bold')
    # Light spike markers
    for st in spike_times_out:
        ax.axvline(st, color='blue', alpha=0.08, linewidth=0.5)

    # ── Panel 3: Output spikes ──
    ax = axes[2]
    ax.eventplot([spike_times_out], lineoffsets=0.5, linelengths=0.8,
                  colors=['#1f77b4'], linewidths=1.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Spikes', fontsize=10)

    n_spk = len(spike_times_out)
    dur_s = total_ms / 1000
    rate = n_spk / dur_s
    if n_spk > 2:
        isis = np.diff(spike_times_out)
        cv = np.std(isis) / np.mean(isis)
        mean_isi = np.mean(isis)
    else:
        cv = 0; mean_isi = 0

    ax.set_title(f'Output Spikes — {n_spk} spikes, {rate:.0f} Hz, '
                 f'ISI CV={cv:.3f}, r(spike, BSA)@20ms = {r20:.3f}',
                 fontsize=12, fontweight='bold')

    # Add info text
    fig.text(0.5, 0.01,
             f'stim={STIM_CURRENT}  tau_e={INPUT_TAU_E}ms  adapt_inc={INPUT_ADAPT_INC}  '
             f'NMDA=off  noise=off  |  sample: {SAMPLE}',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUT_DIR / 'diagnostic_3panel_493.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
