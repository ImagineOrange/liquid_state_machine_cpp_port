#!/usr/bin/env python3
"""
4-panel network activity figure for a single digit presentation:
  1. BSA input spectrogram (freq bin vs time)
  2. Input layer spike raster
  3. Reservoir spike raster
  4. Whole-network PSTH (population spike rate)

Usage:
  python experiments/plot_raster.py [--sample <path_to_npz>] [--stim-current 0.0158]
                                     [--input-tau-e 1.93] [--input-adapt-inc 0.005]
                                     [--output <path.png>]

Runs the C++ binary with --raster-dump, then plots from the dumped CSVs.
"""
import argparse
import subprocess
import json
import tempfile
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
OUT_DIR = CPP_DIR / 'results' / 'raster'

# Default optimal input params from grid search
DEFAULT_STIM = 0.0158
DEFAULT_TAU_E = 1.93
DEFAULT_ADAPT_INC = 0.005

# Pick a default sample
DEFAULT_SAMPLE = 'spike_train_0_george_0.npz'


def run_raster_dump(sample_path, stim, tau_e, adapt_inc, dump_dir):
    cmd = [
        str(CPP_BIN),
        '--snapshot', str(SNAPSHOT),
        '--trace-file', str(sample_path),
        '--raster-dump', str(dump_dir),
        '--stim-current', str(stim),
        '--input-tau-e', str(tau_e),
        '--input-adapt-inc', str(adapt_inc),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr}")
        raise RuntimeError("C++ raster dump failed")


def main():
    parser = argparse.ArgumentParser(description='4-panel network raster figure')
    parser.add_argument('--sample', type=str, default=str(BSA_DIR / DEFAULT_SAMPLE))
    parser.add_argument('--stim-current', type=float, default=DEFAULT_STIM)
    parser.add_argument('--input-tau-e', type=float, default=DEFAULT_TAU_E)
    parser.add_argument('--input-adapt-inc', type=float, default=DEFAULT_ADAPT_INC)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dump-dir', type=str, default=None,
                        help='Reuse existing dump dir (skip C++ run)')
    args = parser.parse_args()

    # Run C++ or reuse existing dump
    if args.dump_dir and Path(args.dump_dir).exists():
        dump_dir = Path(args.dump_dir)
        print(f"Reusing existing dump: {dump_dir}")
    else:
        dump_dir = Path(args.dump_dir) if args.dump_dir else OUT_DIR / 'dump'
        dump_dir.mkdir(parents=True, exist_ok=True)
        run_raster_dump(args.sample, args.stim_current, args.input_tau_e,
                        args.input_adapt_inc, dump_dir)

    # Load data
    spikes = pd.read_csv(dump_dir / 'spikes.csv')
    bsa = pd.read_csv(dump_dir / 'bsa.csv')
    with open(dump_dir / 'meta.json') as f:
        meta = json.load(f)

    audio_end = meta['audio_duration_ms']
    total_ms = meta['total_ms']
    n_input = meta['n_input']
    n_reservoir = meta['n_reservoir']
    digit = meta['digit']
    filename = meta['filename']

    # Separate spikes by zone
    input_spikes = spikes[spikes['zone'] == 'input']
    res_spikes = spikes[spikes['zone'] == 'reservoir']

    # Assign contiguous y-indices for raster plots
    input_ids = sorted(input_spikes['neuron_id'].unique())
    res_ids = sorted(res_spikes['neuron_id'].unique())
    input_id_map = {nid: i for i, nid in enumerate(input_ids)}
    res_id_map = {nid: i for i, nid in enumerate(res_ids)}

    # ── Plot ──
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1.2, 1.8, 0.8]})

    # ── Panel 1: BSA input spectrogram ──
    ax = axes[0]
    bsa_t = bsa['time_ms'].values
    bsa_fb = bsa['freq_bin'].values
    # Plot as scatter — small dots colored by freq bin
    ax.scatter(bsa_t, bsa_fb, s=0.3, c=bsa_fb, cmap='viridis',
               alpha=0.6, rasterized=True)
    ax.set_ylabel('Freq bin')
    ax.set_ylim(-1, 128)
    ax.set_title(f'BSA Input — digit {digit} ({filename}, {len(bsa_t)} spikes)',
                 fontweight='bold')
    ax.axvline(audio_end, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # ── Panel 2: Input layer raster ──
    ax = axes[1]
    if len(input_spikes) > 0:
        y_vals = [input_id_map[nid] for nid in input_spikes['neuron_id']]
        ax.scatter(input_spikes['time_ms'].values, y_vals,
                   s=0.4, c='#1f77b4', alpha=0.5, rasterized=True)
    n_active_input = len(input_ids)
    n_input_spikes = len(input_spikes)
    input_rate = n_input_spikes / (total_ms / 1000) / max(n_active_input, 1)
    ax.set_ylabel('Input neuron')
    ax.set_ylim(-1, max(n_active_input, 1))
    ax.set_title(f'Input Shell Raster — {n_active_input}/{n_input} active, '
                 f'{n_input_spikes} spikes ({input_rate:.0f} Hz/neuron)',
                 fontweight='bold')
    ax.axvline(audio_end, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # ── Panel 3: Reservoir raster ──
    ax = axes[2]
    if len(res_spikes) > 0:
        y_vals = [res_id_map[nid] for nid in res_spikes['neuron_id']]
        ax.scatter(res_spikes['time_ms'].values, y_vals,
                   s=0.3, c='#2ca02c', alpha=0.3, rasterized=True)
    n_active_res = len(res_ids)
    n_res_spikes = len(res_spikes)
    res_rate = n_res_spikes / (total_ms / 1000) / max(n_active_res, 1)
    ax.set_ylabel('Reservoir neuron')
    ax.set_ylim(-1, max(n_active_res, 1))
    ax.set_title(f'Reservoir Raster — {n_active_res}/{n_reservoir} active, '
                 f'{n_res_spikes} spikes ({res_rate:.0f} Hz/neuron)',
                 fontweight='bold')
    ax.axvline(audio_end, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # ── Panel 4: Whole-network PSTH ──
    ax = axes[3]
    bin_width = 5.0  # ms
    bins = np.arange(0, total_ms + bin_width, bin_width)
    all_spike_times = spikes['time_ms'].values

    counts, edges = np.histogram(all_spike_times, bins=bins)
    rate_hz = counts / (bin_width / 1000)  # convert to spk/s
    bin_centers = edges[:-1] + bin_width / 2

    ax.fill_between(bin_centers, rate_hz, alpha=0.4, color='#333333')
    ax.plot(bin_centers, rate_hz, color='#333333', linewidth=0.8)
    ax.set_ylabel('Pop. rate (spk/s)')
    ax.set_xlabel('Time (ms)')
    mean_rate = len(all_spike_times) / (total_ms / 1000)
    ax.set_title(f'Population PSTH ({bin_width:.0f}ms bins) — '
                 f'{len(all_spike_times)} total spikes, '
                 f'{mean_rate:.0f} spk/s mean',
                 fontweight='bold')
    ax.axvline(audio_end, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, total_ms)

    # Footer
    fig.text(0.5, 0.005,
             f'stim={args.stim_current}  tau_e={args.input_tau_e}ms  '
             f'adapt_inc={args.input_adapt_inc}  |  '
             f'{meta["n_total"]} neurons ({n_input} input + {n_reservoir} reservoir)',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f'raster_{filename}.png'

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
