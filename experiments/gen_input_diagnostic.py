#!/usr/bin/env python3
"""
Generate single-neuron diagnostic figure for an input neuron
using optimal grid-search parameters.

Usage:
    python gen_input_diagnostic.py [--neuron 493] [--sample-file spike_train_0_george_0.npz]
"""
import sys, os, subprocess, argparse
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

# Optimal params from MI-primary grid search (8000 pts, pooled 8-quantile MI)
# Best composite=1.236, MI=1.057 bits, r20=0.884, rate=85Hz
# Hard bio filters: rate 5-150Hz, CV 0.3-2.0, refrac <10%, burst <15%
STIM_CURRENT = 0.0518
INPUT_TAU_E = 1.05
INPUT_ADAPT_INC = 0.0
INPUT_STD_U = 0.0


def run_trace(neuron_id, sample_file, trace_csv):
    audio_path = BSA_DIR / sample_file
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        str(CPP_BIN),
        '--snapshot', str(SNAPSHOT),
        '--trace-neuron', str(neuron_id),
        '--trace-file', str(audio_path),
        '--trace-output', str(trace_csv),
        '--data-dir', str(DATA_DIR),
        '--samples-per-digit', '1',
        '--n-workers', '1',
        '--no-noise',
        '--no-input-nmda',
        '--stim-current', str(STIM_CURRENT),
        '--input-tau-e', str(INPUT_TAU_E),
        '--input-adapt-inc', str(INPUT_ADAPT_INC),
    ]
    if INPUT_STD_U > 0:
        cmd += ['--input-std-u', str(INPUT_STD_U)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"C++ trace failed with code {result.returncode}")
    return pd.read_csv(trace_csv)


def plot_diagnostic(df, neuron_id, sample_file, output_path):
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 11,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })

    t = df['t_ms'].values
    spike_mask = df['spiked'].values == 1
    spike_times = t[spike_mask]
    n_spikes = len(spike_times)
    duration_s = t[-1] / 1000.0
    rate_hz = n_spikes / duration_s if duration_s > 0 else 0

    # ISI stats
    if n_spikes > 2:
        isis = np.diff(spike_times)
        isi_cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        mean_isi = np.mean(isis)
    else:
        isi_cv = 0
        mean_isi = 0

    panels = [
        ('Membrane Potential (mV)', 'v', '#1f77b4'),
        ('Excitatory Conductance (g_e)', 'g_e', '#2ca02c'),
        ('Inhibitory Conductance (g_i)', 'g_i', '#d62728'),
        ('NMDA Conductance (g_nmda)', 'g_nmda', '#9467bd'),
        ('Adaptation', 'adaptation', '#8c564b'),
        ('Excitatory Current (i_e)', 'i_e', '#2ca02c'),
        ('Inhibitory Current (i_i)', 'i_i', '#d62728'),
        ('NMDA Current (i_nmda)', 'i_nmda', '#9467bd'),
        ('Adaptation Current (i_adapt)', 'i_adapt', '#8c564b'),
        ('Slow Inh. Conductance (g_i_slow)', 'g_i_slow', '#e377c2'),
    ]

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.2 * n_panels), sharex=True)

    for idx, (ax, (title, col, color)) in enumerate(zip(axes, panels)):
        ax.plot(t, df[col].values, color=color, linewidth=0.6, alpha=0.9)
        ax.set_ylabel(title, fontsize=8)

        if idx == 0:
            # Draw spike peaks on voltage trace
            for st in spike_times:
                si = np.argmin(np.abs(t - st))
                ax.plot([st, st], [df[col].values[si], 20],
                        color=color, linewidth=1.0, alpha=0.7)
            ax.scatter(spike_times, np.full(n_spikes, 25),
                       marker='v', s=18, color=color, zorder=5)
            ax.set_ylim(bottom=ax.get_ylim()[0], top=35)
        else:
            for st in spike_times:
                ax.axvline(st, color='gray', alpha=0.1, linewidth=0.4)

    axes[-1].set_xlabel('Time (ms)')

    fig.suptitle(
        f'Input Neuron {neuron_id} — {sample_file}\n'
        f'stim={STIM_CURRENT}, tau_e={INPUT_TAU_E}ms, adapt_inc={INPUT_ADAPT_INC}, std_u={INPUT_STD_U}\n'
        f'{n_spikes} spikes | {rate_hz:.1f} Hz | ISI CV={isi_cv:.3f} | mean ISI={mean_isi:.1f}ms',
        fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron', type=int, default=493)
    parser.add_argument('--sample-file', default='spike_train_0_george_0.npz')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    out_dir = CPP_DIR / 'results' / 'neuron_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = out_dir / f'trace_cpp_{args.neuron}_optimal.csv'
    output_png = args.output or str(out_dir / f'diagnostic_input_{args.neuron}.png')

    df = run_trace(args.neuron, args.sample_file, str(trace_csv))
    plot_diagnostic(df, args.neuron, args.sample_file, output_png)


if __name__ == '__main__':
    main()
