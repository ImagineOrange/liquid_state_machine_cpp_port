#!/usr/bin/env python3
"""
Publication-quality 5-panel network activity figure:
  A. Mel spectrogram of raw audio
  B. BSA spike raster (freq bin vs time)
  C. Input layer raster (tonotopically sorted)
  D. Reservoir raster (tonotopically sorted by dominant input)
  E. Population PSTH

Usage:
  python experiments/plot_raster.py [--sample <path>] [--dump-dir <dir>] [--output <path>]
"""
import argparse
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy.signal import spectrogram
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
CPP_BIN = CPP_DIR / 'cls_sweep'
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
DATA_DIR = CPP_DIR / 'data'
BSA_DIR = DATA_DIR / 'spike_trains_bsa'
WAV_DIR = DATA_DIR / 'free-spoken-digit-dataset' / 'recordings'
OUT_DIR = CPP_DIR / 'results' / 'raster'

DEFAULT_STIM = 0.0158
DEFAULT_TAU_E = 1.93
DEFAULT_ADAPT_INC = 0.005
DEFAULT_SAMPLE = 'spike_train_0_george_0.npz'

# ── Color palette ──
C_RES = '#2d8a4e'
C_STIM_LINE = '#cc3333'
C_WARMUP = '#4488cc'


def run_raster_dump(sample_path, stim, tau_e, adapt_inc, dump_dir):
    cmd = [
        str(CPP_BIN), '--snapshot', str(SNAPSHOT),
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


def load_neuron_mapping(dump_dir):
    df = pd.read_csv(dump_dir / 'neurons.csv')
    mapping = {}
    for _, row in df.iterrows():
        nid = int(row['neuron_id'])
        bins = [int(x) for x in str(row['freq_bins']).split(';') if x]
        weights = [float(x) for x in str(row['weights']).split(';') if x]
        primary = bins[np.argmax(weights)] if bins and weights else 0
        mapping[nid] = (primary, bins, weights)
    return mapping


def compute_mel_spectrogram(wav_path, warmup_ms, n_mel=128):
    """Compute a mel-scale spectrogram from a WAV file, time-aligned with warmup offset."""
    sr, audio = wavfile.read(wav_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Compute spectrogram
    nperseg = min(256, len(audio))
    noverlap = nperseg // 2
    f, t_spec, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Convert to mel scale via simple triangular filterbank
    f_min, f_max = 0, sr / 2
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    mel_spec = np.zeros((n_mel, Sxx.shape[1]))
    for i in range(n_mel):
        lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, freq in enumerate(f):
            if lo <= freq <= center:
                w = (freq - lo) / max(center - lo, 1e-10)
                mel_spec[i] += w * Sxx[j]
            elif center < freq <= hi:
                w = (hi - freq) / max(hi - center, 1e-10)
                mel_spec[i] += w * Sxx[j]

    mel_spec_db = 10 * np.log10(np.maximum(mel_spec, 1e-10))
    t_spec_ms = t_spec * 1000 + warmup_ms
    audio_duration_ms = t_spec_ms[-1]  # original audio length
    return t_spec_ms, mel_spec_db, audio_duration_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default=str(BSA_DIR / DEFAULT_SAMPLE))
    parser.add_argument('--stim-current', type=float, default=DEFAULT_STIM)
    parser.add_argument('--input-tau-e', type=float, default=DEFAULT_TAU_E)
    parser.add_argument('--input-adapt-inc', type=float, default=DEFAULT_ADAPT_INC)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dump-dir', type=str, default=None)
    args = parser.parse_args()

    # Run or reuse dump
    if args.dump_dir and Path(args.dump_dir).exists():
        dump_dir = Path(args.dump_dir)
    else:
        dump_dir = Path(args.dump_dir) if args.dump_dir else OUT_DIR / 'dump'
        dump_dir.mkdir(parents=True, exist_ok=True)
        run_raster_dump(args.sample, args.stim_current, args.input_tau_e,
                        args.input_adapt_inc, dump_dir)

    spikes = pd.read_csv(dump_dir / 'spikes.csv')
    bsa = pd.read_csv(dump_dir / 'bsa.csv')
    with open(dump_dir / 'meta.json') as f:
        meta = json.load(f)
    neuron_map = load_neuron_mapping(dump_dir)

    audio_end = meta['audio_duration_ms']
    total_ms = meta['total_ms']
    warmup_ms = meta.get('warmup_ms', 0.0)
    n_input = meta['n_input']
    n_reservoir = meta['n_reservoir']
    digit = meta['digit']
    filename = meta['filename']

    # Derive WAV filename from BSA filename: spike_train_0_george_0 -> 0_george_0.wav
    wav_name = filename.replace('spike_train_', '') + '.wav'
    wav_path = WAV_DIR / wav_name

    input_spikes = spikes[spikes['zone'] == 'input']
    res_spikes = spikes[spikes['zone'] == 'reservoir']

    # Tonotopic sort for input neurons
    input_ids_with_bin = [(nid, neuron_map.get(nid, (0, [], []))[0])
                          for nid in sorted(input_spikes['neuron_id'].unique())]
    input_ids_with_bin.sort(key=lambda x: x[1])
    input_id_order = {nid: i for i, (nid, _) in enumerate(input_ids_with_bin)}
    input_primary_bins = {nid: fb for nid, fb in input_ids_with_bin}

    # Sort reservoir neurons by dominant input neuron's tonotopic position
    drive_path = dump_dir / 'input_drive.csv'
    if drive_path.exists():
        drive_df = pd.read_csv(drive_path)
        drive_map = dict(zip(drive_df['neuron_id'], drive_df['input_drive']))
        dominant_map = dict(zip(drive_df['neuron_id'], drive_df['dominant_input']))
        res_inherited_freq = {}
        for nid in res_spikes['neuron_id'].unique():
            dom_inp = dominant_map.get(nid, -1)
            if dom_inp >= 0 and dom_inp in neuron_map:
                res_inherited_freq[nid] = neuron_map[dom_inp][0]
            else:
                res_inherited_freq[nid] = -1
        res_ids_active = list(res_spikes['neuron_id'].unique())
        res_ids_active.sort(key=lambda nid: (res_inherited_freq.get(nid, -1),
                                              drive_map.get(nid, 0.0)))
        res_id_map = {nid: i for i, nid in enumerate(res_ids_active)}
        res_freq_vals = {nid: res_inherited_freq.get(nid, -1) for nid in res_ids_active}
    else:
        res_ids_active = sorted(res_spikes['neuron_id'].unique())
        res_id_map = {nid: i for i, nid in enumerate(res_ids_active)}
        res_freq_vals = {}

    n_active_input = len(input_id_order)
    n_active_res = len(res_id_map)

    # ── Figure setup ──
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.6,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    n_panels = 5
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(n_panels, 1, figure=fig,
                           height_ratios=[0.7, 0.7, 0.8, 1.1, 0.55],
                           hspace=0.25)

    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    def mark_epochs(ax):
        if warmup_ms > 0:
            ax.axvspan(0, warmup_ms, color=C_WARMUP, alpha=0.06, zorder=0)
            ax.axvline(warmup_ms, color=C_WARMUP, ls=':', alpha=0.5, lw=0.7)
        ax.axvline(audio_end, color=C_STIM_LINE, ls='--', alpha=0.6, lw=0.7)

    panel_labels = ['A', 'B', 'C', 'D', 'E']

    # ── A: Mel Spectrogram ──
    ax = axes[0]
    if wav_path.exists():
        from matplotlib.colors import LinearSegmentedColormap
        t_spec, mel_db, _ = compute_mel_spectrogram(wav_path, warmup_ms, n_mel=128)

        # Stretch: pad spectrogram with silence (floor value) to fill total_ms
        if t_spec[-1] < total_ms:
            dt_spec = t_spec[1] - t_spec[0]
            extra_t = np.arange(t_spec[-1] + dt_spec, total_ms + dt_spec, dt_spec)
            t_spec = np.concatenate([t_spec, extra_t])
            pad = np.full((mel_db.shape[0], len(extra_t)), mel_db.min())
            mel_db = np.concatenate([mel_db, pad], axis=1)

        # White-based colormap: white -> light warm -> dark warm
        cmap_warm = LinearSegmentedColormap.from_list('white_warm', [
            '#ffffff', '#fff3e0', '#ffb74d', '#e65100', '#4a1500'
        ])
        vmin = mel_db.max() - 50
        ax.pcolormesh(t_spec, np.arange(128), mel_db,
                      cmap=cmap_warm, vmin=vmin, vmax=mel_db.max(),
                      shading='gouraud', rasterized=True)
        ax.set_ylabel('Mel bin')
        ax.set_ylim(0, 127)
        ax.set_yticks([0, 32, 64, 96, 127])
    else:
        ax.text(0.5, 0.5, f'WAV not found: {wav_name}', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='#999')
        ax.set_ylabel('Mel bin')
    mark_epochs(ax)
    ax.set_title(f'A   Mel spectrogram — digit {digit} ({wav_name})',
                 loc='left', fontsize=9, color='#333', pad=4)
    ax.tick_params(labelbottom=False)

    # ── B: BSA Input ──
    ax = axes[1]
    bt, bf = bsa['time_ms'].values, bsa['freq_bin'].values
    ax.scatter(bt, bf, s=0.5, c=bf, cmap='viridis', alpha=0.9,
               vmin=0, vmax=127, rasterized=True, linewidths=0)
    ax.set_ylabel('Frequency\nbin')
    ax.set_ylim(-2, 130)
    ax.set_yticks([0, 32, 64, 96, 127])
    mark_epochs(ax)
    ax.set_title(f'B   BSA encoding  ({len(bt):,} spikes)',
                 loc='left', fontsize=9, color='#333', pad=4)
    ax.tick_params(labelbottom=False)

    # ── C: Input shell raster (tonotopic) ──
    ax = axes[2]
    if len(input_spikes) > 0:
        y_vals = np.array([input_id_order.get(nid, 0) for nid in input_spikes['neuron_id']])
        c_vals = np.array([input_primary_bins.get(nid, 0) for nid in input_spikes['neuron_id']])
        ax.scatter(input_spikes['time_ms'].values, y_vals,
                   s=0.7, c=c_vals, cmap='viridis', alpha=0.9,
                   vmin=0, vmax=127, rasterized=True, linewidths=0)
    n_input_spikes = len(input_spikes)
    input_rate = n_input_spikes / (total_ms / 1000) / max(n_active_input, 1)
    ax.set_ylabel('Input neuron\n(tonotopic)')
    ax.set_ylim(-2, max(n_active_input, 1) + 2)
    mark_epochs(ax)
    ax.set_title(f'C   Input shell — {n_active_input}/{n_input} active, '
                 f'{input_rate:.0f} Hz/neuron',
                 loc='left', fontsize=9, color='#333', pad=4)
    ax.tick_params(labelbottom=False)

    # ── D: Reservoir raster (tonotopic) ──
    ax = axes[3]
    if len(res_spikes) > 0:
        y_vals = np.array([res_id_map.get(nid, 0) for nid in res_spikes['neuron_id']])
        if res_freq_vals:
            c_freq = np.array([res_freq_vals.get(nid, 0) for nid in res_spikes['neuron_id']])
            ax.scatter(res_spikes['time_ms'].values, y_vals,
                       s=0.7, c=c_freq, cmap='viridis', alpha=0.85,
                       vmin=0, vmax=127, rasterized=True, linewidths=0)
        else:
            ax.scatter(res_spikes['time_ms'].values, y_vals,
                       s=0.4, c=C_RES, alpha=0.6, rasterized=True, linewidths=0)
    n_res_spikes = len(res_spikes)
    res_rate = n_res_spikes / (total_ms / 1000) / max(n_active_res, 1)
    ax.set_ylabel('Reservoir neuron\n(tonotopic)')
    ax.set_ylim(-5, max(n_active_res, 1) + 5)
    mark_epochs(ax)
    ax.set_title(f'D   Reservoir — {n_active_res}/{n_reservoir} active, '
                 f'{res_rate:.0f} Hz/neuron',
                 loc='left', fontsize=9, color='#333', pad=4)
    ax.tick_params(labelbottom=False)

    # ── E: Layer-wise PSTH ──
    ax = axes[4]
    psth_bin = 2.0
    bins_arr = np.arange(0, total_ms + psth_bin, psth_bin)
    centers = bins_arr[:-1] + psth_bin / 2

    # Load E/I identity from snapshot
    snap = np.load(SNAPSHOT)
    is_inh = snap['is_inhibitory'].astype(bool)

    # Split spikes by layer
    input_t = input_spikes['time_ms'].values
    res_exc_mask = (spikes['zone'] == 'reservoir') & \
                   spikes['neuron_id'].map(lambda nid: not is_inh[nid] if nid < len(is_inh) else True)
    res_inh_mask = (spikes['zone'] == 'reservoir') & \
                   spikes['neuron_id'].map(lambda nid: is_inh[nid] if nid < len(is_inh) else False)
    res_exc_t = spikes.loc[res_exc_mask, 'time_ms'].values
    res_inh_t = spikes.loc[res_inh_mask, 'time_ms'].values

    # Gaussian-smoothed firing rates
    from scipy.ndimage import gaussian_filter1d
    sigma = 5.0 / psth_bin  # 5ms smoothing kernel

    layers = [
        ('Input Arc', input_t, '#2d8a4e', n_active_input),
        ('Reservoir E', res_exc_t, '#cc3333', max(1, n_reservoir - int(is_inh.sum()))),
        ('Reservoir I', res_inh_t, '#3a7ebf', max(1, int(is_inh.sum()))),
    ]
    for label, times, color, n_neurons in layers:
        counts, _ = np.histogram(times, bins=bins_arr)
        rate_hz = counts / (psth_bin / 1000) / max(n_neurons, 1)
        smoothed = gaussian_filter1d(rate_hz.astype(float), sigma)
        ax.plot(centers, smoothed, color=color, lw=1.2, label=label)
        ax.fill_between(centers, smoothed, alpha=0.15, color=color, linewidth=0)

    mark_epochs(ax)
    ax.set_ylabel('Rate (Hz)')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.set_title(f'E   Layer PSTH ({psth_bin:.0f}ms bins, per-neuron rate)',
                 loc='left', fontsize=9, color='#333', pad=4)
    ax.set_xlim(0, total_ms)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Epoch annotations on panel A
    ax0 = axes[0]
    if warmup_ms > 0:
        ax0.text(warmup_ms / 2, 132, 'warmup', fontsize=6.5, color=C_WARMUP,
                 ha='center', va='bottom', fontstyle='italic', clip_on=False)
    ax0.text(audio_end, 132, 'offset', fontsize=6.5, color=C_STIM_LINE,
             ha='center', va='bottom', fontstyle='italic', clip_on=False)

    out_path = Path(args.output) if args.output else OUT_DIR / f'raster_{filename}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
