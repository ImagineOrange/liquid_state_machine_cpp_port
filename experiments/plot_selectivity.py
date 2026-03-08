#!/usr/bin/env python3
"""
Frequency selectivity analysis figure (2 panels):
  A. Per-frequency-band firing rate comparison (BSA vs input spikes)
  B. Per-neuron selectivity: r(spike, matched BSA) vs r(spike, unmatched BSA)

Reads from an existing raster dump directory (run plot_raster.py first).

Usage:
  python experiments/plot_selectivity.py [--dump-dir results/raster/dump] [--output <path>]
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = CPP_DIR / 'results' / 'raster'
DEFAULT_DUMP = OUT_DIR / 'dump'

C_BSA = '#e8870e'
C_INPUT = '#1b6ca8'
C_SCATTER = '#1b6ca8'
C_DIAG = '#888888'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-dir', type=str, default=str(DEFAULT_DUMP))
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        print(f"Dump dir not found: {dump_dir}")
        print("Run plot_raster.py first to generate the dump.")
        return

    spikes = pd.read_csv(dump_dir / 'spikes.csv')
    bsa = pd.read_csv(dump_dir / 'bsa.csv')
    with open(dump_dir / 'meta.json') as f:
        meta = json.load(f)
    neuron_map = load_neuron_mapping(dump_dir)

    audio_end = meta['audio_duration_ms']
    digit = meta['digit']
    filename = meta['filename']

    input_spikes = spikes[spikes['zone'] == 'input']
    input_stim = input_spikes[input_spikes['time_ms'] <= audio_end]
    bsa_stim = bsa[bsa['time_ms'] <= audio_end]

    # Tonotopic mapping
    input_primary_bins = {}
    for nid, (primary, _, _) in neuron_map.items():
        input_primary_bins[nid] = primary

    # ── Style ──
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9.5,
        'axes.linewidth': 0.6,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, (ax_band, ax_sel) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ══════════════════════════════════════════════════════════
    # A: Frequency band rate comparison
    # ══════════════════════════════════════════════════════════
    n_bands = 16
    band_edges = np.linspace(0, 128, n_bands + 1).astype(int)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2
    bar_w = (128 / n_bands) * 0.38

    bsa_band_rates = np.zeros(n_bands)
    input_band_rates = np.zeros(n_bands)

    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i + 1]
        mask_bsa = (bsa_stim['freq_bin'] >= lo) & (bsa_stim['freq_bin'] < hi)
        bsa_band_rates[i] = mask_bsa.sum() / (audio_end / 1000)

        nids_in_band = [nid for nid, fb in input_primary_bins.items()
                        if lo <= fb < hi]
        if nids_in_band:
            spk_count = input_stim[input_stim['neuron_id'].isin(nids_in_band)].shape[0]
            input_band_rates[i] = spk_count / (audio_end / 1000) / len(nids_in_band)

    bsa_norm = bsa_band_rates / max(bsa_band_rates.max(), 1)
    input_norm = input_band_rates / max(input_band_rates.max(), 1e-9)
    r_band = np.corrcoef(bsa_norm, input_norm)[0, 1] if bsa_norm.std() > 0 else 0

    ax_band.bar(band_centers - bar_w * 0.55, bsa_norm, width=bar_w, alpha=0.75,
                color=C_BSA, label='BSA input', edgecolor='none')
    ax_band.bar(band_centers + bar_w * 0.55, input_norm, width=bar_w, alpha=0.75,
                color=C_INPUT, label='Input spikes', edgecolor='none')
    ax_band.set_xlabel('Frequency bin')
    ax_band.set_ylabel('Normalized rate')
    ax_band.set_xlim(-2, 130)
    ax_band.set_ylim(0, 1.15)
    ax_band.legend(fontsize=8.5, frameon=False, loc='upper right')
    ax_band.text(0.03, 0.95, 'A', transform=ax_band.transAxes, fontsize=14,
                 fontweight='bold', va='top')
    ax_band.set_title(f'Frequency band rates — $r$ = {r_band:.3f}', fontweight='bold')

    # ══════════════════════════════════════════════════════════
    # B: Per-neuron selectivity scatter
    # ══════════════════════════════════════════════════════════
    bin_width_ms = 20.0
    n_time_bins = max(1, int(audio_end / bin_width_ms))
    all_freq_bins = list(range(128))
    rng = np.random.RandomState(42)

    matched_corrs = []
    random_corrs = []

    for nid, (_, bins_list, _) in neuron_map.items():
        if not bins_list:
            continue
        nid_t = input_stim[input_stim['neuron_id'] == nid]['time_ms'].values
        if len(nid_t) < 3:
            continue

        spk_binned = np.histogram(nid_t, bins=n_time_bins, range=(0, audio_end))[0].astype(float)

        # Matched
        matched_t = bsa_stim[bsa_stim['freq_bin'].isin(bins_list)]['time_ms'].values
        bsa_matched = np.histogram(matched_t, bins=n_time_bins, range=(0, audio_end))[0].astype(float)
        if bsa_matched.std() > 0 and spk_binned.std() > 0:
            r_m, _ = pearsonr(bsa_matched, spk_binned)
        else:
            r_m = 0.0
        matched_corrs.append(r_m)

        # Unmatched
        unmatched = [b for b in all_freq_bins if b not in bins_list]
        rand_bins = list(rng.choice(unmatched, size=min(len(bins_list), len(unmatched)), replace=False))
        rand_t = bsa_stim[bsa_stim['freq_bin'].isin(rand_bins)]['time_ms'].values
        bsa_rand = np.histogram(rand_t, bins=n_time_bins, range=(0, audio_end))[0].astype(float)
        if bsa_rand.std() > 0 and spk_binned.std() > 0:
            r_r, _ = pearsonr(bsa_rand, spk_binned)
        else:
            r_r = 0.0
        random_corrs.append(r_r)

    matched_corrs = np.array(matched_corrs)
    random_corrs = np.array(random_corrs)

    ax_sel.scatter(random_corrs, matched_corrs, s=18, alpha=0.55, c=C_SCATTER,
                   edgecolors='white', linewidths=0.3, zorder=3)

    lim = max(abs(matched_corrs).max(), abs(random_corrs).max(), 0.3) * 1.15
    ax_sel.plot([-lim, lim], [-lim, lim], color=C_DIAG, ls='--', lw=0.8, alpha=0.5, zorder=1)
    ax_sel.set_xlim(-lim, lim)
    ax_sel.set_ylim(-lim, lim)
    ax_sel.set_aspect('equal', adjustable='box')
    ax_sel.set_xlabel('$r$(spike rate, unmatched BSA)')
    ax_sel.set_ylabel('$r$(spike rate, matched BSA)')

    mean_m = matched_corrs.mean()
    mean_r = random_corrs.mean()
    n_above = (matched_corrs > random_corrs).sum()
    n_total = len(matched_corrs)

    ax_sel.text(0.03, 0.95, 'B', transform=ax_sel.transAxes, fontsize=14,
                fontweight='bold', va='top')
    ax_sel.set_title(f'Frequency selectivity — {n_above}/{n_total} above diagonal '
                     f'({100 * n_above / max(n_total, 1):.0f}%)', fontweight='bold')

    # Summary annotation
    ax_sel.text(0.97, 0.05,
                f'matched $\\bar{{r}}$ = {mean_m:.3f}\n'
                f'random $\\bar{{r}}$ = {mean_r:.3f}',
                transform=ax_sel.transAxes, fontsize=8.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout(w_pad=3)

    out_path = (Path(args.output) if args.output
                else OUT_DIR / f'selectivity_{filename}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n--- Frequency Selectivity Summary ---")
    print(f"  Band-rate correlation:        r = {r_band:.4f}")
    print(f"  Mean r(spike, matched BSA):   {mean_m:.4f}")
    print(f"  Mean r(spike, unmatched BSA): {mean_r:.4f}")
    print(f"  Neurons above diagonal:       {n_above}/{n_total}")


if __name__ == '__main__':
    main()
