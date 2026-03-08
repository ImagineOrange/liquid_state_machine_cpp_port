#!/usr/bin/env python3
"""Generate diagnostic traces for top grid search configs to compare."""
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
OUT_DIR = CPP_DIR / 'results' / 'top_config_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEURON = 493
SAMPLE = 'spike_train_0_george_18.npz'

configs = [
    {'stim': 0.0889, 'tau_e': 0.31, 'adapt': 0.005, 'label': 'A: stim=0.089 tau=0.31'},
    {'stim': 0.0158, 'tau_e': 1.93, 'adapt': 0.005, 'label': 'B: stim=0.016 tau=1.93'},
    {'stim': 0.0281, 'tau_e': 1.05, 'adapt': 0.005, 'label': 'C: stim=0.028 tau=1.05'},
    {'stim': 0.0500, 'tau_e': 0.57, 'adapt': 0.005, 'label': 'D: stim=0.050 tau=0.57'},
    {'stim': 0.0089, 'tau_e': 3.55, 'adapt': 0.005, 'label': 'E: stim=0.009 tau=3.55'},
    {'stim': 0.0667, 'tau_e': 0.57, 'adapt': 0.005, 'label': 'F: stim=0.067 tau=0.57'},
]

traces = {}
for i, cfg in enumerate(configs):
    csv_path = OUT_DIR / f'trace_{i}.csv'
    cmd = [
        str(CPP_BIN), '--snapshot', str(SNAPSHOT),
        '--trace-neuron', str(NEURON),
        '--trace-file', str(BSA_DIR / SAMPLE),
        '--trace-output', str(csv_path),
        '--data-dir', str(DATA_DIR),
        '--samples-per-digit', '1', '--n-workers', '1',
        '--no-noise', '--no-input-nmda',
        '--stim-current', str(cfg['stim']),
        '--input-tau-e', str(cfg['tau_e']),
        '--input-adapt-inc', str(cfg['adapt']),
    ]
    print(f"Config {i}: {cfg['label']}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        continue
    traces[i] = pd.read_csv(csv_path)

# Plot comparison: V + g_e + spike raster for all configs
plt.style.use('default')
plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3})

colors = plt.cm.tab10(np.linspace(0, 0.6, len(configs)))

fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

# Panel 1: Spike rasters
ax = axes[0]
for i, cfg in enumerate(configs):
    if i not in traces:
        continue
    df = traces[i]
    t = df['t_ms'].values
    spk_t = t[df['spiked'].values == 1]
    ax.eventplot([spk_t], lineoffsets=i, linelengths=0.8, colors=[colors[i]])
ax.set_yticks(range(len(configs)))
ax.set_yticklabels([c['label'] for c in configs], fontsize=8)
ax.set_title('Spike Rasters — Top 6 Configs (neuron 493)')

# Panel 2: Membrane potential
ax = axes[1]
for i, cfg in enumerate(configs):
    if i not in traces:
        continue
    df = traces[i]
    ax.plot(df['t_ms'], df['v'], color=colors[i], linewidth=0.4, alpha=0.8, label=cfg['label'])
ax.set_ylabel('V (mV)')
ax.set_title('Membrane Potential')
ax.legend(fontsize=7, loc='upper right', ncol=2)

# Panel 3: g_e
ax = axes[2]
for i, cfg in enumerate(configs):
    if i not in traces:
        continue
    df = traces[i]
    ax.plot(df['t_ms'], df['g_e'], color=colors[i], linewidth=0.4, alpha=0.8)
ax.set_ylabel('g_e')
ax.set_title('Excitatory Conductance')

# Panel 4: Adaptation
ax = axes[3]
for i, cfg in enumerate(configs):
    if i not in traces:
        continue
    df = traces[i]
    ax.plot(df['t_ms'], df['adaptation'], color=colors[i], linewidth=0.5, alpha=0.8)
ax.set_ylabel('Adaptation')
ax.set_title('Adaptation Current')
ax.set_xlabel('Time (ms)')

# Stats table
stats_text = []
for i, cfg in enumerate(configs):
    if i not in traces:
        continue
    df = traces[i]
    t = df['t_ms'].values
    spk = t[df['spiked'].values == 1]
    n = len(spk)
    dur = t[-1] / 1000
    rate = n / dur if dur > 0 else 0
    if n > 2:
        isis = np.diff(spk)
        cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        mean_isi = np.mean(isis)
    else:
        cv = 0; mean_isi = 0
    stats_text.append(f"{cfg['label']}: {n} spk, {rate:.0f}Hz, CV={cv:.3f}, ISI={mean_isi:.1f}ms")

fig.text(0.01, 0.01, '\n'.join(stats_text), fontsize=8, family='monospace',
         va='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = OUT_DIR / 'top_configs_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
