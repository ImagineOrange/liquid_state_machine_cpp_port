#!/usr/bin/env python3
"""
Single-neuron tuning curve detail figure.
Shows one input neuron's Gaussian selectivity and overlap with neighbors.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import os

CPP_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
OVERLAP_K = 4
TUNING_SIGMA_CHANNELS = 1.5
TARGET = 493

OUT_DIR = str(CPP_DIR / 'results' / 'gaussian_tuning_curves')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load and compute (same as main script) ---
snap = np.load(SNAPSHOT, allow_pickle=True)
positions = snap['positions']
input_neuron_indices = snap['input_neuron_indices']
n_mel = int(snap['input_neuron_mapping'].shape[0])

neuron_phi = {}
for nid in input_neuron_indices:
    neuron_phi[nid] = np.arctan2(positions[nid][0], positions[nid][2])

sorted_neurons = sorted(input_neuron_indices, key=lambda nid: neuron_phi[nid])
sorted_phis = [neuron_phi[nid] for nid in sorted_neurons]
phi_arr = np.array(sorted_phis)
phi_min, phi_max = phi_arr[0], phi_arr[-1]

channel_centers = np.interp(
    np.linspace(0, len(phi_arr) - 1, n_mel),
    np.arange(len(phi_arr)), phi_arr)

avg_spacing = (phi_max - phi_min) / max(1, len(phi_arr) - 1)
sigma_phi = TUNING_SIGMA_CHANNELS * avg_spacing
inv_2sigma2 = 1.0 / (2.0 * sigma_phi**2)

k = min(OVERLAP_K, n_mel)
neuron_to_bins = {}
bin_to_neurons = {}
for nid in input_neuron_indices:
    nphi = neuron_phi[nid]
    dists = sorted([(abs(channel_centers[m] - nphi), m) for m in range(n_mel)])
    neuron_to_bins[nid] = []
    for j in range(k):
        d, mel = dists[j]
        w = np.exp(-d * d * inv_2sigma2)
        neuron_to_bins[nid].append((mel, w))
        bin_to_neurons.setdefault(mel, []).append((nid, w))

if TARGET not in neuron_to_bins:
    TARGET = sorted_neurons[len(sorted_neurons) // 2]

target_bins = sorted(neuron_to_bins[TARGET], key=lambda x: x[0])
target_mels = [b for b, _ in target_bins]
target_ws = [w for _, w in target_bins]
target_phi = neuron_phi[TARGET]
target_center_bin = max(target_bins, key=lambda x: x[1])[0]

# Neighbors that share bins
neighbor_nids = set()
for mel, _ in target_bins:
    for nid, _ in bin_to_neurons.get(mel, []):
        if nid != TARGET:
            neighbor_nids.add(nid)
neighbor_nids = sorted(neighbor_nids, key=lambda nid: neuron_phi[nid])

print(f"Neuron {TARGET}: center bin={target_center_bin}, phi={target_phi:.4f}")
print(f"  Bins: {target_mels}, weights: {[f'{w:.3f}' for w in target_ws]}")
print(f"  {len(neighbor_nids)} overlapping neighbors")

# --- Figure ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')

x_fine = np.linspace(0, n_mel - 1, 2000)
centers_fine = np.interp(x_fine, np.arange(n_mel), channel_centers)

# Neighbor Gaussians (draw first, behind everything)
neighbor_palette = ['#e8a598', '#a8d8b9', '#c4b7d9', '#f5d6a8',
                    '#9ec5e8', '#f0b8d4', '#b8d4a8', '#d4c4a8']
for ni, nid in enumerate(neighbor_nids):
    nphi = neuron_phi[nid]
    d_n = np.abs(centers_fine - nphi)
    g_n = np.exp(-d_n**2 * inv_2sigma2)
    mask = g_n > 0.15
    color = neighbor_palette[ni % len(neighbor_palette)]
    ax.fill_between(x_fine[mask], g_n[mask], alpha=0.12, color=color)
    ax.plot(x_fine[mask], g_n[mask], color=color, linewidth=1.8, alpha=0.5,
            label=f'Neuron {nid}')

# Target full Gaussian envelope (clip to meaningful region w > 0.05)
d_target = np.abs(centers_fine - target_phi)
g_target = np.exp(-d_target**2 * inv_2sigma2)
mask_t = g_target > 0.05
ax.fill_between(x_fine[mask_t], g_target[mask_t], alpha=0.08, color='#2166ac')
ax.plot(x_fine[mask_t], g_target[mask_t], color='#2166ac', linewidth=2.5, alpha=0.4,
        linestyle='--', zorder=3)

# Target mapped bins as bars
bar_colors = []
for w in target_ws:
    intensity = 0.3 + 0.7 * w  # brighter for higher weight
    bar_colors.append((*matplotlib.colors.to_rgb('#2166ac'), intensity))

ax.bar(target_mels, target_ws, width=0.55, color='#2166ac', alpha=0.9,
       edgecolor='white', linewidth=1.2, zorder=4)

# Weight annotations
for b, w in zip(target_mels, target_ws):
    d_ch = abs(b - target_center_bin)
    ax.annotate(f'{w:.2f}', (b, w + 0.025), ha='center', fontsize=11,
                color='#2166ac', fontweight='bold', zorder=5)

# Center marker
ax.annotate('', xy=(target_center_bin, -0.06), xytext=(target_center_bin, -0.02),
            arrowprops=dict(arrowstyle='->', color='#2166ac', lw=2),
            annotation_clip=False)
ax.text(target_center_bin, -0.09, 'center', ha='center', fontsize=9,
        color='#2166ac', fontweight='bold', clip_on=False)

# Zoom to relevant region
all_bins = set(target_mels)
for nid in neighbor_nids:
    for b, _ in neuron_to_bins[nid]:
        all_bins.add(b)
pad = 4
xmin = max(0, min(all_bins) - pad)
xmax = min(n_mel - 1, max(all_bins) + pad)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.02, 1.12)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['bottom'].set_color('#555555')
ax.tick_params(colors='#555555', labelsize=9)
ax.set_xlabel('Frequency Bin (mel index)', fontsize=11, color='#333333')
ax.set_ylabel('Tuning Weight', fontsize=11, color='#333333')

# Legend
handles, labels = ax.get_legend_handles_labels()
# Add target entry manually at front
from matplotlib.patches import Patch
target_patch = Patch(facecolor='#2166ac', alpha=0.9, edgecolor='white',
                     label=f'Neuron {TARGET} (K={k} mapped bins)')
handles = [target_patch] + handles
labels = [target_patch.get_label()] + labels
ax.legend(handles, labels, fontsize=8.5, loc='upper right',
          framealpha=0.9, edgecolor='#cccccc', fancybox=True)

# Title
ax.set_title(f'Input Neuron {TARGET} — Gaussian Frequency Tuning\n'
             f'K={k} bins/neuron  |  sigma = {TUNING_SIGMA_CHANNELS} ch spacing  |  '
             f'{len(neighbor_nids)} overlapping neighbors',
             fontsize=13, fontweight='bold', color='#222222', pad=12)

# Info box
total_drive = sum(target_ws)
info = (f'Total drive: {total_drive:.2f}x stim_current\n'
        f'Peak weight: {max(target_ws):.3f} (bin {target_center_bin})\n'
        f'Min weight: {min(target_ws):.3f}')
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9)
ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', bbox=props, color='#444444')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'neuron_{TARGET}_tuning_detail.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#fafafa')
plt.close()
print(f"Saved: {out}")
