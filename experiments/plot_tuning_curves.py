#!/usr/bin/env python3
"""
Visualize Gaussian frequency tuning curves for ALL input neurons.
Generates two figures:
  1. Full coverage overview (weight matrix, overlaid curves, coverage stats)
  2. Single-neuron detail (bin selectivity, weight overlaps with neighbors)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import os

CPP_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
OVERLAP_K = 4
TUNING_SIGMA_CHANNELS = 1.5

OUT_DIR = str(CPP_DIR / 'results' / 'gaussian_tuning_curves')
os.makedirs(OUT_DIR, exist_ok=True)

# Load snapshot
snap = np.load(SNAPSHOT, allow_pickle=True)
positions = snap['positions']
input_neuron_indices = snap['input_neuron_indices']
n_mel = int(snap['input_neuron_mapping'].shape[0])
n_input = len(input_neuron_indices)

# Compute phi angles
neuron_phi = {}
for nid in input_neuron_indices:
    x, y, z = positions[nid]
    neuron_phi[nid] = np.arctan2(x, z)

sorted_neurons = sorted(input_neuron_indices, key=lambda nid: neuron_phi[nid])
neuron_rank = {nid: i for i, nid in enumerate(sorted_neurons)}

sorted_phis = sorted(neuron_phi.values())
phi_min, phi_max = sorted_phis[0], sorted_phis[-1]

# Quantile-based channel centers
phi_arr = np.array(sorted_phis)
channel_centers = np.interp(
    np.linspace(0, len(phi_arr) - 1, n_mel),
    np.arange(len(phi_arr)),
    phi_arr
)

avg_spacing = (phi_max - phi_min) / max(1, len(phi_arr) - 1)
sigma_phi = TUNING_SIGMA_CHANNELS * avg_spacing
inv_2sigma2 = 1.0 / (2.0 * sigma_phi**2)

# Build neuron -> K nearest bins mapping
k = min(OVERLAP_K, n_mel)
neuron_to_bins = {}
bin_to_neurons = {}

for nid in input_neuron_indices:
    nphi = neuron_phi[nid]
    dists = [(abs(channel_centers[m] - nphi), m) for m in range(n_mel)]
    dists.sort()
    neuron_to_bins[nid] = []
    for j in range(k):
        d, mel_bin = dists[j]
        w = np.exp(-d * d * inv_2sigma2)
        neuron_to_bins[nid].append((mel_bin, w))
        bin_to_neurons.setdefault(mel_bin, []).append((nid, w))

neurons_per_bin = [len(bin_to_neurons.get(m, [])) for m in range(n_mel)]
uncovered_bins = [m for m in range(n_mel) if m not in bin_to_neurons]
total_weight_per_bin = [sum(w for _, w in bin_to_neurons.get(m, [])) for m in range(n_mel)]

print(f"{n_input} input neurons, {n_mel} mel bins, K={k}, sigma={TUNING_SIGMA_CHANNELS}")
print(f"Bins with coverage: {n_mel - len(uncovered_bins)}/{n_mel}")
print(f"Neurons/bin: min={min(neurons_per_bin)}, max={max(neurons_per_bin)}, "
      f"mean={np.mean(neurons_per_bin):.1f}")

# ========================================================================
# FIGURE 1: Full coverage overview
# ========================================================================
plt.style.use('default')
plt.rcParams.update({
    'font.size': 9, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})

fig, axes = plt.subplots(4, 1, figsize=(18, 16),
                          gridspec_kw={'height_ratios': [4, 1.5, 1.2, 1.2]})

# Panel 1: Weight matrix
ax = axes[0]
weight_matrix = np.zeros((n_input, n_mel))
for nid in sorted_neurons:
    row = neuron_rank[nid]
    for mel_bin, w in neuron_to_bins[nid]:
        weight_matrix[row, mel_bin] = w

im = ax.imshow(weight_matrix, aspect='auto', cmap='inferno', origin='lower',
               interpolation='nearest', vmin=0, vmax=1.0)
ax.set_xlabel('Frequency Bin (mel index)', fontsize=11)
ax.set_ylabel('Input Neuron (sorted by phi)', fontsize=11)
ax.set_title(f'Gaussian Tuning Weight Matrix — {n_input} Neurons x {n_mel} Bins\n'
             f'K={k} bins/neuron, sigma={TUNING_SIGMA_CHANNELS} channel spacing '
             f'| Quantile-based channel centers',
             fontsize=13, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='Tuning Weight')

# Panel 2: All tuning curves overlaid
ax = axes[1]
cmap = plt.cm.viridis
norm = Normalize(vmin=0, vmax=n_input - 1)
x_fine = np.linspace(0, n_mel - 1, 1000)
centers_interp = np.interp(x_fine, np.arange(n_mel), channel_centers)
for i, nid in enumerate(sorted_neurons):
    nphi = neuron_phi[nid]
    d_fine = np.abs(centers_interp - nphi)
    g_fine = np.exp(-d_fine**2 * inv_2sigma2)
    mask = g_fine > 0.01
    ax.plot(x_fine[mask], g_fine[mask], color=cmap(norm(i)), linewidth=0.8, alpha=0.5)
ax.set_xlim(0, n_mel - 1)
ax.set_ylim(0, 1.1)
ax.set_xlabel('Frequency Bin', fontsize=10)
ax.set_ylabel('Weight', fontsize=10)
ax.set_title('All Gaussian Tuning Curves Overlaid', fontsize=11, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, shrink=0.8, label='Neuron index (low->high freq)')

# Panel 3: Neurons per bin
ax = axes[2]
ax.bar(range(n_mel), neurons_per_bin, width=1.0, color='steelblue', alpha=0.7, edgecolor='none')
ax.axhline(np.mean(neurons_per_bin), color='orange', linewidth=2, linestyle='--',
           label=f'Mean = {np.mean(neurons_per_bin):.1f}')
if uncovered_bins:
    for ub in uncovered_bins:
        ax.axvline(ub, color='red', alpha=0.3, linewidth=1)
    ax.plot([], [], color='red', alpha=0.3, label=f'{len(uncovered_bins)} uncovered bins')
ax.set_xlim(0, n_mel - 1)
ax.set_xlabel('Frequency Bin', fontsize=10)
ax.set_ylabel('# Neurons', fontsize=10)
ax.set_title(f'Coverage: Neurons Per Frequency Bin (min={min(neurons_per_bin)}, '
             f'max={max(neurons_per_bin)})', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15, axis='y')

# Panel 4: Total weight per bin
ax = axes[3]
ax.bar(range(n_mel), total_weight_per_bin, width=1.0, color='#9467bd', alpha=0.7, edgecolor='none')
ax.axhline(np.mean(total_weight_per_bin), color='orange', linewidth=2, linestyle='--',
           label=f'Mean = {np.mean(total_weight_per_bin):.2f}')
ax.set_xlim(0, n_mel - 1)
ax.set_xlabel('Frequency Bin', fontsize=10)
ax.set_ylabel('Total Weight', fontsize=10)
ax.set_title('Aggregate Drive Per Bin', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'gaussian_tuning_curves.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ========================================================================
# FIGURE 2: Single-neuron detail — neuron 493 (from diagnostics)
# ========================================================================
# Pick neuron 493 if it's an input neuron, otherwise pick a mid-arc neuron
TARGET = 493
if TARGET not in neuron_to_bins:
    mid_idx = len(sorted_neurons) // 2
    TARGET = sorted_neurons[mid_idx]
    print(f"Neuron 493 not in input set, using neuron {TARGET} instead")

target_bins = sorted(neuron_to_bins[TARGET], key=lambda x: x[0])
target_mels = [b for b, _ in target_bins]
target_ws = [w for _, w in target_bins]
target_phi = neuron_phi[TARGET]
target_center_bin = max(target_bins, key=lambda x: x[1])[0]

# Find neighbors: neurons that share at least one bin with target
neighbor_nids = set()
for mel, _ in target_bins:
    for nid, _ in bin_to_neurons.get(mel, []):
        if nid != TARGET:
            neighbor_nids.add(nid)
neighbor_nids = sorted(neighbor_nids, key=lambda nid: neuron_phi[nid])

print(f"\nNeuron {TARGET}: center bin={target_center_bin}, phi={target_phi:.4f}")
print(f"  Mapped bins: {target_mels}, weights: {[f'{w:.3f}' for w in target_ws]}")
print(f"  Overlapping neighbors: {len(neighbor_nids)}")

fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                          gridspec_kw={'height_ratios': [2.5, 2, 1.5, 1.5]})

# --- Panel A: Target neuron tuning curve with neighbor overlap ---
ax = axes[0]

# Full Gaussian envelope for target
x_fine = np.linspace(0, n_mel - 1, 1000)
centers_fine = np.interp(x_fine, np.arange(n_mel), channel_centers)
d_fine = np.abs(centers_fine - target_phi)
g_fine = np.exp(-d_fine**2 * inv_2sigma2)
ax.fill_between(x_fine, g_fine, alpha=0.15, color='#1f77b4')
ax.plot(x_fine, g_fine, color='#1f77b4', linewidth=2, alpha=0.6, linestyle='--',
        label=f'Neuron {TARGET} (full Gaussian)')

# Target mapped bins as bars
ax.bar(target_mels, target_ws, width=0.6, color='#1f77b4', alpha=0.9,
       edgecolor='white', linewidth=0.8, label=f'Neuron {TARGET} mapped bins (K={k})')
for b, w in zip(target_mels, target_ws):
    ax.annotate(f'{w:.3f}', (b, w + 0.02), ha='center', fontsize=8,
                color='#1f77b4', fontweight='bold')

# Neighbor tuning curves (faded)
neighbor_colors = plt.cm.Set2(np.linspace(0, 1, max(len(neighbor_nids), 1)))
for ni, nid in enumerate(neighbor_nids):
    nphi = neuron_phi[nid]
    d_n = np.abs(centers_fine - nphi)
    g_n = np.exp(-d_n**2 * inv_2sigma2)
    mask = g_n > 0.05
    lbl = f'Neuron {nid}' if ni < 6 else None
    ax.plot(x_fine[mask], g_n[mask], color=neighbor_colors[ni % len(neighbor_colors)],
            linewidth=1.0, alpha=0.4, label=lbl)

# Zoom to relevant region
all_relevant_bins = set(target_mels)
for nid in neighbor_nids:
    for b, _ in neuron_to_bins[nid]:
        all_relevant_bins.add(b)
xmin = min(all_relevant_bins) - 3
xmax = max(all_relevant_bins) + 3
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Tuning Weight', fontsize=10)
ax.set_title(f'Neuron {TARGET} — Gaussian Tuning Curve & Neighbor Overlap\n'
             f'Center bin={target_center_bin}, phi={target_phi:.3f} rad, '
             f'{len(neighbor_nids)} overlapping neighbors',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(True, alpha=0.15)

# --- Panel B: Bin-level breakdown showing all neurons at each of target's bins ---
ax = axes[1]
bar_width = 0.15
all_nids_at_bins = set()
for mel in target_mels:
    for nid, _ in bin_to_neurons.get(mel, []):
        all_nids_at_bins.add(nid)
all_nids_at_bins = sorted(all_nids_at_bins, key=lambda nid: neuron_phi[nid])
nid_to_color = {}
nid_to_color[TARGET] = '#1f77b4'
other_colors = plt.cm.tab10(np.linspace(0.1, 0.9, len(all_nids_at_bins)))
ci = 0
for nid in all_nids_at_bins:
    if nid not in nid_to_color:
        nid_to_color[nid] = other_colors[ci]
        ci += 1

for bi, mel in enumerate(target_mels):
    neurons_here = bin_to_neurons.get(mel, [])
    neurons_here_sorted = sorted(neurons_here, key=lambda x: -x[1])
    for ni, (nid, w) in enumerate(neurons_here_sorted):
        offset = (ni - len(neurons_here_sorted) / 2) * bar_width
        color = nid_to_color.get(nid, 'gray')
        edgecolor = 'black' if nid == TARGET else 'white'
        lw = 1.5 if nid == TARGET else 0.5
        ax.bar(mel + offset, w, width=bar_width * 0.9, color=color,
               alpha=0.85, edgecolor=edgecolor, linewidth=lw)
        if w > 0.3:
            ax.text(mel + offset, w + 0.015, f'{nid}', ha='center', fontsize=5.5,
                    rotation=45, color='dimgray')

ax.set_xlim(min(target_mels) - 1.5, max(target_mels) + 1.5)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Frequency Bin', fontsize=10)
ax.set_ylabel('Weight', fontsize=10)
ax.set_title(f'All Neurons at Neuron {TARGET}\'s Bins — '
             f'Who Else Listens Here? (bold outline = target)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')

# --- Panel C: Target's weight profile (bar chart with bin labels) ---
ax = axes[2]
colors = ['#1f77b4' if w == max(target_ws) else '#aec7e8' for w in target_ws]
bars = ax.bar(range(k), target_ws, color=colors, edgecolor='white', width=0.7)
ax.set_xticks(range(k))
ax.set_xticklabels([f'Bin {m}' for m in target_mels], fontsize=9)
for i, (b, w) in enumerate(zip(target_mels, target_ws)):
    d_channels = abs(b - target_center_bin)
    ax.annotate(f'w={w:.3f}\n({d_channels} ch away)',
                (i, w + 0.02), ha='center', fontsize=8)
ax.set_ylim(0, 1.2)
ax.set_ylabel('Tuning Weight', fontsize=10)
ax.set_title(f'Neuron {TARGET} — Weight Profile Across {k} Mapped Bins',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')

# --- Panel D: Selectivity comparison — target vs its neighbors' total drive ---
ax = axes[3]
# For each of target's bins, show target's share of total weight
target_share = []
for mel, tw in zip(target_mels, target_ws):
    total = total_weight_per_bin[mel]
    share = tw / total if total > 0 else 0
    target_share.append(share)

ax.bar(range(k), target_share, color='#1f77b4', alpha=0.8, edgecolor='white', width=0.7)
ax.set_xticks(range(k))
ax.set_xticklabels([f'Bin {m}' for m in target_mels], fontsize=9)
for i, (sh, mel) in enumerate(zip(target_share, target_mels)):
    n_neurons = neurons_per_bin[mel]
    ax.annotate(f'{sh:.0%}\n({n_neurons} neurons)',
                (i, sh + 0.01), ha='center', fontsize=8)
ax.set_ylim(0, max(target_share) * 1.3 if target_share else 1)
ax.set_ylabel('Share of Total Drive', fontsize=10)
ax.set_title(f'Neuron {TARGET} — Fractional Share of Each Bin\'s Total Drive',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, f'neuron_{TARGET}_tuning_detail.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
