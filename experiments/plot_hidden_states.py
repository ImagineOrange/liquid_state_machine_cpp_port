#!/usr/bin/env python3
"""
Hidden States Figure — Spike vs Adaptation Conductance Trajectories

A. Euclidean separability between digit centroids over time (spikes vs adaptation)
B. 3D PCA of spike-count state trajectories
C. 3D PCA of adaptation-conductance state trajectories

Usage:
    python experiments/plot_hidden_states.py [results_dir]
"""

import sys, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from itertools import combinations

RESULTS_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results/hidden_states")

# --- Style (match project conventions) ---
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

# --- Load data ---
meta = json.loads((RESULTS_DIR / "hidden_states_meta.json").read_text())
n_bins = meta["n_bins"]
n_res = meta["n_reservoir"]
digits = meta["digits"]
bin_ms = meta["bin_ms"]

print(f"Loading data: {n_bins} bins x {n_res} neurons, digits {digits}")

spike_df = pd.read_csv(RESULTS_DIR / "spike_trajectories.csv")
adapt_df = pd.read_csv(RESULTS_DIR / "adapt_trajectories.csv")

neuron_cols = [f"n{i}" for i in range(n_res)]

trial_ids = sorted(spike_df["trial"].unique())
n_trials = len(trial_ids)

def df_to_tensor(df):
    tensors = []
    for tid in trial_ids:
        block = df[df["trial"] == tid].sort_values("time_ms")
        tensors.append(block[neuron_cols].values[:n_bins])
    return np.array(tensors)

spike_tensor = df_to_tensor(spike_df)
adapt_tensor = df_to_tensor(adapt_df)

trial_digits = np.array([spike_df[spike_df["trial"] == t]["digit"].iloc[0] for t in trial_ids])
time_ms = np.arange(n_bins) * bin_ms + bin_ms / 2

print(f"Spike tensor: {spike_tensor.shape}, Adapt tensor: {adapt_tensor.shape}")

# --- Compute Euclidean separability ---
def compute_separability(tensor, trial_digits, digits):
    n_bins = tensor.shape[1]
    sep = np.zeros(n_bins)
    for b in range(n_bins):
        centroids = {}
        for d in digits:
            mask = trial_digits == d
            centroids[d] = tensor[mask, b, :].mean(axis=0)
        dists = []
        for d1, d2 in combinations(digits, 2):
            dists.append(np.linalg.norm(centroids[d1] - centroids[d2]))
        sep[b] = np.mean(dists)
    return sep

spike_sep = compute_separability(spike_tensor, trial_digits, digits)
adapt_sep = compute_separability(adapt_tensor, trial_digits, digits)

# --- PCA ---
spike_flat = spike_tensor.reshape(-1, n_res)
adapt_flat = adapt_tensor.reshape(-1, n_res)

pca_spike = PCA(n_components=3)
spike_pca = pca_spike.fit_transform(spike_flat).reshape(n_trials, n_bins, 3)

pca_adapt = PCA(n_components=3)
adapt_pca = pca_adapt.fit_transform(adapt_flat).reshape(n_trials, n_bins, 3)

# --- Color palette ---
DIGIT_COLORS = {
    0: "#e41a1c",
    1: "#377eb8",
    2: "#4daf4a",
    3: "#984ea3",
    4: "#ff7f00",
    5: "#a65628",
}

# Estimate stimulus end (total - post_stim)
stim_end_ms = time_ms[-1] + bin_ms / 2 - 200

# ================================================================
# FIGURE — 2-row layout: top = two PCA panels, bottom = separability
# ================================================================
fig = plt.figure(figsize=(16, 11))
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.86, wspace=0.30, hspace=0.35)

gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.3, 1], wspace=0.28, hspace=0.32)

# === Panel A: Spike PCA ===
ax_a = fig.add_subplot(gs[0, 0], projection="3d")
ax_a.set_facecolor('white')
ax_a.xaxis.pane.fill = False
ax_a.yaxis.pane.fill = False
ax_a.zaxis.pane.fill = False
ax_a.xaxis.pane.set_edgecolor('#dddddd')
ax_a.yaxis.pane.set_edgecolor('#dddddd')
ax_a.zaxis.pane.set_edgecolor('#dddddd')
ax_a.grid(True, alpha=0.15, linewidth=0.4)

for i in range(n_trials):
    d = trial_digits[i]
    traj = spike_pca[i]
    ax_a.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              color=DIGIT_COLORS[d], alpha=0.65, linewidth=1.3)
    ax_a.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                 color=DIGIT_COLORS[d], s=25, marker="o", edgecolors="white",
                 linewidths=0.4, zorder=5, alpha=0.9)
    ax_a.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                 color=DIGIT_COLORS[d], s=25, marker="s", edgecolors="white",
                 linewidths=0.4, zorder=5, alpha=0.9)

ve = pca_spike.explained_variance_ratio_
ax_a.set_xlabel(f"PC1 ({ve[0]:.0%})", fontsize=8, labelpad=2)
ax_a.set_ylabel(f"PC2 ({ve[1]:.0%})", fontsize=8, labelpad=2)
ax_a.set_zlabel(f"PC3 ({ve[2]:.0%})", fontsize=8, labelpad=2)
ax_a.tick_params(labelsize=7, pad=0)

# === Panel B: Adaptation PCA ===
ax_b = fig.add_subplot(gs[0, 1], projection="3d")
ax_b.set_facecolor('white')
ax_b.xaxis.pane.fill = False
ax_b.yaxis.pane.fill = False
ax_b.zaxis.pane.fill = False
ax_b.xaxis.pane.set_edgecolor('#dddddd')
ax_b.yaxis.pane.set_edgecolor('#dddddd')
ax_b.zaxis.pane.set_edgecolor('#dddddd')
ax_b.grid(True, alpha=0.15, linewidth=0.4)

for i in range(n_trials):
    d = trial_digits[i]
    traj = adapt_pca[i]
    ax_b.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              color=DIGIT_COLORS[d], alpha=0.65, linewidth=1.3)
    ax_b.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                 color=DIGIT_COLORS[d], s=25, marker="o", edgecolors="white",
                 linewidths=0.4, zorder=5, alpha=0.9)
    ax_b.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                 color=DIGIT_COLORS[d], s=25, marker="s", edgecolors="white",
                 linewidths=0.4, zorder=5, alpha=0.9)

ve = pca_adapt.explained_variance_ratio_
ax_b.set_xlabel(f"PC1 ({ve[0]:.0%})", fontsize=8, labelpad=2)
ax_b.set_ylabel(f"PC2 ({ve[1]:.0%})", fontsize=8, labelpad=2)
ax_b.set_zlabel(f"PC3 ({ve[2]:.0%})", fontsize=8, labelpad=2)
ax_b.tick_params(labelsize=7, pad=0)

# === Panel C: Separability curves ===
ax_c = fig.add_subplot(gs[1, :])

# Epoch shading
ax_c.axvspan(0, stim_end_ms, color='#e8f5e9', alpha=0.5, zorder=0, label='_nolegend_')
ax_c.axvspan(stim_end_ms, time_ms[-1] + bin_ms / 2, color='#fff3e0', alpha=0.5, zorder=0, label='_nolegend_')
ax_c.axvline(stim_end_ms, color='#999', ls='--', lw=0.8, alpha=0.7, zorder=1)

ax_c.plot(time_ms, spike_sep, color="#c62828", linewidth=2.2, label="Spike counts", zorder=3)
ax_c.plot(time_ms, adapt_sep, color="#1565c0", linewidth=2.2, label="Adaptation conductance", zorder=3)

# Fill between to highlight the gap
ax_c.fill_between(time_ms, spike_sep, adapt_sep,
                   where=adapt_sep > spike_sep,
                   color="#1565c0", alpha=0.08, zorder=2)

ax_c.set_xlabel("Time from stimulus onset (ms)", fontsize=9)
ax_c.set_ylabel("Mean pairwise Euclidean distance\nbetween digit centroids", fontsize=9)
ax_c.legend(fontsize=8, loc="upper left", frameon=False)
ax_c.grid(True, alpha=0.15, linewidth=0.5)
ax_c.set_xlim(0, time_ms[-1] + bin_ms / 2)
ax_c.set_ylim(bottom=0)

# Epoch labels
ax_c.text(stim_end_ms / 2, ax_c.get_ylim()[1] * 0.95, "stimulus",
          ha="center", va="top", fontsize=8, color="#388e3c", alpha=0.7)
ax_c.text(stim_end_ms + (time_ms[-1] - stim_end_ms) / 2, ax_c.get_ylim()[1] * 0.95,
          "post-stimulus", ha="center", va="top", fontsize=8, color="#e65100", alpha=0.7)

# Annotate ratio at end
ratio_end = adapt_sep[-1] / max(spike_sep[-1], 1e-6)
ax_c.annotate(f"{ratio_end:.1f}x",
              xy=(time_ms[-1], adapt_sep[-1]),
              xytext=(time_ms[-1] - 40, adapt_sep[-1] + adapt_sep.max() * 0.05),
              fontsize=8, color="#1565c0", fontweight="bold",
              arrowprops=dict(arrowstyle="-", color="#1565c0", lw=0.5))

# --- Panel labels ---
# 3D axes need figure-level text for labels
fig.text(0.04, 0.88, "A", fontsize=14, fontweight="bold", va="bottom")
fig.text(0.52, 0.88, "B", fontsize=14, fontweight="bold", va="bottom")
ax_c.text(-0.04, 1.06, "C", transform=ax_c.transAxes,
          fontsize=14, fontweight="bold", va="bottom")

# --- Panel subtitles ---
fig.text(0.27, 0.89, f"Spike-count state trajectories   ({pca_spike.explained_variance_ratio_[:3].sum():.0%} var. in 3 PCs)",
         fontsize=9, fontweight="bold", ha="center", color="#333")
fig.text(0.75, 0.89, f"Adaptation-conductance state trajectories   ({pca_adapt.explained_variance_ratio_[:3].sum():.0%} var. in 3 PCs)",
         fontsize=9, fontweight="bold", ha="center", color="#333")
ax_c.set_title("Inter-digit separability over time", fontsize=10,
               fontweight="bold", loc="left", pad=8)

# --- Shared legend ---
legend_elements = [Line2D([0], [0], color=DIGIT_COLORS[d], linewidth=2.5, label=f"Digit {d}")
                   for d in digits]
legend_elements.append(Line2D([0], [0], marker="o", color="gray", markersize=5,
                               linewidth=0, label="Start"))
legend_elements.append(Line2D([0], [0], marker="s", color="gray", markersize=5,
                               linewidth=0, label="End"))
fig.legend(handles=legend_elements, loc="lower center", ncol=len(digits) + 2,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

# --- Title ---
fig.suptitle("Hidden state representations at WM optimum",
             fontsize=13, fontweight="bold", y=0.94)
fig.text(0.5, 0.90,
         f"$\\Delta_a$={meta['adapt_inc']:.3f}, $\\tau_a$={meta['adapt_tau']:.0f} ms, "
         f"$g_{{tonic}}$={meta['g_tonic']:.3f} nS  |  "
         f"{meta['n_reservoir']} reservoir neurons, {meta['trials_per_digit']} trials/digit, "
         f"{len(digits)} digits",
         ha="center", va="bottom", fontsize=8, color="#666666")

out_path = RESULTS_DIR / "hidden_states_figure.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"\nSaved figure to {out_path}")
plt.close(fig)

# --- Print summary stats ---
print(f"\n--- Separability Summary ---")
print(f"Spike separability:  peak = {spike_sep.max():.2f} at t = {time_ms[np.argmax(spike_sep)]:.0f} ms")
print(f"Adapt separability:  peak = {adapt_sep.max():.2f} at t = {time_ms[np.argmax(adapt_sep)]:.0f} ms")
print(f"Peak ratio (adapt/spike): {adapt_sep.max() / spike_sep.max():.2f}x")
print(f"Late-bin ratio (last 5 bins): {adapt_sep[-5:].mean() / max(spike_sep[-5:].mean(), 1e-6):.2f}x")
print(f"\nSpike PCA variance explained: {pca_spike.explained_variance_ratio_[:3].sum():.1%}")
print(f"Adapt PCA variance explained: {pca_adapt.explained_variance_ratio_[:3].sum():.1%}")
