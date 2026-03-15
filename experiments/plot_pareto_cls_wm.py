#!/usr/bin/env python3
"""Pareto frontier between CLS and WM in parameter space (adapt_inc × adapt_tau heatmaps)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Load data ──
with open("results/classification_adaptation_sweep/classification_adaptation_sweep_checkpoint.json") as f:
    cls_data = json.load(f)
with open("results/wm_adaptation_sweep/wm_adaptation_sweep_checkpoint.json") as f:
    wm_data = json.load(f)

inc_vals = np.array(cls_data["grid"]["unified_inc"])
tau_vals = np.array(cls_data["grid"]["unified_tau"])
n_inc, n_tau = len(inc_vals), len(tau_vals)

# Build 2D grids: (inc_idx, tau_idx) -> accuracy for Branch B
cls_acc = np.full((n_inc, n_tau), np.nan)
wm_acc = np.full((n_inc, n_tau), np.nan)

for r in cls_data["grid_results"]:
    if r["branch"] == "B_matched":
        cls_acc[r["inc_idx"], r["tau_idx"]] = r["classification_accuracy"]

for r in wm_data["grid_results"]:
    if r["branch"] == "B_matched":
        wm_acc[r["inc_idx"], r["tau_idx"]] = r["wm_accuracy"]

# ── Compute Pareto frontier in flattened space ──
valid = ~np.isnan(cls_acc) & ~np.isnan(wm_acc)
c_flat = cls_acc[valid]
w_flat = wm_acc[valid]
ij_flat = [(i, j) for i in range(n_inc) for j in range(n_tau) if valid[i, j]]

order = np.argsort(-c_flat)
pareto_flat_idx = []
best_wm = -np.inf
for idx in order:
    if w_flat[idx] > best_wm:
        pareto_flat_idx.append(idx)
        best_wm = w_flat[idx]

pareto_ij = [ij_flat[k] for k in pareto_flat_idx]

# Build Pareto mask for the 2D grid
pareto_mask = np.zeros((n_inc, n_tau), dtype=bool)
for i, j in pareto_ij:
    pareto_mask[i, j] = True

# Task optima
cls_opt = np.unravel_index(np.nanargmax(cls_acc), cls_acc.shape)
wm_opt = np.unravel_index(np.nanargmax(wm_acc), wm_acc.shape)

# ── Axis labels ──
# For display: replace inc=0 with "0" and show others as floats
inc_labels = [f"{v:.4f}" if v > 0 else "0" for v in inc_vals]
tau_labels = [f"{v:.0f}" for v in tau_vals]

# ── Plot: 3-panel (CLS heatmap, WM heatmap, difference) ──
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

# Transpose so x=tau, y=inc (tau on x-axis since it's the "less interesting" axis)
# Actually: inc on y-axis (rows), tau on x-axis (cols) — natural grid layout
# imshow: rows=y (top to bottom), so flip inc to have 0 at bottom

cls_plot = cls_acc * 100  # shape (n_inc, n_tau)
wm_plot = wm_acc * 100

# Panel 1: CLS heatmap
im1 = axes[0].imshow(cls_plot, aspect="auto", origin="lower",
                      cmap="Blues", interpolation="nearest")
axes[0].set_title("Classification Accuracy (%)", fontsize=13, fontweight="bold")
div1 = make_axes_locatable(axes[0])
cax1 = div1.append_axes("right", size="5%", pad=0.08)
plt.colorbar(im1, cax=cax1)

# Panel 2: WM heatmap
im2 = axes[1].imshow(wm_plot, aspect="auto", origin="lower",
                      cmap="Oranges", interpolation="nearest")
axes[1].set_title("Working Memory Accuracy (%)", fontsize=13, fontweight="bold")
div2 = make_axes_locatable(axes[1])
cax2 = div2.append_axes("right", size="5%", pad=0.08)
plt.colorbar(im2, cax=cax2)

# Panel 3: WM - CLS delta (normalized to each task's range for visual comparison)
# Show raw WM acc since CLS is fairly flat and WM has the big gradient
# Actually: show both contours on a single panel
# Better: show CLS contours + WM contours overlaid, with Pareto points
ax3 = axes[2]
# Use CLS as blue contour fill, WM as orange contour lines
# Or: show the "trade-off" = where you'd pick CLS vs WM
# Simplest informative thing: CLS contours + WM contours + Pareto path
x_grid, y_grid = np.meshgrid(np.arange(n_tau), np.arange(n_inc))

# Fill with CLS
cf = ax3.contourf(x_grid, y_grid, cls_plot, levels=12, cmap="Blues", alpha=0.5)
# WM contour lines
cw = ax3.contour(x_grid, y_grid, wm_plot, levels=10, cmap="Oranges", linewidths=1.5)
ax3.clabel(cw, inline=True, fontsize=7, fmt="%.0f%%")
ax3.set_title("CLS (fill) + WM (contours) + Pareto", fontsize=13, fontweight="bold")

# ── Overlay Pareto points and task optima on all panels ──
for ax_idx, ax in enumerate(axes):
    # Pareto points
    for i, j in pareto_ij:
        ax.plot(j, i, marker="*", markersize=14, color="crimson",
                markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    # Connect Pareto path (sort by inc)
    pareto_sorted = sorted(pareto_ij, key=lambda ij: ij[0])
    pj = [j for _, j in pareto_sorted]
    pi = [i for i, _ in pareto_sorted]
    ax.plot(pj, pi, color="crimson", linewidth=2, linestyle="--", alpha=0.7, zorder=9)

    # Task optima
    ax.plot(cls_opt[1], cls_opt[0], marker="^", markersize=13, color="dodgerblue",
            markeredgecolor="black", markeredgewidth=1.2, zorder=11)
    ax.plot(wm_opt[1], wm_opt[0], marker="s", markersize=11, color="darkorange",
            markeredgecolor="black", markeredgewidth=1.2, zorder=11)

    # Axis ticks
    ax.set_xticks(np.arange(n_tau))
    ax.set_xticklabels(tau_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("adapt_tau (ms)", fontsize=11)
    if ax_idx == 0:
        ax.set_yticks(np.arange(n_inc))
        ax.set_yticklabels(inc_labels, fontsize=7.5)
        ax.set_ylabel("adapt_inc", fontsize=11)

# Legend (manual)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="crimson",
           markersize=14, markeredgecolor="white", label="Pareto frontier"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="dodgerblue",
           markersize=12, markeredgecolor="black", label="CLS optimum"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="darkorange",
           markersize=10, markeredgecolor="black", label="WM optimum"),
]
axes[2].legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Pareto Frontier in Parameter Space (Branch B, rate-matched to 20 Hz)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("results/pareto_cls_wm_paramspace.png", dpi=200, bbox_inches="tight")
print("Saved to results/pareto_cls_wm_paramspace.png")

# Print frontier
print(f"\nPareto frontier ({len(pareto_ij)} points):")
print(f"{'Rank':>4}  {'inc':>8}  {'tau':>8}  {'CLS%':>7}  {'WM%':>7}")
for k, (i, j) in enumerate(sorted(pareto_ij, key=lambda ij: -cls_acc[ij[0], ij[1]])):
    print(f"{k+1:4d}  {inc_vals[i]:8.4f}  {tau_vals[j]:8.1f}  {cls_acc[i,j]*100:7.2f}  {wm_acc[i,j]*100:7.2f}")

plt.show()
