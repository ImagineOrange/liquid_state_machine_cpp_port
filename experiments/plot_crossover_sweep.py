#!/usr/bin/env python3
"""Plot crossover sweep: heatmap of crossover time across (inc, tau) parameter space."""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load data ──
data_dir = sys.argv[1] if len(sys.argv) > 1 else "results/crossover_sweep"
df = pd.read_csv(os.path.join(data_dir, "crossover_sweep.csv"))

inc_vals = sorted(df["inc"].unique())
tau_vals = sorted(df["tau"].unique())

# Build 2D grids
crossover_grid = np.full((len(tau_vals), len(inc_vals)), np.nan)
ratio_grid = np.full((len(tau_vals), len(inc_vals)), np.nan)

for _, row in df.iterrows():
    ti = tau_vals.index(row["tau"])
    ii = inc_vals.index(row["inc"])
    xo = row["crossover_ms"]
    crossover_grid[ti, ii] = xo if xo > 0 else np.nan
    if row["final_spike_sep"] > 0:
        ratio_grid[ti, ii] = row["final_adapt_sep"] / row["final_spike_sep"]

# ── Tick labels — show every other inc to avoid crowding ──
inc_labels = []
for i, v in enumerate(inc_vals):
    if i % 2 == 0:  # every other
        if v == 0:
            inc_labels.append("0")
        elif v >= 0.1:
            inc_labels.append(f"{v:.2f}")
        else:
            inc_labels.append(f"{v:.3f}")
    else:
        inc_labels.append("")

tau_labels = [f"{v/1000:.1f}" for v in tau_vals]

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
fig.subplots_adjust(left=0.07, right=0.96, bottom=0.13, top=0.78, wspace=0.28)

fig.suptitle("Adaptation conductance vs spike-count discriminability across parameter space",
             fontsize=12, fontweight="bold", y=0.96)
fig.text(0.5, 0.92,
         f"6 digits (0-5), 3 trials each  |  {len(inc_vals)} inc x {len(tau_vals)} tau = {len(inc_vals)*len(tau_vals)} points",
         ha="center", fontsize=8, color="#666")

# ── Helper: shared axis formatting ──
def fmt_axes(ax, labels_x, labels_y):
    ax.set_xticks(range(len(inc_vals)))
    ax.set_xticklabels(labels_x, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(tau_vals)))
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_xlabel("Adaptation increment, $I_{adapt}$ (nS)", fontsize=9, labelpad=6)
    ax.set_ylabel("Adaptation time constant, $\\tau_{adapt}$ (s)", fontsize=9, labelpad=4)

# ── Panel A: Crossover time ──
ax1.text(-0.10, 1.04, "A", transform=ax1.transAxes,
         fontsize=14, fontweight="bold", va="bottom")

cmap_xo = plt.cm.RdYlBu_r.copy()
cmap_xo.set_bad(color="#e0e0e0")

im1 = ax1.imshow(crossover_grid, aspect="auto", origin="lower",
                  cmap=cmap_xo, interpolation="nearest")
cb1 = fig.colorbar(im1, ax=ax1, shrink=0.9, pad=0.03)
cb1.set_label("Crossover time (ms)", fontsize=8, labelpad=4)
cb1.ax.tick_params(labelsize=7)

fmt_axes(ax1, inc_labels, tau_labels)
ax1.set_title("Crossover time: when adapt. separability\nexceeds spike separability",
              fontsize=9, fontweight="bold", loc="left", pad=10)

# Annotate "no crossover" cells with a small dash — skip text in cells with values
for ti in range(len(tau_vals)):
    for ii in range(len(inc_vals)):
        if np.isnan(crossover_grid[ti, ii]):
            ax1.text(ii, ti, "-", ha="center", va="center",
                     fontsize=7, color="#999", fontweight="bold")

# ── Panel B: Ratio ──
ax2.text(-0.10, 1.04, "B", transform=ax2.transAxes,
         fontsize=14, fontweight="bold", va="bottom")

cmap_ratio = plt.cm.Blues.copy()
cmap_ratio.set_bad(color="#e0e0e0")

ratio_display = ratio_grid.copy()
vmax = np.nanpercentile(ratio_display, 95) if np.any(~np.isnan(ratio_display)) else 10

im2 = ax2.imshow(ratio_display, aspect="auto", origin="lower",
                  cmap=cmap_ratio, interpolation="nearest",
                  vmin=0, vmax=vmax)
cb2 = fig.colorbar(im2, ax=ax2, shrink=0.9, pad=0.03)
cb2.set_label("Adapt / spike separability ratio", fontsize=8, labelpad=4)
cb2.ax.tick_params(labelsize=7)

fmt_axes(ax2, inc_labels, tau_labels)
ax2.set_title("Post-stimulus advantage:\nadapt. / spike separability ratio",
              fontsize=9, fontweight="bold", loc="left", pad=10)

# Mark WM optimum (inc=0.171, tau=5000) with a star
wm_ii = inc_vals.index(0.171) if 0.171 in inc_vals else None
wm_ti = tau_vals.index(5000.0) if 5000.0 in tau_vals else None
if wm_ii is not None and wm_ti is not None:
    for ax in (ax1, ax2):
        ax.plot(wm_ii, wm_ti, marker="*", markersize=12, color="#ff6600",
                markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax1.annotate("WM opt.", xy=(wm_ii, wm_ti), xytext=(wm_ii + 2, wm_ti + 1.2),
                 fontsize=7, color="#ff6600", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#ff6600", lw=0.8))

out_path = os.path.join(data_dir, "crossover_sweep_figure.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()
