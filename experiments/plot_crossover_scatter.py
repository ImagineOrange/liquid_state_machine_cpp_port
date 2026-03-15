#!/usr/bin/env python3
"""3-panel scatter: crossover time vs ratio, crossover time vs WM accuracy, peak spike vs peak adapt sep."""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Load crossover sweep data ──
xo_dir = sys.argv[1] if len(sys.argv) > 1 else "results/crossover_sweep"
df = pd.read_csv(os.path.join(xo_dir, "crossover_sweep.csv"))

# ── Load WM accuracy from both sweeps (Branch B = rate-matched) ──
wm_lookup = {}

# Source 1: WM adaptation sweep (tau=5000 has all 20 inc values)
wm_path = "results/wm_adaptation_sweep/wm_adaptation_sweep_checkpoint.json"
if os.path.exists(wm_path):
    with open(wm_path) as f:
        wm_data = json.load(f)
    for r in wm_data["grid_results"]:
        if r["branch"] == "B_matched":
            key = (round(r["adapt_inc"], 4), round(r["adapt_tau"], 1))
            wm_lookup[key] = r["wm_accuracy"]

# Source 2: Extended tau sweep (higher tau values)
ext_path = "results/extended_tau_sweep/extended_tau_sweep_checkpoint.json"
if os.path.exists(ext_path):
    with open(ext_path) as f:
        ext = json.load(f)
    for r in ext["grid_results"]:
        if r["branch"] == "B_matched":
            key = (round(r["adapt_inc"], 4), round(r["adapt_tau"], 1))
            if key not in wm_lookup:
                wm_lookup[key] = r["wm_accuracy"]

print(f"WM accuracy available for {len(wm_lookup)} parameter points")

# Merge WM accuracy into crossover df
df["wm_accuracy"] = df.apply(
    lambda row: wm_lookup.get((round(row["inc"], 4), round(row["tau"], 1)), np.nan),
    axis=1
)

# Compute ratio
df["ratio"] = df["final_adapt_sep"] / df["final_spike_sep"].replace(0, np.nan)

# Split into crossover vs no-crossover
has_xo = df[df["crossover_ms"] > 0].copy()
no_xo = df[df["crossover_ms"] <= 0].copy()

# Color by inc (log scale for visual spread)
inc_vals = sorted(df["inc"].unique())
inc_nonzero = [v for v in inc_vals if v > 0]
inc_min, inc_max = np.log10(min(inc_nonzero)), np.log10(max(inc_nonzero))

def inc_color(inc):
    if inc == 0:
        return "#cccccc"
    normed = (np.log10(inc) - inc_min) / (inc_max - inc_min)
    return plt.cm.viridis(normed)

# ── Figure ──
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.82, wspace=0.30)

fig.suptitle("Crossover dynamics across parameter space",
             fontsize=12, fontweight="bold", y=0.96)
fig.text(0.5, 0.90,
         f"140 parameter combinations  |  rate-matched (Branch B)",
         ha="center", fontsize=8, color="#666")

# ── Panel A: Crossover time vs adapt/spike ratio ──
ax1.text(-0.12, 1.06, "A", transform=ax1.transAxes,
         fontsize=14, fontweight="bold", va="bottom")

for _, row in has_xo.iterrows():
    ax1.scatter(row["crossover_ms"], row["ratio"],
                c=[inc_color(row["inc"])], s=30, alpha=0.8,
                edgecolors="white", linewidths=0.3, zorder=3)

# Regression line
if len(has_xo) > 5:
    mask = np.isfinite(has_xo["ratio"])
    x, y = has_xo.loc[mask, "crossover_ms"].values, has_xo.loc[mask, "ratio"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_fit, slope * x_fit + intercept, color="#333", ls="--", lw=1, alpha=0.6)
    ax1.text(0.95, 0.95, f"r = {r:.2f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=8, color="#333")

ax1.set_xlabel("Crossover time (ms)", fontsize=9)
ax1.set_ylabel("Final adapt / spike\nseparability ratio", fontsize=9)
ax1.set_title("Earlier crossover predicts\nlarger post-stimulus advantage",
              fontsize=9, fontweight="bold", loc="left", pad=8)
ax1.grid(True, alpha=0.15, linewidth=0.5)
ax1.tick_params(labelsize=8)

# ── Panel B: Crossover time vs WM accuracy ──
ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
         fontsize=14, fontweight="bold", va="bottom")

has_xo_wm = has_xo.dropna(subset=["wm_accuracy"])
no_xo_wm = no_xo.dropna(subset=["wm_accuracy"])

for _, row in has_xo_wm.iterrows():
    ax2.scatter(row["crossover_ms"], row["wm_accuracy"],
                c=[inc_color(row["inc"])], s=30, alpha=0.8,
                edgecolors="white", linewidths=0.3, zorder=3)

# Regression
if len(has_xo_wm) > 5:
    x, y = has_xo_wm["crossover_ms"].values, has_xo_wm["wm_accuracy"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_fit, slope * x_fit + intercept, color="#333", ls="--", lw=1, alpha=0.6)
    ax2.text(0.95, 0.95, f"r = {r:.2f}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8, color="#333")

# Show no-crossover points as grey strip on the right
if len(no_xo_wm) > 0:
    xmax = has_xo_wm["crossover_ms"].max() + 60 if len(has_xo_wm) > 0 else 500
    for _, row in no_xo_wm.iterrows():
        ax2.scatter(xmax, row["wm_accuracy"],
                    c=["#cccccc"], s=20, alpha=0.5, marker="x",
                    linewidths=0.8, zorder=2)
    ax2.axvline(xmax - 25, color="#ddd", ls=":", lw=0.8, alpha=0.5)
    ylims = ax2.get_ylim()
    ax2.text(xmax, ylims[0] + (ylims[1]-ylims[0])*0.02,
             "no\ncrossover", ha="center", va="bottom", fontsize=6, color="#999")

ax2.set_xlabel("Crossover time (ms)", fontsize=9)
ax2.set_ylabel("WM accuracy\n(rate-matched)", fontsize=9)
ax2.set_title("Crossover time vs\nworking memory performance",
              fontsize=9, fontweight="bold", loc="left", pad=8)
ax2.grid(True, alpha=0.15, linewidth=0.5)
ax2.tick_params(labelsize=8)

# ── Panel C: Peak spike sep vs peak adapt sep ──
ax3.text(-0.12, 1.06, "C", transform=ax3.transAxes,
         fontsize=14, fontweight="bold", va="bottom")

for _, row in df.iterrows():
    ax3.scatter(row["peak_spike_sep"], row["peak_adapt_sep"],
                c=[inc_color(row["inc"])], s=30, alpha=0.8,
                edgecolors="white", linewidths=0.3, zorder=3)

# Diagonal line (equal separability)
lims = [0, max(df["peak_spike_sep"].max(), df["peak_adapt_sep"].max()) * 1.05]
ax3.plot(lims, lims, color="#999", ls=":", lw=0.8, alpha=0.5, label="equal")

ax3.set_xlabel("Peak spike-count separability", fontsize=9)
ax3.set_ylabel("Peak adaptation\nseparability", fontsize=9)
ax3.set_title("Peak discriminability:\nspikes vs adaptation conductance",
              fontsize=9, fontweight="bold", loc="left", pad=8)
ax3.grid(True, alpha=0.15, linewidth=0.5)
ax3.tick_params(labelsize=8)
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)

# ── Shared colorbar ──
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(vmin=inc_min, vmax=inc_max))
sm.set_array([])
cbar_ax = fig.add_axes([0.95, 0.14, 0.012, 0.66])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Adaptation increment, $I_{adapt}$ (nS)", fontsize=8, labelpad=4)
# Log-scale tick labels
tick_vals = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
cb.set_ticks([np.log10(v) for v in tick_vals])
cb.set_ticklabels([str(v) for v in tick_vals])
cb.ax.tick_params(labelsize=7)

# ── Mark WM optimum ──
wm_row = df[(df["inc"].between(0.170, 0.172)) & (df["tau"].between(4999, 5001))]
if len(wm_row) > 0:
    r = wm_row.iloc[0]
    if r["crossover_ms"] > 0:
        ax1.scatter(r["crossover_ms"], r["ratio"], c="none", s=120,
                    edgecolors="#ff6600", linewidths=2, zorder=10)
        ax1.annotate("WM opt.", xy=(r["crossover_ms"], r["ratio"]),
                     xytext=(r["crossover_ms"] + 30, r["ratio"] + 0.3),
                     fontsize=7, color="#ff6600", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color="#ff6600", lw=0.8))
    if not np.isnan(r.get("wm_accuracy", np.nan)):
        ax2.scatter(r["crossover_ms"], r["wm_accuracy"], c="none", s=120,
                    edgecolors="#ff6600", linewidths=2, zorder=10)
    ax3.scatter(r["peak_spike_sep"], r["peak_adapt_sep"], c="none", s=120,
                edgecolors="#ff6600", linewidths=2, zorder=10)
    ax3.annotate("WM opt.", xy=(r["peak_spike_sep"], r["peak_adapt_sep"]),
                 xytext=(r["peak_spike_sep"] + 2, r["peak_adapt_sep"] + 20),
                 fontsize=7, color="#ff6600", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#ff6600", lw=0.8))

out_path = os.path.join(xo_dir, "crossover_scatter_figure.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()
