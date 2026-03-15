#!/usr/bin/env python3
"""Plot adaptation conductance at B onset vs WM accuracy across all gap lengths.

Directly visualizes the exponential decay model: longer gaps → lower residual
adaptation → lower WM accuracy, with tau controlling the decay rate.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

data_path = "results/gap_sweep/gap_sweep.json"
out_dir = "results/gap_sweep"

with open(data_path) as f:
    data = json.load(f)

inc_vals = np.array(data["grid"]["unified_inc"])
tau_vals = np.array(data["grid"]["unified_tau"])
gap_vals = np.array(data["grid"]["gap_lengths_ms"])

# Extract all points with inc > 0 (inc=0 has adapt=0 always)
adapt_b = []
wm = []
gaps = []
taus = []
incs = []

for r in data["grid_results"]:
    if r["adapt_inc"] > 0.005:  # skip inc=0 (no adaptation)
        adapt_b.append(r["adapt_at_b_onset_mean"])
        wm.append(r["wm_accuracy"] * 100)
        gaps.append(r["gap_ms"])
        taus.append(r["adapt_tau"])
        incs.append(r["adapt_inc"])

adapt_b = np.array(adapt_b)
wm = np.array(wm)
gaps = np.array(gaps)
taus = np.array(taus)
incs = np.array(incs)

# ── Figure: 2-panel ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Adaptation conductance at B onset predicts WM accuracy\n"
             "across all gap lengths, tau values, and adaptation strengths",
             fontsize=13, fontweight="bold")

# Panel A: adapt_at_b vs WM, colored by gap
cmap_gap = plt.cm.plasma
norm_gap = Normalize(vmin=np.log10(50), vmax=np.log10(1000))

for gi, gap_ms in enumerate(gap_vals):
    mask = gaps == gap_ms
    c = cmap_gap(norm_gap(np.log10(gap_ms)))
    ax1.scatter(adapt_b[mask], wm[mask], c=[c], s=20, alpha=0.5,
                edgecolor="none", label=f"{gap_ms:.0f} ms")

ax1.set_xlabel("Adaptation conductance at B onset (nS)", fontsize=11)
ax1.set_ylabel("WM accuracy (%)", fontsize=11)
ax1.set_title("A. Conductance residual → WM readout", fontsize=11, fontweight="bold")
ax1.axhline(y=20, color="gray", linestyle=":", alpha=0.4)
ax1.legend(title="Gap length", fontsize=8, title_fontsize=9,
           loc="lower right", framealpha=0.9, markerscale=2)
ax1.grid(True, alpha=0.2)

# Fit line through all points
from numpy.polynomial import polynomial as P
valid = adapt_b > 0.01
coeffs = np.polyfit(adapt_b[valid], wm[valid], 1)
x_fit = np.linspace(0, adapt_b.max() * 1.05, 100)
ax1.plot(x_fit, np.polyval(coeffs, x_fit), "k--", linewidth=1.5, alpha=0.6,
         label=f"r = {np.corrcoef(adapt_b[valid], wm[valid])[0,1]:.3f}")
ax1.legend(title="Gap length", fontsize=8, title_fontsize=9,
           loc="lower right", framealpha=0.9, markerscale=2)

# Panel B: At inc=0.171, show adapt_at_b vs WM for each (tau, gap) combination
# This is the clean view — 4 tau × 6 gap = 24 points
ax2.set_title("B. At inc=0.171: exponential decay model", fontsize=11, fontweight="bold")

tau_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
tau_markers = ["o", "s", "D", "^"]

inc_target = 0.171
inc_idx = int(np.argmin(np.abs(inc_vals - inc_target)))
actual_inc = inc_vals[inc_idx]

for ti, (tau_ms, color, marker) in enumerate(zip(tau_vals, tau_colors, tau_markers)):
    mask = (np.abs(incs - actual_inc) < 0.001) & (np.abs(taus - tau_ms) < 1)
    ab = adapt_b[mask]
    wm_vals = wm[mask]
    gap_subset = gaps[mask]

    # Sort by gap for line
    order = np.argsort(gap_subset)
    ax2.plot(ab[order], wm_vals[order], marker=marker, color=color,
             linewidth=2, markersize=9, markeredgecolor="black",
             markeredgewidth=0.5, label=f"τ = {tau_ms/1000:.1f}k ms",
             alpha=0.85)

    # Annotate gap lengths on tau=10380 line
    if tau_ms == tau_vals[-1]:
        for i in order:
            if gap_subset[i] in [50, 150, 1000]:
                ax2.annotate(f"  {gap_subset[i]:.0f}ms",
                             xy=(ab[i], wm_vals[i]),
                             fontsize=7, color=color, fontweight="bold")

# Add theoretical decay curves
ax2.set_xlabel("Adaptation conductance at B onset (nS)", fontsize=11)
ax2.set_ylabel("WM accuracy (%)", fontsize=11)
ax2.set_ylim(74, 84)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax2.grid(True, alpha=0.2)

# Add annotation about the mechanism
ax2.annotate("Longer gaps → lower\nresidual conductance\n→ lower WM accuracy",
             xy=(1.5, 77), fontsize=9, fontstyle="italic", color="#555",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

ax2.annotate("Higher τ → slower\ndecay → higher residual\n→ better WM at long gaps",
             xy=(2.0, 68), fontsize=9, fontstyle="italic", color="#555",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.92])
out_path = os.path.join(out_dir, "adapt_at_b_onset_vs_wm.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()
