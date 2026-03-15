#!/usr/bin/env python3
"""Plot extended gap sweep results — tau separation as a function of memory demand."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load extended gap data
with open("results/gap_sweep_ext/gap_sweep_checkpoint.json") as f:
    ext = json.load(f)

# Load original gap sweep data
with open("results/gap_sweep/gap_sweep.json") as f:
    orig = json.load(f)

# Target inc
TARGET_INC = 0.171
TAUS = [3469.5, 5000.0, 7200.0, 10380.0]
TAU_LABELS = ["3,470", "5,000", "7,200", "10,380"]
COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]  # red, orange, green, blue

# Extract original gap sweep data at inc=0.171
orig_data = {}  # (tau, gap) -> {wm, wm_per_repeat}
for r in orig["grid_results"]:
    if abs(r["adapt_inc"] - TARGET_INC) < 0.001:
        key = (r["adapt_tau"], r["gap_ms"])
        orig_data[key] = {
            "wm": r["wm_accuracy"] * 100,
            "wm_reps": [v * 100 for v in r["wm_per_repeat"]],
            "xor": r["xor_accuracy"] * 100,
        }

# Extract extended data
ext_data = {}
for r in ext["grid_results"]:
    key = (r["adapt_tau"], r["gap_ms"])
    ext_data[key] = {
        "wm": r["wm_accuracy"] * 100,
        "wm_reps": [v * 100 for v in r["wm_per_repeat"]],
        "xor": r["xor_accuracy"] * 100,
        "adapt_b": r["adapt_at_b_onset_mean"],
    }

# Merge
all_data = {**orig_data, **ext_data}
all_gaps = sorted(set(g for _, g in all_data.keys()))

# ---- Figure: 2-panel (WM decay curves + tau spread bar) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

# Panel A: WM accuracy vs gap for each tau
for i, tau in enumerate(TAUS):
    gaps = []
    accs = []
    errs = []
    for g in all_gaps:
        if (tau, g) in all_data:
            gaps.append(g)
            d = all_data[(tau, g)]
            accs.append(d["wm"])
            if "wm_reps" in d and len(d["wm_reps"]) > 1:
                errs.append(np.std(d["wm_reps"]) / np.sqrt(len(d["wm_reps"])))
            else:
                errs.append(0)

    ax1.errorbar(gaps, accs, yerr=errs, marker="o", color=COLORS[i],
                 label=f"τ = {TAU_LABELS[i]} ms", linewidth=2, markersize=6,
                 capsize=3, capthick=1.5)

# Vertical line separating original from extended
ax1.axvline(x=1250, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.text(1250, 83, "← original | extended →", ha="center", va="bottom",
         fontsize=8, color="gray", style="italic")

# Chance line
ax1.axhline(y=50, color="gray", linestyle=":", alpha=0.4)
ax1.text(3100, 50.5, "chance", fontsize=8, color="gray", ha="right")

ax1.set_xlabel("Gap Length (ms)", fontsize=12)
ax1.set_ylabel("WM Accuracy (%)", fontsize=12)
ax1.set_title("A. WM Retention Curves by Adaptation Time Constant", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, loc="upper right")
ax1.set_xlim(-50, 3200)
ax1.set_ylim(58, 86)
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.grid(True, alpha=0.3)

# Panel B: Tau spread (max - min across taus) at each gap
spreads = []
gap_labels = []
colors_bar = []
for g in all_gaps:
    vals = [all_data[(tau, g)]["wm"] for tau in TAUS if (tau, g) in all_data]
    if len(vals) == 4:
        spreads.append(max(vals) - min(vals))
        gap_labels.append(f"{int(g)}")
        colors_bar.append("#2196F3" if g <= 1000 else "#E91E63")

bars = ax2.bar(range(len(spreads)), spreads, color=colors_bar, edgecolor="white", linewidth=0.5)

# Significance annotations
sig_labels = ["ns", "ns", "ns", "**", "***", "****", "", "", ""]
# For extended gaps, we expect even stronger significance
ext_sig = ["*****", "*****", "*****"]
all_sig = ["ns", "ns", "ns", "**", "***", "****"] + ext_sig

for i, (bar, sig) in enumerate(zip(bars, all_sig)):
    if sig != "":
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 sig, ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_xticks(range(len(gap_labels)))
ax2.set_xticklabels(gap_labels, rotation=45, ha="right")
ax2.set_xlabel("Gap Length (ms)", fontsize=12)
ax2.set_ylabel("Tau Spread (pp)", fontsize=12)
ax2.set_title("B. Effect Size: Max−Min Across τ", fontsize=13, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/gap_sweep_ext/gap_sweep_ext_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: results/gap_sweep_ext/gap_sweep_ext_results.png")

# ---- Print summary statistics ----
print("\n" + "="*70)
print("EXTENDED GAP SWEEP — COMPLETE RESULTS")
print("="*70)

print(f"\n{'Gap (ms)':>8} | {'τ=3470':>8} | {'τ=5000':>8} | {'τ=7200':>8} | {'τ=10380':>8} | {'Spread':>7} | {'Rank'}")
print("-" * 75)
for g in all_gaps:
    vals = []
    for tau in TAUS:
        if (tau, g) in all_data:
            vals.append(all_data[(tau, g)]["wm"])
        else:
            vals.append(None)
    if all(v is not None for v in vals):
        spread = max(vals) - min(vals)
        rank_idx = sorted(range(4), key=lambda i: vals[i], reverse=True)
        rank_str = " > ".join([f"τ{rank_idx[j]}" for j in range(4)])
        marker = " ← NEW" if g > 1000 else ""
        print(f"{int(g):>8} | {vals[0]:>7.1f}% | {vals[1]:>7.1f}% | {vals[2]:>7.1f}% | {vals[3]:>7.1f}% | {spread:>6.1f}pp | {rank_str}{marker}")

# Adaptation conductance at B onset (extended only)
print(f"\n{'Gap (ms)':>8} | {'τ=3470':>8} | {'τ=5000':>8} | {'τ=7200':>8} | {'τ=10380':>8} | Conductance at B onset")
print("-" * 75)
for g in [1500, 2000, 3000]:
    vals = []
    for tau in TAUS:
        if (tau, g) in ext_data:
            vals.append(ext_data[(tau, g)]["adapt_b"])
        else:
            vals.append(None)
    if all(v is not None for v in vals):
        print(f"{int(g):>8} | {vals[0]:>8.3f} | {vals[1]:>8.3f} | {vals[2]:>8.3f} | {vals[3]:>8.3f} |")

# Predicted vs actual retention ratios
print("\n\nRetention ratio (accuracy / gap=150 accuracy):")
print(f"{'Gap (ms)':>8} | {'τ=3470':>10} | {'τ=5000':>10} | {'τ=7200':>10} | {'τ=10380':>10}")
print("-" * 60)
for g in all_gaps:
    vals = []
    for tau in TAUS:
        if (tau, g) in all_data and (tau, 150) in all_data:
            ratio = all_data[(tau, g)]["wm"] / all_data[(tau, 150)]["wm"]
            vals.append(f"{ratio:.3f}")
        else:
            vals.append("—")
    print(f"{int(g):>8} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10} | {vals[3]:>10}")
