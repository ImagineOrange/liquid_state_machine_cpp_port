#!/usr/bin/env python3
"""Plot ablation study results — comprehensive 4-panel figure."""

import json
import numpy as np
import matplotlib.pyplot as plt
with open("results/ablation/ablation_results.json") as f:
    data = json.load(f)

# Parse
conditions = ["baseline", "no_std", "no_nmda", "sfa_only", "bare"]
cond_labels = ["Baseline", "-STD", "-NMDA", "SFA-only", "Bare"]
adapt_levels = ["no_adapt", "cls_opt", "wm_opt"]
adapt_inc = {"no_adapt": 0.0, "cls_opt": 0.071, "wm_opt": 0.171}
adapt_labels_short = ["inc=0", "inc=0.071\n(CLS opt)", "inc=0.171\n(WM opt)"]
adapt_colors = ["#9E9E9E", "#2196F3", "#E91E63"]

results = {}
for r in data["results"]:
    results[(r["condition"], r["adapt_label"])] = r

# ---- Figure: 4-panel ----
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ========================================
# Panel A: WM accuracy by condition (main result)
# ========================================
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(conditions))
width = 0.25

for ai, (alevel, acolor) in enumerate(zip(adapt_levels, adapt_colors)):
    vals, errs = [], []
    for cond in conditions:
        key = (cond, alevel)
        if key in results:
            vals.append(results[key]["wm_accuracy"] * 100)
            errs.append(results[key]["wm_std"] * 100)
        else:
            vals.append(np.nan)
            errs.append(0)
    offset = (ai - 1) * width
    ax_a.bar(x + offset, vals, width, yerr=errs, color=acolor, alpha=0.85,
            capsize=3, edgecolor="white", linewidth=0.5, label=adapt_labels_short[ai])

ax_a.axhline(y=20, color="gray", linestyle=":", alpha=0.4)
ax_a.text(4.4, 21, "chance", fontsize=8, color="gray")
ax_a.set_xticks(x)
ax_a.set_xticklabels(cond_labels, fontsize=10)
ax_a.set_ylabel("WM Accuracy (%)", fontsize=11)
ax_a.set_title("A. Working Memory Across Ablations", fontsize=12, fontweight="bold")
ax_a.set_ylim(15, 90)
ax_a.legend(fontsize=9, loc="upper left")
ax_a.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel B: CLS accuracy by condition
# ========================================
ax_b = fig.add_subplot(gs[0, 1])

for ai, (alevel, acolor) in enumerate(zip(adapt_levels, adapt_colors)):
    vals, errs = [], []
    for cond in conditions:
        key = (cond, alevel)
        if key in results:
            vals.append(results[key]["cls_accuracy"] * 100)
            errs.append(results[key]["cls_std"] * 100)
        else:
            vals.append(np.nan)
            errs.append(0)
    offset = (ai - 1) * width
    ax_b.bar(x + offset, vals, width, yerr=errs, color=acolor, alpha=0.85,
            capsize=3, edgecolor="white", linewidth=0.5, label=adapt_labels_short[ai])

ax_b.set_xticks(x)
ax_b.set_xticklabels(cond_labels, fontsize=10)
ax_b.set_ylabel("CLS Accuracy (%)", fontsize=11)
ax_b.set_title("B. Classification Across Ablations", fontsize=12, fontweight="bold")
ax_b.set_ylim(90, 100)
ax_b.legend(fontsize=9, loc="lower left")
ax_b.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel C: WM delta from baseline (waterfall)
# ========================================
ax_c = fig.add_subplot(gs[1, 0])

# For each ablation condition, show WM change vs baseline at each inc level
abl_conds = ["no_std", "no_nmda", "sfa_only"]
abl_labels = ["-STD", "-NMDA", "SFA-only"]
x2 = np.arange(len(abl_conds))

for ai, (alevel, acolor) in enumerate(zip(adapt_levels, adapt_colors)):
    deltas = []
    for cond in abl_conds:
        bl_key = ("baseline", alevel)
        ab_key = (cond, alevel)
        if bl_key in results and ab_key in results:
            delta = (results[ab_key]["wm_accuracy"] - results[bl_key]["wm_accuracy"]) * 100
            deltas.append(delta)
        else:
            deltas.append(np.nan)
    offset = (ai - 1) * width
    bars = ax_c.bar(x2 + offset, deltas, width, color=acolor, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=adapt_labels_short[ai])
    # Add value labels
    for i, d in enumerate(deltas):
        if not np.isnan(d):
            ax_c.text(x2[i] + offset, d - 0.3 if d < 0 else d + 0.2,
                     f"{d:.1f}", ha="center", va="top" if d < 0 else "bottom",
                     fontsize=7, fontweight="bold")

ax_c.axhline(y=0, color="black", linewidth=0.8)
ax_c.set_xticks(x2)
ax_c.set_xticklabels(abl_labels, fontsize=10)
ax_c.set_ylabel("WM Change vs Baseline (pp)", fontsize=11)
ax_c.set_title("C. WM Impact of Each Ablation", fontsize=12, fontweight="bold")
ax_c.set_ylim(-9, 1)
ax_c.legend(fontsize=9, loc="lower left")
ax_c.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel D: Dynamic range preservation
# ========================================
ax_d = fig.add_subplot(gs[1, 1])

# WM dynamic range (inc=0.171 - inc=0) for each condition
dr_conds = ["baseline", "no_std", "no_nmda", "sfa_only"]
dr_labels = ["Baseline", "-STD", "-NMDA", "SFA-only"]
dr_colors = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

dr_vals = []
for cond in dr_conds:
    wm_high = results[(cond, "wm_opt")]["wm_accuracy"] * 100
    wm_low = results[(cond, "no_adapt")]["wm_accuracy"] * 100
    dr_vals.append(wm_high - wm_low)

bars = ax_d.bar(range(len(dr_conds)), dr_vals, color=dr_colors, alpha=0.85,
               edgecolor="white", linewidth=0.5)

# Add percentage labels
bl_dr = dr_vals[0]
for i, (val, bar) in enumerate(zip(dr_vals, bars)):
    pct = val / bl_dr * 100
    ax_d.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f"{val:.1f} pp\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9,
             fontweight="bold")

# Add bare network floor
bare_wm = results[("bare", "no_adapt")]["wm_accuracy"] * 100
ax_d.axhline(y=bare_wm - 20, color="gray", linestyle="--", alpha=0.5)
ax_d.text(3.5, bare_wm - 20 + 0.5, f"bare floor: {bare_wm:.1f}% WM\n({bare_wm-20:.1f} pp above chance)",
         fontsize=8, color="gray", ha="right")

ax_d.set_xticks(range(len(dr_conds)))
ax_d.set_xticklabels(dr_labels, fontsize=10)
ax_d.set_ylabel("WM Dynamic Range (pp)\n(inc=0.171 minus inc=0)", fontsize=11)
ax_d.set_title("D. SFA Accounts for 93% of WM Dynamic Range", fontsize=12, fontweight="bold")
ax_d.set_ylim(0, 62)
ax_d.grid(True, axis="y", alpha=0.3)

plt.savefig("results/ablation/ablation_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: results/ablation/ablation_results.png")
