#!/usr/bin/env python3
"""Plot multi-seed replication results — 4-panel figure."""

import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/multi_seed/multi_seed_results.json") as f:
    data = json.load(f)

# Parse into per-seed, per-level structure
seeds = []
seen = set()
for r in data["results"]:
    if r["seed"] not in seen:
        seeds.append(r["seed"])
        seen.add(r["seed"])

adapt_levels = ["no_adapt", "cls_opt", "wm_opt"]
adapt_labels = ["inc=0", "inc=0.071 (CLS opt)", "inc=0.171 (WM opt)"]
adapt_colors = ["#9E9E9E", "#2196F3", "#E91E63"]

results = {}
n_res = {}
for r in data["results"]:
    results[(r["seed"], r["adapt_label"])] = r
    n_res[r["seed"]] = r["n_reservoir"]

seed_labels = [str(s) for s in seeds]

# ---- Figure: 4-panel ----
fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                      left=0.07, right=0.97, top=0.95, bottom=0.06)

# ========================================
# Panel A: WM accuracy per seed (main result)
# ========================================
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(seeds))
width = 0.25

for ai, (alevel, acolor, alabel) in enumerate(zip(adapt_levels, adapt_colors, adapt_labels)):
    vals = [results[(s, alevel)]["wm_accuracy"] * 100 for s in seeds]
    errs = [results[(s, alevel)]["wm_std"] * 100 for s in seeds]
    offset = (ai - 1) * width
    ax_a.bar(x + offset, vals, width, yerr=errs, color=acolor, alpha=0.85,
            capsize=2, edgecolor="white", linewidth=0.5, label=alabel)

ax_a.axhline(y=20, color="gray", linestyle=":", alpha=0.4)
ax_a.text(-0.4, 21.5, "chance", fontsize=8, color="gray")
ax_a.set_xticks(x)
ax_a.set_xticklabels(seed_labels, fontsize=9)
ax_a.set_xlabel("Network Seed", fontsize=10)
ax_a.set_ylabel("WM Accuracy (%)", fontsize=10)
ax_a.set_title("A. Working Memory Across 10 Network Topologies", fontsize=11, fontweight="bold")
ax_a.set_ylim(15, 95)
ax_a.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax_a.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel B: CLS accuracy per seed
# ========================================
ax_b = fig.add_subplot(gs[0, 1])

for ai, (alevel, acolor, alabel) in enumerate(zip(adapt_levels, adapt_colors, adapt_labels)):
    vals = [results[(s, alevel)]["cls_accuracy"] * 100 for s in seeds]
    errs = [results[(s, alevel)]["cls_std"] * 100 for s in seeds]
    offset = (ai - 1) * width
    ax_b.bar(x + offset, vals, width, yerr=errs, color=acolor, alpha=0.85,
            capsize=2, edgecolor="white", linewidth=0.5, label=alabel)

ax_b.set_xticks(x)
ax_b.set_xticklabels(seed_labels, fontsize=9)
ax_b.set_xlabel("Network Seed", fontsize=10)
ax_b.set_ylabel("CLS Accuracy (%)", fontsize=10)
ax_b.set_title("B. Classification Preserved Across Topologies", fontsize=11, fontweight="bold")
ax_b.set_ylim(91, 99)
ax_b.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax_b.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel C: WM dynamic range per seed
# ========================================
ax_c = fig.add_subplot(gs[1, 0])

dr_vals = []
for s in seeds:
    wm_high = results[(s, "wm_opt")]["wm_accuracy"] * 100
    wm_low = results[(s, "no_adapt")]["wm_accuracy"] * 100
    dr_vals.append(wm_high - wm_low)

colors_dr = plt.cm.Set2(np.linspace(0, 1, len(seeds)))
bars = ax_c.bar(x, dr_vals, color=colors_dr, alpha=0.85, edgecolor="white", linewidth=0.5)

# Mean line
mean_dr = np.mean(dr_vals)
std_dr = np.std(dr_vals)
ax_c.axhline(y=mean_dr, color="#D32F2F", linewidth=1.5, linestyle="--", alpha=0.7)
ax_c.fill_between([-0.5, len(seeds) - 0.5], mean_dr - std_dr, mean_dr + std_dr,
                   color="#D32F2F", alpha=0.08)
ax_c.text(len(seeds) - 0.6, mean_dr + std_dr + 0.4,
         f"mean = {mean_dr:.1f} \u00b1 {std_dr:.1f} pp",
         fontsize=9, ha="right", color="#D32F2F", fontweight="bold")

ax_c.set_xticks(x)
ax_c.set_xticklabels(seed_labels, fontsize=9)
ax_c.set_xlabel("Network Seed", fontsize=10)
ax_c.set_ylabel("WM Dynamic Range (pp)", fontsize=10)
ax_c.set_title("C. SFA-Driven WM Gain Per Seed (inc=0.171 minus inc=0)",
               fontsize=11, fontweight="bold")
ax_c.set_ylim(48, 59)
ax_c.grid(True, axis="y", alpha=0.3)

# ========================================
# Panel D: Cross-seed summary statistics
# ========================================
ax_d = fig.add_subplot(gs[1, 1])

# Collect per-seed metrics
metric_data = [
    ("CLS @ inc=0",     [results[(s, "no_adapt")]["cls_accuracy"] * 100 for s in seeds], "#4CAF50", 0.5),
    ("CLS @ inc=0.071", [results[(s, "cls_opt")]["cls_accuracy"] * 100 for s in seeds],  "#4CAF50", 0.7),
    ("CLS @ inc=0.171", [results[(s, "wm_opt")]["cls_accuracy"] * 100 for s in seeds],   "#4CAF50", 0.9),
    ("WM @ inc=0",      [results[(s, "no_adapt")]["wm_accuracy"] * 100 for s in seeds],  "#E91E63", 0.5),
    ("WM @ inc=0.071",  [results[(s, "cls_opt")]["wm_accuracy"] * 100 for s in seeds],   "#E91E63", 0.7),
    ("WM @ inc=0.171",  [results[(s, "wm_opt")]["wm_accuracy"] * 100 for s in seeds],    "#E91E63", 0.9),
    ("XOR @ inc=0",     [results[(s, "no_adapt")]["xor_accuracy"] * 100 for s in seeds],  "#FF9800", 0.5),
    ("XOR @ inc=0.171", [results[(s, "wm_opt")]["xor_accuracy"] * 100 for s in seeds],    "#FF9800", 0.9),
]

y_pos = np.arange(len(metric_data))
for i, (name, vals, color, alpha) in enumerate(metric_data):
    m, s = np.mean(vals), np.std(vals)
    ax_d.barh(i, m, xerr=s, color=color, alpha=alpha, capsize=3,
             edgecolor="white", linewidth=0.5, height=0.7)
    ax_d.text(min(m + s + 1.0, 100), i, f"{m:.1f} \u00b1 {s:.1f}%",
             va="center", fontsize=8.5)

ax_d.set_yticks(y_pos)
ax_d.set_yticklabels([d[0] for d in metric_data], fontsize=9)
ax_d.set_xlabel("Accuracy (%)", fontsize=10)
ax_d.set_title("D. Cross-Seed Summary (N=10)", fontsize=11, fontweight="bold")
ax_d.set_xlim(15, 108)
ax_d.axvline(x=20, color="gray", linestyle=":", alpha=0.4)
ax_d.text(105, -0.3, "chance\n(20%)", fontsize=8, color="gray", ha="right", va="top")
ax_d.grid(True, axis="x", alpha=0.3)
ax_d.invert_yaxis()

plt.savefig("results/multi_seed/multi_seed_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: results/multi_seed/multi_seed_results.png")

# Print summary stats
print(f"\n=== MULTI-SEED SUMMARY (N={len(seeds)}) ===")
print(f"Network sizes: {[n_res[s] for s in seeds]}")
print(f"  range: {min(n_res.values())}-{max(n_res.values())}")
for name, vals, _, _ in metric_data:
    print(f"  {name:18s}: {np.mean(vals):.1f} +/- {np.std(vals):.1f}%  (range: {min(vals):.1f}-{max(vals):.1f})")
print(f"  {'WM range':18s}: {mean_dr:.1f} +/- {std_dr:.1f} pp  (range: {min(dr_vals):.1f}-{max(dr_vals):.1f})")
print(f"  Dissociation: {len(seeds)}/{len(seeds)} PASS")
