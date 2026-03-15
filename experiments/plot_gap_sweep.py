#!/usr/bin/env python3
"""Generate gap sweep figures from gap_sweep.json.

Produces 4 figures for Section 12:
  Fig 23: 4-panel summary (degradation, tau spread w/ ANOVA, XOR, degradation-by-inc)
  Fig 24: WM heatmaps (inc x tau) at each of 6 gap lengths
  Fig 25: Mean WM heatmap + degradation curves side-by-side
  Fig 26: Adapt-at-B-onset vs WM accuracy (exponential decay model)
"""

import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

# ── Load data ──
data_path = sys.argv[1] if len(sys.argv) > 1 else "results/gap_sweep/gap_sweep.json"
out_dir = os.path.dirname(data_path) or "results/gap_sweep"

with open(data_path) as f:
    data = json.load(f)

inc_vals = np.array(data["grid"]["unified_inc"])
tau_vals = np.array(data["grid"]["unified_tau"])
gap_vals = np.array(data["grid"]["gap_lengths_ms"])
n_inc, n_tau, n_gaps = len(inc_vals), len(tau_vals), len(gap_vals)

# Build 3D grids
wm_acc = np.full((n_inc, n_tau, n_gaps), np.nan)
xor_acc = np.full((n_inc, n_tau, n_gaps), np.nan)
adapt_at_b = np.full((n_inc, n_tau, n_gaps), np.nan)

for r in data["grid_results"]:
    ii, ti = r["inc_idx"], r["tau_idx"]
    gi = list(gap_vals).index(r["gap_ms"])
    wm_acc[ii, ti, gi] = r["wm_accuracy"]
    xor_acc[ii, ti, gi] = r["xor_accuracy"]
    adapt_at_b[ii, ti, gi] = r["adapt_at_b_onset_mean"]

def find_inc_idx(target):
    return int(np.argmin(np.abs(inc_vals - target)))

wm_idx = find_inc_idx(0.171)
gap_strs = [f"{g:.0f}" for g in gap_vals]


# ============================================================
# Figure 23: 4-panel summary — the key claims in one figure
# ============================================================
print("Generating Figure 23: gap_sweep_6panel.png ...")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Gap Sweep Summary: 480 points (20 inc x 4 tau x 6 gaps), Branch B",
             fontsize=13, fontweight="bold", y=0.97)

# (A) Graceful degradation — best WM per gap
ax = axes[0, 0]
best_wm = np.nanmax(wm_acc.reshape(-1, n_gaps), axis=0) * 100
ax.plot(gap_vals, best_wm, "o-", color="#d95f02", linewidth=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1, zorder=5)
ax.fill_between(gap_vals, 20, best_wm, alpha=0.08, color="#d95f02")
ax.set_xlabel("Gap length (ms)")
ax.set_ylabel("Best WM accuracy (%)")
ax.set_title("A.  Graceful degradation", fontweight="bold", loc="left")
ax.set_xscale("log")
ax.set_xticks(gap_vals)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_ylim(76, 86)
ax.axhline(20, color="gray", ls=":", alpha=0.3)
ax.grid(True, alpha=0.15)
# annotate endpoints
ax.annotate(f"{best_wm[0]:.1f}%", (gap_vals[0], best_wm[0]),
            textcoords="offset points", xytext=(8, 8), fontsize=9, color="#d95f02")
ax.annotate(f"{best_wm[-1]:.1f}%", (gap_vals[-1], best_wm[-1]),
            textcoords="offset points", xytext=(-40, -14), fontsize=9, color="#d95f02")
ax.annotate(f"{best_wm[0]-best_wm[-1]:.1f} pp drop\nover 20x gap range",
            xy=(300, 81), fontsize=8, color="#666", fontstyle="italic",
            bbox=dict(fc="white", ec="none", alpha=0.8))

# (B) Tau spread grows with gap — with ANOVA significance
ax = axes[0, 1]
tau_spread = []
for gi in range(n_gaps):
    vals = wm_acc[wm_idx, :, gi] * 100
    tau_spread.append(np.nanmax(vals) - np.nanmin(vals))

# Compute ANOVA p-values for tau effect at each gap
from scipy import stats as sp_stats
anova_p = []
for gi in range(n_gaps):
    groups = []
    for ti in range(n_tau):
        key = (wm_idx, ti, gap_vals[gi])
        for r in data["grid_results"]:
            if r["inc_idx"] == wm_idx and r["tau_idx"] == ti and r["gap_ms"] == gap_vals[gi]:
                groups.append(np.array(r["wm_per_repeat"]) * 100)
                break
    F, p = sp_stats.f_oneway(*groups)
    anova_p.append(p)

colors_bar = ["#bbb" if p > 0.05 else "#4292c6" if p > 0.001 else "#08519c" for p in anova_p]
bars = ax.bar(range(n_gaps), tau_spread, color=colors_bar, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(n_gaps))
ax.set_xticklabels(gap_strs)
ax.set_xlabel("Gap length (ms)")
ax.set_ylabel("Tau spread (pp)")
ax.set_title("B.  Tau becomes decisive at long gaps", fontweight="bold", loc="left")
ax.set_ylim(0, 6.0)
ax.grid(True, alpha=0.15, axis="y")
# value labels with significance stars
for i, (b, v, p) in enumerate(zip(bars, tau_spread, anova_p)):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(b.get_x() + b.get_width()/2, v + 0.12, f"{v:.1f}\n{star}",
            ha="center", fontsize=7, fontweight="bold", color=colors_bar[i])
ax.text(0.98, 0.95, "at inc = 0.171\nANOVA across 4 tau values", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5, color="#666", fontstyle="italic")

# (C) XOR is gap-insensitive
ax = axes[1, 0]
best_xor = np.nanmax(xor_acc.reshape(-1, n_gaps), axis=0) * 100
ax.plot(gap_vals, best_xor, "s-", color="#2ca02c", linewidth=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1)
ax.set_xlabel("Gap length (ms)")
ax.set_ylabel("Best XOR accuracy (%)")
ax.set_title("C.  XOR is gap-insensitive", fontweight="bold", loc="left")
ax.set_xscale("log")
ax.set_xticks(gap_vals)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.axhline(50, color="gray", ls=":", alpha=0.3, label="Chance")
ax.set_ylim(64, 70)
ax.grid(True, alpha=0.15)
ax.text(0.5, 0.08, f"Total range: {best_xor.max()-best_xor.min():.1f} pp",
        transform=ax.transAxes, ha="center", fontsize=9, color="#2ca02c",
        bbox=dict(fc="white", ec="#2ca02c", alpha=0.9, boxstyle="round,pad=0.3"))

# (D) Degradation by inc — who suffers most?
ax = axes[1, 1]
notable_incs = [0.0, 0.0707, 0.0949, 0.1710, 0.3081, 0.5550]
labels = ["0", "0.071\n(CLS)", "0.095\n(ISI CV)", "0.171\n(WM)", "0.308", "0.555"]
deltas = []
for inc_t in notable_incs:
    ii = find_inc_idx(inc_t)
    wm150 = np.nanmax(wm_acc[ii, :, 2]) * 100  # gap=150
    wm1000 = np.nanmax(wm_acc[ii, :, 5]) * 100  # gap=1000
    deltas.append(wm1000 - wm150)
colors_d = ["#999" if abs(d) < 1 else "#e6550d" if d < -2 else "#fd8d3c" if d < 0 else "#31a354"
             for d in deltas]
bars = ax.bar(range(len(deltas)), deltas, color=colors_d, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(deltas)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_xlabel("Adaptation increment (inc)")
ax.set_ylabel("WM change, gap 150 -> 1000 ms (pp)")
ax.set_title("D.  Peak degradation at CLS optimum", fontweight="bold", loc="left")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(True, alpha=0.15, axis="y")
for i, (b, v) in enumerate(zip(bars, deltas)):
    y_off = -0.3 if v < 0 else 0.1
    ax.text(b.get_x() + b.get_width()/2, v + y_off,
            f"{v:+.1f}", ha="center", fontsize=8, fontweight="bold", color=colors_d[i])

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(out_dir, "gap_sweep_6panel.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"  Saved: {out_dir}/gap_sweep_6panel.png")
plt.close()


# ============================================================
# Figure 24: WM heatmaps at each gap length
# ============================================================
print("Generating Figure 24: gap_sweep_wm_heatmaps.png ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
fig.suptitle("WM Accuracy Across Gap Lengths (Branch B, rate-matched)",
             fontsize=13, fontweight="bold")

# Shared colorbar range
vmin, vmax = 20, 85
# Sparse inc labels: show key values only
key_incs = {0: "0", 5: ".022", 10: ".071", 13: ".171", 16: ".414", 19: "1.0"}
inc_ticks = list(key_incs.keys())
inc_tick_labels = list(key_incs.values())

for gi in range(n_gaps):
    ax = axes[gi // 3, gi % 3]
    gap_ms = gap_vals[gi]

    im = ax.imshow(wm_acc[:, :, gi].T * 100, aspect="auto", origin="lower",
                   cmap="YlOrRd", interpolation="nearest", vmin=vmin, vmax=vmax)

    # Optimum star
    opt_flat = np.nanargmax(wm_acc[:, :, gi])
    oi, oj = np.unravel_index(opt_flat, (n_inc, n_tau))
    best = wm_acc[oi, oj, gi] * 100
    ax.plot(oi, oj, "*", markersize=16, color="cyan", markeredgecolor="black",
            markeredgewidth=0.8, zorder=10)

    # Tau spread annotation
    vals_wm = wm_acc[wm_idx, :, gi] * 100
    spread = np.nanmax(vals_wm) - np.nanmin(vals_wm)

    ax.set_title(f"Gap = {gap_ms:.0f} ms   (best: {best:.1f}%,  tau spread: {spread:.1f} pp)",
                 fontsize=9, fontweight="bold")
    ax.set_xticks(inc_ticks)
    ax.set_xticklabels(inc_tick_labels, fontsize=7)
    ax.set_yticks(range(n_tau))
    ax.set_yticklabels([f"{v/1000:.1f}k" for v in tau_vals], fontsize=8)
    if gi % 3 == 0:
        ax.set_ylabel("adapt_tau (ms)")
    if gi >= 3:
        ax.set_xlabel("adapt_inc")

# Single shared colorbar
cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label="WM accuracy (%)")

fig.savefig(os.path.join(out_dir, "gap_sweep_wm_heatmaps.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"  Saved: {out_dir}/gap_sweep_wm_heatmaps.png")
plt.close()


# Shared colormap for gap-indexed optima (used in Fig 25/26)
cmap = plt.cm.plasma
norm = Normalize(vmin=0, vmax=n_gaps - 1)

# ============================================================
# Figure 25: Combined heatmap + degradation curves
# ============================================================
print("Generating Figure 25: wm_combined_heatmap_with_optima.png ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})
fig.suptitle("WM Landscape and Degradation by Adaptation Strength",
             fontsize=13, fontweight="bold", y=0.98)

# (A) Mean WM heatmap with optima
mean_wm = np.nanmean(wm_acc, axis=2) * 100
im = ax1.imshow(mean_wm.T, aspect="auto", origin="lower",
                cmap="YlOrRd", interpolation="nearest", vmin=20, vmax=85)
cb = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.03)
cb.set_label("Mean WM (%)")

# Overlay per-gap optima as colored dots (no text labels — use legend)
gap_markers = ["o", "s", "D", "^", "v", "P"]  # distinct markers per gap
for gi in range(n_gaps):
    oi, oj = np.unravel_index(np.nanargmax(wm_acc[:, :, gi]), (n_inc, n_tau))
    c = cmap(norm(gi))
    ax1.plot(oi, oj, marker=gap_markers[gi], markersize=9, color=c,
             markeredgecolor="white", markeredgewidth=1.2, zorder=10,
             label=f"WM opt, gap={gap_vals[gi]:.0f}ms")

# CLS marker
cls_ii = find_inc_idx(0.0707)
cls_ti = int(np.argmin(np.abs(tau_vals - 5000)))
ax1.plot(cls_ii, cls_ti, marker="*", markersize=14, color="dodgerblue",
         markeredgecolor="white", markeredgewidth=1.2, zorder=11,
         label="CLS optimum")

ax1.set_xticks(inc_ticks)
ax1.set_xticklabels(inc_tick_labels, fontsize=7)
ax1.set_yticks(range(n_tau))
ax1.set_yticklabels([f"{v/1000:.1f}k" for v in tau_vals])
ax1.set_xlabel("adapt_inc")
ax1.set_ylabel("adapt_tau (ms)")
ax1.set_title("A.  Mean WM landscape + per-gap optima", fontweight="bold", loc="left")

ax1.legend(loc="lower left", fontsize=6, framealpha=0.9, ncol=2)

# (B) Degradation curves — cleaner version
notable_incs = [0.0707, 0.0949, 0.1710, 0.3081, 0.5550]
colors_line = ["#1f77b4", "#2ca02c", "#d95f02", "#d62728", "#9467bd"]
labels_line = ["0.071 (CLS opt)", "0.095 (ISI CV)", "0.171 (WM opt)",
               "0.308", "0.555"]
styles = ["-", "--", "-", "--", ":"]

for inc_t, color, label, ls in zip(notable_incs, colors_line, labels_line, styles):
    ii = find_inc_idx(inc_t)
    best_line = [np.nanmax(wm_acc[ii, :, gi]) * 100 for gi in range(n_gaps)]
    ax2.plot(gap_vals, best_line, "o" + ls[0] if ls == "-" else "s" + ls[0],
             color=color, linewidth=2, markersize=6, label=f"inc = {label}",
             linestyle=ls, alpha=0.85)

ax2.set_xlabel("Gap length (ms)")
ax2.set_ylabel("Best WM accuracy (%, optimized over tau)")
ax2.set_title("B.  WM degradation by adaptation strength", fontweight="bold", loc="left")
ax2.set_xscale("log")
ax2.set_xticks(gap_vals)
ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax2.set_ylim(68, 86)
ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax2.grid(True, alpha=0.15)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, "wm_combined_heatmap_with_optima.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"  Saved: {out_dir}/wm_combined_heatmap_with_optima.png")
plt.close()


# ============================================================
# Figure 26: Adapt-at-B-onset vs WM accuracy
# ============================================================
print("Generating Figure 26: adapt_at_b_onset_vs_wm.png ...")

# Extract all non-zero-inc data
ab_all, wm_all, gap_all, tau_all, inc_all = [], [], [], [], []
for r in data["grid_results"]:
    if r["adapt_inc"] > 0.005:
        ab_all.append(r["adapt_at_b_onset_mean"])
        wm_all.append(r["wm_accuracy"] * 100)
        gap_all.append(r["gap_ms"])
        tau_all.append(r["adapt_tau"])
        inc_all.append(r["adapt_inc"])
ab_all = np.array(ab_all)
wm_all = np.array(wm_all)
gap_all = np.array(gap_all)
tau_all = np.array(tau_all)
inc_all = np.array(inc_all)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Adaptation Conductance at B Onset Predicts WM Accuracy",
             fontsize=13, fontweight="bold", y=0.98)

# (A) Global scatter colored by gap
cmap_gap = plt.cm.plasma
norm_gap = Normalize(vmin=np.log10(50), vmax=np.log10(1000))
for gap_ms in gap_vals:
    mask = gap_all == gap_ms
    c = cmap_gap(norm_gap(np.log10(gap_ms)))
    ax1.scatter(ab_all[mask], wm_all[mask], c=[c], s=15, alpha=0.4, edgecolor="none")

# Trend line
valid = ab_all > 0.01
r_val = np.corrcoef(ab_all[valid], wm_all[valid])[0, 1]
coeffs = np.polyfit(ab_all[valid], wm_all[valid], 1)
x_fit = np.linspace(0, ab_all.max() * 1.05, 100)
ax1.plot(x_fit, np.polyval(coeffs, x_fit), "k--", lw=1.5, alpha=0.5)

ax1.set_xlabel("Adaptation conductance at B onset (nS)")
ax1.set_ylabel("WM accuracy (%)")
ax1.set_title(f"A.  All non-zero points (r = {r_val:.2f})", fontweight="bold", loc="left")
ax1.grid(True, alpha=0.15)

# Gap legend (manual, clean)
for gap_ms in [50, 150, 500, 1000]:
    c = cmap_gap(norm_gap(np.log10(gap_ms)))
    ax1.scatter([], [], c=[c], s=40, label=f"{gap_ms:.0f} ms")
ax1.legend(title="Gap", fontsize=7.5, title_fontsize=8, loc="lower right",
           framealpha=0.9, markerscale=1.5)

# (B) Clean view at inc=0.171
tau_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
tau_markers = ["o", "s", "D", "^"]
actual_inc = inc_vals[find_inc_idx(0.171)]

for ti, (tau_ms, color, marker) in enumerate(zip(tau_vals, tau_colors, tau_markers)):
    mask = (np.abs(inc_all - actual_inc) < 0.001) & (np.abs(tau_all - tau_ms) < 1)
    ab = ab_all[mask]
    wm_v = wm_all[mask]
    gap_sub = gap_all[mask]
    order = np.argsort(gap_sub)
    ax2.plot(ab[order], wm_v[order], marker=marker, color=color,
             linewidth=2, markersize=9, markeredgecolor="white",
             markeredgewidth=1, label=f"tau = {tau_ms/1000:.1f}k ms")

ax2.set_xlabel("Adaptation conductance at B onset (nS)")
ax2.set_ylabel("WM accuracy (%)")
ax2.set_title("B.  At inc = 0.171: exponential decay model", fontweight="bold", loc="left")
ax2.set_ylim(74, 84)
ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax2.grid(True, alpha=0.15)

# Annotation — positioned to avoid data
ax2.text(0.04, 0.06, "Longer gap = lower residual = lower WM\n"
         "Higher tau = slower decay = more retained",
         transform=ax2.transAxes, fontsize=8, color="#555", fontstyle="italic",
         va="bottom", bbox=dict(fc="white", ec="#ccc", alpha=0.9, boxstyle="round,pad=0.4"))

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, "adapt_at_b_onset_vs_wm.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"  Saved: {out_dir}/adapt_at_b_onset_vs_wm.png")
plt.close()


print("\nDone. All 5 figures regenerated.")
