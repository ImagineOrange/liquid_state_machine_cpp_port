#!/usr/bin/env python3
"""
Plot results from the C++ input neuron regime grid search (v2: NMDA-off, expanded metrics).
Usage: python plot_input_grid.py [input_grid_results.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
_default_csv = str(Path(__file__).resolve().parent.parent / 'results' / 'input_grid_search' / 'input_grid_results.csv')
csv_path = sys.argv[1] if len(sys.argv) > 1 else _default_csv
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} grid points from {csv_path}")
print(f"Axes: stim_current={sorted(df.stim_current.unique())}")
print(f"      input_tau_e={sorted(df.input_tau_e.unique())}")
print(f"      input_adapt_inc={sorted(df.input_adapt_inc.unique())}")

# Find top configurations
top = df.nlargest(20, 'composite_score')
print("\n=== TOP 20 CONFIGURATIONS ===")
cols_show = ['stim_current', 'input_tau_e', 'input_adapt_inc',
             'mean_rate_hz', 'mean_isi_cv', 'mean_refrac_frac',
             'spike_bsa_r_20ms', 'mean_mi_bits', 'mean_fano_factor',
             'composite_score']
cols_show = [c for c in cols_show if c in df.columns]
print(top[cols_show].to_string(index=False))

best = df.loc[df.composite_score.idxmax()]
print(f"\n=== BEST ===")
print(f"  stim_current={best.stim_current:.4f}, tau_e={best.input_tau_e:.1f}ms, "
      f"adapt_inc={best.input_adapt_inc:.4f}")
print(f"  Rate={best.mean_rate_hz:.1f}Hz, ISI_CV={best.mean_isi_cv:.3f}, "
      f"Refrac={best.mean_refrac_frac:.3f}, ModDepth={best.mean_mod_depth:.3f}")
r_cols = [c for c in df.columns if c.startswith('spike_bsa_r_')]
for c in r_cols:
    print(f"  {c}={best[c]:.3f}", end='')
print()
if 'mean_mi_bits' in df.columns:
    print(f"  MI={best.mean_mi_bits:.4f} bits, Fano={best.mean_fano_factor:.3f}, "
          f"Burst={best.mean_burst_frac:.3f}, Sparseness={best.mean_temporal_sparseness:.3f}, "
          f"DynRange={best.mean_dynamic_range:.3f}")
print(f"  Score={best.composite_score:.4f}")

adapt_vals = sorted(df.input_adapt_inc.unique())
stim_vals = sorted(df.stim_current.unique())
tau_vals = sorted(df.input_tau_e.unique())

# ---- Figure 1: Heatmaps score per (stim_current x tau_e) for select adapt_inc ----
n_adapt_show = min(5, len(adapt_vals))
ncols = min(3, n_adapt_show + 1)
nrows = int(np.ceil((n_adapt_show + 1) / ncols))
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
fig1.suptitle("Composite Score: stim_current x tau_e (NMDA off on input)",
              fontsize=14, fontweight='bold')
axes_flat = np.array(axes1).flat

# Pick evenly-spaced adapt values to show
show_idx = np.linspace(0, len(adapt_vals) - 1, n_adapt_show, dtype=int)
for ai_plot, ai in enumerate(show_idx):
    adapt = adapt_vals[ai]
    ax = axes_flat[ai_plot]
    sub = df[df.input_adapt_inc == adapt]
    best_scores = sub.groupby(['stim_current', 'input_tau_e']).composite_score.max().reset_index()
    pivot = best_scores.pivot(index='input_tau_e', columns='stim_current', values='composite_score')
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', origin='lower',
                   vmin=df.composite_score.quantile(0.05),
                   vmax=df.composite_score.quantile(0.98))
    ax.set_xticks(range(len(stim_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in stim_vals], fontsize=6, rotation=45)
    ax.set_yticks(range(len(tau_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in tau_vals], fontsize=7)
    ax.set_xlabel("stim_current")
    ax.set_ylabel("tau_e (ms)")
    ax.set_title(f"adapt_inc={adapt:.2f}")

# Best across all
ax = axes_flat[n_adapt_show]
best_all = df.groupby(['stim_current', 'input_tau_e']).composite_score.max().reset_index()
pivot_all = best_all.pivot(index='input_tau_e', columns='stim_current', values='composite_score')
im = ax.imshow(pivot_all.values, aspect='auto', cmap='RdYlGn', origin='lower',
               vmin=df.composite_score.quantile(0.05),
               vmax=df.composite_score.quantile(0.98))
ax.set_xticks(range(len(stim_vals)))
ax.set_xticklabels([f"{v:.2f}" for v in stim_vals], fontsize=6, rotation=45)
ax.set_yticks(range(len(tau_vals)))
ax.set_yticklabels([f"{v:.1f}" for v in tau_vals], fontsize=7)
ax.set_xlabel("stim_current")
ax.set_ylabel("tau_e (ms)")
ax.set_title("BEST across all adapt_inc", fontweight='bold')

# Hide extra axes
for i in range(n_adapt_show + 1, len(list(axes_flat))):
    axes_flat[i].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.92, 0.95])
cbar_ax = fig1.add_axes([0.93, 0.15, 0.02, 0.7])
fig1.colorbar(im, cax=cbar_ax, label='Composite Score')
fig1.savefig(csv_path.replace('.csv', '_heatmaps.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {csv_path.replace('.csv', '_heatmaps.png')}")

# ---- Figure 2: Metric profiles vs stim_current (at a few adapt_inc, mid tau_e) ----
mid_tau = tau_vals[len(tau_vals) // 2]  # pick middle tau_e
fig2, axes2 = plt.subplots(4, 4, figsize=(20, 16))
fig2.suptitle(f"Metric Profiles vs stim_current (tau_e={mid_tau:.1f}ms, NMDA off)",
              fontsize=14, fontweight='bold')

metrics2 = [
    ('mean_rate_hz', 'Firing Rate (Hz)'),
    ('mean_isi_cv', 'ISI CV'),
    ('mean_refrac_frac', 'Refractory Fraction'),
    ('mean_mod_depth', 'Rate Modulation Depth'),
    ('spike_bsa_r_5ms', 'Spike-BSA r (5ms)'),
    ('spike_bsa_r_10ms', 'Spike-BSA r (10ms)'),
    ('spike_bsa_r_20ms', 'Spike-BSA r (20ms)'),
    ('spike_bsa_r_50ms', 'Spike-BSA r (50ms)'),
    ('spike_bsa_r_100ms', 'Spike-BSA r (100ms)'),
    ('mean_burst_frac', 'Burst Fraction'),
    ('mean_fano_factor', 'Fano Factor'),
    ('mean_temporal_sparseness', 'Temporal Sparseness'),
    ('mean_dynamic_range', 'Dynamic Range (log10)'),
    ('mean_mi_bits', 'Mutual Information (bits)'),
    ('n_active_neurons', 'Active Input Neurons'),
    ('composite_score', 'Composite Score'),
]
# Only plot columns that exist
metrics2 = [(c, l) for c, l in metrics2 if c in df.columns]

# Show a subset of adapt values
adapt_show = adapt_vals[::max(1, len(adapt_vals) // 5)]

for mi, (col, label) in enumerate(metrics2):
    if mi >= 16:
        break
    ax = axes2.flat[mi]
    for adapt in adapt_show:
        sub = df[(df.input_adapt_inc == adapt) & (df.input_tau_e == mid_tau)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values('stim_current')
        ax.plot(sub.stim_current, sub[col], 'o-', label=f"a={adapt:.2f}", markersize=3)
    ax.set_xlabel("stim_current")
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=10)
    if mi == 0:
        ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

for i in range(len(metrics2), 16):
    axes2.flat[i].set_visible(False)

plt.tight_layout()
fig2.savefig(csv_path.replace('.csv', '_profiles.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {csv_path.replace('.csv', '_profiles.png')}")

# ---- Figure 3: Effect of tau_e at best adapt_inc ----
best_adapt = best.input_adapt_inc
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 14))
fig3.suptitle(f"Effect of tau_e on Metrics (adapt_inc={best_adapt:.2f}, NMDA off)",
              fontsize=14, fontweight='bold')

te_metrics = [
    ('mean_rate_hz', 'Firing Rate (Hz)'),
    ('mean_isi_cv', 'ISI CV'),
    ('spike_bsa_r_20ms', 'Spike-BSA r (20ms)'),
    ('mean_refrac_frac', 'Refractory Fraction'),
    ('mean_mod_depth', 'Modulation Depth'),
    ('mean_mi_bits', 'Mutual Information (bits)'),
    ('mean_fano_factor', 'Fano Factor'),
    ('mean_temporal_sparseness', 'Temporal Sparseness'),
    ('composite_score', 'Composite Score'),
]
te_metrics = [(c, l) for c, l in te_metrics if c in df.columns]

df_te = df[df.input_adapt_inc == best_adapt]

for mi, (col, label) in enumerate(te_metrics):
    if mi >= 9:
        break
    ax = axes3.flat[mi]
    for tau in tau_vals:
        sub = df_te[df_te.input_tau_e == tau].sort_values('stim_current')
        if len(sub) == 0:
            continue
        ax.plot(sub.stim_current, sub[col], 'o-', label=f"tau_e={tau:.1f}ms", markersize=3)
    ax.set_xlabel("stim_current")
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=10)
    if mi == 0:
        ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

for i in range(len(te_metrics), 9):
    axes3.flat[i].set_visible(False)

plt.tight_layout()
fig3.savefig(csv_path.replace('.csv', '_tau_e.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {csv_path.replace('.csv', '_tau_e.png')}")

# ---- Figure 4: Information preservation landscape ----
# Show MI and best correlation as 2D heatmaps (stim x adapt) at best tau_e
best_tau = best.input_tau_e
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
fig4.suptitle(f"Information Preservation Landscape (tau_e={best_tau:.1f}ms, NMDA off)",
              fontsize=14, fontweight='bold')

info_metrics = [
    ('spike_bsa_r_20ms', 'Spike-BSA r (20ms)'),
    ('mean_mi_bits', 'MI (bits)'),
    ('mean_fano_factor', 'Fano Factor'),
    ('mean_burst_frac', 'Burst Fraction'),
    ('mean_temporal_sparseness', 'Temporal Sparseness'),
    ('composite_score', 'Composite Score'),
]
info_metrics = [(c, l) for c, l in info_metrics if c in df.columns]

df_tau = df[df.input_tau_e == best_tau]

for mi, (col, label) in enumerate(info_metrics):
    if mi >= 6:
        break
    ax = axes4.flat[mi]
    pivot = df_tau.pivot_table(index='input_adapt_inc', columns='stim_current',
                                values=col, aggfunc='mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(stim_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in stim_vals], fontsize=6, rotation=45)
    ax.set_yticks(range(len(adapt_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in adapt_vals], fontsize=7)
    ax.set_xlabel("stim_current")
    ax.set_ylabel("adapt_inc")
    ax.set_title(label, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

for i in range(len(info_metrics), 6):
    axes4.flat[i].set_visible(False)

plt.tight_layout()
fig4.savefig(csv_path.replace('.csv', '_info_landscape.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {csv_path.replace('.csv', '_info_landscape.png')}")

# ---- Figure 5: Correlation across timescales for top configs ----
fig5, ax5 = plt.subplots(figsize=(12, 7))
fig5.suptitle("Spike-BSA Correlation Across Timescales (Top 10 Configs)", fontsize=14, fontweight='bold')

top10 = df.nlargest(10, 'composite_score')
timescales = [5, 10, 20, 50, 100]
r_cols_ordered = [f'spike_bsa_r_{t}ms' for t in timescales]
r_cols_ordered = [c for c in r_cols_ordered if c in df.columns]
ts_available = [t for t, c in zip(timescales, [f'spike_bsa_r_{t}ms' for t in timescales]) if c in df.columns]

for idx, (_, row) in enumerate(top10.iterrows()):
    vals = [row[c] for c in r_cols_ordered]
    label = f"stim={row.stim_current:.2f} tau={row.input_tau_e:.1f} a={row.input_adapt_inc:.2f}"
    ax5.plot(ts_available, vals, 'o-', label=label, markersize=5)

ax5.set_xlabel("Bin Window (ms)")
ax5.set_ylabel("Pearson r (spike rate vs BSA rate)")
ax5.set_title("How well do spikes track BSA at different timescales?")
ax5.legend(fontsize=7, loc='best')
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')
ax5.set_xticks(ts_available)
ax5.set_xticklabels([str(t) for t in ts_available])

plt.tight_layout()
fig5.savefig(csv_path.replace('.csv', '_timescale_corr.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {csv_path.replace('.csv', '_timescale_corr.png')}")

print("\nDone.")
