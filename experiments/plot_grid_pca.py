#!/usr/bin/env python3
"""
t-SNE embedding of input grid search parameter space, colored by composite score.
Highlights top 50 configurations.

Features: the 5 grid search parameters (log-transformed where appropriate).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = CPP_DIR / 'results' / 'input_grid_search' / 'input_grid_results.csv'
OUT_DIR = CPP_DIR / 'results' / 'input_grid_search'

def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(csv_path)

    # Feature columns: the 5 swept parameters
    param_cols = ['stim_current', 'input_tau_e', 'input_adapt_inc',
                  'input_std_u', 'input_std_tau_rec']
    X = df[param_cols].values.copy()

    # Log-transform log-spaced axes (stim_current, tau_e, adapt_inc)
    # Add small epsilon to adapt_inc (has zeros)
    X[:, 0] = np.log10(X[:, 0])
    X[:, 1] = np.log10(X[:, 1])
    X[:, 2] = np.log10(X[:, 2] + 1e-4)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running t-SNE on 5 parameter features (8000 points)...")
    tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto',
                init='pca', random_state=42, max_iter=2000)
    Z = tsne.fit_transform(X_scaled)
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

    score = df['composite_score'].values
    mi = df['mean_mi_bits'].values
    rate = df['mean_rate_hz'].values

    # Top 50 by composite score
    top50_idx = np.argsort(score)[-50:]
    rest_idx = np.argsort(score)[:-50]

    # Sort rest by score so high-score points draw on top
    rest_order = rest_idx[np.argsort(score[rest_idx])]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: Composite Score
    ax = axes[0]
    sc = ax.scatter(Z[rest_order, 0], Z[rest_order, 1],
                    c=score[rest_order], cmap='viridis', s=5, alpha=0.6,
                    vmin=0, vmax=score.max(), rasterized=True)
    ax.scatter(Z[top50_idx, 0], Z[top50_idx, 1],
               c=score[top50_idx], cmap='viridis', s=60, alpha=1.0,
               edgecolors='red', linewidths=1.2,
               vmin=0, vmax=score.max(), zorder=5)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('Composite Score')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Composite Score\n(MI + 0.15*r@20ms + 0.05*mod_depth,\nbio-filtered)', fontsize=10)

    # Panel 2: MI
    ax = axes[1]
    sc2 = ax.scatter(Z[rest_order, 0], Z[rest_order, 1],
                     c=mi[rest_order], cmap='inferno', s=5, alpha=0.6,
                     vmin=0, vmax=mi.max(), rasterized=True)
    ax.scatter(Z[top50_idx, 0], Z[top50_idx, 1],
               c=mi[top50_idx], cmap='inferno', s=60, alpha=1.0,
               edgecolors='cyan', linewidths=1.2,
               vmin=0, vmax=mi.max(), zorder=5)
    cb2 = plt.colorbar(sc2, ax=ax, shrink=0.8)
    cb2.set_label('Mutual Information (bits)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Pooled MI\n(8-quantile, bias-corrected by pooling\nacross 30 samples)', fontsize=10)

    # Panel 3: Firing Rate
    ax = axes[2]
    sc3 = ax.scatter(Z[rest_order, 0], Z[rest_order, 1],
                     c=rate[rest_order], cmap='plasma', s=5, alpha=0.6,
                     vmin=0, vmax=min(200, rate.max()), rasterized=True)
    ax.scatter(Z[top50_idx, 0], Z[top50_idx, 1],
               c=rate[top50_idx], cmap='plasma', s=60, alpha=1.0,
               edgecolors='lime', linewidths=1.2,
               vmin=0, vmax=min(200, rate.max()), zorder=5)
    cb3 = plt.colorbar(sc3, ax=ax, shrink=0.8)
    cb3.set_label('Firing Rate (Hz)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Mean Firing Rate (Hz)', fontsize=10)

    # Legend for top 50 marker
    for ax in axes:
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                  markersize=8, markeredgecolor='red', markeredgewidth=1.2,
                                  label='Top 50')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    fig.suptitle('t-SNE of Input Grid Search Parameter Space (8,000 points)\n'
                 'Features: log(stim_current), log(tau_e), log(adapt_inc), std_u, std_tau_rec\n'
                 'Top 50 configurations highlighted (red/cyan/lime borders)',
                 fontsize=12, y=1.04)
    plt.tight_layout()
    out1 = OUT_DIR / 'tsne_parameter_space.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # --- Print top 50 stats ---
    top50 = df.iloc[top50_idx].sort_values('composite_score', ascending=False)
    print(f"\n{'='*80}")
    print(f"TOP 50 CONFIGURATIONS (by composite score)")
    print(f"{'='*80}")
    print(f"  Score range: {top50['composite_score'].min():.4f} - {top50['composite_score'].max():.4f}")
    print(f"  MI range:    {top50['mean_mi_bits'].min():.4f} - {top50['mean_mi_bits'].max():.4f} bits")
    print(f"  r@20ms range: {top50['spike_bsa_r_20ms'].min():.4f} - {top50['spike_bsa_r_20ms'].max():.4f}")
    print(f"  Rate range:  {top50['mean_rate_hz'].min():.1f} - {top50['mean_rate_hz'].max():.1f} Hz")
    print(f"  ISI CV range: {top50['mean_isi_cv'].min():.4f} - {top50['mean_isi_cv'].max():.4f}")
    print()
    print("  Param distributions in top 50:")
    for col in param_cols:
        vals = sorted(set(float(v) for v in top50[col].unique()))
        print(f"    {col}: {vals}")
    print()
    print("  Top 10:")
    print(top50[['stim_current','input_tau_e','input_adapt_inc','input_std_u',
                  'mean_rate_hz','mean_mi_bits','spike_bsa_r_20ms','composite_score']].head(10).to_string())


if __name__ == '__main__':
    main()
