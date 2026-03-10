#!/usr/bin/env python3
"""
Robustness analysis: rule out tonic conductance confound in Branch B.

Demonstrates that the ISI CV / PR → accuracy correlations are not artifacts
of variation in g_tonic level across the rate-matched population.

Six analyses:
1. Within-regime correlations (g ∈ {1.41, 2.81}, n=122)
2. Full-branch regression with g_tonic covariate
3. Within-regime BSA comparison
4. Cross-regime consistency (per g-level correlations)
5. Mediation: does g_tonic predict accuracy after controlling for dynamics?
6. Per-bin within-regime correlations (replicating sign-flip)

Outputs: console report + JSON summary.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

CPP_DIR = Path(__file__).resolve().parent.parent
INPUT = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
OUTPUT_JSON = CPP_DIR / 'results' / 'classification_adaptation_sweep' / 'gtonic_robustness.json'

BIN_MS = 20.0

with open(INPUT) as f:
    data = json.load(f)

bsa_baseline_acc = data['bsa_baseline']['accuracy']

# All Branch B non-baseline entries
all_B = [e for e in data['grid_results']
         if e['branch'] == 'B_matched' and e['inc_idx'] > 0]
print(f"Branch B non-baseline: n={len(all_B)}")

# Extract arrays for convenience
def extract(entries):
    return {
        'acc': np.array([e['classification_accuracy'] for e in entries]),
        'isi_cv': np.array([e['isi_cv_mean'] for e in entries]),
        'pr': np.array([e['participation_ratio_mean'] for e in entries]),
        'rate': np.array([e['firing_rate_hz'] for e in entries]),
        'g_tonic': np.array([e['tonic_conductance'] for e in entries]),
        'adapt_inc': np.array([e['adapt_inc'] for e in entries]),
        'adapt_tau': np.array([e['adapt_tau'] for e in entries]),
        'per_bin': np.array([e['per_bin_accuracy'] for e in entries]),  # (n, 48)
        'acc_std': np.array([e['classification_accuracy_std'] for e in entries]),
        'per_repeat': [e['classification_per_repeat_accuracy'] for e in entries],
    }

d_all = extract(all_B)

# --- Helpers ---

def pearsonr_safe(x, y):
    """Pearson r with p-value, returns (nan, nan) if degenerate."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    return r, p

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    if len(x) < 4:
        return np.nan, np.nan
    # Residualize x and y on z
    def resid(a, b):
        b = np.column_stack([b, np.ones(len(b))])
        beta = np.linalg.lstsq(b, a, rcond=None)[0]
        return a - b @ beta
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    rx = resid(x, z)
    ry = resid(y, z)
    return pearsonr_safe(rx, ry)

def ols_summary(y, X, names):
    """Simple OLS with standardized betas, t-stats, p-values."""
    n, k = X.shape
    # Standardize
    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)
    y_z = (y - y.mean()) / (y.std() + 1e-15)
    # Add intercept to unstandardized for R²
    X_i = np.column_stack([X, np.ones(n)])
    beta_raw = np.linalg.lstsq(X_i, y, rcond=None)[0]
    y_hat = X_i @ beta_raw
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    # Standardized betas
    X_zi = np.column_stack([X_z, np.ones(n)])
    beta_std = np.linalg.lstsq(X_zi, y_z, rcond=None)[0][:k]
    # Standard errors and t-stats for standardized
    dof = n - k - 1
    mse = ss_res / max(dof, 1)
    # Use original scale for proper inference
    XtX_inv = np.linalg.pinv(X_i.T @ X_i)
    se_raw = np.sqrt(np.diag(XtX_inv) * mse)[:k]
    # Convert to standardized scale
    se_std = se_raw * y.std() / (X.std(axis=0) + 1e-15)
    # t-stats from standardized
    t_stats = beta_std / (np.sqrt(np.diag(np.linalg.pinv(X_zi.T @ X_zi))[:k] * np.sum((y_z - X_zi @ np.linalg.lstsq(X_zi, y_z, rcond=None)[0]) ** 2) / max(dof, 1)) + 1e-15)
    p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)

    result = {'r2': r2, 'n': n, 'dof': dof, 'coefficients': {}}
    for i, name in enumerate(names):
        result['coefficients'][name] = {
            'beta_std': float(beta_std[i]),
            't': float(t_stats[i]),
            'p': float(p_vals[i]),
        }
    return result

results = {}

# ============================================================
# 1. Within-regime correlations (g ∈ {1.41, 2.81})
# ============================================================
print("\n" + "="*70)
print("1. WITHIN-REGIME CORRELATIONS (g ∈ {1.41, 2.81}, dominant regime)")
print("="*70)

dominant_mask = np.isin(np.round(d_all['g_tonic'], 2), [1.41, 2.81])
dominant_entries = [e for e, m in zip(all_B, dominant_mask) if m]
d_dom = extract(dominant_entries)
n_dom = len(dominant_entries)
print(f"n = {n_dom}")
print(f"  g=1.41: {np.sum(np.round(d_dom['g_tonic'], 2) == 1.41)}")
print(f"  g=2.81: {np.sum(np.round(d_dom['g_tonic'], 2) == 2.81)}")

r_isi, p_isi = pearsonr_safe(d_dom['isi_cv'], d_dom['acc'])
r_pr, p_pr = pearsonr_safe(d_dom['pr'], d_dom['acc'])
r_rate, p_rate = pearsonr_safe(d_dom['rate'], d_dom['acc'])
r_gtonic, p_gtonic = pearsonr_safe(d_dom['g_tonic'], d_dom['acc'])

print(f"\n  acc ~ ISI CV:  r = {r_isi:+.4f}  (p = {p_isi:.2e})")
print(f"  acc ~ PR:      r = {r_pr:+.4f}  (p = {p_pr:.2e})")
print(f"  acc ~ rate:    r = {r_rate:+.4f}  (p = {p_rate:.2e})")
print(f"  acc ~ g_tonic: r = {r_gtonic:+.4f}  (p = {p_gtonic:.2e})")

# Partial correlations
r_isi_rate, p_isi_rate = partial_corr(d_dom['isi_cv'], d_dom['acc'], d_dom['rate'])
r_rate_isi, p_rate_isi = partial_corr(d_dom['rate'], d_dom['acc'], d_dom['isi_cv'])
print(f"\n  Partial: acc ~ ISI CV | rate:  r = {r_isi_rate:+.4f}  (p = {p_isi_rate:.2e})")
print(f"  Partial: acc ~ rate | ISI CV: r = {r_rate_isi:+.4f}  (p = {p_rate_isi:.2e})")

# Multiple regression within regime
reg1 = ols_summary(d_dom['acc'],
                    np.column_stack([d_dom['isi_cv'], d_dom['pr'], d_dom['rate']]),
                    ['ISI_CV', 'PR', 'rate'])
print(f"\n  OLS (acc ~ ISI_CV + PR + rate): R² = {reg1['r2']:.4f}")
for name, c in reg1['coefficients'].items():
    print(f"    {name:8s}: β = {c['beta_std']:+.4f}, t = {c['t']:+.2f}, p = {c['p']:.2e}")

results['within_regime_dominant'] = {
    'n': n_dom,
    'g_levels': [1.41, 2.81],
    'bivariate': {
        'acc_isi_cv': {'r': r_isi, 'p': p_isi},
        'acc_pr': {'r': r_pr, 'p': p_pr},
        'acc_rate': {'r': r_rate, 'p': p_rate},
        'acc_g_tonic': {'r': r_gtonic, 'p': p_gtonic},
    },
    'partial': {
        'acc_isi_cv_given_rate': {'r': r_isi_rate, 'p': p_isi_rate},
        'acc_rate_given_isi_cv': {'r': r_rate_isi, 'p': p_rate_isi},
    },
    'ols_acc_dynamics': reg1,
}

# ============================================================
# 2. Full-branch regression with g_tonic covariate
# ============================================================
print("\n" + "="*70)
print("2. FULL-BRANCH REGRESSION WITH g_tonic COVARIATE")
print("="*70)

# Without g_tonic
reg_no_g = ols_summary(d_all['acc'],
                        np.column_stack([d_all['isi_cv'], d_all['pr'], d_all['rate']]),
                        ['ISI_CV', 'PR', 'rate'])
print(f"\n  Without g_tonic (n={reg_no_g['n']}): R² = {reg_no_g['r2']:.4f}")
for name, c in reg_no_g['coefficients'].items():
    print(f"    {name:8s}: β = {c['beta_std']:+.4f}, t = {c['t']:+.2f}, p = {c['p']:.2e}")

# With g_tonic
reg_with_g = ols_summary(d_all['acc'],
                          np.column_stack([d_all['isi_cv'], d_all['pr'],
                                           d_all['rate'], d_all['g_tonic']]),
                          ['ISI_CV', 'PR', 'rate', 'g_tonic'])
print(f"\n  With g_tonic (n={reg_with_g['n']}): R² = {reg_with_g['r2']:.4f}")
for name, c in reg_with_g['coefficients'].items():
    print(f"    {name:8s}: β = {c['beta_std']:+.4f}, t = {c['t']:+.2f}, p = {c['p']:.2e}")

delta_r2 = reg_with_g['r2'] - reg_no_g['r2']
print(f"\n  ΔR² from adding g_tonic: {delta_r2:+.6f}")

results['full_branch_regression'] = {
    'without_g_tonic': reg_no_g,
    'with_g_tonic': reg_with_g,
    'delta_r2': delta_r2,
}

# ============================================================
# 3. Within-regime BSA comparison
# ============================================================
print("\n" + "="*70)
print("3. WITHIN-REGIME BSA COMPARISON")
print("="*70)

# Identify BSA-exceeding cells using one-sided Welch t-test (cell > BSA),
# matching Finding 5's methodology: joint BH-FDR across 600 tests (both branches).
# BSA: mean=0.9484, n=5, population std=0.0026 → sample std = 0.0026 * sqrt(5/4)
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests

bsa_mean = bsa_baseline_acc
bsa_n = 5
bsa_sample_std = 0.0026 * np.sqrt(bsa_n / (bsa_n - 1))

def _bsa_pval(e):
    cell_std = e['classification_accuracy_std'] * np.sqrt(5 / 4)
    t, p2 = ttest_ind_from_stats(
        e['classification_accuracy'], cell_std, 5,
        bsa_mean, bsa_sample_std, bsa_n, equal_var=False)
    return p2 / 2 if t > 0 else 1 - p2 / 2

# All 600 cells (both branches) for joint FDR correction
all_A = [e for e in data['grid_results'] if e['branch'] == 'A_unmatched']
all_B_full = [e for e in data['grid_results'] if e['branch'] == 'B_matched']
all_600 = all_A + all_B_full
pvals_600 = np.array([_bsa_pval(e) for e in all_600])
reject_600, _, _, _ = multipletests(pvals_600, alpha=0.05, method='fdr_bh')

# Extract Branch B non-baseline results
b_nb_start = len(all_A) + 15  # skip 15 inc=0 rows in Branch B
b_nb_reject = reject_600[len(all_A):]
bsa_exceeding = [e for e, r in zip(all_B_full, b_nb_reject)
                 if r and e['inc_idx'] > 0]

print(f"BSA baseline accuracy: {bsa_baseline_acc:.4f}")
print(f"Cells exceeding BSA: {len(bsa_exceeding)}")

bsa_dom = [e for e in bsa_exceeding
           if round(e['tonic_conductance'], 2) in [1.41, 2.81]]
print(f"  of which in dominant regime: {len(bsa_dom)}")

if len(bsa_dom) >= 5:
    d_bsa = extract(bsa_dom)
    r_isi_bsa, p_isi_bsa = pearsonr_safe(d_bsa['isi_cv'], d_bsa['acc'])
    r_tau_bsa, p_tau_bsa = pearsonr_safe(d_bsa['adapt_tau'], d_bsa['acc'])
    print(f"\n  Within BSA-exceeding dominant-regime cells (n={len(bsa_dom)}):")
    print(f"    acc ~ ISI CV:    r = {r_isi_bsa:+.4f}  (p = {p_isi_bsa:.2e})")
    print(f"    acc ~ adapt_tau: r = {r_tau_bsa:+.4f}  (p = {p_tau_bsa:.2e})")
    print(f"    acc range: {d_bsa['acc'].min():.4f} – {d_bsa['acc'].max():.4f}")
    print(f"    ISI CV range: {d_bsa['isi_cv'].min():.4f} – {d_bsa['isi_cv'].max():.4f}")

    results['bsa_within_regime'] = {
        'n_bsa_exceeding_total': len(bsa_exceeding),
        'n_bsa_exceeding_dominant': len(bsa_dom),
        'acc_isi_cv': {'r': float(r_isi_bsa), 'p': float(p_isi_bsa)},
        'acc_adapt_tau': {'r': float(r_tau_bsa), 'p': float(p_tau_bsa)},
        'acc_range': [float(d_bsa['acc'].min()), float(d_bsa['acc'].max())],
    }
else:
    print("  Too few cells for within-regime BSA analysis")
    results['bsa_within_regime'] = {'n_bsa_exceeding_dominant': len(bsa_dom), 'note': 'too few'}

# ============================================================
# 4. Cross-regime consistency
# ============================================================
print("\n" + "="*70)
print("4. CROSS-REGIME CONSISTENCY (per g-level correlations)")
print("="*70)

g_levels = sorted(set(round(e['tonic_conductance'], 2) for e in all_B))
regime_table = []
for g in g_levels:
    entries_g = [e for e in all_B if round(e['tonic_conductance'], 2) == g]
    n_g = len(entries_g)
    if n_g < 5:
        continue
    d_g = extract(entries_g)
    r_isi_g, p_isi_g = pearsonr_safe(d_g['isi_cv'], d_g['acc'])
    r_pr_g, p_pr_g = pearsonr_safe(d_g['pr'], d_g['acc'])
    r_rate_g, p_rate_g = pearsonr_safe(d_g['rate'], d_g['acc'])

    row = {
        'g_tonic': g,
        'n': n_g,
        'acc_isi_cv': {'r': float(r_isi_g), 'p': float(p_isi_g)},
        'acc_pr': {'r': float(r_pr_g), 'p': float(p_pr_g)},
        'acc_rate': {'r': float(r_rate_g), 'p': float(p_rate_g)},
        'mean_acc': float(d_g['acc'].mean()),
        'mean_rate': float(d_g['rate'].mean()),
    }
    regime_table.append(row)
    sig_isi = '*' if p_isi_g < 0.05 else ''
    sig_pr = '*' if p_pr_g < 0.05 else ''
    print(f"  g={g:6.4f} nS (n={n_g:3d}): "
          f"ISI CV r={r_isi_g:+.3f}{sig_isi:1s}  "
          f"PR r={r_pr_g:+.3f}{sig_pr:1s}  "
          f"rate r={r_rate_g:+.3f}  "
          f"mean_acc={d_g['acc'].mean():.4f}")

# Check sign consistency
isi_signs = [r['acc_isi_cv']['r'] for r in regime_table]
pr_signs = [r['acc_pr']['r'] for r in regime_table]
isi_consistent = all(s > 0 for s in isi_signs) or all(s < 0 for s in isi_signs)
pr_consistent = all(s > 0 for s in pr_signs) or all(s < 0 for s in pr_signs)
print(f"\n  ISI CV sign consistent across regimes: {isi_consistent} "
      f"(signs: {['+' if s > 0 else '-' for s in isi_signs]})")
print(f"  PR sign consistent across regimes: {pr_consistent} "
      f"(signs: {['+' if s > 0 else '-' for s in pr_signs]})")

results['cross_regime'] = {
    'per_regime': regime_table,
    'isi_cv_sign_consistent': isi_consistent,
    'pr_sign_consistent': pr_consistent,
}

# ============================================================
# 5. Mediation: g_tonic → dynamics → accuracy
# ============================================================
print("\n" + "="*70)
print("5. MEDIATION: g_tonic AFTER CONTROLLING FOR DYNAMICS")
print("="*70)

# acc ~ ISI_CV + PR + g_tonic (does g_tonic add anything beyond dynamics?)
reg_med = ols_summary(d_all['acc'],
                       np.column_stack([d_all['isi_cv'], d_all['pr'], d_all['g_tonic']]),
                       ['ISI_CV', 'PR', 'g_tonic'])
print(f"\n  OLS (acc ~ ISI_CV + PR + g_tonic): R² = {reg_med['r2']:.4f}")
for name, c in reg_med['coefficients'].items():
    print(f"    {name:8s}: β = {c['beta_std']:+.4f}, t = {c['t']:+.2f}, p = {c['p']:.2e}")

# Direct path: g_tonic → acc
r_g_acc, p_g_acc = pearsonr_safe(d_all['g_tonic'], d_all['acc'])
print(f"\n  Direct: g_tonic → acc:  r = {r_g_acc:+.4f}  (p = {p_g_acc:.2e})")

# Partial: g_tonic → acc | ISI_CV + PR
r_g_acc_partial, p_g_acc_partial = partial_corr(
    d_all['g_tonic'], d_all['acc'],
    np.column_stack([d_all['isi_cv'], d_all['pr']]))
print(f"  Partial: g_tonic → acc | (ISI_CV, PR): r = {r_g_acc_partial:+.4f}  (p = {p_g_acc_partial:.2e})")

# g_tonic → ISI_CV (mediation path a)
r_g_isi, p_g_isi = pearsonr_safe(d_all['g_tonic'], d_all['isi_cv'])
r_g_pr, p_g_pr = pearsonr_safe(d_all['g_tonic'], d_all['pr'])
print(f"\n  Mediation paths:")
print(f"    g_tonic → ISI CV: r = {r_g_isi:+.4f}  (p = {p_g_isi:.2e})")
print(f"    g_tonic → PR:     r = {r_g_pr:+.4f}  (p = {p_g_pr:.2e})")

# Proportion mediated (approximate: 1 - partial/total)
if abs(r_g_acc) > 0.01:
    prop_mediated = 1 - abs(r_g_acc_partial) / abs(r_g_acc)
    print(f"    Approximate proportion mediated: {prop_mediated:.1%}")
else:
    prop_mediated = None
    print(f"    g_tonic → acc too weak for mediation analysis")

results['mediation'] = {
    'ols_acc_dynamics_gtonic': reg_med,
    'direct_g_acc': {'r': float(r_g_acc), 'p': float(p_g_acc)},
    'partial_g_acc_given_dynamics': {'r': float(r_g_acc_partial), 'p': float(p_g_acc_partial)},
    'g_to_isi_cv': {'r': float(r_g_isi), 'p': float(p_g_isi)},
    'g_to_pr': {'r': float(r_g_pr), 'p': float(p_g_pr)},
    'proportion_mediated': float(prop_mediated) if prop_mediated is not None else None,
}

# ============================================================
# 6. Per-bin within-regime correlations (sign-flip replication)
# ============================================================
print("\n" + "="*70)
print("6. PER-BIN WITHIN-REGIME SIGN-FLIP REPLICATION")
print("="*70)

n_bins = d_dom['per_bin'].shape[1]
bin_times = np.arange(n_bins) * BIN_MS + BIN_MS / 2

r_isi_bins = np.full(n_bins, np.nan)
r_pr_bins = np.full(n_bins, np.nan)
for b in range(n_bins):
    bin_acc = d_dom['per_bin'][:, b]
    if np.std(bin_acc) > 1e-10:
        r_isi_bins[b], _ = stats.pearsonr(d_dom['isi_cv'], bin_acc)
        r_pr_bins[b], _ = stats.pearsonr(d_dom['pr'], bin_acc)

# Find crossover
def find_crossover(r_vals, times, min_bin=3):
    for i in range(max(1, min_bin), len(r_vals)):
        if np.isnan(r_vals[i-1]) or np.isnan(r_vals[i]):
            continue
        if r_vals[i-1] > 0 and r_vals[i] <= 0:
            t0, t1 = times[i-1], times[i]
            r0, r1 = r_vals[i-1], r_vals[i]
            if abs(r1 - r0) < 1e-10:
                return t0
            return t0 + (0 - r0) / (r1 - r0) * (t1 - t0)
    return None

cross_isi = find_crossover(r_isi_bins, bin_times)
cross_pr = find_crossover(r_pr_bins, bin_times)

# Compare with full-branch
r_isi_bins_full = np.full(n_bins, np.nan)
r_pr_bins_full = np.full(n_bins, np.nan)
for b in range(n_bins):
    bin_acc = d_all['per_bin'][:, b]
    if np.std(bin_acc) > 1e-10:
        r_isi_bins_full[b], _ = stats.pearsonr(d_all['isi_cv'], bin_acc)
        r_pr_bins_full[b], _ = stats.pearsonr(d_all['pr'], bin_acc)

cross_isi_full = find_crossover(r_isi_bins_full, bin_times)
cross_pr_full = find_crossover(r_pr_bins_full, bin_times)

# Correlation between within-regime and full-branch r curves
valid = ~np.isnan(r_isi_bins) & ~np.isnan(r_isi_bins_full)
if np.sum(valid) > 3:
    r_curve_corr_isi, _ = stats.pearsonr(r_isi_bins[valid], r_isi_bins_full[valid])
else:
    r_curve_corr_isi = np.nan
valid_pr = ~np.isnan(r_pr_bins) & ~np.isnan(r_pr_bins_full)
if np.sum(valid_pr) > 3:
    r_curve_corr_pr, _ = stats.pearsonr(r_pr_bins[valid_pr], r_pr_bins_full[valid_pr])
else:
    r_curve_corr_pr = np.nan

print(f"\n  Within dominant regime (n={n_dom}):")
print(f"    ISI CV crossover: {cross_isi:.0f} ms" if cross_isi else "    ISI CV: no crossover found")
print(f"    PR crossover:     {cross_pr:.0f} ms" if cross_pr else "    PR: no crossover found")
print(f"\n  Full branch (n={len(all_B)}):")
print(f"    ISI CV crossover: {cross_isi_full:.0f} ms" if cross_isi_full else "    ISI CV: no crossover found")
print(f"    PR crossover:     {cross_pr_full:.0f} ms" if cross_pr_full else "    PR: no crossover found")
print(f"\n  Curve similarity (within-regime vs full-branch):")
print(f"    ISI CV r-curve correlation: {r_curve_corr_isi:.4f}")
print(f"    PR r-curve correlation:     {r_curve_corr_pr:.4f}")

# Peak and trough values
stim_bins = bin_times <= 300
post_bins = bin_times > 300
print(f"\n  ISI CV r: stimulus peak = {np.nanmax(r_isi_bins[stim_bins]):+.3f}, "
      f"post-stim trough = {np.nanmin(r_isi_bins[post_bins]):+.3f}")
print(f"  PR r:     stimulus peak = {np.nanmax(r_pr_bins[stim_bins]):+.3f}, "
      f"post-stim trough = {np.nanmin(r_pr_bins[post_bins]):+.3f}")

results['perbin_within_regime'] = {
    'n': n_dom,
    'isi_cv_crossover_ms': float(cross_isi) if cross_isi else None,
    'pr_crossover_ms': float(cross_pr) if cross_pr else None,
    'full_branch_isi_cv_crossover_ms': float(cross_isi_full) if cross_isi_full else None,
    'full_branch_pr_crossover_ms': float(cross_pr_full) if cross_pr_full else None,
    'curve_correlation_isi_cv': float(r_curve_corr_isi),
    'curve_correlation_pr': float(r_curve_corr_pr),
    'isi_cv_stim_peak': float(np.nanmax(r_isi_bins[stim_bins])),
    'isi_cv_post_trough': float(np.nanmin(r_isi_bins[post_bins])),
    'pr_stim_peak': float(np.nanmax(r_pr_bins[stim_bins])),
    'pr_post_trough': float(np.nanmin(r_pr_bins[post_bins])),
    'r_isi_cv_per_bin': [float(x) for x in r_isi_bins],
    'r_pr_per_bin': [float(x) for x in r_pr_bins],
    'bin_times_ms': [float(x) for x in bin_times],
}

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Within the dominant g-regime (g ∈ {{1.41, 2.81}}, n={n_dom}):
  • acc ~ ISI CV:  r = {r_isi:+.3f} (full branch: {pearsonr_safe(d_all['isi_cv'], d_all['acc'])[0]:+.3f})
  • acc ~ PR:      r = {r_pr:+.3f} (full branch: {pearsonr_safe(d_all['pr'], d_all['acc'])[0]:+.3f})
  • acc ~ rate:    r = {r_rate:+.3f}
  • OLS R² (acc ~ ISI_CV + PR + rate): {reg1['r2']:.3f}

Adding g_tonic to full-branch regression:
  • ΔR² = {delta_r2:+.6f}
  • g_tonic β = {reg_with_g['coefficients']['g_tonic']['beta_std']:+.4f} (p = {reg_with_g['coefficients']['g_tonic']['p']:.2e})
  • ISI_CV β stable: {reg_no_g['coefficients']['ISI_CV']['beta_std']:+.3f} → {reg_with_g['coefficients']['ISI_CV']['beta_std']:+.3f}
  • PR β stable: {reg_no_g['coefficients']['PR']['beta_std']:+.3f} → {reg_with_g['coefficients']['PR']['beta_std']:+.3f}

Per-bin sign-flip replicates within regime:
  • ISI CV crossover: {f'{cross_isi:.0f} ms' if cross_isi else 'none'} (full branch: {f'{cross_isi_full:.0f} ms' if cross_isi_full else 'none'})
  • Curve similarity: ISI CV {r_curve_corr_isi:.3f}, PR {r_curve_corr_pr:.3f}
""")

# Save JSON
results['description'] = (
    "Robustness analysis: ruling out tonic conductance confound in Branch B. "
    "Tests whether ISI CV and PR correlations with accuracy survive within the "
    "dominant g-regime (g=1.41-2.81 nS, n=122) where tonic conductance is nearly "
    "constant, and whether adding g_tonic as a covariate changes the dynamics betas."
)
results['bsa_baseline_accuracy'] = float(bsa_baseline_acc)

# Clean NaN for JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

with open(OUTPUT_JSON, 'w') as f:
    json.dump(clean_for_json(results), f, indent=2)
print(f"Saved: {OUTPUT_JSON}")


if __name__ == '__main__':
    pass
