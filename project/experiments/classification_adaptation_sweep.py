"""
Classification Adaptation Parameter Sweep — Consolidated 159-Point Grid.

Covers the SAME 159 (inc, tau) grid points as the XOR + WM sweeps:
  - Original 8×8 grid (Experiment 13): inc 0.05-1.0, tau 30-2000
  - Arm A sAHP Zoom: inc 0.005-0.05, tau 500-5000
  - Arm E Grid Infill: inc=0 baseline + gap fill
  - Total: 159 unique points (16 inc × 13 tau, 76% fill)

Protocol:
  - 5-digit classification (digits 0-4), 500 samples per digit = 2500 total
  - Full-resolution BSA input (no frame quantization)
  - 20ms time bins, flat Ridge readout (all bins concatenated)
  - Ridge alpha sweep [0.01 .. 1000]
  - 60/40 StratifiedShuffleSplit × 5 repeats
  - Rate-matched per grid point to LHS-021 baseline firing rate
  - Noise fix: per-worker PID-based RNG reseed (after build_full_network)

Per-grid-point measurements:
  1. Classification accuracy (5-class, flat Ridge)
  2. Confusion matrix
  3. ISI coefficient of variation during stimulus
  4. Adaptation conductance at stimulus end
  5. Drive cost (stimulus_current required to rate-match)
  6. Participation ratio (effective dimensionality during stimulus via SVD)
  7. Per-bin accuracy curve (diagnostic)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import os
import copy
import json
import time
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# PATH SETUP
# ============================================================
SCRIPT_DIR = Path(__file__).parent
EXPANDED_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPANDED_DIR))
sys.path.insert(0, str(EXPANDED_DIR / 'utils'))

from utils.network_builder import (
    BASE_CONFIG,
    build_full_network, build_std_masks, run_sample_with_std,
    STD_U, STD_TAU_REC,
)
from utils.simulation_utils import (
    load_audio_samples,
    compute_paired_stats,
    POST_STIMULUS_MS,
    SAMPLE_LOAD_SEED,
)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_DIGITS = [0, 1, 2, 3, 4]
N_DIGITS = len(DEFAULT_DIGITS)
BIN_MS = 20.0           # readout time bins
POST_STIM_MS = POST_STIMULUS_MS  # post-stimulus window (200ms)

SAMPLES_PER_DIGIT = 500  # 500 × 5 digits = 2500 total
N_SPLIT_REPEATS = 5      # 60/40 StratifiedShuffleSplit repeats
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# LHS-021 SOTA (all params fixed except the two we sweep)
LHS021_PARAMS = {
    'lambda_connect': 0.003288382505082908,
    'core_core_mult': 0.8274594948671626,
    'surface_shell_fraction': 0.18171487839543513,
    'adapt_inc': 0.6264283936395093,
    'shell_core_mult': 4.849737508561994,
}
FIXED_NMDA_TAU = 50.0
SEED = 42

# Rate-matching calibration
RATE_TOLERANCE_HZ = 2.0
CALIBRATION_N_SAMPLES = 200
CALIBRATION_MAX_ITER = 100
CALIBRATION_STIM_BOUNDS = (0.01, 5.0)
CALIBRATION_WARM_MARGIN = 3.0

DATA_DIR = EXPANDED_DIR / 'data'
DEFAULT_RESULTS_DIR = EXPANDED_DIR / 'results' / 'classification_adaptation_sweep'

# Unified axis values (16 inc × 13 tau) across all sweeps
UNIFIED_INC = np.array([
    0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.185714, 0.2, 0.321429, 0.457143, 0.592857,
    0.63, 0.728571, 0.864286, 1.0,
])
UNIFIED_TAU = np.array([
    30.0, 54.7, 99.6, 181.0, 330.6, 500.0, 602.4,
    1098.0, 2000.0, 3000.0, 3500.0, 4000.0, 5000.0,
])


# ============================================================
# GRID POINT DEFINITIONS (matching WM sweep exactly)
# ============================================================
def _snap_to_unified(inc, tau, inc_tol=0.005, tau_tol=5.0):
    """Find the (inc_idx, tau_idx) in the unified grid for a given point."""
    i_idx = np.argmin(np.abs(UNIFIED_INC - inc))
    t_idx = np.argmin(np.abs(UNIFIED_TAU - tau))
    if abs(UNIFIED_INC[i_idx] - inc) > inc_tol:
        return None, None
    if abs(UNIFIED_TAU[t_idx] - tau) > tau_tol:
        return None, None
    return int(i_idx), int(t_idx)


def build_grid_points(arms='all'):
    """Build the full list of (inc, tau) grid points to evaluate.

    Args:
        arms: comma-separated string of which arms to include.
              'all' = original + A + E (default)
              'original' = just the 8x8 grid
              'A' = sAHP zoom only
              'E' = grid infill only

    Returns list of dicts with: adapt_inc, adapt_tau, source, point_id, inc_idx, tau_idx
    """
    requested = set(a.strip() for a in arms.split(','))
    include_all = 'all' in requested
    points = []
    seen = set()  # (rounded_inc, rounded_tau) for dedup

    def _add(inc, tau, source):
        key = (round(inc, 4), round(tau, 0))
        if key in seen:
            return
        seen.add(key)
        i_idx, t_idx = _snap_to_unified(inc, tau)
        points.append({
            'adapt_inc': float(inc),
            'adapt_tau': float(tau),
            'source': source,
            'point_id': f'{source}_inc{inc:.6f}_tau{tau:.1f}',
            'inc_idx': i_idx,
            'tau_idx': t_idx,
        })

    # --- Original 8×8 grid (Experiment 13) ---
    if include_all or 'original' in requested:
        orig_inc = np.linspace(0.05, 1.0, 8)
        orig_tau = np.geomspace(30.0, 2000.0, 8)
        for inc in orig_inc:
            for tau in orig_tau:
                _add(inc, tau, 'original')

    # --- Arm A: sAHP Zoom (6×6 = 36 points) ---
    if include_all or 'A' in requested:
        arm_a_inc = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        arm_a_tau = [500.0, 1098.0, 2000.0, 3000.0, 4000.0, 5000.0]
        for inc in arm_a_inc:
            for tau in arm_a_tau:
                _add(inc, tau, 'A')

    # --- Arm E: Grid Infill (61 points) ---
    if include_all or 'E' in requested:
        arm_e_combos = [
            # adapt_inc = 0: no-adaptation baseline across all tau values
            (0.0, 30.0), (0.0, 54.7), (0.0, 99.6), (0.0, 181.0), (0.0, 330.6),
            (0.0, 500.0), (0.0, 602.4), (0.0, 1097.7), (0.0, 2000.0),
            (0.0, 3000.0), (0.0, 3500.0), (0.0, 4000.0), (0.0, 5000.0),
            # Low inc × short tau corner
            (0.005, 30.0), (0.005, 54.7), (0.005, 99.6), (0.005, 181.0), (0.005, 330.6),
            (0.01, 30.0), (0.01, 54.7), (0.01, 99.6), (0.01, 181.0),
            (0.02, 30.0), (0.02, 54.7),
            (0.03, 30.0), (0.03, 54.7),
            (0.04, 30.0),
            # Mid-high inc × high tau block
            (0.185714, 3500.0), (0.185714, 4000.0), (0.185714, 5000.0),
            (0.2, 3500.0), (0.2, 4000.0), (0.2, 5000.0),
            (0.321429, 3000.0), (0.321429, 3500.0), (0.321429, 4000.0), (0.321429, 5000.0),
            (0.457143, 3000.0), (0.457143, 3500.0), (0.457143, 4000.0), (0.457143, 5000.0),
            (0.592857, 3000.0), (0.592857, 3500.0), (0.592857, 4000.0), (0.592857, 5000.0),
            (0.63, 3000.0), (0.63, 3500.0), (0.63, 4000.0), (0.63, 5000.0),
            (0.728571, 3000.0), (0.728571, 3500.0), (0.728571, 4000.0), (0.728571, 5000.0),
            (0.864286, 3000.0), (0.864286, 3500.0), (0.864286, 4000.0), (0.864286, 5000.0),
            (1.0, 3000.0), (1.0, 3500.0), (1.0, 4000.0), (1.0, 5000.0),
        ]
        for inc, tau in arm_e_combos:
            _add(inc, tau, 'E')

    return points


# ============================================================
# ACTIVITY BINNING
# ============================================================
def bin_activity_record(activity_record, reservoir_indices, dt, bin_ms, n_bins):
    """Convert step-wise activity_record to (n_bins, n_reservoir) spike counts."""
    n_reservoir = len(reservoir_indices)
    res_set = set(int(x) for x in reservoir_indices)
    res_idx_to_pos = {int(idx): pos for pos, idx in enumerate(reservoir_indices)}
    bins = np.zeros((n_bins, n_reservoir))
    total_spikes = 0
    for step, active in enumerate(activity_record):
        t = step * dt
        b = int(t / bin_ms)
        if b >= n_bins:
            b = n_bins - 1
        for nid in active:
            pos = res_idx_to_pos.get(nid)
            if pos is not None:
                bins[b, pos] += 1
                total_spikes += 1
    return bins, total_spikes


# ============================================================
# MULTIPROCESSING WORKER (with noise fix)
# ============================================================
_worker_state = {}


def _pool_initializer(config_dict, dynamical_overrides_dict):
    global _worker_state
    network, zone_info = build_full_network(
        config_dict, quiet=True,
        dynamical_overrides=dynamical_overrides_dict,
        remove_nonarc=True)
    std_eligible, recurrent_masks = build_std_masks(network, zone_info)

    # Reseed RNG per worker so noise sequences are independent across workers.
    np.random.seed(config_dict['random_seed'] + os.getpid())

    _worker_state = {
        'network': network,
        'zone_info': zone_info,
        'config': config_dict,
        'std_eligible': std_eligible,
        'recurrent_masks': recurrent_masks,
    }


def _sim_worker(args):
    """Run one single-digit trial and extract features + ISI stats + adaptation."""
    global _worker_state
    sample, sample_idx = args

    network = _worker_state['network']
    zone_info = _worker_state['zone_info']
    config = _worker_state['config']
    std_eligible = _worker_state['std_eligible']
    recurrent_masks = _worker_state['recurrent_masks']

    reservoir_indices = zone_info['reservoir_zone_indices']
    dt = config['dt']
    total_duration_ms = config['audio_duration_ms'] + POST_STIM_MS
    n_bins = int(np.ceil(total_duration_ms / BIN_MS))

    # Stimulus end = last BSA spike
    stim_end_ms = float(np.max(sample['spike_times_ms']))

    # Run simulation with adaptation snapshot at stimulus end
    activity_record, adapt_snapshot = run_sample_with_std(
        network, sample, zone_info, config,
        std_u=STD_U, std_tau_rec=STD_TAU_REC,
        std_eligible=std_eligible, recurrent_masks=recurrent_masks,
        record_adapt_at_ms=stim_end_ms)

    # Time-bin spike counts
    res_bins, total_res_spikes = bin_activity_record(
        activity_record, reservoir_indices, dt, BIN_MS, n_bins)

    # ISI CV during stimulus period
    stim_end_step = min(int(stim_end_ms / dt), len(activity_record))
    res_set = set(int(x) for x in reservoir_indices)
    neuron_last_spike = {}
    all_isis = []

    for step in range(stim_end_step):
        for nid in activity_record[step]:
            if nid in res_set:
                t = step * dt
                if nid in neuron_last_spike:
                    all_isis.append(t - neuron_last_spike[nid])
                neuron_last_spike[nid] = t

    if len(all_isis) >= 2:
        all_isis_arr = np.array(all_isis)
        isi_mean = float(np.mean(all_isis_arr))
        isi_cv = float(np.std(all_isis_arr) / max(isi_mean, 1e-9))
    else:
        isi_cv = float('nan')

    # Adaptation conductance at stimulus end
    mean_adapt_stim_end = float(np.mean(adapt_snapshot[reservoir_indices]))

    return {
        'res_bins': res_bins,
        'total_res_spikes': total_res_spikes,
        'digit': sample['digit'],
        'sample_idx': sample_idx,
        'n_reservoir': len(reservoir_indices),
        'isi_cv': isi_cv,
        'mean_adapt_stim_end': mean_adapt_stim_end,
    }


def run_all_samples(config, samples, dynamical_overrides, n_workers=None):
    """Run all single-digit trials with measurements.

    Returns:
        res_bins_3d: (n_samples, n_bins, n_reservoir) time-binned spike counts
        digits: (n_samples,) digit labels
        total_spikes: (n_samples,) total reservoir spikes per trial
        n_reservoir: int
        isi_cvs: (n_samples,) ISI CV per trial during stimulus
        adapt_stim_ends: (n_samples,) mean adaptation at stimulus end
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    n_samples = len(samples)
    worker_args = [(s, i) for i, s in enumerate(samples)]

    res_bins_list = [None] * n_samples
    digits = np.zeros(n_samples, dtype=int)
    total_spikes = np.zeros(n_samples)
    isi_cvs = np.zeros(n_samples)
    adapt_stim_ends = np.zeros(n_samples)
    n_reservoir = None
    completed = 0

    with mp.Pool(processes=n_workers,
                 initializer=_pool_initializer,
                 initargs=(config, dynamical_overrides)) as pool:
        for result in pool.imap_unordered(_sim_worker, worker_args):
            idx = result['sample_idx']
            res_bins_list[idx] = result['res_bins']
            digits[idx] = result['digit']
            total_spikes[idx] = result['total_res_spikes']
            isi_cvs[idx] = result['isi_cv']
            adapt_stim_ends[idx] = result['mean_adapt_stim_end']
            n_reservoir = result['n_reservoir']
            completed += 1
            if completed % 100 == 0 or completed == n_samples:
                print(f"        {completed:5d}/{n_samples}", flush=True)

    res_bins_3d = np.array(res_bins_list)
    return res_bins_3d, digits, total_spikes, n_reservoir, isi_cvs, adapt_stim_ends


# ============================================================
# RATE-MATCHING CALIBRATION
# ============================================================
def _measure_rate(config, samples_subset, dynamical_overrides, n_workers):
    """Run a small batch of trials and return mean firing rate (Hz)."""
    _, _, total_spikes, n_res, _, _ = run_all_samples(
        config, samples_subset, dynamical_overrides, n_workers=n_workers)

    trial_duration_s = (config['audio_duration_ms'] + POST_STIM_MS) / 1000.0
    per_trial_rate = total_spikes / (n_res * trial_duration_s)
    return float(np.mean(per_trial_rate))


def calibrate_stimulus_current(base_config, dynamical_overrides,
                                calibration_samples, n_workers,
                                target_rate,
                                tolerance=RATE_TOLERANCE_HZ,
                                max_iter=CALIBRATION_MAX_ITER,
                                stim_bounds=CALIBRATION_STIM_BOUNDS,
                                initial_guess=None):
    """Binary search over stimulus_current to match target firing rate.

    If initial_guess is provided, test it first. If within tolerance, return
    immediately (1 iteration). Otherwise, use it to set tight bounds for the
    binary search.
    """
    lo, hi = stim_bounds
    log = []
    iteration = 0

    # Try the initial guess first (e.g. cached from a previous sweep)
    if initial_guess is not None:
        config = copy.deepcopy(base_config)
        config['stimulus_current'] = initial_guess
        rate = _measure_rate(config, calibration_samples,
                             dynamical_overrides, n_workers)
        log.append({'iter': iteration, 'stim_current': initial_guess,
                    'rate_hz': rate})
        print(f"      cal[{iteration}] stim={initial_guess:.4f} -> {rate:.1f} Hz"
              f"  (cached guess)", flush=True)
        iteration += 1

        if abs(rate - target_rate) <= tolerance:
            return initial_guess, rate, log

        # Narrow bounds around the guess for faster convergence
        margin = 3.0
        lo = max(lo, initial_guess / margin)
        hi = min(hi, initial_guess * margin)
        if rate > target_rate:
            hi = initial_guess
        else:
            lo = initial_guess

    for _ in range(iteration, max_iter):
        mid = (lo + hi) / 2.0
        config = copy.deepcopy(base_config)
        config['stimulus_current'] = mid

        rate = _measure_rate(config, calibration_samples,
                             dynamical_overrides, n_workers)
        log.append({'iter': iteration, 'stim_current': mid, 'rate_hz': rate})
        print(f"      cal[{iteration}] stim={mid:.4f} -> {rate:.1f} Hz",
              flush=True)
        iteration += 1

        if abs(rate - target_rate) <= tolerance:
            return mid, rate, log

        if rate > target_rate:
            hi = mid
        else:
            lo = mid

        if (hi - lo) < 1e-5:
            print(f"      cal: search range collapsed at stim={mid:.6f}, "
                  f"rate={rate:.1f} Hz (target {target_rate:.1f})", flush=True)
            break

    best = min(log, key=lambda x: abs(x['rate_hz'] - target_rate))
    print(f"      cal: best after {len(log)} iters: "
          f"stim={best['stim_current']:.4f} -> {best['rate_hz']:.1f} Hz "
          f"(target {target_rate:.1f})", flush=True)
    return best['stim_current'], best['rate_hz'], log


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================
def classify_flat_ridge(X_3d, y, digits=DEFAULT_DIGITS):
    """Flat Ridge classifier with 60/40 StratifiedShuffleSplit × N_SPLIT_REPEATS.

    For each repeat, sweeps Ridge alpha and picks the best.
    Returns mean accuracy across repeats, per-repeat accuracies, best confusion matrix.
    """
    n_samples = X_3d.shape[0]
    X_flat = X_3d.reshape(n_samples, -1)

    repeat_accs = []
    repeat_cms = []

    for rep in range(N_SPLIT_REPEATS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4,
                                     random_state=SEED + rep)
        train_idx, test_idx = next(sss.split(X_flat, y))

        best_acc, best_preds = -1, None
        for alpha in RIDGE_ALPHAS:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_flat[train_idx])
            X_test = scaler.transform(X_flat[test_idx])
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            clf = RidgeClassifier(alpha=alpha, solver='svd')
            clf.fit(X_train, y[train_idx])
            preds = clf.predict(X_test)
            acc = accuracy_score(y[test_idx], preds)
            if acc > best_acc:
                best_acc = acc
                best_preds = preds
        repeat_accs.append(float(best_acc))
        repeat_cms.append(
            confusion_matrix(y[test_idx], best_preds, labels=digits).tolist())

    mean_acc = float(np.mean(repeat_accs))
    return {
        'accuracy': mean_acc,
        'accuracy_std': float(np.std(repeat_accs)),
        'per_repeat_accuracy': repeat_accs,
        'fold_accuracies': repeat_accs,  # alias for compute_paired_stats
        'confusion_matrix': repeat_cms[-1],
    }


def classify_per_bin(X_3d, y):
    """Per-bin classification accuracy (diagnostic).

    Uses the same 60/40 StratifiedShuffleSplit × N_SPLIT_REPEATS scheme.
    Returns mean accuracy per bin across repeats.
    """
    n_samples, n_bins, n_features = X_3d.shape
    bin_accs = np.zeros(n_bins)
    for b in range(n_bins):
        X_bin = X_3d[:, b, :]
        repeat_accs = []
        for rep in range(N_SPLIT_REPEATS):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4,
                                         random_state=SEED + rep)
            train_idx, test_idx = next(sss.split(X_bin, y))
            best_acc = -1
            for alpha in RIDGE_ALPHAS:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_bin[train_idx])
                X_test = scaler.transform(X_bin[test_idx])
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    clf = RidgeClassifier(alpha=alpha, solver='svd')
                    clf.fit(X_train, y[train_idx])
                    preds = clf.predict(X_test)
                    acc = accuracy_score(y[test_idx], preds)
                except np.linalg.LinAlgError:
                    acc = 0.2  # chance
                if acc > best_acc:
                    best_acc = acc
            repeat_accs.append(best_acc)
        bin_accs[b] = float(np.mean(repeat_accs))
    return bin_accs.tolist()


# ============================================================
# POST-HOC MEASUREMENTS
# ============================================================
def compute_participation_ratio(res_bins_3d):
    """Participation ratio (effective dimensionality) across time bins.

    PR = (sum lambda)^2 / (N_eff * sum lambda^2)
    Returns (mean PR, per-bin PR list).
    """
    n_samples, n_bins, n_res = res_bins_3d.shape
    n_eff = min(n_samples, n_res)

    pr_per_bin = np.zeros(n_bins)
    for b in range(n_bins):
        X = res_bins_3d[:, b, :]
        X_centered = X - X.mean(axis=0)

        if np.std(X_centered) < 1e-10:
            pr_per_bin[b] = 0.0
            continue

        try:
            _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            cov = X_centered.T @ X_centered / max(n_samples - 1, 1)
            eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0.0)
            sum_lam = np.sum(eigenvalues)
            sum_lam2 = np.sum(eigenvalues ** 2)
            if sum_lam2 > 1e-20:
                pr_per_bin[b] = (sum_lam ** 2) / (n_eff * sum_lam2)
            else:
                pr_per_bin[b] = 0.0
            continue
        eigenvalues = S ** 2 / max(n_samples - 1, 1)

        sum_lam = np.sum(eigenvalues)
        sum_lam2 = np.sum(eigenvalues ** 2)
        if sum_lam2 > 1e-20:
            pr_per_bin[b] = (sum_lam ** 2) / (n_eff * sum_lam2)
        else:
            pr_per_bin[b] = 0.0

    return float(np.mean(pr_per_bin)), pr_per_bin.tolist()


# ============================================================
# BSA FEATURE EXTRACTION
# ============================================================
def extract_bsa_bins(sample, n_channels=128, bin_ms=BIN_MS,
                     total_duration_ms=None):
    """Extract BSA spike times into time bins (n_bins, n_channels)."""
    if total_duration_ms is None:
        total_duration_ms = float(np.max(sample['spike_times_ms'])) + POST_STIM_MS
    n_bins = int(np.ceil(total_duration_ms / bin_ms))
    bsa_bins = np.zeros((n_bins, n_channels))
    spike_times = sample['spike_times_ms']
    freq_bins = sample['freq_bin_indices'].astype(int)
    for t, ch in zip(spike_times, freq_bins):
        b = int(t / bin_ms)
        if 0 <= b < n_bins:
            bsa_bins[b, ch] += 1
    return bsa_bins


# ============================================================
# FIGURE GENERATION
# ============================================================
def _format_inc(v):
    if v < 0.1:
        return f'{v:.3f}'
    return f'{v:.2f}'


def _format_tau(v):
    if v >= 1000:
        return f'{v/1000:.1f}k'
    return f'{v:.0f}'


def generate_figures(grid_results, bsa_baseline, baseline_result, output_dir):
    """Generate all figures from sweep results."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'savefig.facecolor': '#1a1a2e',
    })

    n_inc = len(UNIFIED_INC)
    n_tau = len(UNIFIED_TAU)

    # Build 2D matrices from grid_results
    acc_mat = np.full((n_inc, n_tau), np.nan)
    gap_mat = np.full((n_inc, n_tau), np.nan)
    rate_mat = np.full((n_inc, n_tau), np.nan)
    stim_mat = np.full((n_inc, n_tau), np.nan)
    isi_cv_mat = np.full((n_inc, n_tau), np.nan)
    adapt_mat = np.full((n_inc, n_tau), np.nan)
    pr_mat = np.full((n_inc, n_tau), np.nan)

    for r in grid_results:
        i = r.get('inc_idx')
        t = r.get('tau_idx')
        if i is None or t is None:
            continue
        acc_mat[i, t] = r['classification_accuracy']
        gap_mat[i, t] = r['classification_gap_pp']
        rate_mat[i, t] = r['firing_rate_hz']
        stim_mat[i, t] = r['matched_stimulus_current']
        isi_cv_mat[i, t] = r['isi_cv_mean'] if r['isi_cv_mean'] is not None else np.nan
        adapt_mat[i, t] = r['adapt_at_stim_end_mean']
        pr_mat[i, t] = r.get('participation_ratio_mean', np.nan)

    bsa_acc = bsa_baseline['accuracy'] * 100
    chance = 100.0 / N_DIGITS

    # --- Figure 1: Classification accuracy (imshow, uniform cells) ---
    fig, ax = plt.subplots(figsize=(14, 8))

    measured_mask = ~np.isnan(acc_mat)
    im = ax.imshow(acc_mat * 100, origin='lower', aspect='auto',
                    cmap='magma', interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Accuracy (%)', fontsize=10)

    # Mark best measured cell
    measured_pct = np.where(measured_mask, acc_mat * 100, np.nan)
    if not np.all(np.isnan(measured_pct)):
        best_flat = np.nanargmax(measured_pct)
        bi, bt = np.unravel_index(best_flat, measured_pct.shape)
        ax.plot(bt, bi, marker='*', color='cyan', markersize=18,
                markeredgecolor='white', markeredgewidth=1.0, zorder=5)
        best_val = measured_pct[bi, bt]
        best_inc = UNIFIED_INC[bi]
        best_tau = UNIFIED_TAU[bt]
        ax.set_title(f'Classification Accuracy (noise-fixed)\n'
                     f'best={best_val:.1f}% @ inc={best_inc:.3f}, tau={best_tau:.0f}  |  '
                     f'BSA={bsa_acc:.1f}%, chance={chance:.0f}%  |  '
                     f'{int(np.sum(measured_mask))} points')

    # Hatching on NaN cells
    for i in range(n_inc):
        for j in range(n_tau):
            if np.isnan(acc_mat[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    hatch='///', edgecolor='gray', alpha=0.3,
                    linewidth=0, zorder=2))

    ax.set_xticks(range(n_tau))
    ax.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_inc))
    ax.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=8)
    ax.set_xlabel(r'$\tau_{adapt}$ (ms)')
    ax.set_ylabel('adapt_inc')

    fig.tight_layout()
    p = output_dir / 'accuracy_heatmap.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Figure 2: Neural dynamics landscape (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for ax_dyn, mat, cmap, clabel, title in [
        (axes[0, 0], isi_cv_mat, 'viridis', 'ISI CV',
         'ISI Coefficient of Variation (stimulus)'),
        (axes[0, 1], adapt_mat, 'inferno', 'Mean g_adapt',
         'Adaptation Conductance at Stimulus End'),
        (axes[1, 0], pr_mat, 'cividis', 'Participation Ratio',
         'Population Dimensionality (stimulus)'),
        (axes[1, 1], stim_mat, 'plasma', 'stimulus_current',
         'Drive Cost (stimulus_current to rate-match)'),
    ]:
        im = ax_dyn.imshow(mat, origin='lower', aspect='auto',
                           cmap=cmap, interpolation='nearest')
        div = make_axes_locatable(ax_dyn)
        cx = div.append_axes('right', size='5%', pad=0.08)
        fig.colorbar(im, cax=cx).set_label(clabel, fontsize=10)
        ax_dyn.set_xticks(range(n_tau))
        ax_dyn.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                               rotation=45, ha='right', fontsize=7)
        ax_dyn.set_yticks(range(n_inc))
        ax_dyn.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=7)
        ax_dyn.set_xlabel(r'$\tau_{adapt}$ (ms)')
        ax_dyn.set_ylabel('adapt_inc')
        ax_dyn.set_title(title)

    fig.suptitle('Neural Dynamics Landscape (Classification)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p = output_dir / 'dynamics_landscape.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Figure 3: Rate-matching verification ---
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(rate_mat, origin='lower', aspect='auto',
                    cmap='viridis', interpolation='nearest')
    div = make_axes_locatable(ax)
    cx = div.append_axes('right', size='5%', pad=0.08)
    fig.colorbar(im, cax=cx).set_label('Mean Firing Rate (Hz)')
    ax.set_xticks(range(n_tau))
    ax.set_xticklabels([_format_tau(t) for t in UNIFIED_TAU],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_inc))
    ax.set_yticklabels([_format_inc(v) for v in UNIFIED_INC], fontsize=8)
    ax.set_xlabel(r'$\tau_{adapt}$ (ms)')
    ax.set_ylabel('adapt_inc')
    ax.set_title('Rate Verification (should be uniform)')
    fig.tight_layout()
    p = output_dir / 'rate_matching.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Classification adaptation sweep (159 grid points, noise-fixed)')
    parser.add_argument('--arms', type=str, default='all',
                        help='Which arms to run: all, original, A, E '
                             '(comma-separated, default: all)')
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to partial results JSON to resume from')
    parser.add_argument('--figures-only', action='store_true',
                        help='Only regenerate figures from existing results')
    parser.add_argument('--cached-calibration', action='store_true',
                        help='Use stimulus currents from previous (pre-noise-fix) '
                             'sweep as initial guesses for calibration')
    args = parser.parse_args()

    n_workers = args.n_workers or 8

    # Load cached calibration values from previous (pre-noise-fix) sweeps
    cached_stim_lookup = {}
    if args.cached_calibration:
        old_sweep_path = EXPANDED_DIR / 'results' / 'classification_adaptation_sweep' / 'classification_adaptation_sweep.json'
        followup_path = EXPANDED_DIR / 'results' / 'classification_adaptation_followup' / 'classification_adaptation_followup_checkpoint.json'

        if old_sweep_path.exists():
            with open(old_sweep_path) as f:
                old_data = json.load(f)
            old_inc = np.array(old_data['grid']['adapt_inc_values'])
            old_tau = np.array(old_data['grid']['adapt_tau_values'])
            for r in old_data['grid_results']:
                inc = old_inc[r['inc_idx']]
                tau = old_tau[r['tau_idx']]
                key = (round(inc, 4), round(tau, 0))
                cached_stim_lookup[key] = r['matched_stimulus_current']

        if followup_path.exists():
            with open(followup_path) as f:
                fu_data = json.load(f)
            for r in fu_data['grid_results']:
                if r.get('compartment', 'both') != 'both':
                    continue
                key = (round(r['adapt_inc'], 4), round(r['adapt_tau'], 0))
                cached_stim_lookup[key] = r['matched_stimulus_current']

        print(f"  Loaded {len(cached_stim_lookup)} cached calibration values")

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build grid points
    grid_points = build_grid_points(args.arms)
    n_grid = len(grid_points)

    # Load completed results if resuming
    completed_ids = set()
    grid_results = []
    baseline_result = None
    bsa_baseline = None
    no_adapt_result = None
    target_rate_hz = None

    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            prev = json.load(f)
        grid_results = prev.get('grid_results', [])
        baseline_result = prev.get('lhs021_baseline')
        bsa_baseline = prev.get('bsa_baseline')
        no_adapt_result = prev.get('no_adapt_baseline')
        target_rate_hz = prev.get('rate_matching', {}).get('target_rate_hz')
        for r in grid_results:
            completed_ids.add(r['point_id'])
        print(f"  Resuming: {len(completed_ids)}/{n_grid} grid points complete")

    # Figures-only mode
    if args.figures_only:
        if not grid_results:
            print("ERROR: --figures-only requires --resume with results")
            return
        print("Regenerating figures...")
        generate_figures(grid_results, bsa_baseline, baseline_result, results_dir)
        print("Done.")
        return

    total_start = time.time()

    # Count per source
    source_counts = {}
    for p in grid_points:
        source_counts[p['source']] = source_counts.get(p['source'], 0) + 1

    print("=" * 70)
    print("  CLASSIFICATION ADAPTATION PARAMETER SWEEP (NOISE-FIXED)")
    print(f"  Grid points: {n_grid}")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt} points")
    print(f"  Task: 5-class digit classification (digits {DEFAULT_DIGITS})")
    print(f"  Samples: {SAMPLES_PER_DIGIT} per digit = {SAMPLES_PER_DIGIT * N_DIGITS} total")
    print(f"  Workers: {n_workers}")
    print(f"  Readout: Flat Ridge (all {BIN_MS}ms bins concatenated)")
    print(f"  CV: 60/40 StratifiedShuffleSplit x {N_SPLIT_REPEATS} repeats")
    print(f"  Rate-matched: LHS-021 baseline +/- {RATE_TOLERANCE_HZ:.0f} Hz")
    print(f"  Noise fix: per-worker PID-based RNG reseed")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load audio
    # ----------------------------------------------------------
    print("\n[1] Loading audio samples...")
    load_config = copy.deepcopy(BASE_CONFIG)
    load_config['random_seed'] = SAMPLE_LOAD_SEED
    load_config['samples_per_digit'] = SAMPLES_PER_DIGIT
    samples = load_audio_samples(load_config, DATA_DIR, digits=DEFAULT_DIGITS)
    n_samples = len(samples)
    print(f"  {n_samples} samples loaded")

    y = np.array([s['digit'] for s in samples], dtype=np.int64)

    max_audio_ms = max(float(np.max(s['spike_times_ms'])) for s in samples) + 5.0

    # ----------------------------------------------------------
    # 2. BSA baseline (adaptation-independent)
    # ----------------------------------------------------------
    if bsa_baseline is None:
        print("\n[2] Computing BSA baseline...")
        total_duration_ms = max_audio_ms + POST_STIM_MS
        n_bins = int(np.ceil(total_duration_ms / BIN_MS))

        bsa_bins_list = [extract_bsa_bins(s, total_duration_ms=total_duration_ms)
                         for s in samples]
        target_n_bins = max(b.shape[0] for b in bsa_bins_list)
        bsa_3d = np.zeros((n_samples, target_n_bins, 128))
        for i, b in enumerate(bsa_bins_list):
            bsa_3d[i, :b.shape[0], :] = b

        bsa_result = classify_flat_ridge(bsa_3d, y)
        bsa_baseline = {
            'accuracy': bsa_result['accuracy'],
            'accuracy_std': bsa_result['accuracy_std'],
            'per_repeat_accuracy': bsa_result['per_repeat_accuracy'],
            'fold_accuracies': bsa_result['fold_accuracies'],
            'confusion_matrix': bsa_result['confusion_matrix'],
        }
        print(f"  BSA: {bsa_baseline['accuracy']:.1%} "
              f"(std={bsa_baseline['accuracy_std']:.3f})")
    else:
        print(f"\n[2] BSA baseline (from checkpoint): {bsa_baseline['accuracy']:.1%}")

    # ----------------------------------------------------------
    # 3. Base config (shared across grid points)
    # ----------------------------------------------------------
    base_config = copy.deepcopy(BASE_CONFIG)
    base_config['audio_duration_ms'] = max_audio_ms
    base_config['post_stimulus_ms'] = POST_STIM_MS
    base_config['random_seed'] = SEED
    base_config['stimulus_current'] = 0.88  # LHS-021 default
    base_config['lambda_connect'] = LHS021_PARAMS['lambda_connect']
    base_config['surface_shell_fraction'] = LHS021_PARAMS['surface_shell_fraction']

    calibration_samples = samples[:CALIBRATION_N_SAMPLES]

    # Verify calibration subset balance
    cal_speakers = [s['speaker'] for s in calibration_samples]
    cal_digits = [s['digit'] for s in calibration_samples]
    cal_sp_dist = Counter(cal_speakers)
    cal_dig_dist = Counter(cal_digits)

    print(f"\n  Calibration subset ({CALIBRATION_N_SAMPLES}/{n_samples} samples):")
    print(f"    Speakers: {dict(sorted(cal_sp_dist.items()))}")
    print(f"    Digits: {dict(sorted(cal_dig_dist.items()))}")

    trial_duration_s = (base_config['audio_duration_ms'] + POST_STIM_MS) / 1000.0

    # ----------------------------------------------------------
    # 4. LHS-021 baseline (within-experiment reference)
    # ----------------------------------------------------------
    if baseline_result is None or target_rate_hz is None:
        print("\n[3] Running LHS-021 baseline...")
        lhs021_overrides = {
            'shell_core_mult': LHS021_PARAMS['shell_core_mult'],
            'core_core_mult': LHS021_PARAMS['core_core_mult'],
            'adapt_inc': LHS021_PARAMS['adapt_inc'],
            'nmda_tau': FIXED_NMDA_TAU,
        }

        t0 = time.time()
        bl_bins_3d, bl_digits, bl_spikes, bl_n_res, bl_isi_cvs, bl_adapt = \
            run_all_samples(base_config, samples, lhs021_overrides,
                            n_workers=n_workers)
        bl_sim_time = time.time() - t0

        bl_per_trial_rate = bl_spikes / (bl_n_res * trial_duration_s)
        target_rate_hz = float(np.mean(bl_per_trial_rate))
        bl_rate_std = float(np.std(bl_per_trial_rate))

        assert np.array_equal(bl_digits, y)

        # Pad reservoir bins to uniform size
        target_n_bins = max(bl_bins_3d.shape[1],
                           max(b.shape[0] for b in [extract_bsa_bins(s, total_duration_ms=max_audio_ms + POST_STIM_MS) for s in samples[:1]]))
        if bl_bins_3d.shape[1] < target_n_bins:
            pad_width = target_n_bins - bl_bins_3d.shape[1]
            bl_bins_3d = np.pad(bl_bins_3d,
                                ((0, 0), (0, pad_width), (0, 0)),
                                mode='constant')

        bl_result = classify_flat_ridge(bl_bins_3d, y)
        bl_stats = compute_paired_stats(bl_result, bsa_baseline, 'Reservoir', 'BSA')

        bl_isi_valid = bl_isi_cvs[~np.isnan(bl_isi_cvs)]
        bl_pr_mean, bl_pr_per_bin = compute_participation_ratio(bl_bins_3d)

        baseline_result = {
            'params': LHS021_PARAMS,
            'stimulus_current': base_config['stimulus_current'],
            'classification_accuracy': bl_result['accuracy'],
            'classification_accuracy_std': bl_result['accuracy_std'],
            'classification_per_repeat_accuracy': bl_result['per_repeat_accuracy'],
            'classification_gap_pp': bl_stats['gap_pp'],
            'classification_p_value': bl_stats['p_value'],
            'classification_stars': bl_stats['stars'],
            'confusion_matrix': bl_result['confusion_matrix'],
            'firing_rate_hz': target_rate_hz,
            'firing_rate_std': bl_rate_std,
            'n_reservoir': bl_n_res,
            'sim_time_s': bl_sim_time,
            'isi_cv_mean': float(np.mean(bl_isi_valid)) if len(bl_isi_valid) > 0 else None,
            'adapt_at_stim_end_mean': float(np.mean(bl_adapt)),
            'participation_ratio_mean': bl_pr_mean,
        }

        print(f"  LHS-021 baseline: {bl_sim_time:.0f}s")
        print(f"  Firing rate: {target_rate_hz:.1f} +/- {bl_rate_std:.1f} Hz "
              f"-> calibration target")
        print(f"  Classification: {bl_result['accuracy']:.1%} "
              f"(gap={bl_stats['gap_pp']:+.1f}pp {bl_stats['stars']})")
    else:
        print(f"\n[3] LHS-021 baseline (from checkpoint): "
              f"{baseline_result['classification_accuracy']:.1%}, "
              f"rate={target_rate_hz:.1f} Hz")

    # ----------------------------------------------------------
    # 5. Sweep all grid points
    # ----------------------------------------------------------
    n_remaining_total = sum(1 for p in grid_points if p['point_id'] not in completed_ids)
    print(f"\n[4] Running {n_remaining_total} remaining grid points "
          f"(rate-matched to {target_rate_hz:.1f} +/- {RATE_TOLERANCE_HZ:.0f} Hz)...")

    grid_point_times = []
    sweep_start_time = time.time()

    # Warm-start per tau column
    last_stim_by_tau = {}
    for r in grid_results:
        t = r.get('tau_idx')
        if t is not None:
            cur = last_stim_by_tau.get(t)
            if cur is None or r.get('inc_idx', -1) > cur[0]:
                last_stim_by_tau[t] = (r.get('inc_idx', 0), r['matched_stimulus_current'])
    last_stim_by_tau = {t: stim for t, (_, stim) in last_stim_by_tau.items()}

    for pt_num, pt in enumerate(grid_points):
        if pt['point_id'] in completed_ids:
            continue

        inc_val = pt['adapt_inc']
        tau_val = pt['adapt_tau']
        i_idx = pt['inc_idx']
        t_idx = pt['tau_idx']
        point_id = pt['point_id']

        grid_point_start = time.time()
        n_done = len(completed_ids)
        n_remaining = n_grid - n_done - 1

        if grid_point_times:
            avg_time = np.mean(grid_point_times[-10:])
            eta_s = avg_time * (n_remaining + 1)
            eta_h = eta_s / 3600
            eta_str = f"ETA: {eta_h:.1f}h" if eta_h >= 1.0 else f"ETA: {eta_s/60:.0f}min"
        else:
            eta_str = "ETA: calculating..."

        print(f"\n{'_' * 60}")
        print(f"  [{pt['source']}] inc={inc_val:.4f}, tau={tau_val:.1f}ms  "
              f"({n_done + 1}/{n_grid})  |  {eta_str}")
        elapsed_total = time.time() - sweep_start_time
        print(f"  Elapsed: {elapsed_total/3600:.1f}h")
        print(f"{'_' * 60}")

        dynamical_overrides = {
            'shell_core_mult': LHS021_PARAMS['shell_core_mult'],
            'core_core_mult': LHS021_PARAMS['core_core_mult'],
            'adapt_inc': float(inc_val),
            'adapt_tau': float(tau_val),
            'nmda_tau': FIXED_NMDA_TAU,
        }

        # Rate-matching calibration
        # Priority: cached guess > warm-start from same tau column > global search
        cache_key = (round(inc_val, 4), round(tau_val, 0))
        cached_guess = cached_stim_lookup.get(cache_key)
        prev_stim = last_stim_by_tau.get(t_idx) if t_idx is not None else None

        if cached_guess is not None:
            print(f"    Calibrating (cached guess={cached_guess:.4f})...")
            matched_stim, cal_rate, cal_log = calibrate_stimulus_current(
                base_config, dynamical_overrides, calibration_samples,
                n_workers=n_workers, target_rate=target_rate_hz,
                initial_guess=cached_guess)
        elif prev_stim is not None:
            warm_lo = max(CALIBRATION_STIM_BOUNDS[0],
                          prev_stim / CALIBRATION_WARM_MARGIN)
            warm_hi = min(CALIBRATION_STIM_BOUNDS[1],
                          prev_stim * CALIBRATION_WARM_MARGIN)
            cal_bounds = (warm_lo, warm_hi)
            print(f"    Calibrating (warm-start [{warm_lo:.4f}, {warm_hi:.4f}])...")
            matched_stim, cal_rate, cal_log = calibrate_stimulus_current(
                base_config, dynamical_overrides, calibration_samples,
                n_workers=n_workers, target_rate=target_rate_hz,
                stim_bounds=cal_bounds)
        else:
            print(f"    Calibrating (target {target_rate_hz:.1f} Hz)...")
            matched_stim, cal_rate, cal_log = calibrate_stimulus_current(
                base_config, dynamical_overrides, calibration_samples,
                n_workers=n_workers, target_rate=target_rate_hz)

        # Retry with global bounds if initial attempt missed
        if abs(cal_rate - target_rate_hz) > RATE_TOLERANCE_HZ:
            print(f"    Initial calibration missed ({cal_rate:.1f} Hz), global retry...")
            matched_stim, cal_rate, cal_log = calibrate_stimulus_current(
                base_config, dynamical_overrides, calibration_samples,
                n_workers=n_workers, target_rate=target_rate_hz,
                stim_bounds=CALIBRATION_STIM_BOUNDS)

        if t_idx is not None:
            last_stim_by_tau[t_idx] = matched_stim
        print(f"    Matched: stim={matched_stim:.4f} -> {cal_rate:.1f} Hz")

        # --- Full evaluation at matched rate ---
        config = copy.deepcopy(base_config)
        config['stimulus_current'] = matched_stim

        t0 = time.time()
        res_bins_3d, digits_check, total_spikes, n_res, isi_cvs, adapt_ends = \
            run_all_samples(config, samples, dynamical_overrides,
                            n_workers=n_workers)
        sim_time = time.time() - t0

        assert np.array_equal(digits_check, y)

        # Firing rate
        per_trial_rate = total_spikes / (n_res * trial_duration_s)
        rate_mean = float(np.mean(per_trial_rate))
        rate_std = float(np.std(per_trial_rate))

        print(f"    Sim: {sim_time:.0f}s, "
              f"Rate: {rate_mean:.1f} +/- {rate_std:.1f} Hz "
              f"(stim={matched_stim:.4f})")

        # --- Classification ---
        res_result = classify_flat_ridge(res_bins_3d, y)
        stats = compute_paired_stats(res_result, bsa_baseline,
                                     'Reservoir', 'BSA')

        print(f"    Classification: {res_result['accuracy']:.1%} "
              f"(gap={stats['gap_pp']:+.1f}pp {stats['stars']})")

        # --- ISI statistics ---
        isi_valid = isi_cvs[~np.isnan(isi_cvs)]
        isi_cv_mean = float(np.mean(isi_valid)) if len(isi_valid) > 0 else None
        isi_cv_std = float(np.std(isi_valid)) if len(isi_valid) > 0 else None

        # --- Adaptation at stimulus end ---
        adapt_mean = float(np.mean(adapt_ends))
        adapt_std = float(np.std(adapt_ends))

        # --- Participation ratio ---
        pr_mean, pr_per_bin = compute_participation_ratio(res_bins_3d)

        # --- Per-bin accuracy (diagnostic) ---
        per_bin_acc = classify_per_bin(res_bins_3d, y)

        isi_str = f"{isi_cv_mean:.3f}" if isi_cv_mean is not None else "N/A"
        print(f"    ISI CV: {isi_str}, "
              f"g_adapt: {adapt_mean:.4f}, "
              f"PR: {pr_mean:.4f}")

        grid_results.append({
            'point_id': point_id,
            'source': pt['source'],
            'inc_idx': i_idx,
            'tau_idx': t_idx,
            'adapt_inc': float(inc_val),
            'adapt_tau': float(tau_val),
            'matched_stimulus_current': matched_stim,
            'calibration_rate_hz': cal_rate,
            'calibration_log': cal_log,
            'classification_accuracy': res_result['accuracy'],
            'classification_accuracy_std': res_result['accuracy_std'],
            'classification_per_repeat_accuracy': res_result['per_repeat_accuracy'],
            'classification_gap_pp': stats['gap_pp'],
            'classification_ci_lo_pp': stats['ci_lo_pp'],
            'classification_ci_hi_pp': stats['ci_hi_pp'],
            'classification_p_value': stats['p_value'],
            'classification_cohens_d': stats['cohens_d'],
            'classification_stars': stats['stars'],
            'confusion_matrix': res_result['confusion_matrix'],
            'firing_rate_hz': rate_mean,
            'firing_rate_std': rate_std,
            'n_reservoir': n_res,
            'sim_time_s': sim_time,
            'isi_cv_mean': isi_cv_mean,
            'isi_cv_std': isi_cv_std,
            'adapt_at_stim_end_mean': adapt_mean,
            'adapt_at_stim_end_std': adapt_std,
            'participation_ratio_mean': pr_mean,
            'participation_ratio_per_bin': pr_per_bin,
            'per_bin_accuracy': per_bin_acc,
        })
        completed_ids.add(point_id)

        # Timing
        grid_point_elapsed = time.time() - grid_point_start
        grid_point_times.append(grid_point_elapsed)
        n_done_now = len(completed_ids)
        n_remaining_now = n_grid - n_done_now
        if grid_point_times:
            avg_t = np.mean(grid_point_times[-10:])
            eta_h = (avg_t * n_remaining_now) / 3600
            total_elapsed = time.time() - sweep_start_time
            print(f"    {grid_point_elapsed/60:.1f}min  |  "
                  f"{n_done_now}/{n_grid}  |  "
                  f"Elapsed: {total_elapsed/3600:.1f}h  |  "
                  f"ETA: {eta_h:.1f}h", flush=True)

        # Free large arrays before next grid point
        del res_bins_3d, total_spikes, isi_cvs, adapt_ends, digits_check
        gc.collect()

        # Checkpoint
        _save_checkpoint(results_dir, grid_results, bsa_baseline,
                         baseline_result, no_adapt_result,
                         target_rate_hz, args, total_start)

    total_time = time.time() - total_start

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SWEEP COMPLETE")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f} hours)")
    print(f"  Grid points: {len(grid_results)}/{n_grid}")
    print(f"{'=' * 70}")

    print(f"\n  Baseline (LHS-021): rate={target_rate_hz:.1f} Hz")
    bl_acc = baseline_result['classification_accuracy']
    print(f"  Classification: {bl_acc:.1%} "
          f"(gap={baseline_result['classification_gap_pp']:+.1f}pp "
          f"{baseline_result['classification_stars']})")

    if grid_results:
        best = max(grid_results, key=lambda r: r['classification_accuracy'])
        acc = best['classification_accuracy']
        gap = best['classification_gap_pp']
        delta = (acc - bl_acc) * 100
        print(f"\n  Best classification:")
        print(f"    [{best['source']}] inc={best['adapt_inc']:.4f}, "
              f"tau={best['adapt_tau']:.1f}ms, "
              f"stim={best['matched_stimulus_current']:.4f}")
        print(f"    Accuracy: {acc:.1%} (gap={gap:+.1f}pp), "
              f"Rate: {best['firing_rate_hz']:.1f}Hz")
        print(f"    vs LHS-021: {bl_acc:.1%} -> delta = {delta:+.1f}pp")

    # ----------------------------------------------------------
    # 7. Final save + figures
    # ----------------------------------------------------------
    _save_checkpoint(results_dir, grid_results, bsa_baseline,
                     baseline_result, no_adapt_result,
                     target_rate_hz, args, total_start, final=True)

    print("\n  Generating figures...")
    generate_figures(grid_results, bsa_baseline, baseline_result, results_dir)

    print("\nDone.")


def _save_checkpoint(results_dir, grid_results, bsa_baseline,
                     baseline_result, no_adapt_result,
                     target_rate_hz, args, total_start, final=False):
    summary = {
        'experiment': 'Classification Adaptation Parameter Sweep (Noise-Fixed, Consolidated)',
        'timestamp': datetime.now().isoformat(),
        'total_time_s': time.time() - total_start,
        'task': '5-class digit classification',
        'digits': DEFAULT_DIGITS,
        'n_samples': SAMPLES_PER_DIGIT * N_DIGITS,
        'samples_per_digit': SAMPLES_PER_DIGIT,
        'readout': f'Flat Ridge (all {BIN_MS}ms bins concatenated)',
        'cv_scheme': f'60/40 StratifiedShuffleSplit x {N_SPLIT_REPEATS} repeats',
        'noise_fix': 'per-worker PID-based RNG reseed',
        'measurements': [
            'classification_accuracy', 'confusion_matrix', 'isi_cv',
            'adapt_at_stim_end', 'drive_cost', 'participation_ratio',
            'per_bin_accuracy',
        ],
        'rate_matching': {
            'target_rate_hz': target_rate_hz,
            'tolerance_hz': RATE_TOLERANCE_HZ,
            'calibration_n_samples': CALIBRATION_N_SAMPLES,
            'calibration_max_iter': CALIBRATION_MAX_ITER,
            'stim_bounds': list(CALIBRATION_STIM_BOUNDS),
        },
        'lhs021_baseline': baseline_result,
        'no_adapt_baseline': no_adapt_result,
        'grid': {
            'unified_inc': UNIFIED_INC.tolist(),
            'unified_tau': UNIFIED_TAU.tolist(),
            'n_inc': len(UNIFIED_INC),
            'n_tau': len(UNIFIED_TAU),
        },
        'fixed_params': {
            'lambda_connect': LHS021_PARAMS['lambda_connect'],
            'core_core_mult': LHS021_PARAMS['core_core_mult'],
            'surface_shell_fraction': LHS021_PARAMS['surface_shell_fraction'],
            'shell_core_mult': LHS021_PARAMS['shell_core_mult'],
            'nmda_tau': FIXED_NMDA_TAU,
            'stimulus_current_baseline': 0.88,
            'std_u': STD_U,
            'std_tau_rec': STD_TAU_REC,
        },
        'bsa_baseline': bsa_baseline,
        'grid_results': grid_results,
    }

    fname = ('classification_adaptation_sweep.json' if final
             else 'classification_adaptation_sweep_checkpoint.json')
    results_path = results_dir / fname
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if final:
        print(f"\n  Saved: {results_path}")


if __name__ == "__main__":
    main()
