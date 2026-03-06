#!/usr/bin/env python3
"""
Behavioral verification: compare Python and C++ (snapshot) network responses.

Runs LHS-021 baseline on the same audio samples through both implementations,
then compares per-sample spike counts, firing rates, and classification accuracy.

Usage:
    python verify_behavior.py [--n-workers 4] [--cpp-path ../liquid_state_machine_cpp_port]
"""

import sys
import os
import json
import subprocess
import argparse
import time
import numpy as np
from pathlib import Path
from scipy import stats

EXPANDED_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPANDED_DIR))

from utils.network_builder import (
    BASE_CONFIG, build_full_network, build_std_masks,
    run_sample_with_std, STD_U, STD_TAU_REC,
)
from utils.simulation_utils import load_audio_samples

# Match C++ constants
SEED = 42
SAMPLE_LOAD_SEED = 42
BIN_MS = 20.0
POST_STIM_MS = 200.0
SAMPLES_PER_DIGIT = 100
N_SPLIT_REPEATS = 5
DEFAULT_DIGITS = [0, 1, 2, 3, 4]

# LHS-021 overrides
LHS021_SHELL_CORE_MULT = 4.849737508561994
LHS021_CORE_CORE_MULT = 0.8274594948671626
LHS021_ADAPT_INC = 0.6264283936395093
FIXED_NMDA_TAU = 50.0


def _load_samples_fast(data_dir, digits, samples_per_digit, seed):
    """Load audio samples with timeout protection against corrupted files."""
    import signal
    spike_dir = Path(data_dir) / 'spike_trains_bsa'
    np.random.seed(seed + 100)

    def _timeout(signum, frame):
        raise TimeoutError()

    samples = []
    for digit in digits:
        all_files = sorted(spike_dir.glob(f'spike_train_{digit}_*.npz'))
        valid_files = []
        for f in all_files:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(3)
            try:
                d = np.load(f)
                d.close()
                valid_files.append(f)
            except Exception:
                print(f"  WARNING: Skipping {f.name}")
            finally:
                signal.alarm(0)

        if len(valid_files) <= samples_per_digit:
            selected = valid_files
        else:
            indices = np.random.choice(len(valid_files), samples_per_digit, replace=False)
            selected = [valid_files[i] for i in sorted(indices)]

        for fpath in selected:
            data = np.load(fpath)
            samples.append({
                'spike_times_ms': data['spike_times_ms'].astype(np.float64),
                'freq_bin_indices': data['freq_bin_indices'].astype(np.int32),
                'digit': int(data['digit']),
                'speaker': str(data['speaker']),
                'spectrogram': data['spectrogram'],
                'filename': fpath.stem,
            })

    print(f"  Loaded {len(samples)} samples")
    for d in digits:
        count = sum(1 for s in samples if s['digit'] == d)
        print(f"    Digit {d}: {count}")
    return samples


def run_python_baseline(n_workers):
    """Run Python LHS-021 baseline, return per-sample stats."""
    config = BASE_CONFIG.copy()

    dynamical_overrides = {
        'shell_core_mult': LHS021_SHELL_CORE_MULT,
        'core_core_mult': LHS021_CORE_CORE_MULT,
        'adapt_inc': LHS021_ADAPT_INC,
        'nmda_tau': FIXED_NMDA_TAU,
    }

    print("Building Python network...")
    network, zone_info = build_full_network(
        config, quiet=True,
        dynamical_overrides=dynamical_overrides,
        remove_nonarc=True,
    )
    std_eligible, recurrent_masks = build_std_masks(network, zone_info)

    print("Loading audio samples...")
    data_dir = EXPANDED_DIR / 'data'
    samples = _load_samples_fast(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED)

    n_samples = len(samples)
    reservoir_indices = zone_info['reservoir_zone_indices']
    n_reservoir = len(reservoir_indices)
    dt = config['dt']

    max_audio_ms = max(float(np.max(s['spike_times_ms'])) for s in samples) + 5.0
    config['audio_duration_ms'] = max_audio_ms
    config['post_stimulus_ms'] = POST_STIM_MS
    total_duration_ms = max_audio_ms + POST_STIM_MS
    trial_dur_s = total_duration_ms / 1000.0
    n_bins = int(np.ceil(total_duration_ms / BIN_MS))

    print(f"Running {n_samples} samples through Python network...")
    t0 = time.time()

    def bin_activity_record(activity_record, reservoir_indices, dt, bin_ms, n_bins):
        n_reservoir = len(reservoir_indices)
        res_idx_to_pos = {int(idx): pos for pos, idx in enumerate(reservoir_indices)}
        bins = np.zeros((n_bins, n_reservoir))
        total_spikes = 0
        for step, active in enumerate(activity_record):
            t = step * dt
            b = min(int(t / bin_ms), n_bins - 1)
            for nid in active:
                pos = res_idx_to_pos.get(nid)
                if pos is not None:
                    bins[b, pos] += 1
                    total_spikes += 1
        return bins, total_spikes

    per_sample_spikes = []
    per_sample_rate_hz = []
    per_sample_isi_cv = []
    per_sample_digit = []
    per_sample_adapt_stim_end = []
    per_sample_filename = []
    res_bins_list = []

    # Run sequentially (single network, reset each time)
    np.random.seed(config.get('random_seed', SEED) + os.getpid())

    for i, sample in enumerate(samples):
        stim_end_ms = float(np.max(sample['spike_times_ms']))

        activity_record, adapt_snapshot = run_sample_with_std(
            network, sample, zone_info, config,
            std_u=STD_U, std_tau_rec=STD_TAU_REC,
            std_eligible=std_eligible, recurrent_masks=recurrent_masks,
            record_adapt_at_ms=stim_end_ms)

        res_bins, total_res_spikes = bin_activity_record(
            activity_record, reservoir_indices, dt, BIN_MS, n_bins)

        # ISI CV
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
        isi_cv = float(np.std(all_isis) / max(np.mean(all_isis), 1e-9)) if len(all_isis) >= 2 else float('nan')

        mean_adapt = float(np.mean(adapt_snapshot[reservoir_indices]))

        per_sample_spikes.append(total_res_spikes)
        per_sample_rate_hz.append(total_res_spikes / (n_reservoir * trial_dur_s))
        per_sample_isi_cv.append(isi_cv)
        per_sample_digit.append(sample['digit'])
        per_sample_filename.append(sample['filename'])
        per_sample_adapt_stim_end.append(mean_adapt)
        res_bins_list.append(res_bins)

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"    {i+1}/{n_samples}")

    elapsed = time.time() - t0
    print(f"  Python simulation: {elapsed:.0f}s")

    # Classification
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedShuffleSplit

    res_bins_3d = np.array(res_bins_list)
    X = res_bins_3d.reshape(n_samples, -1)
    y = np.array(per_sample_digit)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    per_repeat_acc = []
    for rep in range(N_SPLIT_REPEATS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED + rep)
        for train_idx, test_idx in sss.split(X, y):
            best_acc = 0
            for alpha in alphas:
                clf = RidgeClassifier(alpha=alpha)
                clf.fit(X[train_idx], y[train_idx])
                acc = clf.score(X[test_idx], y[test_idx])
                if acc > best_acc:
                    best_acc = acc
            per_repeat_acc.append(best_acc)

    cls_accuracy = float(np.mean(per_repeat_acc))
    cls_std = float(np.std(per_repeat_acc))
    mean_rate = float(np.mean(per_sample_rate_hz))

    print(f"  Rate: {mean_rate:.1f} Hz")
    print(f"  Classification: {cls_accuracy*100:.1f}% (std={cls_std:.3f})")

    result = {
        'n_samples': n_samples,
        'n_reservoir': n_reservoir,
        'mean_firing_rate_hz': mean_rate,
        'classification_accuracy': cls_accuracy,
        'classification_accuracy_std': cls_std,
        'per_repeat_accuracy': per_repeat_acc,
        'per_sample_spikes': [float(x) for x in per_sample_spikes],
        'per_sample_rate_hz': [float(x) for x in per_sample_rate_hz],
        'per_sample_isi_cv': [float(x) for x in per_sample_isi_cv],
        'per_sample_digit': per_sample_digit,
        'per_sample_filename': per_sample_filename,
        'per_sample_adapt_stim_end': [float(x) for x in per_sample_adapt_stim_end],
    }

    with open('verify_py.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to verify_py.json")

    return result


def run_cpp_baseline(cpp_path, n_workers):
    """Run C++ LHS-021 baseline with snapshot, return per-sample stats."""
    snapshot = cpp_path / 'network_snapshot.npz'
    if not snapshot.exists():
        print(f"  Snapshot not found at {snapshot}, exporting...")
        subprocess.run([sys.executable, 'export_network.py',
                       '-o', str(snapshot)], check=True)

    binary = cpp_path / 'cls_sweep'
    if not binary.exists():
        print(f"  Building C++ port...")
        subprocess.run(['make', '-C', str(cpp_path)], check=True)

    verify_json = str(EXPANDED_DIR / 'verify_cpp.json')
    data_dir = str(EXPANDED_DIR / 'data')
    cmd = [
        str(binary),
        '--snapshot', str(snapshot),
        '--n-workers', str(n_workers),
        '--samples-per-digit', str(SAMPLES_PER_DIGIT),
        '--data-dir', data_dir,
        '--verify-only',
        '--verify-output', verify_json,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"C++ verify failed with code {result.returncode}")

    with open(verify_json) as f:
        return json.load(f)


def compare_results(py, cpp):
    """Compare Python and C++ results."""
    print("\n" + "=" * 60)
    print("  BEHAVIORAL COMPARISON: Python vs C++ (Snapshot)")
    print("=" * 60)

    # 1. Basic stats
    print(f"\n  {'Metric':<30s} {'Python':>12s} {'C++':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'n_samples':<30s} {py['n_samples']:>12d} {cpp['n_samples']:>12d}")
    print(f"  {'n_reservoir':<30s} {py['n_reservoir']:>12d} {cpp['n_reservoir']:>12d}")
    print(f"  {'mean_firing_rate_hz':<30s} {py['mean_firing_rate_hz']:>12.2f} {cpp['mean_firing_rate_hz']:>12.2f}")
    print(f"  {'classification_accuracy':<30s} {py['classification_accuracy']*100:>11.1f}% {cpp['classification_accuracy']*100:>11.1f}%")
    print(f"  {'classification_std':<30s} {py['classification_accuracy_std']:>12.4f} {cpp['classification_accuracy_std']:>12.4f}")

    # 2. Match samples by filename for paired comparison
    py_by_name = {}
    for i, fn in enumerate(py.get('per_sample_filename', [])):
        py_by_name[fn] = i
    cpp_by_name = {}
    for i, fn in enumerate(cpp.get('per_sample_filename', [])):
        cpp_by_name[fn] = i

    common_names = sorted(set(py_by_name.keys()) & set(cpp_by_name.keys()))
    n_paired = len(common_names)
    print(f"\n  Filename-matched pairs: {n_paired}")

    py_all_spikes = np.array(py['per_sample_spikes'])
    cpp_all_spikes = np.array(cpp['per_sample_spikes'])
    py_all_rates = np.array(py['per_sample_rate_hz'])
    cpp_all_rates = np.array(cpp['per_sample_rate_hz'])

    if n_paired > 0:
        py_idx = [py_by_name[fn] for fn in common_names]
        cpp_idx = [cpp_by_name[fn] for fn in common_names]
        py_matched = py_all_spikes[py_idx]
        cpp_matched = cpp_all_spikes[cpp_idx]
        py_rates_m = py_all_rates[py_idx]
        cpp_rates_m = cpp_all_rates[cpp_idx]

        r, p_corr = stats.pearsonr(py_matched, cpp_matched)
        r_rate, _ = stats.pearsonr(py_rates_m, cpp_rates_m)
        mean_diff = np.mean(py_matched) - np.mean(cpp_matched)
        std_diff = np.std(py_matched - cpp_matched)
    else:
        # Fallback: unpaired
        print("  (No filename matches — falling back to unpaired comparison)")
        n_common = min(len(py_all_spikes), len(cpp_all_spikes))
        py_matched = py_all_spikes[:n_common]
        cpp_matched = cpp_all_spikes[:n_common]
        py_rates_m = py_all_rates[:n_common]
        cpp_rates_m = cpp_all_rates[:n_common]
        r, p_corr = stats.pearsonr(py_matched, cpp_matched)
        r_rate, _ = stats.pearsonr(py_rates_m, cpp_rates_m)
        mean_diff = np.mean(py_all_spikes) - np.mean(cpp_all_spikes)
        std_diff = np.std(py_matched - cpp_matched)

    ks_stat, ks_p = stats.ks_2samp(py_all_spikes, cpp_all_spikes)
    ks_rate, ks_rate_p = stats.ks_2samp(py_all_rates, cpp_all_rates)

    print(f"\n  Per-sample spike count comparison (paired):")
    print(f"    Pearson r:          {r:.6f}  (p={p_corr:.2e})")
    print(f"    KS test:            stat={ks_stat:.4f}, p={ks_p:.4f}")
    print(f"    Mean difference:    {mean_diff:.1f} spikes")
    print(f"    Std of differences: {std_diff:.1f} spikes")

    print(f"\n  Per-sample firing rate comparison (paired):")
    print(f"    Pearson r:          {r_rate:.6f}")
    print(f"    KS test:            stat={ks_rate:.4f}, p={ks_rate_p:.4f}")

    # 4. Classification comparison
    acc_diff = (py['classification_accuracy'] - cpp['classification_accuracy']) * 100
    print(f"\n  Classification gap:   {acc_diff:+.2f} pp")

    # 5. Verdict
    print(f"\n  {'='*54}")
    issues = []
    rate_diff = abs(py['mean_firing_rate_hz'] - cpp['mean_firing_rate_hz'])
    if rate_diff > 5.0:
        issues.append(f"Mean rate difference too large: {rate_diff:.1f} Hz (need <5 Hz)")
    if ks_p < 0.01:
        issues.append(f"Spike distribution differs: KS p={ks_p:.4f} (need >0.01)")
    if abs(acc_diff) > 5:
        issues.append(f"Accuracy gap too large: {acc_diff:+.1f}pp (need <5pp)")

    if not issues:
        print("  VERDICT: PASS — Behavior is statistically equivalent")
    else:
        print("  VERDICT: FAIL")
        for issue in issues:
            print(f"    - {issue}")
    print(f"  {'='*54}")

    # 6. Scatter plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Spike count scatter (matched samples)
        ax = axes[0]
        ax.scatter(py_matched, cpp_matched, alpha=0.3, s=10)
        lims = [min(py_matched.min(), cpp_matched.min()),
                max(py_matched.max(), cpp_matched.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('Python spike count')
        ax.set_ylabel('C++ spike count')
        ax.set_title(f'Per-sample spikes (r={r:.3f})')

        # Firing rate scatter (matched samples)
        ax = axes[1]
        ax.scatter(py_rates_m, cpp_rates_m, alpha=0.3, s=10)
        lims = [min(py_rates_m.min(), cpp_rates_m.min()),
                max(py_rates_m.max(), cpp_rates_m.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('Python rate (Hz)')
        ax.set_ylabel('C++ rate (Hz)')
        ax.set_title(f'Per-sample rate (r={r_rate:.3f})')

        # Spike count distributions
        ax = axes[2]
        bins = np.linspace(min(py_all_spikes.min(), cpp_all_spikes.min()),
                          max(py_all_spikes.max(), cpp_all_spikes.max()), 50)
        ax.hist(py_all_spikes, bins=bins, alpha=0.5, label='Python', density=True)
        ax.hist(cpp_all_spikes, bins=bins, alpha=0.5, label='C++', density=True)
        ax.set_xlabel('Spike count')
        ax.set_ylabel('Density')
        ax.set_title(f'Distributions (KS p={ks_p:.3f})')
        ax.legend()

        plt.tight_layout()
        plt.savefig('verify_behavior.png', dpi=150)
        print(f"\n  Scatter plot saved to verify_behavior.png")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping scatter plot)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral verification: Python vs C++')
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--cpp-path', type=str,
                        default=str(Path(__file__).parent.parent.parent.parent / 'liquid_state_machine_cpp_port'))
    parser.add_argument('--skip-python', action='store_true',
                        help='Skip Python run, load from verify_py.json')
    parser.add_argument('--skip-cpp', action='store_true',
                        help='Skip C++ run, load from verify_cpp.json')
    args = parser.parse_args()

    cpp_path = Path(args.cpp_path)

    print("=" * 60)
    print("  BEHAVIORAL VERIFICATION: Python vs C++ (Snapshot)")
    print("=" * 60)

    # Python side
    if args.skip_python:
        print("\n[Python] Loading from verify_py.json...")
        with open('verify_py.json') as f:
            py_result = json.load(f)
    else:
        print("\n[Python] Running LHS-021 baseline...")
        py_result = run_python_baseline(args.n_workers)

    # C++ side
    if args.skip_cpp:
        print("\n[C++] Loading from verify_cpp.json...")
        with open('verify_cpp.json') as f:
            cpp_result = json.load(f)
    else:
        print(f"\n[C++] Running LHS-021 baseline (snapshot mode)...")
        cpp_result = run_cpp_baseline(cpp_path, args.n_workers)

    # Compare
    compare_results(py_result, cpp_result)
