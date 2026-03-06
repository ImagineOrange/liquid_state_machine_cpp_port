"""
Simulation utilities for temporal XOR experiments.

Replicated from liquid_state_machine_initial/lsm_xor_experiment_novel/:
  - load_audio_samples()     from utils/lsm_BSA_simulation.py
  - organize_samples_by_digit(), create_xor_stimulus(),
    generate_5digit_xor_pairs(), generate_2digit_xor_pairs(),
    compute_epoch_boundaries(), extract_reservoir_epoch_bins(),
    extract_bsa_epoch_bins(), compute_duration_features(),
    extract_digit_a_groups(), extract_speaker_a_groups(),
    classify_binary(), compute_paired_stats()
                              from experiments/temporal_xor_simplified_multiseed_v4.py
                              and experiments/temporal_xor_nonlinear_readout_v5.py

No imports from liquid_state_machine_initial/.
"""

import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONSTANTS
# ============================================================

# Stimulus timing
GAP_MS = 50.0
POST_STIMULUS_MS = 200.0

# Epoch-aligned binning
N_BINS_A = 5
N_BINS_GAP = 1
N_BINS_B = 5
N_BINS_POST = 2
N_EPOCH_BINS = N_BINS_A + N_BINS_GAP + N_BINS_B + N_BINS_POST  # 13

# Epoch bin index ranges
IDX_A_START = 0
IDX_A_END = N_BINS_A
IDX_GAP = N_BINS_A
IDX_B_START = N_BINS_A + N_BINS_GAP
IDX_B_END = N_BINS_A + N_BINS_GAP + N_BINS_B
IDX_POST_START = IDX_B_END
IDX_POST_END = N_EPOCH_BINS

# Classification
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS = 5
N_BOOTSTRAP = 10000

# Pair generation seeds (fixed across all experiments for comparability)
PAIR_SEED = 99
SAMPLE_LOAD_SEED = 42

# Matched-pair conditions
CONDITION_PAIRS = [
    ('R1: Reservoir full',    'B1: BSA full',    (0, N_EPOCH_BINS)),
    ('R2: Reservoir gap+B+P', 'B2: BSA gap+B+P', (IDX_GAP, N_EPOCH_BINS)),
    ('R3: Reservoir B+P',     'B3: BSA B+P',     (IDX_B_START, N_EPOCH_BINS)),
]
EPOCH_LABELS = {
    (0, N_EPOCH_BINS): 'Full (A+Gap+B+Post)',
    (IDX_GAP, N_EPOCH_BINS): 'Gap+B+Post',
    (IDX_B_START, N_EPOCH_BINS): 'B+Post',
}


# ============================================================
# DATA LOADING
# Replicated from lsm_BSA_simulation.py::load_audio_samples()
# ============================================================

def load_audio_samples(config, data_dir, digits=None):
    """Load BSA spike train files.

    Parameters
    ----------
    config : dict
        Must contain 'samples_per_digit' and 'random_seed'.
    data_dir : Path
        Directory containing spike_trains_bsa/ subdirectory.
    digits : list of int, optional
        Which digits to load. Defaults to [0, 1, 2].
    """
    spike_dir = Path(data_dir) / 'spike_trains_bsa'
    target_digits = digits if digits is not None else [0, 1, 2]
    samples_per_digit = config['samples_per_digit']

    np.random.seed(config['random_seed'] + 100)

    samples = []
    for digit in target_digits:
        all_files = sorted(spike_dir.glob(f'spike_train_{digit}_*.npz'))
        files = []
        for f in all_files:
            try:
                d = np.load(f)
                d.close()
                files.append(f)
            except Exception:
                print(f"  WARNING: Skipping corrupted file {f.name}")
        if len(files) < samples_per_digit:
            print(f"  WARNING: Only {len(files)} valid files for digit {digit}")
            selected = files
        else:
            indices = np.random.choice(len(files), samples_per_digit, replace=False)
            selected = [files[i] for i in sorted(indices)]

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

    print(f"  Loaded {len(samples)} audio samples:")
    for d in target_digits:
        count = sum(1 for s in samples if s['digit'] == d)
        print(f"    Digit {d}: {count} samples")

    return samples


# ============================================================
# SAMPLE ORGANIZATION
# ============================================================

def organize_samples_by_digit(samples):
    by_digit = {}
    for s in samples:
        d = s['digit']
        if d not in by_digit:
            by_digit[d] = []
        by_digit[d].append(s)
    return by_digit


# ============================================================
# XOR STIMULUS GENERATION
# ============================================================

def create_xor_stimulus(sample_a, sample_b):
    """Concatenate two BSA samples with a fixed gap after stim A ends."""
    stim_a_end_ms = float(np.max(sample_a['spike_times_ms']))
    digit_b_offset_ms = stim_a_end_ms + GAP_MS
    stim_b_end_ms = digit_b_offset_ms + float(np.max(sample_b['spike_times_ms']))

    return {
        'spike_times_ms': np.concatenate([
            sample_a['spike_times_ms'],
            sample_b['spike_times_ms'] + digit_b_offset_ms,
        ]),
        'freq_bin_indices': np.concatenate([
            sample_a['freq_bin_indices'],
            sample_b['freq_bin_indices'],
        ]),
        'digit_a': sample_a['digit'],
        'digit_b': sample_b['digit'],
        'label': int(sample_a['digit'] == sample_b['digit']),
        'filename': f"{sample_a.get('filename', '?')}+{sample_b.get('filename', '?')}",
        'speaker': f"{sample_a.get('speaker', '?')}+{sample_b.get('speaker', '?')}",
        'digit': -1,
        'stim_a_end_ms': stim_a_end_ms,
        'digit_b_offset_ms': digit_b_offset_ms,
        'stim_b_end_ms': stim_b_end_ms,
    }


def generate_5digit_xor_pairs(samples_by_digit, n_same=2500, n_different=2500,
                               seed=PAIR_SEED):
    """Generate balanced XOR pairs for 5-digit task."""
    rng = np.random.RandomState(seed)
    digits = sorted(samples_by_digit.keys())
    pairs = []

    n_per_digit_same = n_same // len(digits)
    remainder_same = n_same - n_per_digit_same * len(digits)
    for i, d in enumerate(digits):
        pool = samples_by_digit[d]
        n = n_per_digit_same + (1 if i < remainder_same else 0)
        for _ in range(n):
            idx_a, idx_b = rng.choice(len(pool), size=2, replace=False)
            pairs.append(create_xor_stimulus(pool[idx_a], pool[idx_b]))

    diff_combos = [(a, b) for a in digits for b in digits if a != b]
    n_per_combo = n_different // len(diff_combos)
    remainder = n_different - n_per_combo * len(diff_combos)
    for i, (da, db) in enumerate(diff_combos):
        n = n_per_combo + (1 if i < remainder else 0)
        pool_a, pool_b = samples_by_digit[da], samples_by_digit[db]
        for _ in range(n):
            idx_a = rng.choice(len(pool_a))
            idx_b = rng.choice(len(pool_b))
            pairs.append(create_xor_stimulus(pool_a[idx_a], pool_b[idx_b]))

    rng.shuffle(pairs)
    return pairs


def generate_2digit_xor_pairs(samples_by_digit, n_trials=1000, seed=PAIR_SEED):
    """Generate balanced XOR pairs for 2-digit task.

    Forces exact balance: n_trials/4 each of (d0,d0), (d0,d1), (d1,d0), (d1,d1).
    """
    rng = np.random.RandomState(seed)
    digits = sorted(samples_by_digit.keys())
    assert len(digits) == 2, f"Expected 2 digits, got {len(digits)}: {digits}"

    d0, d1 = digits
    n_per_type = n_trials // 4
    pair_types = [(d0, d0), (d0, d1), (d1, d0), (d1, d1)]

    pairs = []
    for da, db in pair_types:
        pool_a = samples_by_digit[da]
        pool_b = samples_by_digit[db]
        for _ in range(n_per_type):
            if da == db:
                idx_a, idx_b = rng.choice(len(pool_a), size=2, replace=False)
            else:
                idx_a = rng.choice(len(pool_a))
                idx_b = rng.choice(len(pool_b))
            pairs.append(create_xor_stimulus(pool_a[idx_a], pool_b[idx_b]))

    rng.shuffle(pairs)
    return pairs


# ============================================================
# TIMING UTILITIES
# ============================================================

def compute_timing_from_pairs(pairs):
    stim_a_ends = np.array([p['stim_a_end_ms'] for p in pairs])
    offsets = np.array([p['digit_b_offset_ms'] for p in pairs])
    stim_b_ends = np.array([p['stim_b_end_ms'] for p in pairs])
    stim_b_durs = stim_b_ends - offsets

    def _stats(arr):
        return {'min': float(arr.min()), 'mean': float(arr.mean()),
                'max': float(arr.max()), 'std': float(arr.std())}

    return {
        'stim_a_end_stats': _stats(stim_a_ends),
        'offset_stats': _stats(offsets),
        'stim_b_dur_stats': _stats(stim_b_durs),
        'max_trial_end_ms': float(stim_b_ends.max()) + POST_STIMULUS_MS,
    }


# ============================================================
# EPOCH BOUNDARY COMPUTATION
# ============================================================

def compute_epoch_boundaries(pair):
    """Compute per-trial epoch bin boundaries from stimulus timing."""
    stim_a_end = pair['stim_a_end_ms']
    b_onset = pair['digit_b_offset_ms']
    b_end = pair['stim_b_end_ms']
    post_end = b_end + POST_STIMULUS_MS

    boundaries = []

    a_bin_ms = stim_a_end / N_BINS_A
    for i in range(N_BINS_A):
        boundaries.append((i * a_bin_ms, (i + 1) * a_bin_ms))

    boundaries.append((stim_a_end, b_onset))

    b_dur = b_end - b_onset
    b_bin_ms = b_dur / N_BINS_B
    for i in range(N_BINS_B):
        boundaries.append((b_onset + i * b_bin_ms, b_onset + (i + 1) * b_bin_ms))

    post_bin_ms = POST_STIMULUS_MS / N_BINS_POST
    for i in range(N_BINS_POST):
        boundaries.append((b_end + i * post_bin_ms, b_end + (i + 1) * post_bin_ms))

    return boundaries


# ============================================================
# FEATURE EXTRACTION (EPOCH-ALIGNED, RATE-NORMALIZED)
# ============================================================

def extract_reservoir_epoch_bins(activity_record, reservoir_indices, dt,
                                  epoch_boundaries):
    """Bin reservoir spikes into epoch-aligned bins, rate-normalized."""
    n_reservoir = len(reservoir_indices)
    n_bins = len(epoch_boundaries)
    res_idx_to_pos = {int(idx): pos for pos, idx in enumerate(reservoir_indices)}

    bin_counts = np.zeros((n_bins, n_reservoir))
    total_spikes = 0

    n_steps = len(activity_record)
    step_to_bin = np.full(n_steps, -1, dtype=np.int32)
    for b, (start_ms, end_ms) in enumerate(epoch_boundaries):
        start_step = int(start_ms / dt)
        end_step = min(int(end_ms / dt), n_steps)
        step_to_bin[start_step:end_step] = b

    for step in range(n_steps):
        b = step_to_bin[step]
        if b < 0:
            continue
        for neuron_idx in activity_record[step]:
            pos = res_idx_to_pos.get(neuron_idx)
            if pos is not None:
                bin_counts[b, pos] += 1
                total_spikes += 1

    for b, (start_ms, end_ms) in enumerate(epoch_boundaries):
        width_ms = end_ms - start_ms
        if width_ms > 0:
            bin_counts[b] /= width_ms

    return bin_counts, total_spikes


def extract_bsa_epoch_bins(xor_stimulus, epoch_boundaries, n_channels=128):
    """Bin BSA input spikes into epoch-aligned bins, rate-normalized."""
    n_bins = len(epoch_boundaries)
    bsa_bins = np.zeros((n_bins, n_channels))

    spike_times = np.asarray(xor_stimulus['spike_times_ms'])
    freq_bins = np.asarray(xor_stimulus['freq_bin_indices'], dtype=int)

    if len(spike_times) == 0:
        return bsa_bins

    starts = np.array([s for s, _ in epoch_boundaries])
    ends = np.array([e for _, e in epoch_boundaries])

    bin_idx = np.searchsorted(starts, spike_times, side='right') - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    valid[valid] &= spike_times[valid] < ends[bin_idx[valid]]

    valid_bins = bin_idx[valid]
    valid_channels = freq_bins[valid]
    np.add.at(bsa_bins, (valid_bins, valid_channels), 1)

    widths = ends - starts
    nonzero = widths > 0
    bsa_bins[nonzero] /= widths[nonzero, np.newaxis]

    return bsa_bins


def compute_duration_features(pairs):
    dur_a = np.array([p['stim_a_end_ms'] for p in pairs])
    dur_b = np.array([p['stim_b_end_ms'] - p['digit_b_offset_ms'] for p in pairs])
    return np.column_stack([dur_a, dur_b])


# ============================================================
# GROUPING FOR CROSS-VALIDATION
# ============================================================

def extract_digit_a_groups(pairs):
    """Extract digit A source recording group IDs for GroupKFold."""
    group_map = {}
    groups = np.zeros(len(pairs), dtype=int)
    for i, p in enumerate(pairs):
        digit_a_file = p['filename'].split('+')[0]
        if digit_a_file not in group_map:
            group_map[digit_a_file] = len(group_map)
        groups[i] = group_map[digit_a_file]
    return groups


def extract_speaker_a_groups(pairs):
    """Extract digit-A speaker IDs for GroupKFold.

    Groups by speaker (e.g. 'george', 'lucas') rather than individual recording.
    This prevents both recording repetition AND speaker leakage across folds.
    """
    speaker_map = {}
    groups = np.zeros(len(pairs), dtype=int)
    for i, p in enumerate(pairs):
        speaker_a = p['speaker'].split('+')[0]
        if speaker_a not in speaker_map:
            speaker_map[speaker_a] = len(speaker_map)
        groups[i] = speaker_map[speaker_a]
    return groups


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_ridge(X, y, cv, groups=None):
    """Ridge classifier with alpha sweep. Returns fold-level results."""
    best_alpha, best_acc = None, -1
    best_preds = None
    best_decisions = None
    best_fold_accs = None

    n_classes = len(np.unique(y))
    multiclass = n_classes > 2

    for alpha in RIDGE_ALPHAS:
        all_preds = np.zeros(len(y), dtype=np.int64)
        if multiclass:
            all_decisions = np.zeros((len(y), n_classes), dtype=np.float64)
        else:
            all_decisions = np.zeros(len(y), dtype=np.float64)
        fold_accs = []
        for train_idx, test_idx in cv.split(X, y, groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            try:
                clf = RidgeClassifier(alpha=alpha)
                clf.fit(X_train, y[train_idx])
                preds = clf.predict(X_test)
                all_preds[test_idx] = preds
                all_decisions[test_idx] = clf.decision_function(X_test)
                fold_accs.append(float(accuracy_score(y[test_idx], preds)))
            except np.linalg.LinAlgError:
                fold_accs.append(1.0 / n_classes)
        acc = accuracy_score(y, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_preds = all_preds.copy()
            best_decisions = all_decisions.copy()
            best_fold_accs = fold_accs

    try:
        if multiclass:
            auc = float(roc_auc_score(y, best_decisions,
                                      multi_class='ovr'))
        else:
            auc = float(roc_auc_score(y, best_decisions))
    except ValueError:
        auc = 0.5

    return {
        'accuracy': float(best_acc),
        'roc_auc': auc,
        'best_alpha': best_alpha,
        'fold_accuracies': best_fold_accs,
        'predictions': best_preds,
        'decision_values': best_decisions,
        'readout': 'Ridge',
    }


# ============================================================
# PAIRED STATISTICS
# ============================================================

def compute_paired_stats(result_a, result_b, label_a='A', label_b='B'):
    """Paired t-test on fold-level accuracies."""
    folds_a = np.array(result_a['fold_accuracies'])
    folds_b = np.array(result_b['fold_accuracies'])
    gaps = folds_a - folds_b

    t_stat, p_value = sp_stats.ttest_rel(folds_a, folds_b)
    d = gaps.mean() / max(gaps.std(ddof=1), 1e-10)

    rng = np.random.RandomState(42)
    boot_gaps = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(gaps), size=len(gaps), replace=True)
        boot_gaps[b] = gaps[idx].mean()

    ci_lo = float(np.percentile(boot_gaps, 2.5))
    ci_hi = float(np.percentile(boot_gaps, 97.5))

    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'

    return {
        'gap_pp': float(result_a['accuracy'] - result_b['accuracy']) * 100,
        'ci_lo_pp': ci_lo * 100,
        'ci_hi_pp': ci_hi * 100,
        'p_value': float(p_value),
        't_stat': float(t_stat),
        'cohens_d': float(d),
        'stars': stars,
        'fold_gaps': gaps.tolist(),
        'label': f'{label_a} vs {label_b}',
    }


def bootstrap_accuracy_ci(predictions, y_true, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    correct = (predictions == y_true).astype(float)
    boot_accs = np.zeros(n_bootstrap)
    n = len(correct)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_accs[b] = correct[idx].mean()
    return float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))
