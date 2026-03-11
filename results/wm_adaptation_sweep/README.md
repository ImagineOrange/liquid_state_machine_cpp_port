# Working Memory / XOR Adaptation Sweep

A parameter sweep over spike-frequency adaptation (adapt_inc, adapt_tau) measuring two higher-order cognitive tasks: **working memory** (5-class recall of digit A from the B+Post epoch) and **temporal XOR** (same vs. different digit pair classification). The central question: **can adaptation sustain a stimulus trace across a silent gap and support nonlinear temporal integration?**

## Key results

### Working memory (5-class, chance = 20%)

| | Best accuracy | adapt_inc | adapt_tau | Rate | vs BSA (26.61%) | vs LHS-021 (35.66%) |
|---|---|---|---|---|---|---|
| **Branch A** (unmatched) | **83.07%** | 0.171 | 5000 ms | 22.1 Hz | +56.46 pp | +47.41 pp |
| **Branch B** (rate-matched) | **81.90%** | 0.171 | 5000 ms | 18.6 Hz | +55.30 pp | +46.24 pp |

Top-3 configurations converge on the same regime: moderate adapt_inc (0.13-0.23), long adapt_tau (3500-5000 ms). Both branches agree on the optimal point (inc=0.171, tau=5000), and Branch B confirms the effect survives rate matching — the +1.17 pp gap between branches is negligible compared to the +55 pp gain over BSA.

### Temporal XOR (2-class, chance = 50%)

| | Best accuracy | adapt_inc | adapt_tau | Rate | vs BSA (51.24%) | vs LHS-021 (51.95%) |
|---|---|---|---|---|---|---|
| **Branch A** (unmatched) | **67.24%** | 0.0707 | 5000 ms | 29.1 Hz | +16.00 pp | +15.29 pp |
| **Branch B** (rate-matched) | **67.57%** | 0.0527 | 5000 ms | 18.3 Hz | +16.33 pp | +15.62 pp |

XOR peaks at lower adaptation strength than WM (inc ~0.05-0.07 vs ~0.17), but the same long time constant (tau=5000 ms). Branch B slightly *outperforms* Branch A, ruling out rate as the explanation. The optimal XOR regime requires weaker adaptation — enough to sustain a trace, but not so much that it dominates B-epoch dynamics and prevents the nonlinear A-vs-B comparison.

## Motivation

Classification (the first sweep) tests whether adaptation steers the reservoir into distinct computational modes during stimulus presentation. But the paper's strongest claim is about *working memory* — the reservoir maintaining a representation of a vanished stimulus across a delay period. This requires a fundamentally different computation: information about digit A must persist through a 150 ms silent gap and remain decodable from activity during digit B and after.

The XOR task goes further: classifying whether two sequentially presented digits are the *same* or *different* requires nonlinear integration of information across both epochs — neither digit alone determines the label.

## Task structure

### Compound stimulus

Each trial presents two spoken digits separated by a silent gap:

```
[─── digit A ───][── 150 ms gap ──][─── digit B ───][── 200 ms post ──]
     5 bins            1 bin             5 bins           2 bins
                                   ◄──── readout window (bins 6-12) ────►
```

- **Digit A**: First spoken digit (BSA-encoded into input spike trains)
- **Gap**: 150 ms silent period — no input spikes, reservoir activity evolves freely
- **Digit B**: Second spoken digit
- **Post-stimulus**: 200 ms after digit B offset
- **Bin size**: 20 ms (rate-normalized spike counts per neuron per bin)

### Pair construction

2,500 digit pairs per grid point, generated deterministically (seed=99):
- **1,250 same-digit pairs** (A = B): label = 1
- **1,250 different-digit pairs** (A ≠ B): label = 0
- Digits drawn from {0, 1, 2, 3, 4} (5-class FSDD subset, 300 recordings per digit)

### Working memory task

- **Goal**: Classify which digit A was presented (5-class)
- **Features**: B+Post epoch only (bins 6-12) — digit A has already ended
- **Pairs used**: Different-digit pairs only (A ≠ B), so digit B identity cannot trivially reveal A
- **Chance level**: 20% (5 classes)
- **What it measures**: Whether the reservoir retains a decodable trace of digit A through the gap and into the B+Post epoch

### XOR task

- **Goal**: Classify same vs. different digit pair (2-class)
- **Features**: B+Post epoch only (bins 6-12)
- **Pairs used**: All 2,500 pairs
- **Chance level**: 50% (2 classes)
- **What it measures**: Nonlinear temporal integration — the label depends on the *relationship* between A and B, not either alone

## The rate-matching confound

Identical to the classification sweep. See [classification sweep README](../classification_adaptation_sweep/README.md) for the full treatment of why single-parameter rate matching always has side effects, and why tonic-conductance-based shunting inhibition is the least-confounded approach.

## Two-branch design

Same two-branch structure as the classification sweep:

### Branch A: Unmatched (natural rate)

- **stimulus_current**: Fixed at INPUT_STIM_CURRENT (0.0518 nA)
- **tonic_conductance**: None (0.0)
- **Rate**: Varies naturally with adaptation parameters
- **Input encoding**: Identical across all grid points (MI = 1.06 bits preserved)
- **Controls for**: Input encoding confound (eliminated)

### Branch B: Tonic-conductance-matched (shunting inhibition)

- **stimulus_current**: Fixed at INPUT_STIM_CURRENT (0.0518 nA)
- **tonic_conductance**: Calibrated per grid point via binary search to match 20 Hz target reservoir firing rate. Applied as `g_tonic * (E_rev - V)` to reservoir neurons only.
- **Rate**: Matched to 20 Hz across all grid points
- **Input encoding**: Identical across all grid points (MI = 1.06 bits preserved)
- **Controls for**: Both input encoding confound and firing rate confound

### Calibration

Rate calibration uses a 200-pair subset for speed. Binary search runs until reservoir firing rate is within +/-2 Hz of the 20 Hz target — no iteration limit.

## Fixed parameters

All parameters identical to the classification sweep (LHS-021 network, input grid search optimum). See [classification sweep README](../classification_adaptation_sweep/README.md) for the full parameter tables.

### WM-specific parameters

| Parameter | Value | Source |
|---|---|---|
| gap_ms | 150.0 ms | Silent period between A and B |
| n_bins_a | 5 | Stimulus A epoch bins |
| n_bins_gap | 1 | Gap epoch bin |
| n_bins_b | 5 | Stimulus B epoch bins |
| n_bins_post | 2 | Post-stimulus epoch bins (200 ms) |
| total_bins | 13 | All epochs combined |
| readout_window | bins 6-12 (B+Post) | 7 bins, excludes A and gap |
| n_pairs | 2500 | Per grid point |
| pair_seed | 99 | Deterministic pair generation |
| calibration_n_pairs | 200 | Subset for rate calibration |

## Sweep grid

Identical grid to the classification sweep — both sweeps share the same adaptation parameter space for direct comparison.

### Grid axes

| Axis | Points | Range | Spacing | Ratio |
|---|---|---|---|---|
| adapt_inc | 20 | 0.0 + 19 values from 0.005 to 1.0 | log | ~1.34x per step |
| adapt_tau | 15 | 30 to 5000 ms | log | ~1.44x per step |

**adapt_inc**: 0.0, 0.005, 0.0067, 0.009, 0.0121, 0.0162, 0.0218, 0.0292, 0.0392, 0.0527, 0.0707, 0.0949, 0.1274, 0.171, 0.2295, 0.3081, 0.4135, 0.555, 0.745, 1.0

**adapt_tau** (ms): 30, 43.2, 62.3, 89.8, 129.4, 186.5, 268.7, 387.3, 558.1, 804.4, 1159.2, 1670.6, 2407.5, 3469.5, 5000

Full grid: 20 x 15 = **300 points**, every combination evaluated. No holes.

### Per grid point

**Branch A (unmatched):**
1. Set reservoir adaptation to (adapt_inc, adapt_tau)
2. stimulus_current = INPUT_STIM_CURRENT (fixed)
3. tonic_conductance = 0.0 (none)
4. Simulate all 2500 pairs (compound A+gap+B+post stimuli)
5. Bin activity into 13 epoch bins
6. Classify WM (5-class, B+Post, different-digit pairs only) and XOR (2-class, B+Post, all pairs)

**Branch B (tonic-conductance-matched):**
1. Set reservoir adaptation to (adapt_inc, adapt_tau)
2. stimulus_current = INPUT_STIM_CURRENT (fixed)
3. Calibrate tonic_conductance to 20 Hz target (200-pair subset)
4. Simulate all 2500 pairs at calibrated tonic_conductance
5. Bin activity into 13 epoch bins
6. Classify WM and XOR as above

## Baselines

| Baseline | WM accuracy | XOR accuracy | Notes |
|---|---|---|---|
| BSA-only | 26.61% | 51.24% | Input spike trains only, no reservoir |
| LHS-021 (default adaptation) | 35.66% | 51.95% | LHS-021 network at native adaptation settings |
| Chance | 20.00% | 50.00% | Random |

## Readout

- **Feature matrix**: 7 B+Post bins × ~604 reservoir neurons = ~4,228 features, flattened per sample
- **Classifier**: One-vs-rest ridge regression (dual-form Cholesky), best alpha from {0.01, 0.1, 1, 10, 100, 1000}
- **Cross-validation**: 5-fold stratified × 5 repeats
- **WM readout**: Different-digit pairs only (~1,250 samples), 5-class
- **XOR readout**: All pairs (2,500 samples), 2-class

## JSON output schema

### Top-level fields

| Field | Description |
|---|---|
| `experiment` | `"Working Memory Adaptation Parameter Sweep (C++ Port)"` |
| `total_time_s` | Total wall-clock time |
| `task` | `"Working memory: classify digit_a from B+Post (different-digit pairs)"` |
| `digits` | `[0, 1, 2, 3, 4]` |
| `n_pairs` | 2500 |
| `wm_chance_level` | 0.20 |
| `xor_chance_level` | 0.50 |
| `epoch_bins` | `{n_bins_a, n_bins_gap, n_bins_b, n_bins_post, total}` |
| `readout_window` | `"B+Post (bins 6-12)"` |
| `gap_ms` | 150.0 |
| `rate_matching` | `{target_rate_hz, tolerance_hz}` |
| `grid` | `{unified_inc, unified_tau, n_inc, n_tau}` |
| `bsa_wm_baseline` | BSA-only WM classification result |
| `bsa_xor_baseline` | BSA-only XOR classification result |
| `lhs021_baseline` | LHS-021 baseline WM + XOR + firing rate |
| `grid_results` | Array of per-grid-point results (see below) |

### Per grid point (both branches)

Each grid point produces two JSON entries (one per branch) with the following fields:

| Field | Type | Description |
|---|---|---|
| `branch` | string | `"A_unmatched"` or `"B_matched"` |
| `point_id` | string | Grid point identifier |
| `inc_idx` | int | Index into unified adapt_inc axis |
| `tau_idx` | int | Index into unified adapt_tau axis |
| `adapt_inc` | float | Spike-frequency adaptation increment |
| `adapt_tau` | float | Adaptation time constant (ms) |
| `stimulus_current` | float | Input stimulus current (nA), fixed at INPUT_STIM_CURRENT |
| `tonic_conductance` | float | Tonic conductance: 0.0 for Branch A, calibrated for Branch B |
| `tonic_reversal` | float | Reversal potential (mV): -80 (inhibitory) or 0 (excitatory) |
| `calibration_rate_hz` | float | Reservoir firing rate achieved during calibration |
| `wm_accuracy` | float | Mean WM accuracy across 5-fold x 5-repeat CV |
| `wm_accuracy_std` | float | Std of per-repeat WM accuracies |
| `xor_accuracy` | float | Mean XOR accuracy across 5-fold x 5-repeat CV |
| `xor_accuracy_std` | float | Std of per-repeat XOR accuracies |
| `firing_rate_hz` | float | Mean reservoir firing rate across all pairs (Hz) |
| `firing_rate_std` | float | Std of per-pair reservoir firing rates |
| `n_reservoir` | int | Number of reservoir neurons |
| `sim_time_s` | float | Wall-clock simulation time (seconds) |
| `isi_cv_mean` | float | Mean ISI coefficient of variation (reservoir, during B epoch) |
| `adapt_at_b_onset_mean` | float | Mean adaptation variable at digit B onset |
| `participation_ratio_mean` | float | SVD-based dimensionality of reservoir activity |
| `per_bin_accuracy` | float[] | WM classification accuracy per epoch bin (length 13) |
| `wm_per_repeat_accuracy` | float[] | Per-repeat WM accuracy (length 5) |
| `xor_per_repeat_accuracy` | float[] | Per-repeat XOR accuracy (length 5) |

### Notable differences from classification sweep schema

| Field | Classification sweep | WM sweep |
|---|---|---|
| `classification_accuracy` | Present | Absent — replaced by `wm_accuracy` and `xor_accuracy` |
| `classification_gap_pp` etc. | Paired stats vs BSA | Not included (baselines stored at top level) |
| `adapt_at_stim_end_mean` | At stimulus offset | `adapt_at_b_onset_mean` — at digit B onset (captures adaptation state entering the WM readout epoch) |
| `per_bin_accuracy` | Per 20ms bin during stimulus | Per epoch bin across all 13 bins (A, gap, B, post) |

## Code references

| Component | File | Key functions/constants |
|---|---|---|
| Sweep entry point | `src/src/wm_sweep.cpp` | `run_wm_sweep()` |
| WM constants | `src/src/wm_sweep.cpp` | `GAP_MS`, `N_BINS_A/GAP/B/POST`, `N_WM_PAIRS`, `PAIR_SEED` |
| Pair generation | `src/src/wm_sweep.cpp` | `generate_xor_pairs()`, `XorPair` struct |
| Epoch binning | `src/src/wm_sweep.cpp` | `bin_activity_epochs()` |
| WM classification | `src/src/wm_sweep.cpp` | `classify_wm_bpost()` |
| XOR classification | `src/src/wm_sweep.cpp` | `classify_xor_bpost()` |
| Simulation worker | `src/src/wm_sweep.cpp` | `wm_sim_worker()` |
| Parallel pair runner | `src/src/wm_sweep.cpp` | `run_all_pairs()` |
| Rate calibration | `src/src/wm_sweep.cpp` | Binary search in `run_wm_sweep()` (same approach as classification sweep) |
| Rate target | `src/inc/experiments.h` | `RATE_TARGET_HZ = 20.0`, `RATE_TOLERANCE_HZ = 2.0` |
| Shunting inhibition | `src/src/network.cpp` | `update_network()` — `i_tonic = g_tonic * (E_i - V)` |
| Network base config | `src/src/builder.cpp` | `make_base_config()` |
| LHS-021 overrides | `src/inc/experiments.h` | `LHS021_SHELL_CORE_MULT`, `LHS021_CORE_CORE_MULT`, etc. |
| Dual-form ridge | `src/src/ml.cpp` | `ridge_fold_prepare()`, `ridge_fold_solve()` |
| CLI entry | `src/src/main.cpp` | `--wm-sweep` flag |
| Triplet heatmap plot | `experiments/plot_triplet_bsa_comparison.py` | 6-panel heatmap (classification + WM + XOR) |

## Reproduction

```bash
# Run full 300-point grid x 2 branches
./wm_sweep --n-workers 8

# Results saved to:
#   results/wm_adaptation_sweep/wm_adaptation_sweep.json
# Checkpoint saved after each grid point:
#   results/wm_adaptation_sweep/wm_adaptation_sweep_checkpoint.json

# Generate 6-panel triplet heatmap (requires classification sweep results too)
python experiments/plot_triplet_bsa_comparison.py
# Outputs: results/triplet_bsa_comparison.png
```
