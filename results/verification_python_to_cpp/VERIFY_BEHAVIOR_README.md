# Behavioral Verification: Python vs C++ (Snapshot) Network

## Motivation

The C++ port of the classification sweep (`liquid_state_machine_cpp_port`) was producing different baseline firing rates than the Python implementation, despite using the same seed (42). Root cause: C++'s `std::mt19937_64` and NumPy's MT19937 produce different random sequences even with identical seeds (different bit width, state initialization). This means the network topology—neuron parameters, connection weights, positions—diverges between implementations.

## Solution: Snapshot Export/Import

Rather than attempting to match RNG sequences across languages, we export the Python-built network structure to an `.npz` file and load it directly in C++, bypassing all random number generation during network construction.

Python's `export_network.py` builds the raw network (`create_ring_zone_network` + `compact_network`) and saves 28 arrays to an `.npz` file with no dynamical overrides applied. The C++ binary loads this snapshot directly via `load_network_snapshot()`, bypassing all C++-side network construction. Both implementations then apply identical LHS-021 overrides at runtime:
- `shell_core_mult = 4.85` (input→reservoir weight scaling)
- `core_core_mult = 0.83` (excitatory reservoir→reservoir scaling)
- `adapt_inc = 0.626` (spike-frequency adaptation increment)
- `nmda_tau = 50.0` (NMDA receptor time constant)

### Exported Arrays (28 total)

| Category | Arrays |
|----------|--------|
| Scalars | `n_neurons`, `sphere_radius` |
| Per-neuron params | `v_rest`, `v_threshold`, `v_reset`, `tau_m`, `tau_ref`, `tau_e`, `tau_i`, `tau_nmda`, `adaptation_increment`, `tau_adaptation`, `e_reversal_arr`, `i_reversal_arr`, `k_reversal_arr` |
| Per-neuron flags | `is_inhibitory`, `is_slow_inhibitory` |
| Per-neuron noise | `v_noise_amp_arr`, `i_noise_amp_arr`, `tau_i_slow` |
| Connectivity | `weights` (n×n), `delays` (n×n) |
| Geometry | `positions` (n×3) |
| Zone info | `input_zone_indices`, `reservoir_zone_indices`, `input_neuron_indices`, `input_neuron_mapping`, `y_threshold` |

## Behavioral Verification Experiment

`verify_behavior.py` runs the LHS-021 baseline through both implementations on 500 audio samples (100 per digit, digits 0–4) and compares aggregate statistics.

### What differs between implementations

Runtime noise is stochastic (different RNG libraries), so exact spike-by-spike match is impossible. Voltage noise (`v_noise_amp`) and current noise (`i_noise_amp`) are drawn from different generators each timestep. The **input** (BSA-encoded audio) and **network structure** (weights, delays, neuron params) are identical.

### Metrics compared

| Metric | Method | Pass criterion |
|--------|--------|----------------|
| Mean firing rate | Direct comparison | < 5 Hz difference |
| Spike count distribution | Two-sample KS test | p > 0.01 |
| Per-sample spike counts | Paired Pearson r (filename-matched) | Informational |
| Classification accuracy | Flat Ridge, 5 repeats | < 5pp gap |

### Results (2026-03-06)

```
Metric                         Python        C++
------------------------------------------------------
n_samples                         500        497
n_reservoir                       604        604
mean_firing_rate_hz             34.33      35.34
classification_accuracy         88.0%      85.6%
classification_std              0.019      0.020

Filename-matched pairs: 155

Per-sample spike count (paired by filename):
  Pearson r:  0.992   (p=3.4e-140)   → near-perfect correlation
  KS test:    stat=0.037, p=0.874    → same distribution

Per-sample firing rate (paired by filename):
  Pearson r:  0.992
  KS test:    stat=0.053, p=0.465    → same distribution

Classification gap: +2.37pp           → within noise

VERDICT: PASS — Behavior is statistically equivalent
```

![Behavioral Verification Scatter Plot](verify_behavior.png)

With filename-matched pairing (155 samples that both implementations loaded), the per-sample Pearson r is **0.992** — the same audio clip through the same network produces nearly identical spike counts despite different runtime noise RNGs. The 1 Hz mean rate difference and 2.4pp classification gap are within the noise floor (~2 standard deviations of the combined classification variance).

### Accuracy gap analysis

The 2.4pp accuracy gap (88.0% vs 85.6%) is within statistical noise. The combined standard deviation of the difference is sqrt(0.019² + 0.020²) ≈ 2.8pp, placing the gap at less than 1σ — not statistically significant.

Three factors account for it:

1. **Runtime noise divergence** — Voltage noise and current noise are drawn from different RNGs each timestep (NumPy vs `std::mt19937_64`). Same network and input, but slightly different noise realizations produce slightly different spike timing and state vectors fed to the Ridge classifier.
2. **Small sample size** — 500 samples with 5-fold cross-validation means each fold trains on ~400 and tests on ~100. A handful of borderline samples flipping changes accuracy by 1–2pp.
3. **Ridge classifier variance** — The 5-repeat std is ~2% for both sides independently, confirming that re-shuffling the train/test split alone moves accuracy by that much.

The per-sample spike count correlation (r = 0.992) is the real signal — the two implementations produce nearly identical responses to the same inputs. The accuracy gap is classifier variance on top of that.

## Usage

```bash
# 1. Export raw network snapshot
python export_network.py -o ../liquid_state_machine_cpp_port/network_snapshot.npz --verify

# 2. Build C++
cd ../liquid_state_machine_cpp_port && make

# 3. Run behavioral verification (500 samples)
cd ../liquid_state_machine_expanded
python verify_behavior.py --n-workers 4

# Skip previously-completed side:
python verify_behavior.py --skip-python   # reuse verify_py.json
python verify_behavior.py --skip-cpp      # reuse verify_cpp.json
```

## Files

| File | Description |
|------|-------------|
| `export_network.py` | Builds raw network (seed=42, compacted) and saves to .npz |
| `verify_behavior.py` | Runs both implementations, compares, generates scatter plot |
| `verify_behavior.png` | 3-panel scatter: spike counts, firing rates, distributions |
| `verify_py.json` | Python per-sample stats (cached) |
| `verify_cpp.json` | C++ per-sample stats (cached) |

## C++ Port Changes

| File | Change |
|------|--------|
| `src/inc/builder.h` | Added `load_network_snapshot()`, `print_network_fingerprint()` |
| `src/src/builder.cpp` | Implemented snapshot loading (~170 lines) |
| `src/src/main.cpp` | Added `--snapshot`, `--verify-only`, `--verify-output`, `--samples-per-digit`, `--data-dir` flags |
