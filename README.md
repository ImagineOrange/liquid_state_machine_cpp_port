# Liquid State Machine — C++ Port

<p align="center">
  <img src="results/verification_python_to_cpp/image.png" alt="alt text" width="500">
</p>

C++ implementation of a spiking Liquid State Machine (LSM) for spoken digit classification. Ports and extends the Python reservoir computing system from `phase_1_Networks_of_LIF_neurons/liquid_state_machine_expanded/`, with ~100x speedup for parameter sweeps.

## Network architecture

The reservoir is a 604-neuron LIF network (compacted from 1000) embedded in a 3D sphere. Neurons use conductance-based synapses with three excitatory channels (AMPA, NMDA with Mg2+ block) and two inhibitory channels (GABA-A, GABA-B). Spike-frequency adaptation is modeled via an afterhyperpolarization current. Biophysical parameters are jittered across neurons.

The sphere is divided into two zones:

- **Input shell** — excitatory neurons on the outer surface, arranged into a 300-degree azimuthal arc, tonotopically mapped to 128 mel-frequency bins via Gaussian tuning curves.
- **Reservoir core** — interior neurons receiving feedforward input from the shell. Intra-shell and feedback connections are removed to enforce unidirectional signal flow.

Excitatory recurrent connections undergo short-term depression (Tsodyks-Markram, U=0.1, tau_rec=500ms). Transmission delays are distance-dependent via ring buffer.

Topology is loaded from a deterministic Python-exported snapshot (`network_snapshot.npz`) for bit-identical structure across implementations.

## Input encoding

Raw audio (WAV) is converted to current injection into the input shell through the following stages:

**1. BSA spike encoding** — WAV files are converted offline to spike trains via Ben's Spiker Algorithm. Each spike has a time (ms) and frequency bin index (0–127, mel-scale). Stored as `.npz` files with arrays `spike_times_ms` and `freq_bin_indices`.

**2. Warmup** — BSA spike times are shifted forward by 50 ms to allow network transients to decay before stimulus onset.

**3. Tonotopic mapping** — Input neurons are mapped to frequency bins via Gaussian tuning curves:

- 128 mel bin centers are placed at evenly-spaced quantiles of the input neuron azimuthal angle distribution, compressing the full spectrum onto the populated arc.
- Each neuron selects its K=4 nearest bin centers by angular distance.
- Weights are Gaussian: `w = exp(-d² / (2σ²))`, σ = 1.5 × mean neuron spacing. Weights at offsets: center = 1.0, ±1 bin = 0.80, ±2 = 0.41, ±3 = 0.14.
- K=4 with quantile centers achieves 128/128 bin coverage, mean weight 0.859.
- Constants in `src/inc/builder.h`: `OVERLAP_K = 4`, `TUNING_SIGMA_CHANNELS = 1.5`.

**4. Current injection** — BSA spikes in bin `m` are injected into all neurons mapped to that bin, scaled by tuning weight: `I = stim_current × w`. Stimulus current is 0.0518 nA (from grid search). NMDA is disabled on input synapses (`skip_stim_nmda = true`).

**5. Input neuron parameters** — Input neurons are LIF with:

- `tau_e = 1.05 ms` — fast excitatory decay prevents conductance saturation while preserving temporal structure.
- `adaptation_increment = 0.0` — adaptation degrades temporal information at this stage.
- No short-term depression (STD U = 0.0).
- Applied via `apply_input_neuron_regime()` in `builder.cpp`.

**6. Shell-to-core projection** — Input shell neurons project to reservoir core through distance-dependent connectivity, with weights scaled by 4.85x (`LHS021_SHELL_CORE_MULT`). Intra-shell and feedback connections are removed.

### Input parameter optimization

Input neuron parameters (`stim_current`, `tau_e`, `adaptation_increment`, STD) were selected via an 8,000-point grid search (`--input-grid` mode) maximizing mutual information between BSA input and output spikes under biological plausibility constraints.

**Search space:** `stim_current` (20 log-spaced, 0.01–5.0) × `tau_e` (10 log-spaced, 0.05–12.0 ms) × `adapt_inc` (8 values, 0–5.0) × STD pairs (5 combinations) = 8,000 points × 30 audio samples = 240,000 simulations.

**Objective:** `score = MI + 0.15 × r@20ms + 0.05 × modulation_depth` with hard gates on firing rate (5–150 Hz), ISI CV (0.3–2.0), refractory fraction (< 10%), and burst fraction (< 15%).

**Optimal parameters** (rank 1, composite score 1.236):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `stim_current` | 0.0518 | ~85 Hz firing rate with tau_e=1.05 |
| `tau_e` | 1.05 ms | Conductance does not saturate at this time constant |
| `adapt_inc` | 0.0 | MI drops steeply above 0.05; top 50 configs all ≤ 0.016 |
| `input_std_u` | 0.0 | Top 4 configs have no STD |
| Input NMDA | disabled | NMDA τ ≈ 50 ms; disabling yields higher MI |

The top 50 configurations occupy a narrow band (score 1.172–1.236) along a constant-rate isocline at ~80–95 Hz. MI peaks at ~1.06 bits (3.0 theoretical max at 8-quantile binning). Refinement with 20 samples/digit confirmed rankings and showed MI estimates are binning-limited: q8=1.19, q16=1.50, q32=1.93 bits.

### Frequency selectivity verification

Single-digit presentation (digit 0, "george_0") confirms frequency information propagates through the input layer:

- Band-rate correlation between BSA and input spikes across 16 frequency bands: r = 0.982.
- 121/129 neurons (94%) show higher correlation with matched BSA bins than unmatched bins.
- Mean matched r: 0.91 vs unmatched: 0.50.

### Figures

- `results/raster/raster_spike_train_0_george_0.png` — mel spectrogram, BSA raster, input shell raster, reservoir raster, population PSTH.
- `results/raster/selectivity_spike_train_0_george_0.png` — frequency band rates and per-neuron selectivity.
- `results/gaussian_tuning_curves/` — tuning coverage and single-neuron detail.
- `results/input_grid_search/` — grid search heatmaps, t-SNE embedding.
- `results/neuron_diagnostics/` — single-neuron state traces.

### Code references

| Component | File | Key symbols |
|-----------|------|-------------|
| Tuning curve construction | `src/src/builder.cpp` | `create_ring_zone_network()`, `load_network_snapshot()` |
| Tuning constants | `src/inc/builder.h` | `OVERLAP_K=4`, `TUNING_SIGMA_CHANNELS=1.5` |
| Input regime defaults | `src/inc/builder.h` | `INPUT_STIM_CURRENT=0.0518`, `INPUT_TAU_E=1.05`, `INPUT_ADAPT_INC=0.0` |
| Input regime application | `src/src/builder.cpp` | `apply_input_neuron_regime()` |
| BSA-to-neuron injection | `src/src/builder.cpp` | `run_sample_with_std()` |
| Grid search | `src/src/input_grid.cpp` | `run_input_grid()`, `run_mi_refine()` |
| Raster dump | `src/src/classification.cpp` | `run_raster_dump()` |
| Selectivity analysis | `experiments/plot_selectivity.py` | |
| Raster figure | `experiments/plot_raster.py` | |

## Experiments

### 1. Python-to-C++ verification

Both implementations run 500 samples through identical topology. Per-sample spike count correlation: r = 0.992. Accuracy gap (2.4pp) is within classifier variance (combined SD ≈ 2.8pp).

C++ fires at 35.3 Hz (vs 34.3 Hz Python), classifies at 85.6% (vs 88.0%). Statistically equivalent.

Files: `results/verification_python_to_cpp/`

### 2. Input regime grid search

8,000-point search over `(stim_current, tau_e, adapt_inc, STD)` maximizing MI between BSA input and output spikes under biological plausibility constraints. See [Input parameter optimization](#input-parameter-optimization) above.

Optimal: `stim=0.0518, tau_e=1.05ms, adapt_inc=0.0, no STD` (score 1.236, MI=1.057 bits, r@20ms=0.884, 85 Hz).

Files: `results/input_grid_search/`

### 3. Top configuration comparison

Traces input neuron 493 through 6 candidate configurations on the same audio sample to verify the grid search optimum produces non-degenerate membrane dynamics.

Files: `results/top_config_comparison/`

### 4. Single-neuron diagnostics

State-variable traces (V, g_e, g_i, g_nmda, adaptation, currents) for input and reservoir neurons at optimal parameters. Input neuron 493: r(g_e, BSA) = 0.920, r(spike, BSA)@20ms = 0.906.

Files: `results/neuron_diagnostics/`

### 5. Classification adaptation sweep

300-point sweep over `(adaptation_increment × adaptation_tau)` — 20 inc × 15 tau — measuring 5-class spoken digit classification accuracy. Two conditions: unmatched (natural firing rate) and tonic-conductance-matched (rate-controlled at 20 Hz).

Readout: dual-form ridge regression (one-vs-rest), 20ms bins × ~604 neurons = 36,240 features per sample, 5-fold stratified CV × 5 repeats, best alpha from {0.01, 0.1, 1, 10, 100, 1000}.

When n << p (1200 samples × 36,240 features), solving the n×n dual system via Cholesky decomposition is ~16.7x faster than p×p SVD with identical predictions. See `src/src/ml.cpp`.

Files: `results/classification_adaptation_sweep/`

### 6. Readout method comparison

30+ readout methods compared on BSA-encoded data: linear (Ridge, Logistic Regression, Linear SVM, LDA), nonlinear (Extra Trees, Random Forest, KNN, HistGBM), and dimensionality reduction pipelines (Truncated SVD + Ridge, PCA + Ridge, coarser bins).

Nonlinear methods reach higher accuracy (Extra Trees 97.1%, KNN k=1 96.5%). Linear readout is used for LSM evaluation per convention. Dual-form ridge matches SVD-based ridge at 3.8–16.7x speedup.

Files: `experiments/readout_benchmark.py`

## Project structure

```
├── src/
│   ├── inc/
│   │   ├── common.h          # RNG, matrix ops, LAPACK (SVD, Cholesky), JSON helpers
│   │   ├── network.h         # SphericalNetwork: LIF dynamics, CSR connectivity, ring buffer
│   │   ├── builder.h         # Network construction, zone topology, tuning curves, sim driver
│   │   ├── ml.h              # Dual-form ridge classifier, StandardScaler, stratified CV, stats
│   │   ├── npz_reader.h      # NumPy .npz file reader (ZIP + zlib)
│   │   └── experiments.h     # Shared constants, types, and helpers for all experiment modes
│   └── src/
│       ├── main.cpp           # CLI parsing and dispatch
│       ├── input_grid.cpp     # Input neuron grid search + MI refinement
│       ├── classification.cpp # Adaptation sweep, trace, verify, classify, calibrate
│       ├── network.cpp        # Spiking dynamics, conductance updates, stimulation
│       ├── builder.cpp        # Ring-zone topology, Gaussian tuning, weight overrides, STD
│       ├── ml.cpp             # Dual-form ridge (Cholesky), ML pipeline, statistical tests
│       └── npz_reader.cpp     # NPZ/NPY parsing
├── docker/
│   └── Dockerfile             # Debian Trixie slim build environment
├── Makefile                   # C++17, -O3, LAPACK/BLAS, zlib, OpenMP
├── dev.sh                     # Docker dev container launcher
├── network_snapshot.npz       # Deterministic Python-exported network topology
├── experiments/               # Python analysis and figure generation scripts
├── data/                      # symlink → external BSA spike train data (see below)
└── results/
    ├── verification_python_to_cpp/
    ├── gaussian_tuning_curves/
    ├── network_snapshot/
    ├── neuron_diagnostics/
    ├── input_grid_search/
    └── classification_adaptation_sweep/
```

## Build and run

```bash
make                    # produces ./cls_sweep
./dev.sh                # or: build in Docker container

# Adaptation sweep
./cls_sweep --arms all --n-workers 8

# Input grid search
./cls_sweep --input-grid --n-workers 8

# MI refinement
./cls_sweep --mi-refine --mi-refine-top 50 --mi-refine-samples 20 --n-workers 8

# Single neuron trace
./cls_sweep --trace-neuron 493 --trace-file data/spike_trains_bsa/spike_train_0_george_0.npz \
    --trace-output trace.csv --no-noise

# Behavioral verification
./cls_sweep --verify-only --verify-output verify_cpp.json --samples-per-digit 100
```

## Data

`data/` is a symlink to BSA-encoded spike train data from `Spiking-Neural-Network-Experiments/phase_1_Networks_of_LIF_neurons/liquid_state_machine_expanded/data/`. Contains ~3000 `.npz` files (~7 GB), 500 samples per digit (digits 0–9).

Each `.npz` has:
- `spike_times_ms` — spike times in milliseconds
- `freq_bin_indices` — mel-frequency bin indices (0–127), parallel to `spike_times_ms`

Setup on a new machine:

```bash
ln -sf /path/to/liquid_state_machine_expanded/data ./data
```

## Dependencies

- C++17 compiler (g++ or clang++)
- LAPACK/BLAS (Accelerate on macOS, liblapack/libblas on Linux)
- zlib
- OpenMP (optional, for parallel simulation)
- Python 3 with numpy, pandas, matplotlib, scipy (for experiment scripts)
