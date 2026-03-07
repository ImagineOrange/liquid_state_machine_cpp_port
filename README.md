# Liquid State Machine — C++ Port

A high-performance C++ implementation of a biologically-detailed Liquid State Machine (LSM) for spoken digit classification. This port replicates and extends the Python-based reservoir computing system from `phase_1_Networks_of_LIF_neurons/liquid_state_machine_expanded/`, providing ~100x speedup for parameter sweeps and grid searches.

## The network

The reservoir is a **604-neuron LIF spiking network** (compacted from 1000) embedded in a 3D sphere. Neurons have conductance-based synapses with three excitatory channels (fast AMPA, slow NMDA with voltage-dependent Mg2+ block) and two inhibitory channels (fast GABA-A, slow GABA-B). Spike-frequency adaptation is modeled as a potassium-like afterhyperpolarization current. All biophysical parameters are jittered across neurons.

The sphere is divided into two functional zones:

- **Input shell** — excitatory neurons on the outer surface, arranged into a 300-degree azimuthal arc. These are tonotopically mapped to 128 mel-frequency bins via Gaussian tuning curves (see below).
- **Reservoir core** — interior neurons receiving feedforward input from the shell. Intra-shell and feedback connections are removed, forcing signal flow inward.

Excitatory recurrent connections undergo short-term depression (Tsodyks-Markram, U=0.1, tau_rec=500ms). Transmission delays are distance-dependent via ring buffer.

The network is loaded from a deterministic Python-exported snapshot (`network_snapshot.npz`) for bit-identical topology across implementations.

## Gaussian frequency tuning curves

Input neurons receive BSA-encoded audio via **Gaussian-weighted frequency tuning**, replacing the original flat/equal overlap scheme. This preserves frequency-specific information through the input layer.

**How it works:**

1. **Quantile-based channel centers.** 128 mel-frequency bin centers are placed at evenly-spaced quantiles of the input neuron phi (azimuthal angle) distribution. This compresses the full frequency spectrum onto the populated arc, eliminating coverage gaps from irregular neuron spacing.

2. **Neuron-to-bins mapping.** Each input neuron selects its K=4 nearest frequency bin centers (by phi distance). This guarantees every neuron maps to exactly K bins — uniform receptive field size.

3. **Gaussian weighting.** Each mapping gets weight `w = exp(-d^2 / (2*sigma^2))` where `d` is the phi-distance between neuron and channel center, and `sigma = 1.5 * avg_neuron_spacing`. At this sigma: center weight = 1.0, 1 channel away = 0.80, 2 channels = 0.41, 3 channels = 0.14.

4. **Injection.** During simulation, BSA spikes in frequency bin `m` are injected into all neurons mapped to bin `m`, scaled by their tuning weight: `I = stim_current * w`.

**Coverage:** K=4 with quantile centers achieves 128/128 bin coverage (every bin has at least one mapped neuron), with mean weight 0.859 and zero wasted weights.

**Key parameters** (defined in `src/inc/builder.h`):
- `OVERLAP_K = 4` — bins per neuron
- `TUNING_SIGMA_CHANNELS = 1.5` — Gaussian sigma in units of channel spacing

## Experiments

### 1. Python-to-C++ behavioral verification

Confirms the C++ port is behaviorally equivalent to the Python implementation — a prerequisite for trusting all downstream experiments. Both implementations run 500 samples through identical topology; per-sample spike count correlation is r = 0.992 and the 2.4pp accuracy gap is within classifier variance.

**Results:** C++ fires at 35.3 Hz (vs 34.3 Hz Python), classifies at 85.6% (vs 88.0%). Statistically equivalent.

**Files:** `results/verification_python_to_cpp/`

### 2. Input neuron regime grid search

The input shell converts continuous BSA spike trains into the reservoir's spiking language — if it destroys frequency information, no downstream processing can recover it. This 8,000-point grid search over `(stim_current, tau_e, adapt_inc, STD params)` finds the input neuron dynamical regime that maximizes mutual information between BSA input and output spikes while staying in a biologically plausible firing regime.

**Results:** Optimal params are `stim=0.0158, tau_e=1.93ms, adapt_inc=0.005, no STD`. MI ≥ 1.18 bits (8-quantile, 20ms bins); refinement at higher quantile counts (q16: ~1.50, q32: ~1.93) shows the estimate is binning-limited. The top 50 configs form a broad plateau with <0.05 bit spread — the optimum is robust, not fragile.

**Files:** `results/input_grid_search/`

### 3. Top configuration comparison

Visual sanity check — traces a single input neuron (493) through 6 candidate configurations on the same audio sample to verify that the grid search optimum produces qualitatively reasonable membrane dynamics rather than a degenerate regime.

**Files:** `results/top_config_comparison/`

### 4. Single-neuron diagnostics

Detailed state-variable traces (V, g_e, g_i, g_nmda, adaptation, all currents) for input and reservoir neurons at the optimal grid search parameters. Confirms the full BSA→conductance→spike transformation chain works as expected.

**Results:** Input neuron 493 achieves r(g_e, BSA) = 0.920 and r(spike, BSA)@20ms = 0.906 — excitatory conductance faithfully tracks BSA input, and spike output preserves most of that correlation.

**Files:** `results/neuron_diagnostics/`

### 5. Classification adaptation sweep (main experiment)

The core scientific question: does spike-frequency adaptation improve reservoir computation? This 159-point sweep over `(adaptation_increment, adaptation_tau)` measures 5-class spoken digit classification accuracy, with each grid point rate-matched to a baseline firing rate via binary search over stimulus current to isolate the effect of adaptation dynamics.

**Readout:** SVD-based ridge regression (one-vs-rest), 20ms time bins flattened, 5-fold stratified CV x 5 repeats.

**Files:** `results/classification_adaptation_sweep/`

## Project structure

```
├── src/
│   ├── inc/
│   │   ├── common.h          # RNG, matrix ops, SVD, JSON helpers
│   │   ├── network.h         # SphericalNetwork: LIF dynamics, CSR connectivity, ring buffer
│   │   ├── builder.h         # Network construction, zone topology, tuning curves, sim driver
│   │   ├── ml.h              # Ridge classifier, StandardScaler, stratified split, stats
│   │   ├── npz_reader.h      # NumPy .npz file reader (ZIP + zlib)
│   │   └── experiments.h     # Shared constants, types, and helpers for all experiment modes
│   └── src/
│       ├── main.cpp           # CLI parsing and dispatch (~120 lines)
│       ├── input_grid.cpp     # Input neuron grid search + MI refinement
│       ├── classification.cpp # Adaptation sweep, trace, verify, classify, calibrate
│       ├── network.cpp        # Spiking dynamics, conductance updates, stimulation
│       ├── builder.cpp        # Ring-zone topology, Gaussian tuning, weight overrides, STD
│       ├── ml.cpp             # ML pipeline and statistical tests
│       └── npz_reader.cpp     # NPZ/NPY parsing
├── docker/
│   └── Dockerfile             # Debian Trixie slim build environment
├── Makefile                   # C++17, -O3, LAPACK/BLAS, zlib, OpenMP
├── dev.sh                     # Docker dev container launcher
├── network_snapshot.npz       # Deterministic Python-exported network topology
├── experiments/               # Python analysis and figure generation scripts
│   ├── gen_input_diagnostic.py    # 10-panel input neuron diagnostic generator
│   ├── gen_3panel_diagnostic.py   # 3-panel BSA→conductance→spikes diagnostic
│   ├── gen_top_configs.py         # Top grid search config comparison figure
│   ├── plot_input_grid.py         # Grid search results visualization
│   ├── plot_tuning_curves.py      # Gaussian tuning coverage overview figure
│   └── plot_tuning_detail.py      # Single-neuron tuning detail figure
├── data/                      # symlink → external BSA spike train data (see below)
└── results/
    ├── verification_python_to_cpp/    # Experiment 1
    ├── gaussian_tuning_curves/        # Gaussian frequency tuning curve figures
    ├── network_snapshot/              # Deterministic network topology
    ├── neuron_diagnostics/            # Experiment 4
    ├── input_grid_search/             # Experiments 2 & 3
    └── classification_adaptation_sweep/ # Experiment 5
```

## Build and run

```bash
make                    # produces ./cls_sweep
./dev.sh                # or: build in Docker container

# Adaptation sweep (main experiment)
./cls_sweep --arms all --n-workers 8

# Input neuron grid search (outputs to results/input_grid_search/)
./cls_sweep --input-grid --n-workers 8

# MI refinement of top grid search configs
./cls_sweep --mi-refine --mi-refine-top 50 --mi-refine-samples 20 --n-workers 8

# Single neuron trace
./cls_sweep --trace-neuron 493 --trace-file data/spike_trains_bsa/spike_train_0_george_0.npz \
    --trace-output trace.csv --no-noise --no-input-nmda \
    --stim-current 0.0158 --input-tau-e 1.93 --input-adapt-inc 0.005

# Behavioral verification
./cls_sweep --verify-only --verify-output verify_cpp.json --samples-per-digit 100
```

## Data dependency

The `data/` directory is a symlink to the BSA-encoded audio spike train data produced by the Python preprocessing pipeline in `Spiking-Neural-Network-Experiments/phase_1_Networks_of_LIF_neurons/liquid_state_machine_expanded/data/`. It contains ~3000 `.npz` files (~7 GB) with 500 samples per digit (digits 0-9).

Each `.npz` file has two arrays:
- `spike_times_ms` — spike times in milliseconds
- `freq_bin_indices` — mel-frequency bin indices (0-127), parallel to spike_times_ms

Both the C++ binary and all experiment scripts resolve data through `data/` relative to the project root. To set up on a new machine:

```bash
ln -sf /path/to/liquid_state_machine_expanded/data ./data
```

## Dependencies

- C++17 compiler (g++ or clang++)
- LAPACK/BLAS (Accelerate on macOS, liblapack/libblas on Linux)
- zlib
- OpenMP (optional, for parallel simulation)
- Python 3 with numpy, pandas, matplotlib, scipy (for experiment scripts)
