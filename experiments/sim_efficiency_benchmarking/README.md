# Simulation Efficiency Benchmarking

Standalone benchmark suite measuring performance of the neuron update loop
and caller-level simulation code. All variants replicate exact production
computations and are verified bitwise-identical to baseline. No core code
is modified until benchmarks confirm the optimization is worthwhile.

## Benchmarks

| File | Tests |
|------|-------|
| `sim_efficiency_benchmarking.cpp` | Neuron loop: memory layout (SoA/AoS/Hybrid) + noise pre-generation |
| `bench_caller_loop.cpp` | Caller loop: allocation patterns, STD lookups, activity recording |

## Build & Run

```bash
# Neuron loop benchmark
c++ -O3 -std=c++17 -o sim_efficiency_benchmarking \
    experiments/sim_efficiency_benchmarking/sim_efficiency_benchmarking.cpp -lm
./sim_efficiency_benchmarking [n_neurons] [n_steps] [n_trials]

# Caller loop benchmark
c++ -O3 -std=c++17 -o bench_caller_loop \
    experiments/sim_efficiency_benchmarking/bench_caller_loop.cpp -lm
./bench_caller_loop [n_neurons] [n_steps] [n_trials]

# Defaults: 604 neurons, 10000 steps, 5 trials
```

## Verification

All variants verified **bitwise-identical** to baseline:
- Per-step spike counts match across all verification steps
- All 7 dynamic state vectors compared via `memcmp` — zero mismatches
- Activity records compared element-by-element across all steps

---

## Results (604 neurons, 10K steps, 5 trials, Apple Silicon)

### 1. Neuron loop: memory layout

| Layout | Min (ms) | ns/neuron | Speedup |
|--------|----------|-----------|---------|
| SoA (baseline) | 357.2 | 59.1 | 1.00x |
| AoS | 344.1 | 57.0 | 1.04x |
| Hybrid | 347.2 | 57.5 | 1.03x |
| **Buffered (K=1)** | **250.9** | **41.5** | **1.42x** |

**Verdict:** At 604 neurons all data fits in L2, so layout changes (AoS/Hybrid)
give only 3-4%. Not worth the refactoring risk.

### 2. Neuron loop: multi-step noise K-sweep

| K | Buffer | Cache tier | Min (ms) | Speedup |
|---|--------|-----------|----------|---------|
| 1 | 14 KB | L1 | 249.2 | 1.43x |
| 5 | 71 KB | L2 | 251.5 | 1.42x |
| 10 | 142 KB | L2 | 249.3 | 1.43x |
| 25 | 354 KB | L2 | 251.0 | 1.42x |
| 50 | 708 KB | L3 | 249.3 | 1.43x |
| 100 | 1.4 MB | L3 | 252.2 | 1.42x |
| 250 | 3.5 MB | L3 | 242.7 | 1.47x |
| 500 | 6.9 MB | L3 | 252.3 | 1.42x |

**Verdict:** No benefit over K=1. All K values cluster at 1.42-1.43x.
RNG overhead is fully amortized by the single-step separation.

### 3. Caller loop: allocation + lookup optimizations

Tested four optimizations applied together:
1. **Persistent spike buffer** — reuse `vector<int>` vs `new` per step
2. **CSR-flat activity record** — single flat vector + offsets vs `vector<vector<int>>`
3. **Flat STD lookups** — `vector<bool>` O(1) vs `set::find`/`map::find` O(log n)
4. **Pre-allocated input scratch** — reuse buffers vs per-step allocation

| Variant | Min (ms) | ns/neuron | Speedup |
|---------|----------|-----------|---------|
| Baseline (current patterns) | 233.0 | 38.6 | 1.00x |
| Optimized (#1-#4 combined) | 228.6 | 37.8 | 1.02x |

**Verdict:** Only 2% improvement. At this neuron count, modern allocators handle
small vectors efficiently, and the tree-based containers (382 entries, ~9
comparisons) are fast enough. Not worth the code complexity.

---

## Summary

| Optimization | Speedup | Status |
|---|---|---|
| **Pre-buffered noise (K=1)** | **1.42x** | **Implemented in core** |
| AoS memory layout | 1.04x | Not worth it |
| Hybrid memory layout | 1.03x | Not worth it |
| Multi-step noise K>1 | ~1.00x | No benefit over K=1 |
| Caller-loop #1-#4 | 1.02x | Marginal — not implemented |

### Summary

The measured speedup comes from separating RNG calls from the neuron
arithmetic loop. With 3 `rng_normal()` calls interleaved in the membrane
equation, function call barriers prevent auto-vectorization. Pre-generating all
normals into a small buffer (14 KB, fits L1) allows the compiler to vectorize the
pure-arithmetic neuron loop.

### Projected sweep savings (pre-buffered noise only)

| Metric | Value |
|--------|-------|
| Per grid point (before) | 268 s |
| Per grid point (after) | 182 s |
| Savings per point | 86 s (32%) |
| 300-point sweep | ~7.2 hours saved |
