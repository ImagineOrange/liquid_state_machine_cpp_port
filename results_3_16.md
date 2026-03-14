# Spike-Frequency Adaptation Modulates Computational Mode in a Spiking Liquid State Machine

## A triple dissociation across classification, working memory, and temporal XOR tasks

---

## Abstract

A single, structurally frozen 604-neuron spiking reservoir can support qualitatively different computational strategies — instantaneous pattern discrimination, persistent working memory, and temporal comparison — by modulating only two parameters of spike-frequency adaptation (SFA). A 300-point sweep over adaptation increment and time constant, evaluated on three tasks with rate-matched controls, reveals a statistically verified triple dissociation: classification (CLS) peaks at low adaptation, temporal XOR at weak-to-moderate adaptation, and working memory (WM) at moderate-to-strong adaptation, with all tasks preferring long time constants. The dissociation is not a parametric performance curve where one end is "better" — per-bin temporal decomposition shows that adaptation changes the *kind* of computation, not the *amount*. Mechanistic interpretability analysis traces working memory to a distributed adaptation-conductance code that is near-perfectly decodable (98%) but bottlenecked by the spike-count readout (82%). These results demonstrate that neuromodulatory control of a single biophysical parameter — the afterhyperpolarization current — can function as a mode-switching mechanism that reconfigures a cortical circuit between temporal discrimination and working memory without synaptic plasticity.

---

## 1. Network Architecture

The reservoir is a 604-neuron leaky integrate-and-fire (LIF) network (compacted from 1000) embedded in a 3D sphere. Neurons use conductance-based synapses with three excitatory channels (AMPA, NMDA with Mg2+ block) and two inhibitory channels (GABA-A, GABA-B). Spike-frequency adaptation is modeled via an afterhyperpolarization (AHP) current. Biophysical parameters are jittered across neurons.

The sphere is divided into two functional zones:

- **Input shell** — excitatory neurons on the outer surface, arranged into a 300-degree azimuthal arc, tonotopically mapped to 128 mel-frequency bins via Gaussian tuning curves.
- **Reservoir core** — interior neurons receiving feedforward input from the shell. Intra-shell and feedback connections are removed to enforce unidirectional signal flow.

Excitatory recurrent connections undergo short-term depression (Tsodyks-Markram, U=0.1, tau_rec=500 ms). Transmission delays are distance-dependent via ring buffer.

Topology is loaded from a deterministic Python-exported snapshot (`network_snapshot.npz`) for bit-identical structure across implementations. The C++ port was validated against the original Python implementation (per-sample spike count correlation r = 0.992, accuracy gap < 1σ; see Section 3.1).

### 1.1 Input encoding

Raw audio (WAV) is converted to current injection into the input shell:

1. **BSA spike encoding** — WAV files are converted offline to spike trains via Ben's Spiker Algorithm. Each spike has a time (ms) and frequency bin index (0–127, mel-scale).
2. **Warmup** — BSA spike times are shifted forward by 50 ms to allow network transients to decay.
3. **Tonotopic mapping** — Input neurons are mapped to frequency bins via Gaussian tuning curves. 128 mel bin centers are placed at evenly-spaced quantiles of the input neuron azimuthal angle distribution. Each neuron selects its K=4 nearest bin centers. Weights are Gaussian: `w = exp(-d² / (2σ²))`, σ = 1.5 × mean neuron spacing. K=4 with quantile centers achieves 128/128 bin coverage, mean weight 0.859.
4. **Current injection** — BSA spikes in bin *m* are injected into all neurons mapped to that bin, scaled by tuning weight: `I = stim_current × w`. Stimulus current is 0.0518 nA. NMDA is disabled on input synapses.
5. **Input neuron parameters** — `tau_e = 1.05 ms` (fast excitatory decay prevents conductance saturation), `adaptation_increment = 0.0`, no short-term depression.
6. **Shell-to-core projection** — Input shell neurons project to reservoir core through distance-dependent connectivity, with weights scaled by 4.85×.

### 1.2 Input parameter optimization

Input neuron parameters were selected via an 8,000-point grid search maximizing mutual information between BSA input and output spikes under biological plausibility constraints.

**Search space:** `stim_current` (20 log-spaced, 0.01–5.0) × `tau_e` (10 log-spaced, 0.05–12.0 ms) × `adapt_inc` (8 values, 0–5.0) × STD pairs (5 combinations) = 8,000 points × 30 audio samples = 240,000 simulations.

**Objective:** `score = MI + 0.15 × r@20ms + 0.05 × modulation_depth` with hard gates on firing rate (5–150 Hz), ISI CV (0.3–2.0), refractory fraction (< 10%), and burst fraction (< 15%).

**MI estimation:** Per-neuron spike and BSA time series are binned at 20 ms, then pooled across all 30 samples before computing MI with 8-quantile discretization. Pooling gives ~800 joint observations per neuron (vs ~27 per sample), reducing Miller-Madow bias from ~1.7 bits to ~0.06 bits.

**Optimal parameters** (rank 1, composite score 1.236):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `stim_current` | 0.0518 nA | ~85 Hz firing rate with tau_e=1.05 |
| `tau_e` | 1.05 ms | Conductance does not saturate at this time constant |
| `adapt_inc` | 0.0 | MI drops steeply above 0.05; top 50 configs all ≤ 0.016 |
| `input_std_u` | 0.0 | Top 4 configs have no STD |
| Input NMDA | disabled | NMDA τ ≈ 50 ms; disabling yields higher MI |

The top 50 configurations occupy a narrow band (score 1.172–1.236) along a constant-rate isocline at ~80–95 Hz. MI peaks at ~1.06 bits (3.0 theoretical max at 8-quantile binning). No parameter combination exceeded ~1.07 bits.

### 1.3 Frequency selectivity verification

Single-digit presentation (digit 0, "george_0") confirms frequency information propagates through the input layer:
- Band-rate correlation between BSA and input spikes across 16 frequency bands: r = 0.982
- 121/129 neurons (94%) show higher correlation with matched BSA bins than unmatched bins
- Mean matched r: 0.91 vs unmatched: 0.50

---

## 2. Experimental Design

### 2.1 Three tasks, one reservoir

Three tasks are evaluated on the same LHS-021 spiking reservoir across a shared 20 × 15 grid of adaptation parameters (adapt_inc × adapt_tau = 300 grid points). Each grid point is run in two branches: Branch A (unmatched — natural firing rate) and Branch B (tonic-conductance-matched — reservoir rate clamped to 20 Hz). Input encoding is held fixed at the grid-search optimum across all conditions. Readout is a linear ridge classifier (5-fold × 5-repeat stratified CV).

The three tasks share the same reservoir and adaptation parameters but differ in what they ask the readout to extract:

- **Classification (CLS)**: Identify which of 5 digits was presented, using all temporal bins during a single stimulus. A pure stimulus-discrimination task with no memory requirement. 1,500 samples (300 per digit × 5 digits), 48 bins × 604 neurons = 36,240 features.

- **Working memory (WM)**: Identify digit A from reservoir activity during the B+Post epoch, after a 150 ms silent gap and an interfering digit B. Only different-digit pairs are used. 2,500 digit pairs per grid point; readout from B+Post epoch only (7 bins × 604 neurons = 4,228 features). Chance = 20%.

- **Temporal XOR**: Classify whether a digit pair is same or different, reading out from the B+Post epoch. The label depends on the relationship between A and B — neither alone is sufficient. 2,500 pairs; chance = 50%.

### 2.2 Rate matching

Changing adaptation changes reservoir firing rate. To isolate adaptation dynamics from rate effects, the experiment uses a two-branch design:

**Branch A (unmatched):** Stimulus current fixed at 0.0518 nA, no tonic conductance, rate varies naturally with adaptation.

**Branch B (rate-matched):** Same fixed stimulus current, but a tonic conductance `g_tonic × (E_rev − V)` is calibrated per grid point via binary search to clamp reservoir firing to 20 Hz (midpoint of auditory cortex evoked rates). Reversal is automatically chosen: inhibitory (−80 mV) if rate too high, excitatory (0 mV) if too low. This acts on reservoir neurons only, preserving input encoding (MI = 1.06 bits) across all conditions.

Tonic conductance was chosen over alternatives (stimulus current modulation, weight scaling, threshold adjustment) because it acts on reservoir neurons only, does not alter input encoding quality, and competes with synaptic drive on equal terms as a conductance-based mechanism.

### 2.3 Readout

One-vs-rest ridge regression using the dual-form Cholesky solver. The Gram matrix K = XX^T (n×n, where n = number of samples) is precomputed once per fold; when n << p (e.g., 1200 samples × 36,240 features), this is ~16.7× faster than p×p SVD with identical predictions. Best alpha selected from {0.01, 0.1, 1, 10, 100, 1000}. 5-fold stratified × 5 repeats.

Readout benchmark (30+ methods on BSA-encoded data):

| Method | Accuracy | Type |
|--------|----------|------|
| Extra Trees | 97.1% | Nonlinear |
| KNN k=1 | 96.5% | Nonlinear |
| Logistic Regression (L2) | 95.8% | Linear |
| **Dual Ridge (ours)** | **95.2%** | **Linear** |
| Linear SVM | 94.8% | Linear |
| LDA | 93.6% | Linear |

Linear readout is used for LSM evaluation per convention (Maass et al. 2002). Nonlinear methods reach up to 97.1%.

### 2.4 Sweep grid

Both axes are log-spaced:

| Axis | Points | Range | Ratio per step |
|------|--------|-------|----------------|
| adapt_inc | 20 | 0.0 + 19 values from 0.005 to 1.0 | ~1.34× |
| adapt_tau | 15 | 30 to 5000 ms | ~1.44× |

Full grid: 20 × 15 = **300 points** per branch, every combination evaluated.

---

## 3. Verification and Engineering

### 3.1 Python-to-C++ behavioral verification

The C++ port was validated against the original Python implementation. Both run 500 samples through identical topology (loaded from the same `.npz` snapshot). Runtime noise diverges (different RNG libraries) so exact spike-by-spike match is impossible, but aggregate statistics confirm equivalence:

| Metric | Python | C++ |
|--------|--------|-----|
| Mean firing rate | 34.3 Hz | 35.3 Hz |
| Classification accuracy | 88.0% | 85.6% |
| Per-sample spike count r (155 matched pairs) | 0.992 | |
| KS test (spike counts) | p = 0.874 | |
| Accuracy gap | +2.4 pp (< 1σ of combined SD ≈ 2.8 pp) | |

**Verdict: PASS.** Statistically equivalent.

### 3.2 Simulation efficiency

Benchmarking of the neuron update loop identified one worthwhile optimization: pre-buffered noise generation (separating RNG calls from the neuron arithmetic loop) yields a 1.42× speedup by enabling auto-vectorization. Memory layout changes (AoS, Hybrid) gave only 3–4% at 604 neurons. Caller-loop optimizations (persistent buffers, CSR activity record, flat STD lookups) gave only 2%.

| Optimization | Speedup | Status |
|---|---|---|
| **Pre-buffered noise (K=1)** | **1.42×** | **Implemented** |
| AoS/Hybrid layout | 1.03–1.04× | Not worth refactoring risk |
| Caller-loop optimizations | 1.02× | Marginal |

Projected savings: ~7.2 hours across a 300-point sweep (268s → 182s per grid point).

---

## 4. Results: Task Optima Occupy Distinct Regions of Adaptation Space

### 4.1 Global optima

| Task | Branch | Peak inc | Peak tau (ms) | Peak accuracy | Top-5 centroid inc | Top-5 centroid tau |
|------|--------|----------|---------------|---------------|--------------------|--------------------|
| CLS | B | 0.0707 | 5000 | 96.33% | 0.042 | 3102 |
| WM | B | 0.1710 | 5000 | 81.90% | 0.186 | 4391 |
| XOR | B | 0.0527 | 5000 | 67.57% | 0.095 | 4389 |
| CLS | A | 0.0292 | 2408 | 95.75% | 0.043 | 2741 |
| WM | A | 0.1710 | 5000 | 83.07% | 0.194 | 3876 |
| XOR | A | 0.0707 | 5000 | 67.24% | 0.057 | 4693 |

All three tasks prefer long adaptation time constants (tau = 3000–5000 ms). They separate primarily along adapt_inc: CLS peaks at low adaptation strength (inc ≈ 0.03–0.07), XOR at weak-to-moderate (inc ≈ 0.05–0.07), and WM at moderate-to-strong (inc ≈ 0.17). WM's optimum is identical across branches (inc = 0.171, tau = 5000).

### 4.2 Statistical verification of the triple dissociation

Three independent statistical tests confirm the dissociation:

**Bootstrap of optimal inc (tau=5000 slice, Branch B, 10,000 iterations):** CLS landed at inc=0.0707 in 100% of iterations, XOR at 0.0527 in 99.8%, WM at 0.171 in 87.8% (remainder at 0.2295). No overlap between any pair. Kruskal-Wallis H = 29,522, p ≈ 0.

**Friedman test with softmax center-of-mass (Branch B, tau ≥ 558 ms):** At each of 7 tau values, a softmax-weighted center-of-mass (T=50) estimates the optimal adaptation strength as a continuous variable. Friedman χ² = 14.0, p = 0.0009. All post-hoc Wilcoxon signed-rank tests significant after Bonferroni correction (all W = 0, p = 0.016). The ordering CLS < XOR < WM held at every tau slice. Mean softmax CoM: CLS = 0.096, XOR = 0.141, WM = 0.293.

**Pareto frontier paired t-tests (tau=5000):** CLS advantage at CLS-optimal vs WM-optimal: 1.49 pp, t = 59.3, df = 4, p < 0.0001. WM advantage at WM-optimal vs CLS-optimal: 6.61 pp, t = 10.0, p = 0.0006. Both significant and in opposite directions.

### 4.3 The inc × tau interaction

For WM, the optimal inc shifts with tau: at tau = 558 ms, the WM argmax is inc = 0.41; at tau = 5000 ms, it falls to inc = 0.17. This is consistent with a constraint on the equilibrium adaptation *state* rather than the raw parameters. The product inc × tau is not a sufficient statistic — along iso-product lines, WM accuracy varies by 15–32 pp.

An additive model log(inc) + log(tau) explains 84% of WM variance with a tau/inc beta ratio of 1.87, compared to 9.0 for XOR and a negative inc coefficient for CLS.

---

## 5. The Pareto Frontier Between Classification and Working Memory

### 5.1 Frontier structure

In Branch B, only 5 of 300 grid points are Pareto-optimal for CLS × WM:

| Rank | inc | tau | CLS | WM | CLS rank | WM rank |
|------|-----|-----|-----|-----|----------|---------|
| 1 | 0.0707 | 5000 | 96.33% | 75.30% | 1/300 | 30/300 |
| 2 | 0.0949 | 5000 | 95.54% | 77.58% | 34/300 | 19/300 |
| 3 | 0.1274 | 3470 | 95.51% | 79.92% | 36/300 | 9/300 |
| 4 | 0.1274 | 5000 | 95.23% | 81.15% | 70/300 | 3/300 |
| 5 | 0.1710 | 5000 | 94.84% | 81.90% | 125/300 | 1/300 |

### 5.2 Asymmetry

Moving from CLS-optimal to WM-optimal costs CLS 1.49 pp (2.4% of its above-chance range) while gaining WM 6.61 pp (10.7% of its above-chance range). CLS at the WM optimum (94.84%) still exceeds CLS at zero adaptation (94.50%). The total CLS dynamic range across the 300-point grid is only 6.4 pp, compared to 57 pp for WM.

### 5.3 Trade-off costs at each task's optimum

| Config | CLS | WM | XOR |
|--------|-----|-----|-----|
| CLS-optimal (inc=0.0707, tau=5000) | **96.33%** | 75.30% | 66.32% |
| WM-optimal (inc=0.171, tau=5000) | 94.84% | **81.90%** | 64.71% |
| XOR-optimal (inc=0.0527, tau=5000) | 95.88% | 71.24% | **67.57%** |

---

## 6. Per-Bin Temporal Decomposition of Classification

### 6.1 The sign flip

More adaptation increases the *average* per-bin accuracy (by boosting late bins) but *decreases* overall classification accuracy. Regressing on log(inc) across 285 Branch B grid points:

| Metric | R² | Slope | Direction |
|--------|-----|-------|-----------|
| Overall classification accuracy | 0.51 | −0.0154 | NEGATIVE |
| Per-bin mean (all 48 bins) | 0.52 | +0.0681 | POSITIVE |
| Peak bins (6–15) | 0.05 | −0.0144 | NEGATIVE |
| Late bins (30–47) | 0.57 | +0.1294 | POSITIVE |

### 6.2 Two opposing effects of adaptation on classification

**Effect 1 — Improved peak discrimination (positive).** Peak-epoch accuracy (bins 6–15, 120–300 ms) rises from 63.6% at inc = 0 to 74.8% at inc = 0.1274. Adaptation enhances temporal contrast during the stimulus.

**Effect 2 — Destroyed temporal feature diversity (negative).** The temporal profile standard deviation collapses from 19.1% (inc = 0) to 1.9% (inc = 1.0). Late bins go from chance (~20%) to 43–57% (informative but redundant). The integration bonus shrinks from 55.5 pp to 31.8 pp. The full-feature classifier needs *diverse* temporal patterns; adaptation flattens the profile.

The overall CLS accuracy is the net: at low inc (0–0.07), Effect 1 dominates slightly; at moderate inc (0.07–0.13), the effects cancel; at high inc (> 0.13), Effect 2 dominates.

### 6.3 Epoch comparison at CLS-optimal vs WM-optimal

At tau = 5000, Branch B:

| Epoch | Bins | CLS-opt (inc=0.07) | WM-opt (inc=0.17) | Delta |
|-------|------|---------------------|--------------------|-------|
| Early rise | 0–5 | 64.92% | 65.04% | +0.12 pp |
| Peak | 6–15 | 72.62% | 73.69% | +1.06 pp |
| Transition | 16–24 | 47.39% | 60.66% | +13.27 pp |
| Late | 25–47 | 22.73% | 43.72% | +20.99 pp |
| **Overall** | all | **96.33%** | **94.84%** | **−1.49 pp** |

The WM-optimal configuration wins in every temporal epoch by per-bin accuracy, yet loses overall classification by 1.49 pp. The +21 pp late-bin gain adds redundant, not complementary, features.

### 6.4 Within-trial migration of optimal adaptation strength

The optimal adapt_inc is not constant across the stimulus:

- **Peak epoch (120–300 ms):** optimal inc ≈ 0.10–0.14 — burst-pause regime (ISI CV > 2.1), adjacent to WM optimum
- **Transition (320–440 ms):** optimal inc rises sharply from 0.14 to 0.50
- **Late epoch (500–940 ms):** optimal inc saturates near 0.89–0.94

No single static adaptation strength serves all temporal phases. The CLS optimum at inc=0.07 prioritizes peak-epoch temporal diversity; the WM optimum at inc=0.17 trades slightly lower peak encoding for readable persistence. If adaptation could be modulated within a trial (weak during stimulus, strong during maintenance), the CLS/WM trade-off would dissolve — a biologically plausible scenario given neuromodulatory timescales.

---

## 7. ISI Coefficient of Variation Reveals Two Spiking Regimes

### 7.1 ISI CV as a mediator

ISI CV is only 17% explained by log(inc) + log(tau), meaning it captures residual spike-pattern variation beyond adaptation parameters. Adding ISI CV to an inc+tau regression yields large incremental R²:

| CLS temporal feature | Base R² (inc+tau) | R² with ISI CV | Δ R² |
|---|---|---|---|
| Peak accuracy | 0.305 | 0.700 | +0.395 |
| Profile std | 0.590 | 0.939 | +0.349 |
| Late accuracy | 0.666 | 0.884 | +0.218 |
| Overall CLS | 0.549 | 0.761 | +0.211 |

Participation ratio adds at most Δ R² = 0.06 — it is effectively a proxy for adaptation strength.

### 7.2 The sign reversal defines two regimes

The within-inc correlation between ISI CV and peak accuracy flips sign:

| Inc range | ISI CV range | r(ISI CV, peak) | Interpretation |
|-----------|-------------|-----------------|----------------|
| 0.007–0.095 | 2.1–2.4 | +0.73 to +0.97 | Burst-pause: burstier = better discrimination |
| 0.17–1.0 | 1.3–2.1 | −0.40 to −0.98 | Adapted steady-state: burstiness = failure to equilibrate |

**Regime 1 — Burst-pause (ISI CV > 2.1, inc ≈ 0.01–0.10).** Adaptation creates burst-pause cycles. Different stimuli drive different burst timing. Profile std is high (19–22%). Late bins decay to chance. The integration bonus is maximal. CLS thrives.

**Regime 2 — Adapted steady-state (ISI CV < 2.0, inc > 0.15).** Neurons converge toward regularized equilibrium. The temporal profile flattens. Stimulus history is encoded in the *level* of the adapted state (a DC signal across neurons). WM excels.

### 7.3 Stimulus-epoch vs B-epoch ISI CV divergence

| Inc | CLS ISI (stim) | WM ISI (B-epoch) | Gap | Interpretation |
|-----|--------|---------|-----|------|
| 0.0 | 1.83 | 1.78 | +0.05 | No adaptation, similar |
| 0.05 | 2.35 | 2.08 | +0.27 | Stimulus burstiness decays |
| 0.17 | 2.09 | 1.66 | +0.43 | Transitioning to adapted SS |
| 0.55 | 1.50 | 1.68 | −0.19 | Reversed: stimulus suppressed |
| 1.0 | 1.26 | 1.75 | −0.49 | Fully reversed |

At extreme adaptation, stimulus-epoch ISI CV drops *below* B-epoch ISI CV — massive adaptation immediately suppresses firing during the stimulus, while digit B partially reactivates against a deep adaptation background.

---

## 8. Variance Decomposition

For Branch B, inc > 0:

| Predictor(s) | CLS R² | WM R² | XOR R² |
|---|---|---|---|
| log(inc) | 0.513 | 0.194 | 0.010 |
| log(tau) | 0.037 | 0.649 | 0.747 |
| log(inc) + log(tau) | 0.549 | 0.843 | 0.756 |
| + CLS ISI CV | 0.761 | — | 0.803 |
| + CLS PR | 0.684 | — | 0.765 |

CLS variance is dominated by inc (negative: more adaptation hurts). WM is dominated by tau (positive: longer memory helps). XOR is almost entirely driven by tau alone (R² = 0.75).

---

## 9. Participation Ratio Is a Proxy, Not a Mediator

Participation ratio (PR, effective dimensionality) ranges from ~2 to 12 across the grid. PR increases monotonically with adaptation, yet overall CLS *decreases* (R² = 0.61, negative slope). This paradox resolves through the temporal homogenization story: higher PR accompanies persistent activity across more neurons (higher static-snapshot dimensionality) but collapsed temporal diversity (lower effective dimensionality of the time × neuron feature matrix the classifier uses).

PR adds negligible incremental power beyond inc+tau for any task (Δ R² < 0.025). It is a downstream consequence of adaptation, not an independent dynamical variable.

---

## 10. Mechanistic Interpretability: How Adaptation Implements Working Memory

Analysis at 4 diagnostic grid points (tau=5000, Branch B) with 100 compound A+gap+B trials probes the internal mechanism:

| Condition | adapt_inc | g_tonic | WM (spikes) | WM (adapt state) |
|-----------|-----------|---------|-------------|-------------------|
| no_adapt | 0.000 | 4.375 nS | 26.9% (chance) | 20.0% (chance) |
| cls_optimum | 0.071 | 1.875 nS | 75.6% | 98.2% |
| perbin_optimum | 0.127 | 1.250 nS | 80.3% | 98.3% |
| wm_optimum | 0.171 | 0.625 nS | 81.3% | 97.9% |

### 10.1 Adaptation conductance is the memory

The 604-dimensional adaptation conductance vector at B onset predicts digit-A identity at 97.9–98.3% accuracy — vastly exceeding the spike-count readout (75.6–81.3%). The spike-count classifier recovers only a fraction of what is stored in the conductance state. The stimulus-to-adaptation correlation (A-epoch spike counts → adaptation at B onset) ranges from r = 0.81 to 0.88.

### 10.2 The classifier reads B-epoch spikes

The ridge classifier concentrates 96–98% of its weight on B-epoch bins, with only 2–4% on post-stimulus bins. Adaptation equalizes weight allocation across input-weight quartiles: without adaptation, Q4 (high-input) neurons carry ~36% of ridge weight while Q1 carries ~16%; at WM optimum, all quartiles converge toward 25%.

### 10.3 Population equalization

Adaptation compresses firing rate inequality via activity-dependent negative feedback:

- Q4 (strongest input) mean rate: 29.5 → 25.6 spk/ms (suppressed)
- Q1 (weakest input): 4.6 → 5.8 spk/ms (recruited)
- Q2: 4.4 → 10.8 spk/ms (2.5× increase)
- Firing rate Gini: 0.74 → 0.47
- Top-10% spike share: 58.3% → 31.8%

### 10.4 The mode switch: concentrated rate code → distributed conductance code

| Condition | Weight Gini | Input wt / ridge wt corr | WM accuracy |
|-----------|-------------|---------------------------|-------------|
| no_adapt | 0.52 | +0.60 | 26.9% |
| cls_optimum | 0.42 | +0.42 | 75.6% |
| perbin_optimum | 0.35 | +0.26 | 80.3% |
| wm_optimum | 0.31 | +0.11 | 81.3% |

Without adaptation, the classifier concentrates on a few strongly-driven neurons. With adaptation, weight distributes uniformly. The mode switch is not only temporal (transient vs persistent) but also spatial (concentrated vs distributed).

PCA on 5-class mean adaptation vectors shows 2 PCs capture >95% of variance. The representation is low-dimensional but perfectly linearly separable.

### 10.5 Calibration controls

Five lines of evidence confirm that population equalization is adaptation-driven, not a rate-matching artifact: (A) tonic conductance is uniform across neurons and cannot reshape the distribution; (B) g_tonic *decreases* with adaptation (strongest calibration at no_adapt, weakest at WM optimum); (C) firing rate Gini drops monotonically; (D) top-10% spike share drops; (E) Q4/Q1 dominance ratio collapses from 6.4× to 4.4×.

---

## 11. Discussion

### 11.1 The static network as a substrate for adaptation-gated computation

These results demonstrate that a single, structurally frozen spiking reservoir supports qualitatively different computational strategies by modulating only two parameters of spike-frequency adaptation. The network weights, connectivity, neuron model, and input encoding are identical across all 600 conditions. Yet the *type* of computation that the readout can extract shifts systematically.

This is not a parametric performance curve where one end is "better." The per-bin analysis reveals that what changes is not the *amount* of computation but its *kind*. Low adaptation produces a temporally rich, rapidly evolving representation suited for moment-by-moment stimulus discrimination. High adaptation produces a temporally persistent, slowly evolving representation that carries stimulus history across gaps. These are distinct coding strategies in the same physical network.

### 11.2 The burst-pause to adapted-steady-state transition

The ISI CV analysis identifies a qualitative boundary near ISI CV ≈ 2.1 (inc ≈ 0.10–0.15 at long tau) separating two spiking regimes. The sign reversal in the within-inc correlation confirms a genuine regime change, not a smooth continuum. In the burst-pause regime, burstier firing improves discrimination (r ≈ +0.95). In the adapted-steady-state regime, residual burstiness *hurts* (r ≈ −0.95).

The mechanistic interpretability adds a population dimension: the mode switch is temporal (transient vs persistent) *and* spatial (concentrated vs distributed). The burst-pause regime relies on a few strongly-driven neurons creating stimulus-specific temporal patterns. The adapted-steady-state regime stamps a persistent, distributed pattern across the entire population.

### 11.3 Why the CLS/WM trade-off is mechanistic, not trivial

A naive reading would suggest CLS barely cares about adaptation (6.4 pp range) while WM cares deeply (57 pp range). The per-bin decomposition rejects this: CLS is the site of two large, opposing effects (+11 pp peak discrimination gain, −13 pp temporal integration loss), netting a modest −1.5 pp.

The competition is specifically between temporal feature diversity (which the concatenated-bin classifier needs) and temporal persistence (which WM needs). These are the same physical phenomenon — persistent activity in late bins — producing opposite effects. For CLS, persistent late bins add redundant features. For WM, they *are* the signal. This is a genuine mechanistic incompatibility.

If the CLS readout were restricted to peak bins only, its optimal adaptation would shift to inc ≈ 0.13 — nearly coinciding with the WM optimum — and the Pareto frontier would largely collapse. The trade-off depends not only on reservoir dynamics but on the readout's temporal integration window.

### 11.4 XOR as an intermediate computational demand

XOR's optimal adaptation sits between CLS and WM (inc ≈ 0.05–0.07), driven almost entirely by tau (R² = 0.75 from log(tau) alone). XOR needs enough temporal history to compare A and B, but not the deep persistence WM requires. The XOR/WM grid correlation is r = 0.94, but the residual after regressing out WM is predicted by inc with a negative sign — controlling for memory strength, less adaptation improves XOR discrimination.

### 11.5 Biological implications

Neuromodulatory control of SFA — achievable through a single molecular target (e.g., calcium-activated potassium channels mediating the AHP) — could function as a mode-switching mechanism that reconfigures a cortical circuit between temporal discrimination and working memory without synaptic plasticity.

The burst-pause regime resembles stimulus-locked dynamics in primary auditory cortex during passive listening. The adapted-steady-state regime resembles sustained firing in prefrontal working memory maintenance.

The rate-matching analysis (Branch B) is critical: the CLS/WM dissociation, ISI CV boundary, and per-bin structure all survive rate clamping. The computational mode shift is intrinsic to adaptation dynamics, not secondary to rate changes.

### 11.6 Limitations

- ISI CV is a population average, potentially obscuring heterogeneity across neuron types or positions
- Participation ratio cannot capture temporal dimensionality (the relevant quantity for the concatenated-bin readout)
- Grid resolution (1.34× ratio per step in inc, 1.44× in tau) introduces ±half-step uncertainty in optimal values
- All readout is linear; nonlinear readouts (up to 97.1%) could reshape the trade-off landscape
- The readout bottleneck (98% stored in adaptation conductance, 82% recovered by spikes) may narrow with different decoding strategies

---

## 12. Future Directions

### 12.1 Expanded tau space

All three tasks converge on the longest available time constants (tau = 3470–5000 ms). Extending to tau = 10,000–50,000 ms would test whether the dissociation is genuinely one-dimensional (inc-only, with all tasks preferring maximal tau) or whether a second axis of separation emerges, producing 2D clusters in the (inc, tau) plane.

### 12.2 Variable WM gap length

The current 150 ms gap produces a single WM optimum. The adaptation trace decays exponentially with time constant tau, so the required adaptation strength should scale with gap duration. Sweeping gap length would map how the WM optimum migrates through (inc, tau) space as a function of memory demand, turning the static Pareto frontier into a continuous family of frontiers parameterized by memory load.

---

## Project Structure

```
├── src/
│   ├── inc/
│   │   ├── common.h          # RNG, matrix ops, LAPACK (SVD, Cholesky), JSON
│   │   ├── network.h         # SphericalNetwork: LIF dynamics, CSR connectivity
│   │   ├── builder.h         # Network construction, zone topology, tuning curves
│   │   ├── ml.h              # Dual-form ridge classifier, StandardScaler, CV
│   │   ├── npz_reader.h      # NumPy .npz file reader
│   │   └── experiments.h     # Shared constants and types
│   └── src/
│       ├── main.cpp           # CLI parsing and dispatch
│       ├── input_grid.cpp     # Input neuron grid search + MI refinement
│       ├── classification.cpp # CLS adaptation sweep, raster dump
│       ├── wm_sweep.cpp       # WM/XOR adaptation sweep
│       ├── mechanistic_interp.cpp # Mechanistic interpretability analysis
│       ├── network.cpp        # Spiking dynamics, conductance updates
│       ├── builder.cpp        # Ring-zone topology, Gaussian tuning, overrides
│       ├── ml.cpp             # Dual-form ridge (Cholesky), ML pipeline
│       └── npz_reader.cpp     # NPZ/NPY parsing
├── experiments/               # Python analysis and plotting scripts
├── results/
│   ├── verification_python_to_cpp/   # Python ↔ C++ validation
│   ├── input_grid_search/            # 8,000-point input optimization
│   ├── classification_adaptation_sweep/  # CLS sweep results
│   ├── wm_adaptation_sweep/          # WM/XOR sweep results
│   ├── mechanistic_interp/           # Mechanistic interpretability figures
│   ├── raster/                       # Spike raster visualizations
│   ├── gaussian_tuning_curves/       # Tuning coverage figures
│   └── neuron_diagnostics/           # Single-neuron state traces
├── network_snapshot.npz       # Deterministic Python-exported topology
├── Makefile                   # C++17, -O3, LAPACK/BLAS, zlib, OpenMP
└── data/                      # symlink → BSA spike train data (~7 GB)
```

## Build and Run

```bash
make                    # produces ./cls_sweep

# Classification adaptation sweep (300 points × 2 branches)
./cls_sweep --n-workers 8

# Working memory / XOR sweep
./cls_sweep --wm-sweep --n-workers 8

# Input grid search (8,000 points)
./cls_sweep --input-grid --n-workers 8

# Mechanistic interpretability
./cls_sweep --mech-interp --n-workers 8

# Behavioral verification (500 samples)
./cls_sweep --verify-only --verify-output verify_cpp.json --samples-per-digit 100
```

## Dependencies

- C++17 compiler (g++ or clang++)
- LAPACK/BLAS (Accelerate on macOS, liblapack/libblas on Linux)
- zlib
- OpenMP (optional, for parallel simulation)
- Python 3 with numpy, pandas, matplotlib, scipy (for analysis scripts)

## Data

`data/` is a symlink to BSA-encoded spike train data (~3,000 `.npz` files, ~7 GB, 500 samples per digit, digits 0–9). Each `.npz` contains `spike_times_ms` and `freq_bin_indices` (mel-frequency bin indices 0–127).

```bash
ln -sf /path/to/liquid_state_machine_expanded/data ./data
```
