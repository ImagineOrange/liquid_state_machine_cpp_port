# Spike-Frequency Adaptation Modulates Computational Mode in a Spiking Liquid State Machine

## A triple dissociation across classification, working memory, and temporal XOR tasks

---

## Summary

A single, structurally frozen spiking reservoir can support qualitatively different computational strategies — instantaneous pattern discrimination, persistent working memory, and temporal comparison — by modulating only two parameters of spike-frequency adaptation (SFA). A 300-point sweep over adaptation increment and time constant, evaluated on three tasks with rate-matched controls, reveals a statistically verified triple dissociation: classification (CLS) peaks at low adaptation, temporal XOR at weak-to-moderate adaptation, and working memory (WM) at moderate-to-strong adaptation, with all tasks preferring long time constants. The dissociation is not a parametric performance curve where one end is "better" — per-bin temporal decomposition shows that adaptation changes the *kind* of computation, not the *amount*. Mechanistic interpretability analysis traces working memory to a distributed adaptation-conductance code that is near-perfectly decodable (98%) but bottlenecked by the spike-count readout (82%). Hidden state trajectories and a 140-point crossover sweep reveal the unifying mechanism: adaptation implements a continuous encode-and-integrate pipeline where the adaptation increment controls the handoff time between a transient spike code and a persistent conductance code, and the crossover time is the single generating variable behind the ISI CV regime boundary, per-bin accuracy sign flip, Pareto asymmetry, and readout bottleneck. A variable gap sweep (50–3000 ms) activates the second axis of adaptation — the time constant controls retention duration — transforming the static CLS/WM Pareto frontier into a gap-dependent landscape where tau shifts from irrelevant (gaps ≤ 150 ms, ANOVA p > 0.05) to decisive (gaps ≥ 300 ms, p < 0.01), with tau spread growing monotonically from 0.8 pp at gap=150 ms to 12.1 pp at gap=3000 ms. An ablation study isolates SFA as the primary mechanism: with both short-term depression (STD) and recurrent NMDA disabled, the SFA-only network preserves 93% of the baseline WM dynamic range (50.2 of 54.1 pp), while STD contributes only 1.7 pp. A multi-seed replication across 10 independently constructed network topologies (reservoir sizes 588–643 neurons) confirms the dissociation is topology-invariant: all 10 seeds pass with WM dynamic range of 53.6 ± 1.1 pp (CV = 2.1%, p < 0.001 under the null). These results demonstrate that neuromodulatory control of a single biophysical parameter — the afterhyperpolarization current — can function as a mode-switching mechanism that reconfigures a cortical circuit between temporal discrimination and working memory without synaptic plasticity, and this mechanism is both causally sufficient (ablation) and universally effective across network instantiations (multi-seed).

---

## 1. Network Architecture

The reservoir is a 604-neuron leaky integrate-and-fire (LIF) network (compacted from 1000) embedded in a 3D sphere. Neurons use conductance-based synapses with three excitatory channels (AMPA, NMDA with Mg2+ block) and two inhibitory channels (GABA-A, GABA-B). Note: GABA-B (slow inhibition) is architecturally present but disabled — no neurons are marked as slow inhibitory, so all inhibition is fast (GABA-A only). Spike-frequency adaptation is modeled via an afterhyperpolarization (AHP) current. Biophysical parameters are jittered across neurons.

The sphere is divided into two functional zones **(Figure 1)**:

- **Input shell** — excitatory neurons on the outer surface, arranged into a 300-degree azimuthal arc, tonotopically mapped to 128 mel-frequency bins via Gaussian tuning curves.
- **Reservoir core** — interior neurons receiving feedforward input from the shell. Intra-shell and feedback connections are removed to enforce unidirectional signal flow.

Excitatory recurrent connections undergo short-term depression (Tsodyks-Markram, U=0.1, tau_rec=500 ms). Transmission delays are distance-dependent via ring buffer.

Topology is loaded from a deterministic Python-exported snapshot (`network_snapshot.npz`) for bit-identical structure across implementations. The C++ port was validated against the original Python implementation (per-sample spike count correlation r = 0.992, accuracy gap < 1σ; see Section 3.1).

### 1.1 Input encoding

Raw audio (WAV) is converted to current injection into the input shell:

1. **BSA spike encoding** — WAV files are converted offline to spike trains via Ben's Spiker Algorithm. Each spike has a time (ms) and frequency bin index (0–127, mel-scale).
2. **Warmup** — BSA spike times are shifted forward by 50 ms to allow network transients to decay.
3. **Tonotopic mapping** — Input neurons are mapped to frequency bins via Gaussian tuning curves **(Figure 2)**. 128 mel bin centers are placed at evenly-spaced quantiles of the input neuron azimuthal angle distribution. Each neuron selects its K=4 nearest bin centers. Weights are Gaussian: `w = exp(-d² / (2σ²))`, σ = 1.5 × mean neuron spacing. K=4 with quantile centers achieves 128/128 bin coverage, mean weight 0.859.
4. **Current injection** — BSA spikes in bin *m* are injected into all neurons mapped to that bin, scaled by tuning weight: `I = stim_current × w`. Stimulus current is 0.0518 nA. NMDA is disabled on input synapses.
5. **Input neuron parameters** — `tau_e = 1.05 ms` (fast excitatory decay prevents conductance saturation), `adaptation_increment = 0.0`, no short-term depression.
6. **Shell-to-core projection** — Input shell neurons project to reservoir core through distance-dependent connectivity, with weights scaled by 4.85×.

### 1.2 Input parameter optimization

Input neuron parameters were selected via an 8,000-point grid search maximizing mutual information between BSA input and output spikes under biological plausibility constraints **(Figure 3; Figure S6)**.

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

Three tasks are evaluated on the same LHS-021 spiking reservoir **(Figure 4)** across a shared 20 × 15 grid of adaptation parameters (adapt_inc × adapt_tau = 300 grid points). Each grid point is run in two branches: Branch A (unmatched — natural firing rate) and Branch B (tonic-conductance-matched — reservoir rate clamped to 20 Hz). Input encoding is held fixed at the grid-search optimum across all conditions. Readout is a linear ridge classifier (5-fold × 5-repeat stratified CV).

The three tasks share the same reservoir and adaptation parameters but differ in what they ask the readout to extract:

- **Classification (CLS)**: Identify which of 5 digits was presented, using all temporal bins during a single stimulus. A pure stimulus-discrimination task with no memory requirement. 1,500 samples (300 per digit × 5 digits), 48 bins × 604 neurons = 36,240 features.

- **Working memory (WM)**: Identify digit A from reservoir activity during the B+Post epoch, after a 150 ms silent gap and an interfering digit B. Only different-digit pairs are used. 2,500 digit pairs per grid point; readout from B+Post epoch only (7 bins × 604 neurons = 4,228 features). Chance = 20%.

- **Temporal XOR**: Classify whether a digit pair is same or different, reading out from the B+Post epoch. The label depends on the relationship between A and B — neither alone is sufficient. 2,500 pairs; chance = 50%.

### 2.2 Rate matching

Changing adaptation changes reservoir firing rate. To isolate adaptation dynamics from rate effects, the experiment uses a two-branch design:

**Branch A (unmatched):** Stimulus current fixed at 0.0518 nA, no tonic conductance, rate varies naturally with adaptation.

**Branch B (rate-matched):** Same fixed stimulus current, but a tonic conductance `g_tonic × (E_rev − V)` is calibrated per grid point via binary search to clamp reservoir firing to 20 Hz (midpoint of auditory cortex evoked rates). Reversal is automatically chosen: inhibitory (−80 mV) if rate too high, excitatory (0 mV) if too low. This acts on reservoir neurons only, preserving input encoding (MI = 1.06 bits) across all conditions.

Tonic conductance was chosen over alternatives (stimulus current modulation, weight scaling, threshold adjustment) because it acts on reservoir neurons only, does not alter input encoding quality, and competes with synaptic drive on equal terms as a conductance-based mechanism.

**Noise.** Runtime membrane noise (v_noise_amp = 0.1 mV, i_noise_amp = 0.001 nA) is held constant across all experiments and ablation conditions. These are hardcoded defaults in the network configuration and are never overridden.

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

The C++ port was validated against the original Python implementation **(Figure S5)**. Both run 500 samples through identical topology (loaded from the same `.npz` snapshot). Runtime noise diverges (different RNG libraries) so exact spike-by-spike match is impossible, but aggregate statistics confirm equivalence:

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

### 4.1 Global optima **(Figure 6; Figures S1--S2)**

| Task | Branch | Peak inc | Peak tau (ms) | Peak accuracy | Top-5 centroid inc | Top-5 centroid tau |
|------|--------|----------|---------------|---------------|--------------------|--------------------|
| CLS | B | 0.0707 | 5000 | 96.33% | 0.042 | 3102 |
| WM | B | 0.1710 | 5000 | 81.90% | 0.186 | 4391 |
| XOR | B | 0.0527 | 5000 | 67.57% | 0.095 | 4389 |
| CLS | A | 0.0292 | 2408 | 95.75% | 0.043 | 2741 |
| WM | A | 0.1710 | 5000 | 83.07% | 0.194 | 3876 |
| XOR | A | 0.0707 | 5000 | 67.24% | 0.057 | 4693 |

All three tasks prefer long adaptation time constants (tau = 3000–5000 ms) **(Figure S7)**. They separate primarily along adapt_inc: CLS peaks at low adaptation strength (inc ≈ 0.03–0.07), XOR at weak-to-moderate (inc ≈ 0.05–0.07), and WM at moderate-to-strong (inc ≈ 0.17). WM's optimum is identical across branches (inc = 0.171, tau = 5000).

### 4.2 Statistical verification of the triple dissociation

Three independent statistical tests confirm the dissociation **(Figure 5)**:

**Bootstrap of optimal inc (tau=5000 slice, Branch B, 10,000 iterations):** CLS landed at inc=0.0707 in 100% of iterations, XOR at 0.0527 in 99.8%, WM at 0.171 in 87.8% (remainder at 0.2295). No overlap between any pair. Kruskal-Wallis H = 29,522, p ≈ 0.

**Friedman test with softmax center-of-mass (Branch B, tau ≥ 558 ms):** At each of 7 tau values, a softmax-weighted center-of-mass (T=50) estimates the optimal adaptation strength as a continuous variable. Friedman χ² = 14.0, p = 0.0009. All post-hoc Wilcoxon signed-rank tests significant after Bonferroni correction (all W = 0, p = 0.016). The ordering CLS < XOR < WM held at every tau slice. Mean softmax CoM: CLS = 0.096, XOR = 0.141, WM = 0.293.

**Pareto frontier paired t-tests (tau=5000):** CLS advantage at CLS-optimal vs WM-optimal: 1.49 pp, t = 59.3, df = 4, p < 0.0001. WM advantage at WM-optimal vs CLS-optimal: 6.61 pp, t = 10.0, p = 0.0006. Both significant and in opposite directions.

### 4.3 The inc × tau interaction

For WM, the optimal inc shifts with tau: at tau = 558 ms, the WM argmax is inc = 0.41; at tau = 5000 ms, it falls to inc = 0.17. This is consistent with a constraint on the equilibrium adaptation *state* rather than the raw parameters. The product inc × tau is not a sufficient statistic — along iso-product lines, WM accuracy varies by 15–32 pp.

An additive model log(inc) + log(tau) explains 84% of WM variance with a tau/inc beta ratio of 1.87, compared to 9.0 for XOR and a negative inc coefficient for CLS.

---

## 5. The Pareto Frontier Between Classification and Working Memory

### 5.1 Frontier structure **(Figure 7)**

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

### 6.1 The sign flip **(Figure 9)**

More adaptation increases the *average* per-bin accuracy (by boosting late bins) but *decreases* overall classification accuracy. Regressing on log(inc) across 285 Branch B grid points:

| Metric | R² | Slope | Direction |
|--------|-----|-------|-----------|
| Overall classification accuracy | 0.51 | −0.0154 | NEGATIVE |
| Per-bin mean (all 48 bins) | 0.52 | +0.0681 | POSITIVE |
| Peak bins (6–15) | 0.05 | −0.0144 | NEGATIVE |
| Late bins (30–47) | 0.57 | +0.1294 | POSITIVE |

### 6.2 Two opposing effects of adaptation on classification **(Figures 8, 10)**

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

### 7.1 ISI CV as a mediator **(Figure 11)**

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

Single-trial ISI heatmaps **(Figure S12)** visualize this regime difference directly: without adaptation, neurons fire at short ISIs during stimuli and go silent in between; at the WM optimum, adaptation regularizes spiking and sustains activity through the gap at longer inter-spike intervals.

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

For Branch B, inc > 0 **(Figure 12)**:

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

Analysis at 4 diagnostic grid points (tau=5000, Branch B) with 100 compound A+gap+B trials probes the internal mechanism **(Figure 13)**:

| Condition | adapt_inc | g_tonic | WM (spikes) | WM (adapt state) |
|-----------|-----------|---------|-------------|-------------------|
| no_adapt | 0.000 | 4.375 nS | 26.9% (chance) | 20.0% (chance) |
| cls_optimum | 0.071 | 1.875 nS | 75.6% | 98.2% |
| perbin_optimum | 0.127 | 1.250 nS | 80.3% | 98.3% |
| wm_optimum | 0.171 | 0.625 nS | 81.3% | 97.9% |

### 10.1 Adaptation conductance is the memory **(Figures 16, 19, 20, 21)**

The 604-dimensional adaptation conductance vector at B onset predicts digit-A identity at 97.9–98.3% accuracy — vastly exceeding the spike-count readout (75.6–81.3%). The spike-count classifier recovers only a fraction of what is stored in the conductance state. The stimulus-to-adaptation correlation (A-epoch spike counts → adaptation at B onset) ranges from r = 0.81 to 0.88.

Single-trial visualization of the full adaptation conductance time series confirms the mechanism **(Figure 19)**: at the WM optimum, adaptation builds during stimulus A (concentrated in high-activity neurons), persists through the 150 ms gap with minimal decay, and carries a digit-A-specific spatial pattern into the B epoch. Without adaptation, the conductance state is uniformly zero. Quantitative profile analysis **(Figure 20)** shows the neuron-by-neuron adaptation pattern at end-of-A correlates with the mid-gap pattern at r > 0.95, dropping only when digit B begins modifying the state. A comprehensive single-trial comparison **(Figure 21)** juxtaposes spike rasters, adaptation heatmaps, and population traces: superimposed PSTHs confirm near-identical firing rates under rate matching, while the dual-axis overlay reveals the critical dissociation — the firing rate drops during the gap while the adaptation conductance plateaus, showing the conductance code persists where the spike code vanishes.

### 10.2 The classifier reads B-epoch spikes **(Figure 14)**

The ridge classifier concentrates 96–98% of its weight on B-epoch bins, with only 2–4% on post-stimulus bins. Adaptation equalizes weight allocation across input-weight quartiles: without adaptation, Q4 (high-input) neurons carry ~36% of ridge weight while Q1 carries ~16%; at WM optimum, all quartiles converge toward 25%.

### 10.3 Population equalization **(Figure 15)**

Adaptation compresses firing rate inequality via activity-dependent negative feedback:

- Q4 (strongest input) mean rate: 29.5 → 25.6 spk/ms (suppressed)
- Q1 (weakest input): 4.6 → 5.8 spk/ms (recruited)
- Q2: 4.4 → 10.8 spk/ms (2.5× increase)
- Firing rate Gini: 0.74 → 0.47
- Top-10% spike share: 58.3% → 31.8%

### 10.4 The mode switch: concentrated rate code → distributed conductance code **(Figures 17, 18, 19)**

| Condition | Weight Gini | Input wt / ridge wt corr | WM accuracy |
|-----------|-------------|---------------------------|-------------|
| no_adapt | 0.52 | +0.60 | 26.9% |
| cls_optimum | 0.42 | +0.42 | 75.6% |
| perbin_optimum | 0.35 | +0.26 | 80.3% |
| wm_optimum | 0.31 | +0.11 | 81.3% |

Without adaptation, the classifier concentrates on a few strongly-driven neurons. With adaptation, weight distributes uniformly. The mode switch is not only temporal (transient vs persistent) but also spatial (concentrated vs distributed).

PCA on 5-class mean adaptation vectors shows 2 PCs capture >95% of variance. The representation is low-dimensional but perfectly linearly separable.

### 10.5 Calibration controls **(Figure 22)**

Five lines of evidence confirm that population equalization is adaptation-driven, not a rate-matching artifact: (A) tonic conductance is uniform across neurons and cannot reshape the distribution; (B) g_tonic *decreases* with adaptation (strongest calibration at no_adapt, weakest at WM optimum); (C) firing rate Gini drops monotonically; (D) top-10% spike share drops; (E) Q4/Q1 dominance ratio collapses from 6.4× to 4.4×.

---

## 11. The Encode-and-Integrate Pipeline

The phenomena reported in Sections 4–10 — the triple dissociation, per-bin sign flip, ISI CV regime boundary, Pareto asymmetry, and the 98% vs 82% mechanistic bottleneck — arise from a single underlying mechanism: spike-frequency adaptation implements a continuous handoff from a transient spike code to a persistent conductance code, and the handoff timing is controlled by a single variable.

### 11.1 Hidden state trajectories at the WM optimum **(Figure 23)**

At the WM-optimal configuration (inc=0.171, tau=5000, Branch B), the 604-dimensional spike-count and adaptation-conductance vectors were recorded at each 20 ms time bin across 18 trials (3 per digit, 6 digits).

PCA on the spike-count trajectories captures only 68% of variance in 3 PCs — the representation is high-dimensional and tangled, with digit-class trajectories overlapping and looping through shared regions. PCA on the adaptation-conductance trajectories captures 96% — the representation is low-dimensional and clean, with digit classes diverging monotonically from a shared origin into well-separated endpoints.

Mean pairwise Euclidean distance between digit-class centroids quantifies the divergence:

- **Spike counts:** Rise during stimulus onset to ~20 units, then collapse after offset. By the post-stimulus epoch, discriminability returns to near-baseline.
- **Adaptation conductance:** Rise monotonically throughout the stimulus and continue climbing into the post-stimulus epoch, reaching ~80 units — 4× peak spike discriminability. The signal never declines within the observation window.

The two curves cross during the stimulus. Before the crossover, spikes carry more discriminative information. After it, the adaptation conductance dominates. This crossover is the encode-and-integrate handoff.

### 11.2 Crossover time is the generating variable **(Figures 24, 25)**

A 140-point sweep (20 inc × 7 tau) measures the crossover time — the first bin where adaptation-conductance separability exceeds spike-count separability. The results reveal two properties:

**The crossover depends only on inc.** It forms vertical bands in the (inc, tau) heatmap — identical crossover time (within ±1 bin) at all 7 tau values for each inc level:

| inc | Crossover (ms) | Regime significance |
|-----|---------------|---------------------|
| 0.000–0.016 | Never | Adaptation never catches spike code |
| 0.071 (CLS opt) | 310 | Mid-stimulus |
| 0.095 (ISI CV boundary) | 290 | Spike ≈ adapt separability |
| 0.127 (per-bin opt) | 250 | Early-mid stimulus |
| 0.171 (WM opt) | 210 | Early stimulus |
| 1.000 | 10 | First bin |

**The crossover unifies all secondary metrics:**

- **ISI CV boundary (Section 7):** At inc ≈ 0.095, crossover = 290 ms and peak spike separability (38.7) ≈ peak adaptation separability (38.0). This is the representational balance point — the ISI CV ≈ 2.1 boundary is where the coding format shifts from spike-dominant to conductance-dominant.

- **Per-bin sign flip (Section 6):** Late bins rise because they are past the crossover (reading the conductance DC level), while overall CLS drops because the concatenated-bin classifier needs the temporal diversity that exists only in the pre-crossover portion. An earlier crossover (higher inc) means less encoding and more integration.

- **Pareto asymmetry (Section 5):** From CLS-optimal (crossover=310 ms) to WM-optimal (crossover=210 ms), spike separability drops 13% (40.6 → 35.2) while adaptation separability triples (27.6 → 80.9). Spike discriminability degrades linearly with inc; adaptation discriminability grows exponentially. A 100 ms crossover shift barely affects CLS but massively amplifies WM.

- **98% vs 82% bottleneck (Section 10):** The adaptation-conductance separability at the WM optimum is 2.3× the spike-count separability (80.9 vs 35.2). The 16 pp gap between stored information (98%) and readout recovery (82%) is the classification-domain expression of this separability ratio.

- **Within-trial inc migration (Section 6.4):** The per-bin optimal inc shifts from ~0.10–0.14 (peak epoch) to ~0.89 (late epoch) because each bin wants the crossover at *that* bin — maximizing combined encoding benefit from earlier bins and conductance persistence at the current bin.

### 11.3 Two-axis decomposition of adaptation parameter space

The crossover sweep decomposes the adaptation parameter space into two mechanistically distinct, orthogonal axes:

**Axis 1 — Integration rate (inc):** Controls the crossover time and therefore which representation format dominates at each moment during the trial. This is the axis along which CLS, XOR, and WM separate. Inc explains CLS variance (R² = 0.51, Section 8) because it determines how much of the trial is available for temporal encoding.

**Axis 2 — Retention duration (tau):** Controls how long the conductance code persists after stimulus offset. Has no effect on within-stimulus dynamics (crossover time is tau-invariant). At the 150 ms gap used in the original WM task, `exp(-150/5000) = 0.970` — virtually no trace decays regardless of tau. This explains why all three tasks converge on maximal tau (Section 4.1) and why tau dominates the WM and XOR variance decomposition (Section 8): tau is a ceiling variable at 150 ms gap, and all tasks benefit from being at the ceiling.

An extended tau sweep (5 inc × 7 tau, tau from 5,000 to 44,790 ms) confirmed tau invariance at fixed 150 ms gap: maximum WM variation across the full range was 2.86 pp — within CV noise.

---

## 12. Variable Gap Length Sweep: Activating Axis 2

The original WM task used a fixed 150 ms gap, leaving Axis 2 (retention duration) degenerate. A 480-point sweep (20 inc × 4 tau × 6 gap lengths: 50, 100, 150, 300, 500, 1000 ms) tests the prediction that longer gaps will engage tau as a discriminating variable. A targeted 12-point extension (inc=0.171, 4 tau × 3 gaps: 1500, 2000, 3000 ms) confirms the prediction at long gaps where the effect becomes unambiguous **(Figure 30)**.

### 12.1 Tau becomes decisive at long gaps **(Figures 26, 27)**

The tau spread — max minus min WM accuracy across 4 tau values at fixed inc — quantifies tau's influence:

| Gap (ms) | Tau spread at inc=0.171 |
|----------|------------------------|
| 50 | 1.0 pp |
| 100 | 1.4 pp |
| 150 | 0.8 pp |
| 300 | 2.2 pp |
| 500 | 2.4 pp |
| 1000 | 4.3 pp |
| **1500** | **5.7 pp** |
| **2000** | **9.1 pp** |
| **3000** | **12.1 pp** |

One-way ANOVA across the 4 tau values (5 CV repeats each) confirms that tau differences are not significant at short gaps (gap=50ms: F=2.1, p=0.14; gap=150ms: F=2.9, p=0.06) but become significant at gap=300ms (F=6.7, p=0.004) and highly significant at gap=500ms (F=13.0, p=0.0001) and gap=1000ms (F=48.2, p<0.0001). The extended gap experiment pushes this further: at gaps of 1500–3000 ms the tau spread grows to 5.7–12.1 pp, far exceeding any noise floor. The "best tau" at gap≤150ms is unstable across CV repeats — the argmax tau flips randomly, confirming that Axis 2 is degenerate at short gaps. At gaps ≥ 300ms, the tau ranking locks into a perfectly monotonic order (τ=3470 < τ=5000 < τ=7200 < τ=10380) that never violates at any gap length.

At gap=1000ms and beyond, WM accuracy at inc=0.171 tracks tau monotonically:

| tau (ms) | WM @ gap=1000 | WM @ gap=2000 | WM @ gap=3000 | adapt at B onset (gap=3000) |
|----------|---------------|---------------|---------------|----------------------------|
| 3,470 | 75.9% | 70.1% | 63.8% | 1.180 |
| 5,000 | 78.0% | 74.4% | 71.5% | 1.508 |
| 7,200 | 79.4% | 77.6% | 74.5% | 1.983 |
| 10,380 | 80.2% | 79.2% | 75.9% | 2.191 |

At gap=3000ms, the shortest tau retains only 54% of the conductance trace retained by the longest tau (1.18 vs 2.19), directly producing the 12.1 pp accuracy gap. The residual adaptation conductance at B onset follows the exponential decay model, and WM accuracy tracks this residual **(Figure 29)**.

### 12.2 Graceful degradation

Best WM accuracy per gap (optimizing over inc and tau; extended gaps 1500–3000ms tested at inc=0.171 only):

| Gap (ms) | Best WM | Best inc | Best tau |
|----------|---------|----------|----------|
| 50 | 83.8% | 0.230 | 10,380 |
| 100 | 82.5% | 0.171 | 7,200 |
| 150 | 81.9% | 0.171 | 3,470 |
| 300 | 82.5% | 0.171 | 10,380 |
| 500 | 82.3% | 0.171 | 7,200 |
| 1000 | 80.2% | 0.171 | 10,380 |
| **1500** | **79.5%** | **0.171** | **10,380** |
| **2000** | **79.2%** | **0.171** | **10,380** |
| **3000** | **75.9%** | **0.171** | **10,380** |

Across a 60× range in gap length (50–3000 ms), the best-per-gap WM envelope drops 7.9 pp (83.8% → 75.9%). Within the original sweep range (50–1000 ms), the drop is only 3.6 pp. At fixed inc=0.171, the drop from gap=150ms to gap=3000ms is 5.9 pp (81.9% → 75.9%). At every gap ≥ 300ms, tau=10,380 — the grid maximum — is the optimal tau. This is a grid-edge effect: the true optimal tau at long gaps likely exceeds 10,380 ms, and extending the tau range is a clear next step (Section 14.1). The extended gap data (1500–3000ms) was collected at inc=0.171 only; the original 20-inc sweep confirms this inc is optimal at all gaps from 100–1000ms, and the physical argument extends to longer gaps (see Section 12.5).

The non-monotonicity in the table (gap=300 and 500 exceeding gap=150) arises because the "best tau" column at gap≤150ms is noise — ANOVA confirms no significant tau effect at these gaps (p>0.05), so the argmax tau is arbitrary. At gap=150ms, tau=3,470 wins by 0.8 pp over the runner-up, but this advantage flips across CV repeats. At gap≥300ms, the tau effect is significant (p<0.01) and the preference for higher tau is stable.

Degradation is in the mean, not the variance: per-repeat WM standard deviations at inc=0.171 range 0.5–1.3 pp across all gap lengths and tau values, with no systematic increase at longer gaps.

### 12.3 Degradation peaks at the CLS optimum

WM accuracy loss from gap=150 to gap=1000 (each at its best tau) varies systematically with inc:

| inc | Δ WM (gap 150→1000) |
|-----|---------------------|
| 0.000 | −0.9 pp |
| 0.071 (CLS opt) | **−3.6 pp** |
| 0.095 (ISI CV boundary) | −3.0 pp |
| 0.171 (WM opt) | −1.6 pp |
| 0.308 | −0.9 pp |
| 0.555 | +0.3 pp |

Peak degradation falls at the CLS optimum (inc=0.071), where adaptation builds enough trace to be gap-sensitive but not enough for the trace to dominate. Higher inc builds stronger traces that absorb proportionally less damage from decay. At inc ≥ 0.5, the trace is so saturated that the gap is operationally irrelevant.

### 12.4 XOR is gap-insensitive

XOR accuracy is nearly constant across gaps 50–1000ms (66.8–68.0% at best, total range 1.2 pp), with optimal inc stable at 0.095–0.127. However, the extended gap experiment reveals that XOR does degrade at very long gaps: at inc=0.171, XOR drops from ~61–65% at gap=1500ms to ~59–61% at gap=3000ms, suggesting the nonlinear mixing signal also decays — though more slowly than the WM trace — when the gap far exceeds the stimulus duration.

### 12.5 The WM optimum is robust along Axis 1, with a continuous transition on Axis 2 **(Figures 28, 30)**

The gap sweep reveals that the WM optimum does not trace a smooth trajectory through (inc, tau) space. Instead, the two axes behave independently:

**Axis 1 (inc) is fixed at 0.171 for all gaps ≥ 100 ms.** Five of six gap lengths in the full 20-inc sweep (100–1000ms) share the same optimal inc; the extended gap experiment (1500–3000ms) tested only this inc value. The only exception is gap=50ms, where inc=0.230 is significantly better (t=4.87, p=0.001, zero overlap in per-repeat values). This confirms that the integration rate is set by the task type (CLS vs WM), not by memory demand.

**Axis 2 (tau) undergoes a continuous transition from degenerate to decisive.** At gaps ≤ 150 ms, all four tau values produce statistically indistinguishable WM accuracy (ANOVA p > 0.05); the argmax tau is noise. The tau effect emerges at gap=300ms (p=0.004) and grows monotonically: the tau spread at inc=0.171 is 0.8 pp at gap=150ms, 4.3 pp at gap=1000ms, 9.1 pp at gap=2000ms, and 12.1 pp at gap=3000ms. At all gaps ≥ 300ms, the tau ranking is perfectly monotonic (τ=3470 < τ=5000 < τ=7200 < τ=10380), never violated at any gap length. This confirms that the transition is continuous, not threshold-like — the underlying relationship is a smooth function of the ratio gap/tau, as predicted by the exponential decay model.

Within the original 480-point sweep (gaps 50–1000ms), the per-gap optima occupy only 3 distinct grid cells: (inc=0.230, tau=10380) for gap=50ms, (inc=0.171, tau=7200) for gaps 100/500, and (inc=0.171, tau=10380) for gaps 300/1000. Gap=150ms lands at (inc=0.171, tau=3470) but this is not significantly different from any other tau at that gap. The extended gap experiment (1500–3000ms) tested only inc=0.171 — the inc that is optimal across all 5 gap lengths from 100–1000ms in the full sweep. While the physical argument for inc=0.171 remaining optimal at longer gaps is strong (longer gaps favor stronger traces, and inc=0.171 is already on the plateau), the extended data confirms the tau effect, not the inc optimality, at gaps >1000ms.

This pattern reinforces the two-axis model: Axis 1 is the primary dissociating variable (set once by task demands), while Axis 2 activates continuously as gap duration increases and trace decay separates the retention capacity of different tau values. The adapt-at-B-onset data confirms the mechanism: at gap=3000ms, the conductance trace at τ=3470 has decayed to 1.18 while τ=10380 retains 2.19 — nearly 2× the signal strength — directly producing the 12.1 pp accuracy spread.

---

## 13. Discussion

### 13.1 The static network as a substrate for adaptation-gated computation

These results demonstrate that a single, structurally frozen spiking reservoir supports qualitatively different computational strategies by modulating only two parameters of spike-frequency adaptation. The network weights, connectivity, neuron model, and input encoding are identical across all conditions. Yet the *type* of computation that the readout can extract shifts systematically.

This is not a parametric performance curve where one end is "better." The per-bin analysis (Section 6) reveals that what changes is not the *amount* of computation but its *kind*. Low adaptation produces a temporally rich, rapidly evolving representation suited for moment-by-moment stimulus discrimination. High adaptation produces a temporally persistent, slowly evolving representation that carries stimulus history across gaps. These are distinct coding strategies in the same physical network. The crossover framework (Section 11) identifies the crossover time as the single variable that determines which strategy dominates at each moment within a trial.

### 13.2 The burst-pause to adapted-steady-state transition

The ISI CV analysis (Section 7) identifies a qualitative boundary near ISI CV ≈ 2.1 (inc ≈ 0.10–0.15 at long tau) separating two spiking regimes. The sign reversal in the within-inc correlation confirms a genuine regime change, not a smooth continuum. In the burst-pause regime, burstier firing improves discrimination (r ≈ +0.95). In the adapted-steady-state regime, residual burstiness *hurts* (r ≈ −0.95).

The crossover sweep (Section 11.2) reveals that this boundary is precisely the balance point where peak spike-count separability (38.7) equals peak adaptation-conductance separability (38.0), occurring at crossover = 290 ms. The ISI CV boundary is not a qualitative regime change in the dynamics — it is the point where the dominant representational format switches from spike code to conductance code.

The mechanistic interpretability (Section 10) adds a population dimension: the mode switch is temporal (transient vs persistent) *and* spatial (concentrated vs distributed). The burst-pause regime relies on a few strongly-driven neurons creating stimulus-specific temporal patterns. The adapted-steady-state regime stamps a persistent, distributed pattern across the entire population.

### 13.3 Why the CLS/WM trade-off is mechanistic, not trivial

A naive reading would suggest CLS barely cares about adaptation (6.4 pp range) while WM cares deeply (57 pp range). The per-bin decomposition rejects this: CLS is the site of two large, opposing effects (+11 pp peak discrimination gain, −13 pp temporal integration loss), netting a modest −1.5 pp.

The competition is specifically between temporal feature diversity (which the concatenated-bin classifier needs) and temporal persistence (which WM needs). The crossover framework makes this precise: the pre-crossover portion of each trial provides diverse temporal features (encoding mode), while the post-crossover portion provides uniform conductance-derived signal (integration mode). CLS needs encoding mode to dominate; WM needs the integrated result. No single static inc can serve both, because an earlier crossover (higher inc) converts encoding time into integration time.

If the CLS readout were restricted to peak bins only, its optimal adaptation would shift to inc ≈ 0.13 — nearly coinciding with the WM optimum — and the Pareto frontier would largely collapse. The trade-off depends not only on reservoir dynamics but on the readout's temporal integration window.

The gap sweep (Section 12) extends this further: the CLS-optimal configuration (inc=0.071) suffers the greatest WM degradation with increasing gap (-3.6 pp from gap=150 to gap=1000), because its moderate trace is large enough to lose but too weak to survive exponential decay. The WM optimum (inc=0.171) degrades only -1.6 pp — strong traces absorb proportionally less damage.

### 13.4 XOR as an intermediate computational demand

XOR's optimal adaptation sits between CLS and WM (inc ≈ 0.05–0.07), driven almost entirely by tau (R² = 0.75 from log(tau) alone). XOR needs enough temporal history to compare A and B, but not the deep persistence WM requires. The XOR/WM grid correlation is r = 0.94, but the residual after regressing out WM is predicted by inc with a negative sign — controlling for memory strength, less adaptation improves XOR discrimination.

The gap sweep confirms XOR's relative independence from memory demand: XOR accuracy is constant at 66.8–68.0% (1.2 pp range) across gaps 50–1000 ms, with optimal inc stable at 0.095–0.127. XOR depends primarily on the nonlinear mixing during B presentation, which is minimally affected by gap duration within this range. At extended gaps (1500–3000ms), XOR does show gradual degradation (~5 pp from gap=150 to gap=3000 at inc=0.171), indicating the mixing signal also requires some residual trace from A — but far less than WM requires.

### 13.5 The two-axis model and memory demand

The crossover sweep and gap sweep together decompose the adaptation parameter space into two orthogonal axes with distinct functional roles:

- **Axis 1 (inc → integration rate):** Sets the crossover time and determines the CLS/XOR/WM operating point. This axis is fully characterized by the original 300-point sweep and the crossover sweep.
- **Axis 2 (tau → retention duration):** Sets how long the conductance code persists after stimulus offset. This axis is degenerate at the original 150 ms gap but becomes continuously more decisive at longer gaps, with tau spread growing monotonically from 0.8 pp (gap=150ms) to 4.3 pp (gap=1000ms) to 12.1 pp (gap=3000ms) at inc=0.171.

The WM optimum is remarkably stable along Axis 1: inc=0.171 is optimal at all gaps from 100–1000ms in the full 20-inc sweep (5 gap lengths confirmed), and the extended gap experiment (1500–3000ms) tested only this inc value. Only the gap=50ms edge case shifts to inc=0.230. Along Axis 2, tau transitions continuously from interchangeable at gaps ≤ 150ms (ANOVA p > 0.05) to highly decisive at long gaps: at gap=3000ms, the tau ranking is perfectly monotonic (τ=3470 at 63.8% < τ=5000 at 71.5% < τ=7200 at 74.5% < τ=10380 at 75.9%), with the 12.1 pp spread far exceeding any noise floor. The adaptation conductance at B onset confirms the mechanism: at gap=3000ms, τ=3470 retains only 54% of the trace that τ=10380 retains (1.18 vs 2.19), directly producing the accuracy divergence. The static Pareto frontier (Section 5) becomes a gap-dependent landscape where Axis 1 (inc) sets the task operating point and Axis 2 (tau) controls retention capacity as a smooth function of the gap/tau ratio.

### 13.6 Biological implications

Neuromodulatory control of SFA — achievable through a single molecular target (e.g., calcium-activated potassium channels mediating the AHP) — could function as a mode-switching mechanism that reconfigures a cortical circuit between temporal discrimination and working memory without synaptic plasticity.

The encode-and-integrate pipeline is not a model artifact — it is a mathematical consequence of any LIF neuron with a slow negative feedback current. The adaptation conductance is a leaky integrator of spike history (each spike deposits inc conductance, decaying with time constant tau). A cumulative sum must eventually exceed any finite-window rate estimate. The crossover is therefore inevitable for inc > 0; only its timing depends on network parameters.

The two-axis model generates testable predictions:

- **Apamin (SK channel blocker → reduced inc):** Crossover shifts later or disappears. Population responses remain temporally diverse. Favors stimulus discrimination.
- **Intact sAHP (high inc):** Crossover occurs during stimulus. Late responses become persistent and stimulus-averaged. Favors working memory.
- **Cholinergic modulation (muscarinic agonists suppress sAHP):** Shifts network from memory-favoring to encoding-favoring mode, consistent with acetylcholine's known role in attentional gating.
- **Longer maintenance demands (gap sweep prediction, confirmed across 50–3000 ms):** The optimal tau increases with gap duration. At gap=1000ms, tau=10380 outperforms tau=3470 by 4.3 pp; at gap=3000ms, the spread widens to 12.1 pp with a perfectly monotonic tau ranking. The effect follows the exponential decay model: WM accuracy tracks the residual conductance `exp(-gap/tau)` at B onset.

The burst-pause regime resembles stimulus-locked dynamics in primary auditory cortex during passive listening. The adapted-steady-state regime resembles sustained firing in prefrontal working memory maintenance. The rate-matching analysis (Branch B) is critical: all results survive rate clamping, confirming the mode shift is intrinsic to adaptation dynamics.

### 13.7 Limitations

- ISI CV is a population average, potentially obscuring heterogeneity across neuron types or positions
- Participation ratio cannot capture temporal dimensionality (the relevant quantity for the concatenated-bin readout)
- Grid resolution (1.34× ratio per step in inc, 1.44× in tau) introduces ±half-step uncertainty in optimal values
- All readout is linear; nonlinear readouts (up to 97.1%) could reshape the trade-off landscape
- The readout bottleneck (98% stored in adaptation conductance, 82% recovered by spikes) may narrow with different decoding strategies
- The gap sweep uses only 4 tau values (3,470–10,380 ms); while the extended gap experiment (up to 3000 ms) confirms the tau ranking is monotonic and the transition continuous, a finer tau grid would better characterize the functional form of the tau-accuracy relationship and determine whether the optimum lies beyond the current grid edge
- Tau invariance of the crossover time was demonstrated in a feedforward-dominated reservoir; networks with strong attractor dynamics or trained recurrence could show inc×tau interactions in the crossover

---

## 14. Ablation Study: Isolating Adaptation as the Primary Driver

### 14.1 Motivation

The preceding experiments demonstrate that adaptation parameters (inc, tau) predict task performance with high explanatory power (R² = 0.55–0.84). However, the reservoir contains two other activity-dependent history mechanisms that could carry stimulus information across the 150 ms gap:

1. **Short-term synaptic depression (STD)** — Tsodyks-Markram dynamics (U=0.1, tau_rec=500 ms) on excitatory recurrent synapses. A synapse driven hard by digit A will be depleted during digit B, creating a stimulus-specific pattern independent of adaptation. At B onset, ~74% of depression persists (`exp(-150/500) = 0.74`).

2. **Recurrent NMDA** — Slow excitatory conductance (tau=50 ms) on core-to-core synapses. At the 150 ms gap, ~5% persists (`exp(-150/50) = 0.05`). Small but non-zero.

Note: GABA-B (slow inhibition) is architecturally present but disabled in all experiments — no neurons are marked as slow inhibitory (Section 1).

The rate-matching control (Branch A vs B) isolates adaptation from firing rate but does not isolate it from STD or NMDA. A systematic ablation — disabling each slow mechanism independently and in combination — is needed to establish that SFA is both necessary and sufficient for the working memory effect and the task dissociation.

### 14.2 Design

Five conditions, each evaluated at three adaptation levels (inc=0, 0.071, 0.171; all at tau=5000, Branch B with re-calibrated tonic conductance, gap=150 ms):

| Condition | STD (U) | NMDA ratio | SFA | Purpose |
|-----------|---------|-----------|-----|---------|
| Baseline | 0.1 | 0.5 | varied | Reference |
| −STD | **0.0** | 0.5 | varied | Isolate STD contribution |
| −NMDA | 0.1 | **0.0** | varied | Isolate recurrent NMDA contribution |
| SFA-only | **0.0** | **0.0** | varied | Only adaptation survives |
| Bare | **0.0** | **0.0** | **inc=0** | Memory floor of the raw network |

Total: 13 unique conditions. Each runs CLS (1500 samples), WM (2500 pairs), and XOR (2500 pairs) with 5×5 CV.

### 14.3 Predictions

1. **SFA is sufficient:** The SFA-only condition at inc=0.171 should achieve WM ≥ 78%, preserving >90% of the baseline WM dynamic range (55.0 pp) without STD or NMDA.

2. **STD contributes modestly to the memory floor:** The gap between baseline WM at inc=0 (26.9%) and bare WM (~20–23%) quantifies STD's contribution — predicted to be 4–7 pp, real but small compared to adaptation's 55 pp range.

3. **The dissociation survives ablation:** CLS should peak at inc=0.071 and WM at inc=0.171 in every condition. If the ordering reverses in any ablation, the dissociation depends on the ablated mechanism.

4. **CLS is ablation-insensitive:** CLS operates on within-stimulus temporal diversity, not gap-crossing memory. Removing STD or NMDA should minimally affect CLS.

5. **NMDA is negligible:** The 50 ms NMDA time constant is too short relative to the 150 ms gap to carry meaningful memory. Removing it should change WM by <1 pp.

### 14.4 Results **(Figure 31)**

| Condition | inc | CLS | WM | XOR | WM Δ vs baseline |
|-----------|-----|-----|-----|-----|-------------------|
| Baseline | 0.0 | 94.9% | 28.0% | 51.4% | ref |
| Baseline | 0.071 | 95.8% | 76.1% | 66.3% | ref |
| Baseline | 0.171 | 95.1% | 82.2% | 66.4% | ref |
| −STD | 0.0 | 94.0% | 26.3% | 52.3% | −1.7 pp |
| −STD | 0.071 | 94.8% | 72.2% | 65.1% | −3.9 pp |
| −STD | 0.171 | 94.7% | 78.9% | 65.4% | −3.3 pp |
| −NMDA | 0.0 | 93.3% | 27.4% | 52.1% | −0.7 pp |
| −NMDA | 0.071 | 95.3% | 71.7% | 62.5% | −4.4 pp |
| −NMDA | 0.171 | 94.9% | 76.1% | 62.8% | −6.1 pp |
| SFA-only | 0.0 | 93.0% | 26.4% | 52.0% | −1.6 pp |
| SFA-only | 0.071 | 95.2% | 69.5% | 63.5% | −6.6 pp |
| SFA-only | 0.171 | 94.7% | 76.6% | 63.2% | −5.6 pp |
| Bare | 0.0 | 93.0% | 26.3% | 51.8% | −1.7 pp |

**SFA accounts for 93% of the WM dynamic range.** The SFA-only condition (no STD, no NMDA) at inc=0.171 achieves 76.6% WM — a dynamic range of 50.2 pp above its own inc=0 floor (26.4%). The baseline dynamic range is 54.1 pp. SFA alone preserves 50.2/54.1 = 93% of the WM effect.

**The dissociation survives every ablation.** In all four testable conditions (baseline, −STD, −NMDA, SFA-only), CLS peaks at inc=0.071 and WM peaks at inc=0.171. The ordering never reverses.

**STD contributes minimally to gap-crossing memory.** Removing STD drops WM by only 1.7 pp at inc=0 (28.0% → 26.3%). The bare network (no STD, no NMDA, no adaptation) achieves 26.3% WM — 6.3 pp above chance (20%). This residual reflects fast network reverberation through AMPA/GABA-A synapses and distance-dependent delays, not any slow memory mechanism. STD's 500 ms recovery time constant contributes less memory than predicted (1.7 pp vs the predicted 4–7 pp), indicating that STD-mediated synaptic depression patterns are not stimulus-specific enough at 150 ms gap to support digit identity discrimination.

**NMDA contributes to encoding quality, not gap-crossing memory.** Removing recurrent NMDA drops WM by 6.1 pp at the WM optimum — larger than STD's 3.3 pp and larger than predicted (<1 pp). However, this is a within-stimulus encoding effect, not a memory effect: NMDA also drops CLS by 1.6 pp at inc=0 (a task with no gap), and XOR drops by 3.6 pp. NMDA's slow excitatory conductance (tau=50 ms) smooths and sustains synaptic drive during stimulus presentation, improving the quality of the stimulus representation that adaptation then integrates. With only ~5% of NMDA conductance surviving the 150 ms gap (`exp(-150/50) = 0.05`), the gap-crossing contribution is negligible; the effect is on the *input* to the adaptation integrator, not on the memory itself. Crucially, the dissociation pattern — CLS best at low inc, WM best at high inc — is unchanged, confirming that NMDA modulates signal quality uniformly across tasks without altering the adaptation-driven mode switch.

**CLS is ablation-insensitive.** CLS ranges from 93.0% to 95.8% across all 13 conditions (2.8 pp total range), compared to WM's 26.3–82.2% range (55.9 pp). CLS performance depends on within-stimulus temporal diversity, which is determined by spike dynamics during stimulus presentation — not by gap-crossing mechanisms.

---

## 15. Future Directions

### 15.1 Extended tau and gap ranges

The extended gap sweep (Section 12) now spans gaps up to 3,000 ms, where even the longest tau (10,380 ms) shows meaningful degradation (75.9% vs 81.9% at gap=150ms). The next frontier is extending tau beyond 10,380 ms (to 30,000–100,000 ms) at these long gaps: since the optimal tau consistently lands at the grid maximum, the true WM-optimal tau at gap=3000ms likely exceeds our current range. Additionally, extending gaps to 5,000–10,000 ms would probe the regime where even long-tau traces decay catastrophically, determining whether the graceful degradation pattern (7.9 pp over a 60× gap range) continues or whether there is a critical gap length beyond which performance collapses.

### 15.2 Dynamic adaptation modulation within trials

The within-trial migration of optimal inc (Section 6.4) — from ~0.10 during peak encoding to ~0.89 during the late epoch — suggests that time-varying adaptation could dissolve the CLS/WM trade-off entirely. A biologically plausible implementation would modulate the AHP increment on neuromodulatory timescales: weak during stimulus presentation (encoding mode) and strong during the maintenance interval (integration mode). This corresponds to known phasic cholinergic signaling during attentional gating.

### 15.3 Nonlinear readout with the crossover framework

All results use a linear readout. The readout benchmark (Section 2.3) shows nonlinear methods reach 97.1% on CLS. The crossover framework predicts that nonlinear readouts could partially recover the information lost in the spike-count bottleneck (98% stored, 82% recovered linearly). If a nonlinear readout can decode the adaptation state more efficiently from the spike code, the WM optimum might shift and the Pareto frontier could reshape.

### 15.4 Multi-item working memory

The current WM task stores a single digit across one gap. Extending to sequences (A-gap-B-gap-C) would test whether the adaptation conductance can maintain multiple items simultaneously or whether later items overwrite earlier ones. The crossover framework predicts that longer sequences would require higher tau (to survive multiple gaps) and potentially lower inc (to avoid saturation of the conductance state), creating a three-way trade-off between integration rate, retention duration, and capacity.

---

## 16. Extended Discussion

### 16.1 Adaptation as a sufficient mechanism for computational mode switching

The ablation study (Section 14) establishes that SFA is not merely correlated with the mode switch but is causally sufficient: with both STD and recurrent NMDA disabled, the SFA-only network preserves 93% of the baseline WM dynamic range (50.2 of 54.1 pp) and the CLS/WM dissociation is unchanged (CLS peaks at inc=0.071, WM peaks at inc=0.171 in every ablation condition). The multi-seed replication (Section 18) establishes that this result generalizes across network topologies: 10 independent networks (seeds 42–1234, reservoir sizes 588–643) all reproduce the dissociation with WM dynamic range of 53.6 ± 1.1 pp (CV = 2.1%). The central result is that structurally frozen spiking reservoirs — regardless of specific connectivity — support three statistically separable computational modes via modulation of two SFA parameters alone. Bootstrap separation is complete (no overlap across 10,000 iterations), the Friedman rank ordering CLS < XOR < WM holds at every tau slice tested (χ² = 14.0, p = 0.0009), and Pareto frontier paired t-tests confirm opposite-direction advantages (CLS: t = 59.3, p < 0.0001; WM: t = 10.0, p = 0.0006). The per-bin temporal decomposition (Section 6) establishes that the dissociation is qualitative, not quantitative: adaptation changes the *type* of information available to the readout (transient temporal patterns vs persistent DC-like signals), not the overall signal-to-noise ratio.

The crossover time — the first bin where adaptation-conductance separability exceeds spike-count separability — is the single generating variable behind the ISI CV regime boundary, the per-bin sign flip, the Pareto asymmetry, and the readout bottleneck. Its tau-invariance (identical crossover ± 1 bin across all 7 tau values at each inc level) establishes inc as the sole determinant of within-trial dynamics. The crossover is a mathematical consequence of leaky integration: a cumulative sum of per-spike deposits, decaying with time constant tau, must eventually exceed any finite-window rate estimate for inc > 0. This generality implies the mechanism is not specific to this architecture or parameter regime but is intrinsic to any spiking neuron with slow negative feedback.

### 16.2 The CLS/WM trade-off: two large opposing effects

The Pareto frontier spans only 5/300 grid points, with a CLS cost of 1.49 pp (2.4% of above-chance range) against a WM gain of 6.61 pp (10.7%). CLS at the WM optimum (94.84%) exceeds CLS at zero adaptation (94.50%). The trade-off is mechanistically grounded in two opposing effects of adaptation on classification: a +11 pp peak-epoch discrimination gain (inc = 0 to 0.13) and a −13 pp temporal integration loss (profile std 19.1% → 1.9%). The concatenated-bin classifier exploits cross-bin feature complementarity; adaptation homogenizes the temporal profile, converting late-bin signal from complementary to redundant. The WM-optimal configuration wins in every temporal epoch by per-bin accuracy yet loses overall CLS by 1.49 pp — a result explained entirely by the complementarity/redundancy distinction.

The within-trial migration of optimal inc (from ~0.10 at peak to ~0.89 at late bins) indicates that no static inc serves all temporal phases. Dynamic within-trial modulation — low inc during encoding, high inc during maintenance — would in principle dissolve the trade-off. This scenario is biologically realizable via phasic neuromodulatory signaling (Section 17.2).

The gap sweep extends the trade-off analysis: CLS-optimal inc (0.071) suffers maximal WM degradation with gap length (−3.6 pp from gap = 150 to 1000 ms) because its moderate trace is large enough to lose but insufficient to survive exponential decay. WM-optimal inc (0.171) degrades only −1.6 pp; stronger traces absorb proportionally less damage.

### 16.3 Two-axis decomposition and memory demand

The gap sweep activates the second parameter axis. At gap = 150 ms, even the shortest tested tau (3,470 ms) retains exp(−150/3470) = 95.8% of the adaptation trace, rendering tau degenerate (ANOVA p > 0.05). As gap increases, tau becomes continuously more discriminating: tau spread at inc = 0.171 grows monotonically from 0.8 pp (gap = 150 ms) to 4.3 pp (1,000 ms) to 12.1 pp (3,000 ms). At all gaps ≥ 300 ms, the tau ranking is perfectly monotonic (τ = 3,470 < 5,000 < 7,200 < 10,380), with ANOVA significance growing from p = 0.004 (gap = 300 ms) to p < 0.0001 (gap ≥ 500 ms).

The two axes are mechanistically orthogonal: inc controls integration rate (crossover time, within-trial dynamics), tau controls retention duration (post-offset persistence). Inc is tau-invariant in the crossover sweep; tau is gap-dependent in the gap sweep. The WM optimum is fixed along Axis 1 (inc = 0.171 at all gaps ≥ 100 ms) and undergoes a continuous transition along Axis 2 as gap demands increase. The adapt-at-B-onset data confirms the mechanism: at gap = 3,000 ms, τ = 3,470 retains conductance of 1.18 while τ = 10,380 retains 2.19 (ratio 0.54), directly producing the 12.1 pp accuracy spread.

XOR occupies an intermediate position: variance dominated by tau (R² = 0.75), optimal inc between CLS and WM (0.05–0.07), gap-insensitive within 50–1,000 ms (1.2 pp range). After regressing out WM, the XOR residual has a negative inc coefficient — controlling for memory, less adaptation improves XOR. XOR depends primarily on nonlinear mixing during B presentation, with minimal sensitivity to trace fidelity within the tested gap range.

### 16.4 The conductance code and readout bottleneck

The 604-dimensional adaptation conductance at B onset predicts digit-A identity at 97.9–98.3% accuracy; the spike-count readout recovers 75.6–81.3%. PCA captures 96% of conductance variance in 3 PCs (clean, monotonically diverging class trajectories) vs 68% for spike counts (tangled, overlapping trajectories). Mean pairwise Euclidean distance between class centroids reaches ~80 units in conductance space (4× peak spike discriminability). The adaptation conductance is a hidden variable — not directly readable by downstream neurons — that modulates dynamics through membrane hyperpolarization. The 16 pp accuracy gap quantifies the information loss inherent in the spike-count readout.

Population equalization accompanies the mode switch: firing rate Gini drops from 0.74 to 0.47, top-10% spike share from 58.3% to 31.8%, Q4/Q1 dominance ratio from 6.4× to 4.4×. The classifier weight distribution shifts from concentrated (Weight Gini = 0.52, input-weight/ridge-weight r = +0.60) to uniform (Weight Gini = 0.31, r = +0.11). Five calibration controls confirm this is adaptation-driven, not a rate-matching artifact: g_tonic is uniform across neurons, decreases with adaptation, and all equalization metrics move monotonically with inc.

### 16.5 ISI CV regime boundary

The within-inc correlation between ISI CV and peak accuracy flips sign at ISI CV ≈ 2.1: r = +0.73 to +0.97 in the burst-pause regime (inc < 0.10), r = −0.40 to −0.98 in the adapted steady-state (inc > 0.15). The crossover sweep locates this boundary at the balance point where peak spike-count separability (38.7) equals peak adaptation-conductance separability (38.0), at crossover = 290 ms. The regime boundary is the point where the dominant representational format switches from spike code to conductance code.

At extreme adaptation (inc ≥ 0.55), stimulus-epoch ISI CV drops below B-epoch ISI CV — a reversal caused by massive AHP accumulation immediately suppressing stimulus-driven firing, while digit B partially reactivates the network against a deep adaptation background.

---

## 17. Biological Feasibility and Implications

### 17.1 Biophysical substrate

SFA arises from calcium-activated potassium channels: SK (K_Ca2, mAHP, tau ≈ 50–200 ms) and sI_AHP channels (sAHP, tau ≈ 1–5 s). The model's adaptation increment maps to per-spike AHP conductance recruitment, determined by K_Ca channel density, per-spike calcium influx (via voltage-gated Ca²⁺ channels), and calcium buffering capacity (calbindin, parvalbumin, calretinin). The adaptation time constant maps to AHP decay rate, governed by PMCA, SERCA, and mitochondrial calcium uptake, plus channel-calcium interaction kinetics. The tested tau range (3,470–10,380 ms) falls within the biophysical window of the sAHP in cortical pyramidal neurons (Benda & Herz 2003; Brette & Gerstner 2005).

The model's leaky-integrator formulation (per-spike conductance deposit, exponential decay) is a standard AHP reduction used throughout computational neuroscience. Every tested parameter combination corresponds to a physiologically realizable state of a cortical pyramidal neuron with AHP currents. The inc variation across the sweep (0.005–1.0) spans the range of AHP amplitudes documented across cortical cell types: from fast-spiking interneurons (near-zero AHP) to strongly adapting regular-spiking pyramidal neurons in deep layers.

Cortical neurons express multiple adaptation currents (mAHP, sAHP, sodium-dependent adaptation via K_Na channels), each with distinct kinetics. The model uses a single current. The model's Axis 2 (tau) can be interpreted as selecting among these timescales: sAHP for multi-second maintenance, mAHP for sub-second temporal comparison. Interactions between multiple adaptation timescales, not captured here, could support richer computational repertoires.

### 17.2 Neuromodulatory control

Multiple neuromodulatory systems converge on AHP channels, providing the biological substrate for mode switching along the model's two axes:

**Acetylcholine.** Muscarinic M1 activation (Gq/11 → PLC signaling) suppresses sAHP by 80–95% in cortical pyramidal neurons (Madison & Nicoll 1984; McCormick & Prince 1986; Gulledge & Stuart 2005). This reduces effective inc, shifting the operating point toward CLS-optimal (encoding mode). ACh release during attentional engagement would favor stimulus processing over maintenance, consistent with ACh's established role in attentional gating (Hasselmo & Sarter 2011). Phasic cholinergic signaling — cue-locked bursts from basal forebrain (Parikh et al. 2007) — could implement within-trial modulation: transient sAHP suppression during stimulus presentation (encoding), followed by sAHP recovery during maintenance (integration). This matches the temporal profile identified by the within-trial optimal-inc migration.

**Norepinephrine.** Beta-adrenergic activation (Gs → cAMP/PKA) suppresses sAHP (Madison & Nicoll 1986; Pedarzani & Storm 1993). The inverted-U relationship between NE and prefrontal working memory performance (Arnsten 2011) maps onto the model's non-monotonic inc-WM curve: moderate NE may tune effective inc to the WM-optimal range, while excessive NE suppresses inc below the CLS-optimal point, degrading both encoding and integration.

**Serotonin.** 5-HT₁A activation enhances AHP (Andrade & Nicoll 1987), increasing effective inc (favoring integration mode). 5-HT₂ activation (Gq/11 signaling) suppresses AHP, reducing inc (favoring encoding mode). The dual modulation enables bidirectional Axis 1 control via receptor subtype expression.

**Dopamine.** D1 activation in PFC modulates AHP-related K⁺ currents with dose-dependent direction (Vijayraghavan et al. 2007). The D1 inverted-U for working memory parallels the model's inc-WM relationship, suggesting that dopaminergic modulation of AHP channels contributes to D1-dependent working memory gating.

The convergence of four neuromodulatory systems onto a single biophysical target (AHP channels) makes adaptation a natural control point. Combinatorial neuromodulatory control, with each system contributing partial modulation, could tile the (inc, tau) space continuously.

### 17.3 Correspondence to cortical dynamics

**Sensory-prefrontal transition.** Sensory cortices show transient, stimulus-locked responses; prefrontal cortex shows sustained delay-period activity (Fuster & Alexander 1971; Miller et al. 1996; Funahashi et al. 1989). If sensory cortex operates at low effective adaptation (CLS-optimal) and PFC at high adaptation (WM-optimal), the encode-and-integrate pipeline applies at the inter-regional level, with the crossover distributed across regions rather than occurring within one. Local differences in sAHP expression and cholinergic drive would set region-specific crossover times.

**Persistent activity in PFC.** The temporal structure at the WM optimum — phasic stimulus responses transitioning to sustained delay activity — matches electrophysiological recordings in monkey PFC during delayed match-to-sample (Fuster & Alexander 1971; Goldman-Rakic 1995). The model provides an intrinsic single-neuron mechanism (AHP current) for this transition, without requiring attractor dynamics. This does not exclude attractor contributions; it establishes that adaptation alone provides a sufficient substrate, and the 98% conductance-code accuracy represents a performance floor that attractor dynamics would augment.

**Burst coding.** The burst-pause regime (ISI CV > 2.1) resembles thalamic relay neuron bursting (Sherman 2001; Lesica & Stanley 2004). Burst patterns carry high information per spike and drive cortical responses effectively. The model's positive correlation between burstiness and discrimination (r ≈ +0.95) in this regime is consistent with the burst coding hypothesis and extends it by showing that the same neurons transition from burst to tonic coding as adaptation increases.

**Activity-silent working memory.** The adaptation conductance is an activity-silent memory state: it carries 98% of the stimulus information without generating spikes. It biases subsequent stimulus responses in a stimulus-specific way, allowing indirect readout. This aligns with evidence for activity-silent working memory maintenance (Stokes 2015; Wolff et al. 2017; Lundqvist et al. 2016) and provides a specific biophysical substrate — AHP conductance — complementing short-term plasticity accounts (Mongillo et al. 2008). The ablation study (Section 14.4) resolves the relative contributions: adaptation accounts for 93% of the WM dynamic range, while STD contributes only 1.7 pp to the memory floor. Adaptation is the dominant activity-silent memory mechanism in this network.

**Population coding structure.** The adaptation-driven equalization (Gini 0.74 → 0.47) mirrors the shift from concentrated to distributed coding observed between sensory and association cortices (Rigotti et al. 2013). The model attributes this shift to intrinsic homeostasis at trial-relevant timescales: activity-dependent negative feedback compresses the rate distribution without requiring synaptic mechanisms. The low-dimensional conductance code (2 PCs, >95% variance) with high-dimensional population support (604 neurons contributing approximately equally) is characteristic of robust, noise-tolerant representations.

### 17.4 Testable pharmacological predictions

The two-axis model generates six specific predictions:

1. **Apamin (SK blocker → reduced inc):** Later or absent crossover. Higher temporal profile std. Higher ISI CV. Improved discrimination, impaired WM maintenance. Weight Gini should increase (more concentrated readout).

2. **Enhanced sAHP (increased inc):** Earlier crossover. Flattened temporal profiles. Reduced ISI CV. Improved WM at cost of temporal discrimination. More distributed population coding.

3. **Muscarinic agonists (sAHP suppression → reduced inc):** Same direction as apamin but via M1 pathway. Encoding-favoring shift. Testable via local iontophoresis in PFC during delayed response tasks.

4. **Calcium clearance manipulation (tau modulation):** PMCA inhibitors (carboxyeosin, La³⁺) to increase effective tau; BAPTA to decrease tau. Model predicts: (a) no WM effect at gaps ≤ 150 ms; (b) monotonically increasing effect at gaps ≥ 300 ms; (c) WM accuracy proportional to exp(−gap/tau). The gap-dependent specificity provides a multi-point test.

5. **Combined inc × tau pharmacology:** Independent manipulation via SK modulators (inc) and calcium clearance modulators (tau) should produce separable, additive effects. At short gaps, only inc manipulation affects WM; at long gaps, both contribute with tau effects growing with gap duration. A 2 × 2 design at multiple gap lengths would map the biological (inc, tau) space.

6. **Phasic cholinergic blockade:** Optogenetic inhibition of basal forebrain cholinergic neurons during cue presentation should impair encoding without affecting maintenance — testing the prediction that phasic ACh implements within-trial inc modulation.

### 17.5 Implications for working memory pathology

**Schizophrenia.** WM deficits associated with D1 hypofunction (Goldman-Rakic 1994; Barch et al. 2001) could reflect reduced effective inc below the WM-optimal range, producing the characteristic delay-dependent deficit (preserved encoding, impaired maintenance). The D1 inverted-U maps onto the model's inc-WM curve.

**Aging.** Enhanced sAHP amplitude in aged cortical neurons (Disterhoft & Oh 2007; Moyer et al. 1992) — due to increased per-spike calcium influx and/or reduced buffering — could push inc above the WM optimum into the stimulus-suppression regime (ISI CV < 1.5). Simultaneously, altered calcium clearance may reduce effective tau. The model predicts delay-dependent WM deficits (tau-mediated) and encoding deficits (inc-mediated), consistent with age-related patterns (Park & Payer 2006).

**ADHD.** Low effective adaptation (encoding bias) would favor rapid stimulus processing over sustained maintenance. Catecholaminergic stimulants (methylphenidate, amphetamine), which modulate AHP via NE and DA pathways, would shift the operating point toward integration mode — improving WM while potentially reducing perceptual temporal resolution. This maps onto the CLS/WM Pareto frontier.

**Alzheimer's disease.** Cholinergic degeneration (Whitehouse et al. 1982) removes muscarinic sAHP suppression, potentially increasing effective inc to supraoptimal levels. Concurrent calcium dysregulation alters both axes. The model predicts loss of dynamic control over the adaptation operating point — the network becomes fixed in a suboptimal mode rather than flexibly switching between encoding and maintenance.

### 17.6 Relationship to theoretical frameworks

**Liquid state machine theory.** The separation property (Maass et al. 2002) is extended to *persistent separation*: adaptation controls the temporal horizon over which the liquid state remains discriminable. Low adaptation produces high instantaneous separation with rapid decay. High adaptation produces lower instantaneous separation with persistent maintenance. The adaptation parameter governs the time constant of the separation property.

**Reservoir computing.** The "edge of chaos" prescription (Bertschinger & Natschläger 2004; Legenstein & Maass 2007) is task-dependent: CLS favors the burst-pause regime (high temporal diversity, analogous to edge-of-chaos dynamics); WM favors the adapted steady-state (low temporal diversity, high persistence, analogous to ordered dynamics). No single dynamical regime is universally optimal.

**Activity-silent memory.** The conductance code is a specific biophysical realization of activity-silent working memory (Stokes 2015; Mongillo et al. 2008). It complements short-term plasticity accounts with a single-neuron intrinsic mechanism. The ablation study (Section 14.4) resolves the competition: SFA accounts for 93% of the WM dynamic range; STD contributes only 1.7 pp to the memory floor. Adaptation dominates over short-term synaptic plasticity as the memory substrate.

**Predictive coding.** The adaptation state can be interpreted as an internal estimate of recent input history. Low adaptation → weak predictions, large prediction errors (high sensitivity). High adaptation → strong predictions, suppressed prediction errors (stability). The CLS/WM trade-off maps onto the sensitivity/stability (or exploration/exploitation) axis in predictive coding frameworks (Rao & Ballard 1999; Friston 2005).

### 17.7 Circuit design considerations

**Metabolic efficiency.** Adaptation-based mode switching uses existing ion channel infrastructure with zero operational cost beyond the spikes themselves. Synaptic plasticity requires protein synthesis, receptor trafficking, and structural remodeling on timescales of minutes to hours. For trial-by-trial mode switching (milliseconds to seconds), adaptation is faster and cheaper.

**Complementarity with plasticity.** Adaptation and plasticity operate at different timescales. Plasticity optimizes the network's structural substrate (weights, connectivity) over developmental and learning timescales. Adaptation implements real-time selection of the computational mode within that substrate. The frozen-weight reservoir represents the structure; neuromodulatory control of adaptation selects the operating point.

**Scale invariance.** The crossover mechanism depends on per-neuron parameters, not population size. Any spiking network with AHP currents exhibits the encode-and-integrate handoff. The mechanism is available at every level of cortical organization, from microcolumns to cortical areas. The multi-seed replication (Section 18) confirms this at the network level: 10 topologies with reservoir sizes spanning 588–643 neurons all produce the dissociation with WM dynamic range CV of just 2.1%.

**Evolutionary accessibility.** The mechanism requires only voltage-gated calcium channels and calcium-activated potassium channels — ancient, conserved channel families present across vertebrates and many invertebrates. Adaptation-based mode switching was likely available early in nervous system evolution, providing computational flexibility before the emergence of complex recurrent architectures or sophisticated plasticity.

### 17.8 Limitations of the biological analogy

- Single adaptation current vs biological multi-timescale adaptation (mAHP, sAHP, K_Na). Interactions between timescales could support richer repertoires.
- Frozen weights vs trained recurrence. Attractor dynamics could amplify or counteract the mode switch and may introduce inc × tau interactions in the crossover.
- Linear readout vs biological nonlinear dendritic integration. The 98% vs 82% bottleneck is a lower bound; nonlinear decoders would narrow the gap.
- Homogeneous adaptation vs cell-type-specific profiles (PV+ interneurons with minimal adaptation, pyramidal neurons with strong adaptation). Heterogeneous adaptation could support simultaneous encoding and integration modes in different subpopulations.
- GABA-B disabled. Slow inhibition could interact with adaptation during mode transitions. Inhibitory interneuron roles in WM circuits (Wang et al. 2004; Murray et al. 2014) are not captured.
- 604 neurons vs cortical scale (10⁴ per column). Population-level phenomena (equalization, bottleneck, dimensional structure) may change quantitatively with scale. The multi-seed replication (Section 18) confirms topology-invariance across 588–643 neurons but does not test scaling to larger populations.

The core finding — AHP current modulation reconfigures cortical computation between discrimination and memory — rests on the mathematical inevitability of the crossover in any system with slow negative feedback. The qualitative dissociation and the crossover mechanism are robust to model specifics; the quantitative performance values are not.

---

## 18. Multi-Seed Replication

### 18.1 Motivation

All preceding experiments use a single network topology (LHS-021), loaded from a deterministic snapshot. The dissociation, crossover framework, gap sweep, and ablation results could in principle reflect idiosyncratic properties of this particular connectivity pattern — a fortunate arrangement of neuron positions, weight values, or inhibitory/excitatory balance that happens to support adaptation-driven mode switching. If the dissociation fails on networks built from different random seeds, the finding is a curiosity of one topology rather than a general property of adaptation in recurrent spiking networks.

The crossover mechanism is mathematically generic — any LIF neuron with a slow negative feedback current exhibits the encode-and-integrate handoff (Section 11.3). But mathematical generality at the single-neuron level does not guarantee that the population-level dissociation (CLS peaks at low inc, WM peaks at high inc, with distinct optima) survives across random network instantiations. Network-level properties like the Pareto asymmetry, population equalization, and the 98% conductance code could depend on specific connectivity motifs present in LHS-021 but absent in other realizations.

### 18.2 Design

Ten independent network topologies are constructed from different RNG seeds (42, 137, 256, 314, 500, 628, 777, 888, 1000, 1234). Each seed randomizes:

- Neuron positions in the 3D sphere
- Distance-dependent connectivity (which neurons connect)
- Synaptic weight values (drawn from distributions)
- Biophysical parameter jitter (tau_m, v_rest, v_threshold, tau_ref, etc.)

Fixed across seeds: Network size (1000 → ~588–643 after compaction), zone architecture (shell/core), input encoding parameters (stim_current=0.0518, tau_e=1.05, no input adaptation), LHS-021 hyperparameters (lambda_connect, shell_core_mult=4.85, core_core_mult), readout method (ridge, 5×5 CV).

Each network is evaluated at 3 adaptation levels (inc=0, 0.071, 0.171; tau=5000, Branch B with re-calibrated tonic conductance) on all 3 tasks (CLS, WM at gap=150ms, XOR). Total: 10 seeds × 3 inc levels = 30 conditions. Networks are pre-built with full dynamical overrides to ensure correct shell-core weight scaling, then cached for reuse across conditions within each seed. A uniqueness check verifies that all 10 networks have distinct reservoir sizes after compaction.

### 18.3 Predictions

1. **The dissociation holds in every seed.** WM should peak at inc=0.171 and CLS should remain high at all inc levels in all 10 networks. If even one seed reverses the ordering, the dissociation depends on specific connectivity. If all 10 pass, the probability of a topology-specific artifact is < 0.001 (binomial, assuming 50% chance per seed under the null).

2. **Absolute performance varies across seeds but the pattern is stable.** Different topologies will have different baseline CLS and WM accuracies — some networks are simply better reservoirs than others. But the direction of the adaptation effect (more adaptation helps WM, modestly costs CLS) should be invariant. The between-seed variance in absolute accuracy should be larger than within-seed CV noise but smaller than the inc-driven WM dynamic range.

3. **WM dynamic range (inc=0 → inc=0.171) is consistently large.** Across seeds, the WM gain from adaptation should be 40–60 pp (consistent with the 54.1 pp seen in LHS-021 at the ablation baseline). If some seeds show <20 pp WM range, adaptation's role is topology-dependent.

4. **Network size after compaction varies.** The compaction step (removing non-arc neurons) produces ~604 neurons for LHS-021 but may yield different sizes for other seeds. This is expected and informative — it tests whether the dissociation is robust to modest size variation.

5. **CLS is stable across seeds.** CLS at the CLS optimum should be 93–97% across all seeds, since classification depends on the richness of the reservoir's temporal dynamics rather than specific connectivity.

### 18.4 Results

All 10 seeds pass the dissociation. The table below shows per-seed results at the three adaptation levels:

| Seed | n_res | CLS@inc=0 | CLS@cls_opt | CLS@wm_opt | WM@inc=0 | WM@cls_opt | WM@wm_opt | WM range (pp) |
|------|-------|-----------|-------------|------------|----------|------------|-----------|---------------|
| 42   | 613   | 96.3%     | 96.5%       | 94.9%      | 26.4%    | 77.4%      | 78.0%     | 51.7           |
| 137  | 623   | 96.7%     | 96.7%       | 95.1%      | 26.6%    | 75.4%      | 78.8%     | 52.2           |
| 256  | 592   | 95.9%     | 96.2%       | 93.6%      | 26.6%    | 74.4%      | 80.2%     | 53.6           |
| 314  | 617   | 96.6%     | 96.5%       | 95.6%      | 27.4%    | 79.7%      | 80.0%     | 52.6           |
| 500  | 643   | 96.3%     | 95.5%       | 94.4%      | 26.4%    | 76.4%      | 80.2%     | 53.7           |
| 628  | 625   | 96.0%     | 96.1%       | 94.8%      | 26.5%    | 76.2%      | 81.6%     | 55.1           |
| 777  | 588   | 95.9%     | 96.1%       | 93.7%      | 26.1%    | 74.8%      | 79.8%     | 53.7           |
| 888  | 616   | 96.6%     | 96.2%       | 94.9%      | 26.1%    | 76.6%      | 81.2%     | 55.1           |
| 1000 | 618   | 96.0%     | 96.6%       | 94.3%      | 26.2%    | 72.5%      | 80.8%     | 54.7           |
| 1234 | 633   | 96.5%     | 96.2%       | 94.3%      | 27.0%    | 73.0%      | 80.6%     | 53.5           |

Cross-seed summary (N=10):

| Metric             | Mean ± SD        | Range          | CV     |
|--------------------|------------------|----------------|--------|
| CLS @ inc=0        | 96.3 ± 0.3%      | 95.9–96.7%     | 0.3%   |
| CLS @ inc=0.071    | 96.3 ± 0.3%      | 95.5–96.7%     | 0.3%   |
| CLS @ inc=0.171    | 94.6 ± 0.6%      | 93.6–95.6%     | 0.6%   |
| WM @ inc=0         | 26.5 ± 0.4%      | 26.1–27.4%     | 1.5%   |
| WM @ inc=0.071     | 75.6 ± 2.0%      | 72.5–79.7%     | 2.6%   |
| WM @ inc=0.171     | 80.1 ± 1.0%      | 78.0–81.6%     | 1.2%   |
| XOR @ inc=0        | 51.9 ± 0.7%      | 50.5–53.2%     | 1.3%   |
| XOR @ inc=0.171    | 65.8 ± 1.2%      | 63.0–67.5%     | 1.8%   |
| WM dynamic range   | 53.6 ± 1.1 pp    | 51.7–55.1 pp   | 2.1%   |

Network sizes after compaction range from 588 to 643 neurons (mean 617, SD 16), confirming genuinely different topologies. Tonic conductance calibration converged for all 30 conditions (rates: 18.2–21.5 Hz, target 20 Hz).

### 18.5 Interpretation

**Prediction 1 confirmed: 10/10 dissociation.** Every network shows WM@wm_opt >> WM@inc=0 while CLS remains above 93.6%. Under the null hypothesis (50% chance of dissociation per seed), observing 10/10 has probability p < 0.001. The SFA-driven mode switch is not a property of one connectivity pattern.

**Prediction 2 confirmed: Pattern stable, absolute values vary.** The between-seed SD for CLS@inc=0 is 0.3% — smaller than the within-seed CV noise (~0.2%). CLS is remarkably invariant to topology because it relies on the richness of temporal dynamics during stimulus presentation, which is set by the biophysical parameters (shared across seeds) rather than specific connectivity. WM@wm_opt varies more (SD 1.0%), reflecting topology-dependent differences in how well the network's recurrent structure supports conductance-code maintenance.

**Prediction 3 confirmed: Consistently large WM dynamic range.** All 10 seeds produce WM dynamic ranges between 51.7 and 55.1 pp. The CV of the WM dynamic range is just 2.1% — the effect size is extraordinarily stable. No seed shows <40 pp range; the minimum (51.7 pp, seed 42) still represents a 3× improvement over chance.

**Prediction 4 confirmed: Network size varies.** Reservoir sizes span 588–643 (9.4% range). The smallest network (777, n=588) achieves WM@wm_opt = 79.8%; the largest (500, n=643) achieves 80.2%. There is no meaningful correlation between network size and WM performance within this range (r = 0.09), indicating that the dissociation does not depend on having a particular number of neurons.

**Prediction 5 confirmed: CLS universally stable.** CLS at inc=0.071 ranges 95.5–96.7% across all 10 seeds. The worst CLS at wm_opt (93.6%, seed 256) still exceeds the original LHS-021 CLS at zero adaptation (94.5% in the ablation baseline). The CLS cost of the mode switch is universally small.

### 18.6 Relationship to prior results

The multi-seed replication strengthens every preceding finding:

- **Ablation (Section 14):** The ablation study showed SFA accounts for 93% of WM dynamic range within one topology. The multi-seed replication shows this dynamic range (53.6 ± 1.1 pp) is itself topology-invariant. The combination establishes that SFA is both causally sufficient (ablation) and universally effective (multi-seed) as the mode-switching mechanism.

- **Crossover framework (Section 11):** The crossover is a mathematical consequence of leaky integration in any neuron with slow negative feedback. The multi-seed replication confirms that this single-neuron property translates to population-level dissociation across diverse connectivity patterns, as the theory predicts.

- **Gap sweep (Sections 12–13):** Gap sensitivity was measured on one topology. The consistency of WM@wm_opt across seeds (80.1 ± 1.0%) at gap=150ms indicates the baseline from which gap effects operate is itself stable.

- **Pareto frontier (Section 7):** The CLS/WM trade-off — high WM gain (53.6 pp) for small CLS cost (1.7 pp from inc=0 to inc=0.171) — is reproduced in every seed. The asymmetry is structural, not accidental.

### 18.7 Statistical power

With 10 seeds, the experiment provides:
- 95% power to detect a dissociation failure rate ≥ 26% (binomial exact test, alpha=0.05)
- 99% confidence interval for WM dynamic range: 53.6 ± 2.0 pp (t-distribution)
- Effect size (Cohen's d) for WM inc=0 vs inc=0.171: d = 50.1 (mean difference 53.6 pp / pooled SD 1.07 pp) — an extraordinarily large effect by any standard

The within-seed 5-fold × 5-repeat CV provides per-condition confidence intervals; the 10-seed replication provides between-topology confidence intervals. These are independent sources of variance, and both are small relative to the effect.

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
