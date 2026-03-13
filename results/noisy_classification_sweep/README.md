# Noisy Classification Experiment: Encoding Under Uncertainty

## Motivation

### The Problem with Clean Inputs

The central claim of this paper is that spike-frequency adaptation (SFA) continuously steers a fixed spiking reservoir through distinct computational modes. Our data demonstrates that SFA creates a massive working memory benefit (+56pp) and enables nonlinear temporal integration (XOR, +16pp). However, single-digit classification shows only a marginal encoding benefit (+1.5pp), and the per-bin temporal analysis reveals why: SFA never degrades encoding because single-stimulus classification has no interference source. Every spike in every temporal bin reflects the current digit only. SFA can only help (via temporal heterogeneity) or be irrelevant. There is no mechanism for it to hurt.

This creates a narrative problem. We claim "distinct computational modes," but the data shows one mode that matters (memory) and one that doesn't care (encoding). That's not a dissociation — it's a single effect plus a ceiling.

### Why Clean Inputs Mask the Encoding Cost

The BSA baseline for 5-class digit classification is 94.8%. The reservoir at optimal SFA achieves 96.3%. The input representation itself is nearly sufficient. Two PCs of the BSA spectral features explain 54% of variance, and the digits — while overlapping in low-dimensional projections — are separable in the full 6144-dimensional space (128 channels × 48 bins) because each digit has a unique spectral-temporal fingerprint.

SFA's within-digit benefit (breaking population saturation, creating temporal heterogeneity) adds marginal value on top of an already-solved problem. There is no room for SFA to hurt encoding because there is no competing signal whose adaptation trace could corrupt the current representation.

### The Biological Parallel

In cortical circuits, acetylcholine (ACh) and norepinephrine (NE) modulate spike-frequency adaptation and produce well-documented effects on sensory encoding. Crucially, these neuromodulators are released during states of uncertainty, attentional demand, and degraded sensory conditions — not during easy, unambiguous perception. ACh enhances signal-to-noise ratio by suppressing non-selective activity. NE sharpens tuning curves under noisy conditions.

Testing SFA's encoding effects with clean, unambiguous inputs is like measuring the effect of pupil dilation in bright sunlight. The biological system deploys these mechanisms precisely when inputs are degraded, and that is where we should test them.

### The Theoretical Prediction

SFA creates a fundamental tradeoff between two properties a neural population can have:

1. **Temporal redundancy**: Without SFA, the population response is stable across time bins. The same neurons fire at similar rates throughout the stimulus. A classifier can average across bins to cancel noise, because each bin is an independent noisy sample of the same underlying signal.

2. **Temporal heterogeneity**: With SFA, progressively-adapting neurons create a trajectory through state space. Each bin is different. This provides richer features for clean inputs (each bin adds new information) but destroys the redundancy needed to average through noise.

With clean inputs, temporal heterogeneity is strictly beneficial — more diverse features, better classification. But with noisy inputs, the classifier needs redundancy to average out the corruption, and SFA destroys exactly that.

This predicts: **under structured noise, SFA will hurt encoding.** Combined with the existing WM sweep showing SFA helps memory, this completes the dissociation between computational modes.

---

## Experimental Design

### Two Noise Conditions

We introduce two forms of input degradation that test different mechanistic predictions:

#### Condition 1: Structured Distractor (Superimposed Digit)

A second digit's BSA spike train is superimposed on the target digit at a controlled power ratio. The distractor creates structured interference that occupies the same representational space as the signal — it drives the same tonotopically-organized input channels with spectrally coherent patterns. The network cannot distinguish distractor-driven spikes from target-driven spikes.

This is the cocktail party problem. It tests whether the shared adaptation traces from both digits create an ambiguous excitability landscape that degrades encoding.

**Prediction**: SFA hurts encoding under structured distractors because adaptation accumulates from both target and distractor indiscriminately. The resulting suppression pattern reflects a mixture of two digit identities. The temporal heterogeneity SFA creates is contaminated, and the classifier loses the stable redundant features it needs to extract the target.

#### Condition 2: Unstructured Noise (Random Spikes)

Random spikes are injected uniformly across frequency channels at a controlled rate. The noise is spatially uniform and temporally uncorrelated — it has no spectral structure and creates no consistent adaptation pattern across trials.

This is the low-SNR sensory condition. It tests whether SFA-mediated gain normalization can enhance the signal-to-noise ratio by suppressing non-selective (noise-driven) activity, analogous to cholinergic enhancement of sensory encoding.

**Prediction**: SFA helps encoding under unstructured noise because it preferentially suppresses high-rate non-selective neurons driven by noise rather than signal. This is population-level gain normalization: the same mechanism as ACh-mediated enhancement.

### Stimulus Construction

#### Target Digit
Standard BSA spike train, same as all previous experiments. Full power (all spikes included).

#### Structured Distractor
A different digit's BSA spike train, temporally aligned to the target (onset-aligned, looped to fill the target duration if shorter). Distractor spikes are included at a probability proportional to the power ratio:

- **Power ratio 0.0**: Clean target (baseline)
- **Power ratio 0.25**: 25% of distractor spikes randomly included
- **Power ratio 0.50**: 50% of distractor spikes included
- **Power ratio 0.75**: 75% of distractor spikes included
- **Power ratio 1.0**: All distractor spikes included (equal power)

Each target-distractor pair is drawn with target ≠ distractor enforced. The distractor digit identity is recorded for confusability analysis.

#### Unstructured Noise
Random spikes are added uniformly across all 128 frequency channels. The noise rate is defined relative to the mean BSA spike rate:

- **Noise ratio 0.0**: Clean target (baseline)
- **Noise ratio 0.25**: 25% additional random spikes relative to mean BSA rate
- **Noise ratio 0.50**: 50% additional
- **Noise ratio 0.75**: 75% additional
- **Noise ratio 1.0**: Equal noise and signal spike rates

For each noise spike: time drawn uniformly within the stimulus duration, channel drawn uniformly from 0–127.

### Readout Structure

For each trial, we extract:

1. **All-bins**: Full stimulus window, flattened (max_bins × 604). Standard CLS readout.
2. **Last-1-bin**: Final 20ms stimulus bin only (604 features). Temporal endpoint encoding.
3. **Per-bin profiles**: Accuracy computed independently for each 20ms bin. Reveals the temporal structure of how noise and SFA interact.

### Grid Design

#### Pilot (recommended first)
- **Noise levels**: 5 values per condition (0.0, 0.25, 0.5, 0.75, 1.0)
- **SFA grid**: tau=5000 column only (20 inc values)
- **Branches**: A (unmatched) + B (rate-matched)
- **Total**: 5 noise × 20 inc × 2 branch × 2 conditions = 400 grid points
- **Estimated time**: ~20 hours at 3 min/point (no post-stim extension)

#### Full Sweep
Extend to 3 tau values (558, 1670, 5000) if pilot results warrant.

### Samples and Trials

- **Digits**: 0–4 (5-class, matching original CLS experiment)
- **Samples per digit**: 300 (matching original)
- **Trials per grid point**: 1500 (matching original CLS)
- **For structured distractor**: Target ≠ distractor enforced, distractor identity recorded
- **CV**: 5-fold × 5-repeat stratified ridge regression (matching original)
- **Ridge alphas**: Same alpha sweep as original

### BSA Baselines

For each noise level:

- **BSA CLS baseline**: Ridge classification on BSA features (128 channels × max_bins) with noise applied. Tells us how much the raw input representation degrades.

The reservoir's value is measured as accuracy above the corresponding BSA baseline at each noise level.

---

## Predictions and Success Criteria

### Primary Prediction: Structured Distractor Reveals Encoding Cost of SFA

At moderate-to-high SFA (inc ≈ 0.02–0.10) with structured distractor (power ≥ 0.5):

- **Encoding degrades**: All-bins CLS accuracy drops below the no-SFA baseline at the same noise level

**Success criterion**: At least one noise level where the SFA-vs-no-SFA encoding gap is significantly negative (p < 0.05). Combined with the existing WM sweep showing significant positive SFA benefit at the same inc values, this constitutes the encoding-memory dissociation.

### Secondary Prediction: Unstructured Noise Shows SFA-Mediated Sharpening

At moderate SFA (inc ≈ 0.01–0.05) with unstructured noise (ratio ≥ 0.5):

- **Encoding improves more than clean**: The SFA encoding benefit (vs. no-SFA at same noise level) is larger under noise than under clean conditions

**Success criterion**: Interaction between SFA and noise level — the SFA encoding benefit increases monotonically with noise level.

### Tertiary Prediction: Confusability Modulates Distractor Effect

For structured distractors, the encoding cost of SFA should be larger when the distractor digit is spectrally similar to the target. This is tested by analyzing accuracy separately by target-distractor pair confusability.

### Null Results and Interpretation

- If encoding doesn't degrade under structured distractors: SFA's temporal heterogeneity benefit outweighs the redundancy cost even with competing stimuli. The "free lunch" interpretation stands — SFA always helps or is neutral for encoding.

- If both noise types show the same pattern: The structured vs. unstructured distinction doesn't matter, and the effect is driven by SNR rather than interference structure.

---

## Relationship to Paper Narrative

This experiment transforms the paper's central claim from "SFA creates distinct computational modes" (weakly supported by +1.5pp CLS) to "SFA navigates a tradeoff between encoding precision and memory persistence, and this tradeoff becomes visible under ecologically realistic conditions of input uncertainty."

The clean-input results become the baseline: under ideal conditions, SFA adds memory without cost. The noisy-input results reveal the underlying tradeoff that clean conditions mask. This is biologically grounded — the nervous system operates under noise, and neuromodulatory systems that control SFA are deployed precisely during uncertain, demanding conditions.

The structured vs. unstructured noise comparison provides a mechanistic dissection: structured interference reveals the cost of SFA (temporal redundancy loss), while unstructured noise reveals the benefit (gain normalization). Two sides of the same biophysical mechanism, producing opposite encoding effects depending on the nature of the challenge.