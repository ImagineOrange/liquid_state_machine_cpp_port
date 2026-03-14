# Mechanistic interpretability of the adaptation-mediated working memory trace

## How adaptation conductance implements working memory in a spiking reservoir

---

This analysis probes the internal mechanism by which spike-frequency adaptation (SFA) enables working memory (WM) in a 604-neuron LIF reservoir. It complements the behavioral-level findings in [Z_CLS_WM_XOR_README](../../Z_CLS_WM_XOR_README), which established that CLS, WM, and XOR tasks require distinct adaptation regimes and identified a Pareto frontier between CLS and WM. Here we open the black box and trace exactly how the adaptation state carries digit-A identity through the gap and into the B+Post readout epoch.

All results are from 4 diagnostic grid points at tau=5000 ms, Branch B (rate-matched at 20 Hz), using 100 A+gap+B compound trials with 5 digit-A classes:

| Label | adapt_inc | g_tonic (nS) | WM accuracy | Adapt-state WM |
|-------|-----------|-------------|-------------|----------------|
| no_adapt | 0.000 | 4.375 | 26.9% (chance) | 20.0% (chance) |
| cls_optimum | 0.071 | 1.875 | 75.6% | 98.2% |
| perbin_optimum | 0.127 | 1.250 | 80.3% | 98.3% |
| wm_optimum | 0.171 | 0.625 | 81.3% | 97.9% |

---

## 1. Adaptation conductance is the memory (Fig. 1)

The 604-dimensional adaptation conductance vector at B onset is a near-perfect predictor of digit-A identity (97.9--98.3% ridge accuracy across all adaptation levels), vastly exceeding the spike-count readout (75.6--81.3%). The spike-count classifier recovers only a fraction of the information stored in the conductance state.

**Link to the main analysis:** Z_CLS_WM_XOR_README Section 5 identified the adapted-steady-state regime (ISI CV < 2.0, inc > 0.15) as the one where "information is carried by the *level* of the adapted state rather than in temporal patterning." The mechanistic interp confirms this directly: the adaptation conductance *is* the memory variable, and the spike-count readout is an imperfect decoder of it.

The stimulus-to-adaptation correlation (Pearson r between A-epoch spike counts and adaptation state at B onset) ranges from r=0.81 to r=0.88 across the three adapted conditions, with per-digit values ranging from 0.67 to 0.96. This quantifies the first link in the memory chain: stimulus drive -> spike accumulation -> adaptation buildup.

## 2. Ridge weight structure: the classifier reads adaptation through B-epoch spikes (Fig. 2)

The ridge classifier concentrates 96--98% of its absolute weight energy on B-epoch bins (B1--B5), with only 2--4% on post-stimulus bins. This holds across all four grid points.

**Link to the main analysis:** Z_CLS_WM_XOR_README Section 4.6 showed that at the WM optimum, per-bin accuracy in the B epoch reaches 73.7% while late bins reach 43.7%. The mechanistic interp shows *why*: the classifier overwhelmingly reads B-epoch activity because that is where the adaptation-modulated rate signal is strongest and most diverse.

Decomposing ridge weight by input-weight quartile reveals that adaptation equalizes weight allocation. Without adaptation, Q4 (high-input) neurons carry ~36% of ridge weight while Q1 carries ~16%. At the WM optimum, all quartiles converge toward 25% (the uniform expectation). Adaptation recruits the full population into the readout.

## 3. Population equalization: adaptation compresses firing rate inequality (Fig. 3)

Neurons binned by their total input connection weight from the input layer (computed from the 728x728 weight matrix in `network_snapshot.npz`) show dramatic convergence in firing rates with increasing adaptation:

- Q4 (strongest input) mean rate: 29.5 -> 25.6 spk/ms (suppressed)
- Q1 (weakest input) mean rate: 4.6 -> 5.8 spk/ms (recruited)
- Q2 rises from 4.4 -> 10.8 spk/ms (2.5x increase)

The discrimination index (CV of per-digit rates) compresses across all quartiles but remains nonzero, meaning neurons become more similar in mean rate while retaining digit selectivity.

**Link to the main analysis:** Z_CLS_WM_XOR_README Section 6 noted that participation ratio increases monotonically with adaptation but adds no independent explanatory power. The mechanistic interp reveals *what* participation ratio is measuring: adaptation's compression of the rate distribution brings more neurons above the activity threshold where they carry discriminative information. PR is a downstream readout of this equalization process.

## 4. Representation structure: low-dimensional, high-fidelity adaptation code (Fig. 4)

PCA on the 5-class mean adaptation vectors (5 digits x 604 neurons) shows that 2 PCs capture >95% of variance at all adapted conditions. The adaptation representation of digit identity is low-dimensional but high-fidelity -- perfectly separable by a linear classifier.

The selectivity correlation matrix at the WM optimum shows between-digit correlations of r=0.94--0.97. The digit representations are highly similar in their overall pattern but reliably distinguishable. This is consistent with the adapted-steady-state coding scheme: all neurons participate in a shared activity mode, with digit identity encoded in small but systematic deviations from the population mean.

## 5. The mode switch: from concentrated rate code to distributed conductance code (Fig. 5)

The Lorenz curves of ridge weight across neurons reveal a fundamental shift in coding strategy:

| Condition | Weight Gini | Input wt / ridge wt corr | WM accuracy |
|-----------|-------------|---------------------------|-------------|
| no_adapt | 0.52 | +0.60 | 26.9% |
| cls_optimum | 0.42 | +0.42 | 75.6% |
| perbin_optimum | 0.35 | +0.26 | 80.3% |
| wm_optimum | 0.31 | +0.11 | 81.3% |

Without adaptation, the classifier concentrates on a few strongly-driven neurons (Gini=0.52, high correlation between input weight and ridge weight). With adaptation, weight distributes uniformly across the population (Gini=0.31, correlation collapses to 0.11). This is the mode switch: a concentrated rate code where a minority of neurons dominate the readout gives way to a distributed conductance code where all neurons contribute equally.

**Link to the main analysis:** Z_CLS_WM_XOR_README Discussion identified two computational modes separated by the ISI CV boundary at ~2.1. The burst-pause regime (ISI CV > 2.1) produces "high temporal diversity and rapid information decay." The adapted-steady-state regime (ISI CV < 2.0) produces "persistent activity and reduced temporal variability." The mechanistic interp adds a population-level dimension to this: the mode switch is not only temporal (transient vs persistent) but also spatial (concentrated vs distributed). The burst-pause regime relies on a few strongly-driven neurons to create stimulus-specific temporal patterns. The adapted-steady-state regime stamps a persistent, distributed pattern across the entire population.

The readout bottleneck quantifies the cost of this transition. The adaptation conductance stores digit-A identity at 97.9--98.3% accuracy regardless of adaptation strength, but the spike-count readout recovers only 26.9--81.3%. The gap between stored and recovered information narrows with adaptation because the distributed code provides more independent spike-count samples of the underlying conductance state.

## 6. Raster visualization: the mode switch in single-trial activity (Fig. 6)

Side-by-side spike rasters from a single trial (digit A=0, digit B=3) at the no-adapt and WM-optimum conditions, with neurons sorted by average spike count in shared order. Three subpanels per condition:

- **Raster**: Event plot showing individual spikes with epoch shading (A, Gap, B, Post)
- **PSTH**: Gaussian-smoothed (sigma=10 ms) population firing rate over time
- **Ridge weight profile**: Binned mean classifier weight per neuron group (high-activity on left)

The no-adapt raster shows activity concentrated in a band of ~150 high-firing neurons, with the majority of the population nearly silent. The ridge weight profile confirms: the classifier reads almost exclusively from the high-activity neurons. The WM-optimum raster shows activity distributed across the full population, with the ridge weight profile flat.

The PSTH reveals a key temporal difference: the no-adapt condition shows sharp rate peaks during A and B stimuli that drop to baseline during the gap. The WM-optimum condition shows modulated rate peaks with persistent, adaptation-shaped dynamics -- the rate doesn't fully recover between stimuli because the adaptation conductance accumulated during A suppresses subsequent activity in a digit-dependent way.

## 7. Calibration control: redistribution is adaptation-driven, not a rate-matching artifact (Fig. 7)

A critical concern is whether the population equalization is an artifact of the tonic conductance calibration that rate-matches all conditions to 20 Hz. Five lines of evidence rule this out:

**A. Tonic conductance is uniform across neurons.** The calibration applies the same conductance to all 604 reservoir neurons. A uniform additive current cannot change the *relative* firing rate distribution -- it shifts the mean without reshaping the distribution.

**B. Tonic conductance decreases with adaptation** (4.375 nS at no_adapt -> 0.625 nS at WM optimum). More adaptation means *less* calibration drive is needed. If calibration were causing equalization, the effect should be strongest where g_tonic is largest (no_adapt), not where it is smallest (WM optimum). The data shows the opposite.

**C. The firing rate Gini coefficient drops monotonically** (0.74 -> 0.47). This measures inequality in the rate distribution itself, not in classifier weights. Tonic conductance cannot produce this compression.

**D. The top-10% spike share drops from 58.3% to 31.8%.** The dominant neurons contribute a progressively smaller fraction of total network activity. This is the hallmark of activity-dependent negative feedback (high-firing neurons accumulate more adaptation, which suppresses them), not of uniform boosting.

**E. The Q4/Q1 rate dominance ratio collapses from 6.4x to 4.4x.** Quartiles defined by input connection weight show convergence: Q4 (strongest input) is suppressed while Q1-Q3 are recruited. A uniform conductance shift cannot produce differential effects across quartiles defined by a structural property (input weight).

The mechanistic chain is: strong input -> high firing -> large adaptation accumulation -> rate suppression -> weakly-driven neurons fire relatively more -> activity distributes across population -> ridge classifier uses the whole population -> better WM accuracy.

---

## Relationship to the main analysis

The mechanistic interp results fill in the "how" behind the "what" established in [Z_CLS_WM_XOR_README](../../Z_CLS_WM_XOR_README):

| Main analysis finding | Mechanistic interp resolution |
|----------------------|------------------------------|
| WM requires inc > 0.15 while CLS peaks at inc ~ 0.07 (Section 2) | WM reads a persistent adaptation-conductance trace that only forms at sufficient adaptation strength. CLS reads temporal spike patterns that adaptation homogenizes. |
| The Pareto frontier between CLS and WM is mechanistically non-trivial (Section 3) | CLS relies on a concentrated rate code (few high-activity neurons, diverse temporal patterns). WM relies on a distributed conductance code (all neurons contributing, persistent patterns). These are incompatible readout strategies for the same population. |
| ISI CV boundary at ~2.1 separates burst-pause and adapted-steady-state regimes (Section 5) | The population-level mode switch (Gini collapse, Q4/Q1 convergence) maps onto this same boundary. Below the boundary, a few strongly-driven neurons create diverse temporal patterns. Above it, adaptation compresses the population into a distributed steady state. |
| PR increases with adaptation but adds no explanatory power (Section 6) | PR measures the consequence of rate equalization: more neurons participating in activity. It is a readout of the adaptation-driven redistribution, not an independent variable. |
| The CLS classifier exploits temporal diversity that adaptation destroys (Section 4) | The mechanistic reason: adaptation distributes activity uniformly, making all time bins carry redundant copies of the same conductance-state signal. CLS needs diverse bins; WM needs persistent ones. |
| Late-bin accuracy rises with adaptation but hurts CLS (Section 4.5) | The late bins carry adaptation-state information (97.9% decodable from conductance). This helps WM but adds redundant features for CLS. |

---

## Reproduction

All figures are generated by a single script:

```bash
# Generate figures 1-5, 7 (from mechanistic_interp.json + network_snapshot.npz)
python3 results/mechanistic_interp/plot_mechanistic_interp.py

# Figure 6 additionally requires example_rasters.json, generated by:
cd src && make && ./cls_sweep --mech-raster --data-dir ../data --output-dir ../results/mechanistic_interp
```

### Data files

| File | Source | Contents |
|------|--------|----------|
| `mechanistic_interp.json` | `./cls_sweep --mech-interp` | Per-grid-point accuracies, ridge weights, selectivity vectors, adaptation states, calibration info |
| `example_rasters.json` | `./cls_sweep --mech-raster` | Single-trial spike times and neuron positions for no_adapt and wm_optimum |
| `../network_snapshot/network_snapshot.npz` | `./cls_sweep --snapshot` | 728x728 weight matrix, input/reservoir neuron indices |

### Figures

| Figure | Filename | Content |
|--------|----------|---------|
| Fig. 1 | `fig1_headline.png` | Adaptation-state vs spike-count WM accuracy; stimulus-adaptation correlation |
| Fig. 2 | `fig2_weight_structure.png` | Temporal weight distribution; B vs Post epoch share; quartile weight allocation |
| Fig. 3 | `fig3_population_equalization.png` | Quartile firing rates, discrimination index, between-class correlation |
| Fig. 4 | `fig4_representation_structure.png` | PCA on adaptation vectors; digit selectivity correlation matrix |
| Fig. 5 | `fig5_mode_switch.png` | Lorenz curves; readout bottleneck; decile weight/selectivity profiles; mode switch metrics |
| Fig. 6 | `fig6_rasters.png` | Side-by-side rasters with PSTH and ridge weight profiles |
| Fig. 7 | `fig7_calibration_control.png` | Calibration control: g_tonic, rate Gini, top-10% share, quartile convergence, Q4/Q1 ratio |
