#include "experiments.h"
#include <atomic>

// ============================================================
// EXTENDED TAU SWEEP + VARIABLE GAP LENGTH SWEEP
//
// Two experiments addressing reviewer objections:
//
// 1. Extended tau (--extended-tau): Extends adapt_tau from 5000
//    to ~45000 ms to test whether the triple dissociation
//    (CLS/WM/XOR) is 1D (inc-only, shared tau preference) or
//    2D (tasks separate along both inc and tau). Runs all three
//    tasks at each grid point with both branches.
//
// 2. Variable gap (--gap-sweep): Sweeps the silent gap between
//    digit A and digit B from 50 to 1000 ms to show the WM
//    optimum migrates through (inc, tau) space as a function
//    of memory demand. Branch B only.
// ============================================================

// ============================================================
// GRID AXES
// ============================================================

// Same inc axis as existing sweeps
static const std::vector<double> EXT_INC = {
    0.0,
    0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000,
};

// Extended tau: 5000 (overlap with existing) + 6 new values
// continuing the ~1.44x log spacing
static const std::vector<double> EXT_TAU = {
    5000.0, 7200.0, 10380.0, 14960.0, 21550.0, 31080.0, 44790.0,
};

// Gap sweep tau: finer grid spanning catastrophic-decay to ceiling
// 12 values log-spaced from 1000 to 15000 ms
// Covers 37% to 94% trace retention at gap=1000ms
static const std::vector<double> GAP_SWEEP_TAUS = {
    1000.0, 1300.0, 1650.0, 2100.0, 2700.0,
    3469.5, 4400.0, 5600.0, 7200.0, 9200.0,
    10380.0, 15000.0,
};

// Gap lengths to sweep (added 200ms to pinpoint ANOVA transition)
static const std::vector<double> GAP_LENGTHS_MS = {
    50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0,
};

// ============================================================
// SHARED CONSTANTS
// ============================================================
static constexpr int N_BINS_A = 5;
static constexpr int N_BINS_GAP = 1;
static constexpr int N_BINS_B = 5;
static constexpr int N_BINS_POST = 2;
static constexpr int N_EPOCH_BINS = N_BINS_A + N_BINS_GAP + N_BINS_B + N_BINS_POST;  // 13
static constexpr int IDX_B_START = N_BINS_A + N_BINS_GAP;  // 6
static constexpr int IDX_BPOST_END = N_EPOCH_BINS;          // 13
static constexpr int N_BPOST_BINS = IDX_BPOST_END - IDX_B_START;  // 7

static constexpr int N_WM_PAIRS = 2500;
static constexpr int PAIR_SEED = 99;
static constexpr int CAL_N_PAIRS = 200;
static constexpr int CAL_N_SAMPLES = 200;

static constexpr double G_TONIC_HI = 5.0;

// ============================================================
// GRID POINT
// ============================================================
struct ExtGridPoint {
    double adapt_inc, adapt_tau;
    std::string point_id;
    int inc_idx, tau_idx;
};

static std::vector<ExtGridPoint> build_ext_grid(const std::vector<double>& inc_axis,
                                                 const std::vector<double>& tau_axis) {
    std::vector<ExtGridPoint> pts;
    pts.reserve(inc_axis.size() * tau_axis.size());
    for (int ii = 0; ii < (int)inc_axis.size(); ii++) {
        for (int ti = 0; ti < (int)tau_axis.size(); ti++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "inc%.4f_tau%.1f", inc_axis[ii], tau_axis[ti]);
            pts.push_back({inc_axis[ii], tau_axis[ti], buf, ii, ti});
        }
    }
    return pts;
}

// ============================================================
// XOR PAIR — with parameterized gap
// ============================================================
struct XorPairExt {
    AudioSample compound;
    int digit_a, digit_b;
    int label;                  // 1 = same, 0 = different
    std::string speaker_a, speaker_b;
    double stim_a_end_ms;
    double stim_b_offset_ms;
    double stim_b_end_ms;
    double total_stim_end_ms;
};

static std::vector<XorPairExt> generate_pairs_with_gap(
    const std::vector<AudioSample>& samples,
    int n_pairs, int seed, double gap_ms)
{
    std::map<int, std::vector<int>> by_digit;
    for (int i = 0; i < (int)samples.size(); i++)
        by_digit[samples[i].digit].push_back(i);

    std::vector<int> digits;
    for (auto& [d, _] : by_digit) digits.push_back(d);
    std::sort(digits.begin(), digits.end());
    int n_digits = (int)digits.size();

    int n_same = n_pairs / 2;
    int n_diff = n_pairs - n_same;

    std::mt19937 rng(seed);
    std::vector<XorPairExt> pairs;
    pairs.reserve(n_pairs);

    // Same-digit pairs
    for (int p = 0; p < n_same; p++) {
        int d = digits[p % n_digits];
        auto& pool = by_digit[d];
        std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
        int ia = pool[dist(rng)], ib = pool[dist(rng)];
        const auto& sa = samples[ia];
        const auto& sb = samples[ib];

        double a_end = *std::max_element(sa.spike_times_ms.begin(), sa.spike_times_ms.end());
        double b_offset = a_end + gap_ms;

        AudioSample compound;
        compound.digit = sa.digit;
        compound.speaker = sa.speaker;
        compound.filename = sa.filename + "+" + sb.filename;
        compound.spike_times_ms = sa.spike_times_ms;
        compound.freq_bin_indices = sa.freq_bin_indices;
        for (size_t k = 0; k < sb.spike_times_ms.size(); k++) {
            compound.spike_times_ms.push_back(sb.spike_times_ms[k] + b_offset);
            compound.freq_bin_indices.push_back(sb.freq_bin_indices[k]);
        }
        double b_end_abs = b_offset + *std::max_element(sb.spike_times_ms.begin(),
                                                          sb.spike_times_ms.end());
        pairs.push_back({std::move(compound), sa.digit, sb.digit, 1,
                          sa.speaker, sb.speaker, a_end, b_offset, b_end_abs, b_end_abs});
    }

    // Different-digit pairs
    for (int p = 0; p < n_diff; p++) {
        int di_a = p % n_digits;
        int di_b = (di_a + 1 + (p / n_digits) % (n_digits - 1)) % n_digits;
        if (di_b == di_a) di_b = (di_a + 1) % n_digits;
        int da = digits[di_a], db = digits[di_b];

        auto& pool_a = by_digit[da];
        auto& pool_b = by_digit[db];
        std::uniform_int_distribution<int> dist_a(0, (int)pool_a.size() - 1);
        std::uniform_int_distribution<int> dist_b(0, (int)pool_b.size() - 1);
        int ia = pool_a[dist_a(rng)], ib = pool_b[dist_b(rng)];
        const auto& sa = samples[ia];
        const auto& sb = samples[ib];

        double a_end = *std::max_element(sa.spike_times_ms.begin(), sa.spike_times_ms.end());
        double b_offset = a_end + gap_ms;

        AudioSample compound;
        compound.digit = sa.digit;
        compound.speaker = sa.speaker;
        compound.filename = sa.filename + "+" + sb.filename;
        compound.spike_times_ms = sa.spike_times_ms;
        compound.freq_bin_indices = sa.freq_bin_indices;
        for (size_t k = 0; k < sb.spike_times_ms.size(); k++) {
            compound.spike_times_ms.push_back(sb.spike_times_ms[k] + b_offset);
            compound.freq_bin_indices.push_back(sb.freq_bin_indices[k]);
        }
        double b_end_abs = b_offset + *std::max_element(sb.spike_times_ms.begin(),
                                                          sb.spike_times_ms.end());
        pairs.push_back({std::move(compound), sa.digit, sb.digit, 0,
                          sa.speaker, sb.speaker, a_end, b_offset, b_end_abs, b_end_abs});
    }

    std::shuffle(pairs.begin(), pairs.end(), rng);
    return pairs;
}

// ============================================================
// EPOCH BOUNDS
// ============================================================
struct EpochBoundsExt {
    double start_ms[N_EPOCH_BINS];
    double end_ms[N_EPOCH_BINS];
    double dur_ms[N_EPOCH_BINS];
};

static EpochBoundsExt compute_epoch_bounds(const XorPairExt& pair) {
    EpochBoundsExt eb;
    int b = 0;
    double a_bin = pair.stim_a_end_ms / N_BINS_A;
    for (int i = 0; i < N_BINS_A; i++, b++) {
        eb.start_ms[b] = i * a_bin;
        eb.end_ms[b] = (i + 1) * a_bin;
    }
    eb.start_ms[b] = pair.stim_a_end_ms;
    eb.end_ms[b] = pair.stim_b_offset_ms;
    b++;
    double b_dur = pair.stim_b_end_ms - pair.stim_b_offset_ms;
    double b_bin = b_dur / N_BINS_B;
    for (int i = 0; i < N_BINS_B; i++, b++) {
        eb.start_ms[b] = pair.stim_b_offset_ms + i * b_bin;
        eb.end_ms[b] = pair.stim_b_offset_ms + (i + 1) * b_bin;
    }
    double post_bin = POST_STIM_MS / N_BINS_POST;
    for (int i = 0; i < N_BINS_POST; i++, b++) {
        eb.start_ms[b] = pair.stim_b_end_ms + i * post_bin;
        eb.end_ms[b] = pair.stim_b_end_ms + (i + 1) * post_bin;
    }
    for (int i = 0; i < N_EPOCH_BINS; i++)
        eb.dur_ms[i] = eb.end_ms[i] - eb.start_ms[i];
    return eb;
}

// ============================================================
// EPOCH BINNING — rate-normalized
// ============================================================
static Mat bin_activity_epochs(
    const std::vector<std::vector<int>>& activity_record,
    const std::vector<int>& neuron_indices,
    double dt, const EpochBoundsExt& eb)
{
    int n_ch = (int)neuron_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_ch; p++) idx_to_pos[neuron_indices[p]] = p;

    Mat bins(N_EPOCH_BINS, n_ch, 0.0);
    int n_steps = (int)activity_record.size();

    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        int b = -1;
        for (int e = 0; e < N_EPOCH_BINS; e++) {
            if (t >= eb.start_ms[e] && t < eb.end_ms[e]) { b = e; break; }
        }
        if (b < 0) {
            if (t >= eb.start_ms[N_EPOCH_BINS - 1]) b = N_EPOCH_BINS - 1;
            else continue;
        }
        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end())
                bins(b, it->second) += 1.0;
        }
    }
    for (int e = 0; e < N_EPOCH_BINS; e++) {
        if (eb.dur_ms[e] > 1e-6) {
            for (int c = 0; c < n_ch; c++)
                bins(e, c) /= eb.dur_ms[e];
        }
    }
    return bins;
}

// ============================================================
// CLS ACTIVITY BINNING — fixed 20ms bins
// ============================================================
static Mat bin_activity_fixed(
    const std::vector<std::vector<int>>& activity_record,
    const std::vector<int>& reservoir_indices,
    double dt, double bin_ms, int n_bins)
{
    int n_res = (int)reservoir_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_res; p++) idx_to_pos[reservoir_indices[p]] = p;

    Mat bins(n_bins, n_res, 0.0);
    int n_steps = (int)activity_record.size();
    for (int step = 0; step < n_steps; step++) {
        int b = (int)(step * dt / bin_ms);
        if (b < 0 || b >= n_bins) continue;
        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end())
                bins(b, it->second) += 1.0;
        }
    }
    // Rate-normalize
    for (int b = 0; b < n_bins; b++)
        for (int c = 0; c < n_res; c++)
            bins(b, c) /= bin_ms;
    return bins;
}

// ============================================================
// TONIC CONDUCTANCE HELPERS
// ============================================================
static void set_tonic(SphericalNetwork& net, const ZoneInfo& zi,
                       double g, double rev) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zi.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g;
        net.tonic_reversal[nid] = rev;
    }
}

// ============================================================
// WM SIMULATION: run all pairs (OpenMP)
// ============================================================
struct WmResult {
    std::vector<Mat> epoch_bins_list;
    std::vector<int> digit_a_list, digit_b_list, label_list;
    std::vector<double> total_spikes, isi_cvs, adapt_b_onsets;
    int n_reservoir;
};

static WmResult run_wm_pairs(const NetworkConfig& cfg,
                               const std::vector<XorPairExt>& pairs,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& sim_cfg, int n_workers,
                               double g_tonic = 0.0, double tonic_rev = -80.0,
                               bool verbose = true,
                               double std_u_override = -1.0,
                               double nmda_ratio_override = -1.0,
                               int construction_seed = -1) {
    double eff_std_u = (std_u_override >= 0.0) ? std_u_override : STD_U;
    int n_pairs = (int)pairs.size();
    WmResult out;
    out.epoch_bins_list.resize(n_pairs);
    out.digit_a_list.resize(n_pairs);
    out.digit_b_list.resize(n_pairs);
    out.label_list.resize(n_pairs);
    out.total_spikes.resize(n_pairs);
    out.isi_cvs.resize(n_pairs);
    out.adapt_b_onsets.resize(n_pairs);
    out.n_reservoir = 0;

    std::atomic<int> completed(0);

    #pragma omp parallel num_threads(n_workers)
    {
        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
            apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        } else {
            // Seed RNG before network construction so each seed produces a unique topology
            if (construction_seed >= 0)
                rng_seed((uint64_t)construction_seed);
            else
                rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                         (uint64_t)getpid());
            NetworkConfig cfg_copy = cfg;
            build_full_network(net, zone_info, cfg_copy, sim_cfg.dt, true,
                              &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (nmda_ratio_override >= 0.0) net.nmda_ratio = nmda_ratio_override;
        if (g_tonic != 0.0) set_tonic(net, zone_info, g_tonic, tonic_rev);

        // Re-seed for simulation randomness (unique per thread)
        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid() + (construction_seed >= 0 ? (uint64_t)construction_seed * 10000 : 0));

        #pragma omp single
        { out.n_reservoir = (int)zone_info.reservoir_zone_indices.size(); }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_pairs; i++) {
            SimConfig pair_sim = sim_cfg;
            pair_sim.audio_duration_ms = pairs[i].total_stim_end_ms + 5.0;

            auto result = run_sample_with_std(net, pairs[i].compound, zone_info, pair_sim,
                                               eff_std_u, STD_TAU_REC,
                                               masks, {pairs[i].stim_b_offset_ms});

            EpochBoundsExt eb = compute_epoch_bounds(pairs[i]);
            auto bins = bin_activity_epochs(result.activity_record,
                                             zone_info.reservoir_zone_indices,
                                             sim_cfg.dt, eb);

            // ISI CV during B epoch
            int b_start_step = (int)(eb.start_ms[IDX_B_START] / sim_cfg.dt);
            int b_end_step = std::min((int)(eb.end_ms[IDX_B_START + N_BINS_B - 1] / sim_cfg.dt),
                                       (int)result.activity_record.size());
            std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                                   zone_info.reservoir_zone_indices.end());
            std::unordered_map<int, double> last_spike;
            std::vector<double> all_isis;
            for (int step = b_start_step; step < b_end_step; step++) {
                double t = step * sim_cfg.dt;
                for (int nid : result.activity_record[step]) {
                    if (res_set.count(nid)) {
                        auto it = last_spike.find(nid);
                        if (it != last_spike.end()) all_isis.push_back(t - it->second);
                        last_spike[nid] = t;
                    }
                }
            }
            double isi_cv = std::nan("");
            if (all_isis.size() >= 2) {
                double mn = 0;
                for (double v : all_isis) mn += v;
                mn /= all_isis.size();
                double var = 0;
                for (double v : all_isis) var += (v - mn) * (v - mn);
                var /= all_isis.size();
                if (mn > 1e-9) isi_cv = std::sqrt(var) / mn;
            }

            // Adaptation at B onset
            double adapt_mean = 0;
            if (!result.adapt_snapshots.empty() && !result.adapt_snapshots[0].empty()) {
                for (int nid : zone_info.reservoir_zone_indices)
                    adapt_mean += result.adapt_snapshots[0][nid];
                adapt_mean /= zone_info.reservoir_zone_indices.size();
            }

            // Count spikes
            int total = 0;
            for (auto& step_spikes : result.activity_record)
                for (int nid : step_spikes)
                    if (res_set.count(nid)) total++;

            out.epoch_bins_list[i] = std::move(bins);
            out.digit_a_list[i] = pairs[i].digit_a;
            out.digit_b_list[i] = pairs[i].digit_b;
            out.label_list[i] = pairs[i].label;
            out.total_spikes[i] = total;
            out.isi_cvs[i] = isi_cv;
            out.adapt_b_onsets[i] = adapt_mean;

            int c = ++completed;
            if (verbose && (c % 100 == 0 || c == n_pairs)) {
                #pragma omp critical
                printf("        %5d/%d pairs\n", c, n_pairs);
            }
        }
    }
    return out;
}

// ============================================================
// CLS SIMULATION: run all single samples (OpenMP)
// ============================================================
struct ClsResult {
    std::vector<Mat> bins_list;
    std::vector<int> digits;
    std::vector<double> total_spikes;
    int n_reservoir;
    int n_bins;
};

static ClsResult run_cls_samples(const NetworkConfig& cfg,
                                  const std::vector<AudioSample>& samples,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers,
                                  double g_tonic = 0.0, double tonic_rev = -80.0,
                                  bool verbose = true,
                                  double std_u_override = -1.0,
                                  double nmda_ratio_override = -1.0,
                                  int construction_seed = -1) {
    double eff_std_u = (std_u_override >= 0.0) ? std_u_override : STD_U;
    int n_samples = (int)samples.size();
    ClsResult out;
    out.bins_list.resize(n_samples);
    out.digits.resize(n_samples);
    out.total_spikes.resize(n_samples);
    out.n_reservoir = 0;
    out.n_bins = 0;

    std::atomic<int> completed(0);

    #pragma omp parallel num_threads(n_workers)
    {
        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
            apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        } else {
            if (construction_seed >= 0)
                rng_seed((uint64_t)construction_seed);
            else
                rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                         (uint64_t)getpid());
            NetworkConfig cfg_copy = cfg;
            build_full_network(net, zone_info, cfg_copy, sim_cfg.dt, true,
                              &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (nmda_ratio_override >= 0.0) net.nmda_ratio = nmda_ratio_override;
        if (g_tonic != 0.0) set_tonic(net, zone_info, g_tonic, tonic_rev);

        // Seed for simulation (unique per thread, incorporates construction_seed if set)
        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid() + (construction_seed >= 0 ? (uint64_t)construction_seed * 10000 : 0));

        int n_res = (int)zone_info.reservoir_zone_indices.size();
        #pragma omp single
        { out.n_reservoir = n_res; }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_samples; i++) {
            auto result = run_sample_with_std(net, samples[i], zone_info, sim_cfg,
                                               eff_std_u, STD_TAU_REC, masks, {});

            double total_dur = sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms;
            int n_bins = std::max(1, (int)(total_dur / BIN_MS));
            auto bins = bin_activity_fixed(result.activity_record,
                                            zone_info.reservoir_zone_indices,
                                            sim_cfg.dt, BIN_MS, n_bins);

            // Count reservoir spikes
            std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                                   zone_info.reservoir_zone_indices.end());
            int total = 0;
            for (auto& step_spikes : result.activity_record)
                for (int nid : step_spikes)
                    if (res_set.count(nid)) total++;

            out.bins_list[i] = std::move(bins);
            out.digits[i] = samples[i].digit;
            out.total_spikes[i] = total;

            #pragma omp critical
            { if (n_bins > out.n_bins) out.n_bins = n_bins; }

            int c = ++completed;
            if (verbose && (c % 100 == 0 || c == n_samples)) {
                #pragma omp critical
                printf("        %5d/%d samples\n", c, n_samples);
            }
        }
    }

    // Ensure n_bins is set correctly (max across samples)
    for (auto& b : out.bins_list)
        if (b.rows > out.n_bins) out.n_bins = b.rows;

    return out;
}

// ============================================================
// RATE MEASUREMENT
// ============================================================
static double measure_rate_wm(const NetworkConfig& cfg,
                                const std::vector<XorPairExt>& pairs,
                                const DynamicalOverrides& dyn_ovr,
                                const SimConfig& sim_cfg, int n_workers,
                                double g_tonic = 0.0, double rev = -80.0,
                                int construction_seed = -1) {
    auto res = run_wm_pairs(cfg, pairs, dyn_ovr, sim_cfg, n_workers, g_tonic, rev, false,
                             -1.0, -1.0, construction_seed);
    double sum = 0;
    for (int i = 0; i < (int)pairs.size(); i++) {
        double dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * dur_s);
    }
    return sum / pairs.size();
}

static double measure_rate_cls(const NetworkConfig& cfg,
                                const std::vector<AudioSample>& samples,
                                const DynamicalOverrides& dyn_ovr,
                                const SimConfig& sim_cfg, int n_workers,
                                double g_tonic = 0.0, double rev = -80.0) {
    auto res = run_cls_samples(cfg, samples, dyn_ovr, sim_cfg, n_workers, g_tonic, rev, false);
    double sum = 0;
    for (int i = 0; i < (int)samples.size(); i++) {
        double dur_s = (sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * dur_s);
    }
    return sum / samples.size();
}

// ============================================================
// TONIC CONDUCTANCE CALIBRATION
// ============================================================
struct CalResult {
    double g_tonic;
    double reversal;
    double rate_hz;
};

// Template-like calibration that works with either paradigm
static CalResult calibrate_tonic(
    const NetworkConfig& cfg, const DynamicalOverrides& dyn_ovr,
    const SimConfig& sim_cfg, int n_workers, double target_rate,
    // Function to measure rate at a given (g_tonic, reversal)
    std::function<double(double, double)> measure_fn,
    double initial_guess = 0.0, double prev_rev = -80.0)
{
    double natural_rate = measure_fn(0.0, 0.0);
    printf("    cal[0] g=0 (natural) -> %.1f Hz (target=%.1f)\n", natural_rate, target_rate);

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate};

    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    printf("    cal: natural=%.1f, target=%.1f -> %s\n",
           natural_rate, target_rate, reversal < -1 ? "inhibitory" : "excitatory");

    double lo = 0.0, hi = G_TONIC_HI;
    int iteration = 1;

    if (initial_guess > 0.0 && prev_rev == reversal) {
        double rate = measure_fn(initial_guess, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, initial_guess, rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, reversal, rate};
        bool overshot = (reversal < -1) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = initial_guess; else lo = initial_guess;
    }

    for (;; iteration++) {
        double mid = (lo + hi) / 2.0;
        double rate = measure_fn(mid, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, mid, rate);

        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {mid, reversal, rate};

        bool overshot = (reversal < -1) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = mid; else lo = mid;

        if (hi - lo < 1e-6) {
            hi = std::max(mid, 1.0) + mid;
            printf("    cal: interval collapsed, widening hi to %.4f\n", hi);
        }
    }
}

// ============================================================
// CLASSIFICATION — ridge CV (WM: 5-class from B+Post)
// ============================================================
struct ClassifyResult {
    double accuracy;
    double accuracy_std;
    std::vector<double> per_repeat;
};

static ClassifyResult classify_wm(const std::vector<Mat>& epoch_bins,
                                    const std::vector<int>& digit_a,
                                    const std::vector<int>& digit_b,
                                    int n_feat_per_bin) {
    // Different-digit pairs only
    std::vector<int> diff_idx;
    for (int i = 0; i < (int)digit_a.size(); i++)
        if (digit_a[i] != digit_b[i]) diff_idx.push_back(i);

    int n_diff = (int)diff_idx.size();
    int n_feat = N_BPOST_BINS * n_feat_per_bin;

    Mat X(n_diff, n_feat, 0.0);
    std::vector<int> y(n_diff);
    for (int di = 0; di < n_diff; di++) {
        int idx = diff_idx[di];
        y[di] = digit_a[idx];
        for (int b = 0; b < N_BPOST_BINS; b++) {
            int eb = IDX_B_START + b;
            if (eb < epoch_bins[idx].rows)
                for (int r = 0; r < std::min(n_feat_per_bin, epoch_bins[idx].cols); r++)
                    X(di, b * n_feat_per_bin + r) = epoch_bins[idx](eb, r);
        }
    }

    std::vector<double> rep_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double correct = 0; int total = 0;
        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& sp = folds[f];
            Mat Xtr((int)sp.train.size(), n_feat), Xte((int)sp.test.size(), n_feat);
            std::vector<int> ytr(sp.train.size()), yte(sp.test.size());
            for (int i = 0; i < (int)sp.train.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xtr(i, j) = X(sp.train[i], j);
                ytr[i] = y[sp.train[i]];
            }
            for (int i = 0; i < (int)sp.test.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xte(i, j) = X(sp.test[i], j);
                yte[i] = y[sp.test[i]];
            }
            cls::StandardScaler sc;
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte);
            cls::nan_to_num(Xtr); cls::nan_to_num(Xte);

            auto ctx = cls::ridge_fold_prepare(Xtr, ytr, Xte, yte, DEFAULT_DIGITS);
            double best_acc = -1; std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(ctx, Xte, yte, alpha);
                if (rr.accuracy > best_acc) { best_acc = rr.accuracy; best_preds = rr.predictions; }
            }
            for (int i = 0; i < (int)sp.test.size(); i++)
                if (best_preds[i] == y[sp.test[i]]) correct++;
            total += (int)sp.test.size();
        }
        rep_accs.push_back(correct / total);
    }
    double mean = 0; for (double a : rep_accs) mean += a; mean /= rep_accs.size();
    double var = 0; for (double a : rep_accs) var += (a - mean) * (a - mean); var /= rep_accs.size();
    return {mean, std::sqrt(var), rep_accs};
}

static ClassifyResult classify_xor(const std::vector<Mat>& epoch_bins,
                                     const std::vector<int>& labels,
                                     int n_feat_per_bin) {
    int n = (int)epoch_bins.size();
    int n_feat = N_BPOST_BINS * n_feat_per_bin;

    Mat X(n, n_feat, 0.0);
    std::vector<int> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = labels[i];
        for (int b = 0; b < N_BPOST_BINS; b++) {
            int eb = IDX_B_START + b;
            if (eb < epoch_bins[i].rows)
                for (int r = 0; r < std::min(n_feat_per_bin, epoch_bins[i].cols); r++)
                    X(i, b * n_feat_per_bin + r) = epoch_bins[i](eb, r);
        }
    }

    std::vector<int> xor_classes = {0, 1};
    std::vector<double> rep_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double correct = 0; int total = 0;
        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& sp = folds[f];
            Mat Xtr((int)sp.train.size(), n_feat), Xte((int)sp.test.size(), n_feat);
            std::vector<int> ytr(sp.train.size()), yte(sp.test.size());
            for (int i = 0; i < (int)sp.train.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xtr(i, j) = X(sp.train[i], j);
                ytr[i] = y[sp.train[i]];
            }
            for (int i = 0; i < (int)sp.test.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xte(i, j) = X(sp.test[i], j);
                yte[i] = y[sp.test[i]];
            }
            cls::StandardScaler sc;
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte);
            cls::nan_to_num(Xtr); cls::nan_to_num(Xte);

            auto ctx = cls::ridge_fold_prepare(Xtr, ytr, Xte, yte, xor_classes);
            double best_acc = -1; std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(ctx, Xte, yte, alpha);
                if (rr.accuracy > best_acc) { best_acc = rr.accuracy; best_preds = rr.predictions; }
            }
            for (int i = 0; i < (int)sp.test.size(); i++)
                if (best_preds[i] == y[sp.test[i]]) correct++;
            total += (int)sp.test.size();
        }
        rep_accs.push_back(correct / total);
    }
    double mean = 0; for (double a : rep_accs) mean += a; mean /= rep_accs.size();
    double var = 0; for (double a : rep_accs) var += (a - mean) * (a - mean); var /= rep_accs.size();
    return {mean, std::sqrt(var), rep_accs};
}

// CLS classification: concatenate all bins into flat feature vector
static ClassifyResult classify_cls(const std::vector<Mat>& bins_list,
                                     const std::vector<int>& digits,
                                     int n_reservoir, int n_bins) {
    int n = (int)bins_list.size();
    int n_feat = n_bins * n_reservoir;

    Mat X(n, n_feat, 0.0);
    std::vector<int> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = digits[i];
        for (int b = 0; b < std::min(n_bins, bins_list[i].rows); b++)
            for (int r = 0; r < std::min(n_reservoir, bins_list[i].cols); r++)
                X(i, b * n_reservoir + r) = bins_list[i](b, r);
    }

    std::vector<double> rep_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double correct = 0; int total = 0;
        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& sp = folds[f];
            Mat Xtr((int)sp.train.size(), n_feat), Xte((int)sp.test.size(), n_feat);
            std::vector<int> ytr(sp.train.size()), yte(sp.test.size());
            for (int i = 0; i < (int)sp.train.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xtr(i, j) = X(sp.train[i], j);
                ytr[i] = y[sp.train[i]];
            }
            for (int i = 0; i < (int)sp.test.size(); i++) {
                for (int j = 0; j < n_feat; j++) Xte(i, j) = X(sp.test[i], j);
                yte[i] = y[sp.test[i]];
            }
            cls::StandardScaler sc;
            Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte);
            cls::nan_to_num(Xtr); cls::nan_to_num(Xte);

            auto ctx = cls::ridge_fold_prepare(Xtr, ytr, Xte, yte, DEFAULT_DIGITS);
            double best_acc = -1; std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(ctx, Xte, yte, alpha);
                if (rr.accuracy > best_acc) { best_acc = rr.accuracy; best_preds = rr.predictions; }
            }
            for (int i = 0; i < (int)sp.test.size(); i++)
                if (best_preds[i] == y[sp.test[i]]) correct++;
            total += (int)sp.test.size();
        }
        rep_accs.push_back(correct / total);
    }
    double mean = 0; for (double a : rep_accs) mean += a; mean /= rep_accs.size();
    double var = 0; for (double a : rep_accs) var += (a - mean) * (a - mean); var /= rep_accs.size();
    return {mean, std::sqrt(var), rep_accs};
}

// ============================================================
// HELPERS: compute rate stats from simulation result
// ============================================================
static std::pair<double, double> compute_rate_stats(
    const std::vector<double>& total_spikes, int n_reservoir,
    const std::vector<XorPairExt>& pairs) {
    int n = (int)pairs.size();
    double mean = 0;
    for (int i = 0; i < n; i++) {
        double dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        mean += total_spikes[i] / (n_reservoir * dur_s);
    }
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; i++) {
        double dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        double r = total_spikes[i] / (n_reservoir * dur_s);
        var += (r - mean) * (r - mean);
    }
    return {mean, std::sqrt(var / n)};
}

static std::pair<double, double> compute_rate_stats_cls(
    const std::vector<double>& total_spikes, int n_reservoir,
    double total_dur_s) {
    int n = (int)total_spikes.size();
    double mean = 0;
    for (int i = 0; i < n; i++) mean += total_spikes[i] / (n_reservoir * total_dur_s);
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; i++) {
        double r = total_spikes[i] / (n_reservoir * total_dur_s);
        var += (r - mean) * (r - mean);
    }
    return {mean, std::sqrt(var / n)};
}

// ============================================================
// JSON HELPERS
// ============================================================
static void write_double_array(FILE* f, const std::vector<double>& v) {
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.10f", v[i]);
    }
    fprintf(f, "]");
}

// ============================================================
// EXTENDED TAU: CHECKPOINT
// ============================================================
static void ext_save_checkpoint(const std::string& dir,
                                  const std::vector<std::string>& results,
                                  const std::vector<double>& inc_axis,
                                  const std::vector<double>& tau_axis,
                                  double start_time, bool final_save) {
    std::string fname = final_save ? "extended_tau_sweep.json"
                                    : "extended_tau_sweep_checkpoint.json";
    std::string path = dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Extended Tau Sweep (CLS + WM + XOR)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", now_seconds() - start_time);
    fprintf(f, "  \"tasks\": [\"classification\", \"working_memory\", \"temporal_xor\"],\n");

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); write_double_array(f, inc_axis);
    fprintf(f, ",\n    \"unified_tau\": "); write_double_array(f, tau_axis);
    fprintf(f, ",\n    \"n_inc\": %d, \"n_tau\": %d\n", (int)inc_axis.size(), (int)tau_axis.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.1f, \"tolerance_hz\": %.1f},\n",
            RATE_TARGET_HZ, RATE_TOLERANCE_HZ);
    fprintf(f, "  \"wm_gap_ms\": 150.0,\n");

    fprintf(f, "  \"grid_results\": [\n");
    for (size_t i = 0; i < results.size(); i++) {
        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    %s", results[i].c_str());
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    if (final_save) printf("\n  Saved: %s\n", path.c_str());
}

// ============================================================
// GAP SWEEP: CHECKPOINT
// ============================================================
static void gap_save_checkpoint(const std::string& dir,
                                  const std::vector<std::string>& results,
                                  const std::vector<double>& inc_axis,
                                  const std::vector<double>& tau_axis,
                                  const std::vector<double>& gap_axis,
                                  double start_time, bool final_save) {
    std::string fname = final_save ? "gap_sweep.json" : "gap_sweep_checkpoint.json";
    std::string path = dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Variable Gap Length Sweep (WM + XOR)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", now_seconds() - start_time);
    fprintf(f, "  \"branch\": \"B_matched\",\n");

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); write_double_array(f, inc_axis);
    fprintf(f, ",\n    \"unified_tau\": "); write_double_array(f, tau_axis);
    fprintf(f, ",\n    \"gap_lengths_ms\": "); write_double_array(f, gap_axis);
    fprintf(f, ",\n    \"n_inc\": %d, \"n_tau\": %d, \"n_gaps\": %d\n",
            (int)inc_axis.size(), (int)tau_axis.size(), (int)gap_axis.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.1f, \"tolerance_hz\": %.1f},\n",
            RATE_TARGET_HZ, RATE_TOLERANCE_HZ);

    fprintf(f, "  \"grid_results\": [\n");
    for (size_t i = 0; i < results.size(); i++) {
        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    %s", results[i].c_str());
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    if (final_save) printf("\n  Saved: %s\n", path.c_str());
}

// ============================================================
// BUILD JSON STRING FOR ONE GRID RESULT
// ============================================================
static std::string build_result_json(
    const std::string& branch, const ExtGridPoint& pt,
    double stim_current, double g_tonic, double tonic_rev, double cal_rate,
    double cls_acc, double cls_std, const std::vector<double>& cls_reps,
    double wm_acc, double wm_std, const std::vector<double>& wm_reps,
    double xor_acc, double xor_std, const std::vector<double>& xor_reps,
    double rate_mean, double rate_std, int n_reservoir,
    double sim_time, double isi_cv, double adapt_b_onset,
    double gap_ms = 150.0)
{
    std::ostringstream o;
    o << std::fixed;
    o << "{";
    o << "\"branch\": \"" << branch << "\", ";
    o << "\"point_id\": \"" << pt.point_id << "\", ";
    o << "\"inc_idx\": " << pt.inc_idx << ", ";
    o << "\"tau_idx\": " << pt.tau_idx << ", ";
    o.precision(10);
    o << "\"adapt_inc\": " << pt.adapt_inc << ", ";
    o << "\"adapt_tau\": " << pt.adapt_tau << ", ";
    o << "\"gap_ms\": " << gap_ms << ", ";
    o << "\"stimulus_current\": " << stim_current << ", ";
    o << "\"tonic_conductance\": " << g_tonic << ", ";
    o << "\"tonic_reversal\": " << tonic_rev << ", ";
    o << "\"calibration_rate_hz\": " << cal_rate << ", ";

    o << "\"classification_accuracy\": " << cls_acc << ", ";
    o.precision(6);
    o << "\"classification_accuracy_std\": " << cls_std << ", ";
    o << "\"classification_per_repeat\": [";
    for (size_t i = 0; i < cls_reps.size(); i++) { if (i) o << ", "; o << cls_reps[i]; }
    o << "], ";

    o.precision(10);
    o << "\"wm_accuracy\": " << wm_acc << ", ";
    o.precision(6);
    o << "\"wm_accuracy_std\": " << wm_std << ", ";
    o << "\"wm_per_repeat\": [";
    for (size_t i = 0; i < wm_reps.size(); i++) { if (i) o << ", "; o << wm_reps[i]; }
    o << "], ";

    o.precision(10);
    o << "\"xor_accuracy\": " << xor_acc << ", ";
    o.precision(6);
    o << "\"xor_accuracy_std\": " << xor_std << ", ";
    o << "\"xor_per_repeat\": [";
    for (size_t i = 0; i < xor_reps.size(); i++) { if (i) o << ", "; o << xor_reps[i]; }
    o << "], ";

    o << "\"firing_rate_hz\": " << rate_mean << ", ";
    o << "\"firing_rate_std\": " << rate_std << ", ";
    o << "\"n_reservoir\": " << n_reservoir << ", ";
    o << "\"sim_time_s\": " << sim_time;

    if (!std::isnan(isi_cv)) o << ", \"isi_cv_mean\": " << isi_cv;
    else o << ", \"isi_cv_mean\": null";
    o << ", \"adapt_at_b_onset_mean\": " << adapt_b_onset;

    o << "}";
    return o.str();
}

// Simplified JSON for gap sweep (no CLS)
static std::string build_gap_result_json(
    const ExtGridPoint& pt, double gap_ms,
    double g_tonic, double tonic_rev, double cal_rate,
    double wm_acc, double wm_std, const std::vector<double>& wm_reps,
    double xor_acc, double xor_std, const std::vector<double>& xor_reps,
    double rate_mean, double rate_std, int n_reservoir,
    double sim_time, double isi_cv, double adapt_b_onset)
{
    std::ostringstream o;
    o << std::fixed;
    o << "{";
    o << "\"branch\": \"B_matched\", ";
    o << "\"point_id\": \"" << pt.point_id << "\", ";
    o << "\"inc_idx\": " << pt.inc_idx << ", ";
    o << "\"tau_idx\": " << pt.tau_idx << ", ";
    o.precision(10);
    o << "\"adapt_inc\": " << pt.adapt_inc << ", ";
    o << "\"adapt_tau\": " << pt.adapt_tau << ", ";
    o << "\"gap_ms\": " << gap_ms << ", ";
    o << "\"tonic_conductance\": " << g_tonic << ", ";
    o << "\"tonic_reversal\": " << tonic_rev << ", ";
    o << "\"calibration_rate_hz\": " << cal_rate << ", ";

    o << "\"wm_accuracy\": " << wm_acc << ", ";
    o.precision(6);
    o << "\"wm_accuracy_std\": " << wm_std << ", ";
    o << "\"wm_per_repeat\": [";
    for (size_t i = 0; i < wm_reps.size(); i++) { if (i) o << ", "; o << wm_reps[i]; }
    o << "], ";

    o.precision(10);
    o << "\"xor_accuracy\": " << xor_acc << ", ";
    o.precision(6);
    o << "\"xor_accuracy_std\": " << xor_std << ", ";
    o << "\"xor_per_repeat\": [";
    for (size_t i = 0; i < xor_reps.size(); i++) { if (i) o << ", "; o << xor_reps[i]; }
    o << "], ";

    o << "\"firing_rate_hz\": " << rate_mean << ", ";
    o << "\"firing_rate_std\": " << rate_std << ", ";
    o << "\"n_reservoir\": " << n_reservoir << ", ";
    o << "\"sim_time_s\": " << sim_time;

    if (!std::isnan(isi_cv)) o << ", \"isi_cv_mean\": " << isi_cv;
    else o << ", \"isi_cv_mean\": null";
    o << ", \"adapt_at_b_onset_mean\": " << adapt_b_onset;

    o << "}";
    return o.str();
}

// ============================================================
// ENTRY POINT: EXTENDED TAU SWEEP
// ============================================================
int run_extended_tau(int argc, char** argv,
                      int n_workers, const std::string& output_dir,
                      const std::string& data_dir) {
    auto grid = build_ext_grid(EXT_INC, EXT_TAU);
    int n_grid = (int)grid.size();
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  EXTENDED TAU SWEEP (CLS + WM + XOR)\n");
    printf("  Grid: %d inc x %d tau = %d points x 2 branches\n",
           (int)EXT_INC.size(), (int)EXT_TAU.size(), n_grid);
    printf("  Tau range: %.0f - %.0f ms\n", EXT_TAU.front(), EXT_TAU.back());
    printf("  Workers: %d\n", n_workers);
    if (!g_snapshot_path.empty()) printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // 2. Generate WM pairs (gap=150ms, matching existing sweep)
    printf("\n[2] Generating WM pairs (gap=150ms)...\n");
    auto pairs = generate_pairs_with_gap(samples, N_WM_PAIRS, PAIR_SEED, 150.0);
    int n_same = 0, n_diff = 0;
    for (auto& p : pairs) { if (p.label == 1) n_same++; else n_diff++; }
    printf("  %d pairs (%d same, %d different)\n", (int)pairs.size(), n_same, n_diff);

    // Max audio duration for CLS sim config
    double max_audio_ms = 0;
    for (auto& s : samples)
        if (!s.spike_times_ms.empty()) {
            double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
            if (mx > max_audio_ms) max_audio_ms = mx;
        }
    max_audio_ms += 5.0;

    // Base configs
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig cls_sim;
    cls_sim.dt = 0.1;
    cls_sim.audio_duration_ms = max_audio_ms;
    cls_sim.post_stimulus_ms = POST_STIM_MS;
    cls_sim.stimulus_current = INPUT_STIM_CURRENT;

    SimConfig wm_sim;
    wm_sim.dt = 0.1;
    wm_sim.audio_duration_ms = 0;  // set per pair
    wm_sim.post_stimulus_ms = POST_STIM_MS;
    wm_sim.stimulus_current = INPUT_STIM_CURRENT;

    // Find max pair duration for wm_sim
    double max_pair_ms = 0;
    for (auto& p : pairs)
        if (p.total_stim_end_ms > max_pair_ms) max_pair_ms = p.total_stim_end_ms;
    wm_sim.audio_duration_ms = max_pair_ms + 5.0;

    // Calibration subsets
    auto cls_cal_samples = std::vector<AudioSample>(
        samples.begin(), samples.begin() + std::min(CAL_N_SAMPLES, (int)samples.size()));
    auto wm_cal_pairs = std::vector<XorPairExt>(
        pairs.begin(), pairs.begin() + std::min(CAL_N_PAIRS, (int)pairs.size()));

    // 3. Sweep
    printf("\n[3] Running %d grid points x 2 branches...\n", n_grid);
    printf("    Branch A: Unmatched (natural rate)\n");
    printf("    Branch B: Tonic-conductance-matched (target=%.0f Hz)\n", RATE_TARGET_HZ);

    std::vector<std::string> results_json;
    std::vector<double> point_times;
    std::map<int, std::pair<double, double>> last_wm_cal;  // tau_idx -> (g, rev)

    for (int gi = 0; gi < n_grid; gi++) {
        auto& pt = grid[gi];
        double gp_start = now_seconds();

        std::string eta = "...";
        if (!point_times.empty()) {
            int last_n = std::min((int)point_times.size(), 10);
            double avg = 0;
            for (int i = (int)point_times.size() - last_n; i < (int)point_times.size(); i++)
                avg += point_times[i];
            avg /= last_n;
            char buf[64]; snprintf(buf, sizeof(buf), "%.1fh", avg * (n_grid - gi) / 3600.0);
            eta = buf;
        }

        printf("\n  [%d/%d] inc=%.4f, tau=%.1f  |  ETA: %s\n",
               gi + 1, n_grid, pt.adapt_inc, pt.adapt_tau, eta.c_str());

        DynamicalOverrides dyn;
        dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn.adapt_inc = pt.adapt_inc;
        dyn.adapt_tau = pt.adapt_tau;
        dyn.nmda_tau = FIXED_NMDA_TAU;

        for (const char* branch : {"A_unmatched", "B_matched"}) {
            bool is_b = (std::string(branch) == "B_matched");

            double cls_gt = 0, cls_rev = -80.0, cls_cal_rate = 0;
            double wm_gt = 0, wm_rev = -80.0, wm_cal_rate = 0;

            if (is_b) {
                // CLS calibration
                auto cls_measure = [&](double g, double r) -> double {
                    return measure_rate_cls(base_cfg, cls_cal_samples, dyn, cls_sim, n_workers, g, r);
                };
                printf("    [B] CLS calibration:\n");
                auto cls_cal = calibrate_tonic(base_cfg, dyn, cls_sim, n_workers,
                                                RATE_TARGET_HZ, cls_measure);
                cls_gt = cls_cal.g_tonic; cls_rev = cls_cal.reversal; cls_cal_rate = cls_cal.rate_hz;

                // WM calibration (warm-start from previous tau)
                double prev_gt = 0, prev_rev = -80.0;
                if (last_wm_cal.count(pt.tau_idx)) {
                    prev_gt = last_wm_cal[pt.tau_idx].first;
                    prev_rev = last_wm_cal[pt.tau_idx].second;
                }
                auto wm_measure = [&](double g, double r) -> double {
                    return measure_rate_wm(base_cfg, wm_cal_pairs, dyn, wm_sim, n_workers, g, r);
                };
                printf("    [B] WM calibration:\n");
                auto wm_cal = calibrate_tonic(base_cfg, dyn, wm_sim, n_workers,
                                               RATE_TARGET_HZ, wm_measure, prev_gt, prev_rev);
                wm_gt = wm_cal.g_tonic; wm_rev = wm_cal.reversal; wm_cal_rate = wm_cal.rate_hz;
                last_wm_cal[pt.tau_idx] = {wm_gt, wm_rev};
            } else {
                // Measure natural rates
                cls_cal_rate = measure_rate_cls(base_cfg, cls_cal_samples, dyn, cls_sim, n_workers);
                wm_cal_rate = measure_rate_wm(base_cfg, wm_cal_pairs, dyn, wm_sim, n_workers);
                printf("    [A] Natural rates: CLS=%.1f Hz, WM=%.1f Hz\n", cls_cal_rate, wm_cal_rate);
            }

            // --- Run CLS ---
            double t0 = now_seconds();
            auto cls_res = run_cls_samples(base_cfg, samples, dyn, cls_sim, n_workers,
                                            cls_gt, cls_rev, true);
            double cls_time = now_seconds() - t0;

            auto cls_class = classify_cls(cls_res.bins_list, cls_res.digits,
                                           cls_res.n_reservoir, cls_res.n_bins);

            // --- Run WM + XOR ---
            double t1 = now_seconds();
            auto wm_res = run_wm_pairs(base_cfg, pairs, dyn, wm_sim, n_workers,
                                         wm_gt, wm_rev, true);
            double wm_time = now_seconds() - t1;

            auto [wm_rate, wm_rate_std] = compute_rate_stats(wm_res.total_spikes,
                                                               wm_res.n_reservoir, pairs);

            auto wm_class = classify_wm(wm_res.epoch_bins_list, wm_res.digit_a_list,
                                          wm_res.digit_b_list, wm_res.n_reservoir);
            auto xor_class = classify_xor(wm_res.epoch_bins_list, wm_res.label_list,
                                            wm_res.n_reservoir);

            // ISI CV mean
            double isi_cv = 0; int isi_n = 0;
            for (auto v : wm_res.isi_cvs)
                if (!std::isnan(v)) { isi_cv += v; isi_n++; }
            if (isi_n > 0) isi_cv /= isi_n; else isi_cv = std::nan("");

            double adapt_mean = 0;
            for (auto v : wm_res.adapt_b_onsets) adapt_mean += v;
            adapt_mean /= wm_res.adapt_b_onsets.size();

            printf("    [%s] CLS: %.1f%% | WM: %.1f%% | XOR: %.1f%% | Rate: %.1f Hz\n",
                   branch, cls_class.accuracy * 100, wm_class.accuracy * 100,
                   xor_class.accuracy * 100, wm_rate);

            auto json = build_result_json(
                branch, pt, INPUT_STIM_CURRENT,
                is_b ? wm_gt : 0.0, is_b ? wm_rev : -80.0,
                is_b ? wm_cal_rate : wm_rate,
                cls_class.accuracy, cls_class.accuracy_std, cls_class.per_repeat,
                wm_class.accuracy, wm_class.accuracy_std, wm_class.per_repeat,
                xor_class.accuracy, xor_class.accuracy_std, xor_class.per_repeat,
                wm_rate, wm_rate_std, wm_res.n_reservoir,
                cls_time + wm_time, isi_cv, adapt_mean);
            results_json.push_back(json);
        }

        double gp_time = now_seconds() - gp_start;
        point_times.push_back(gp_time);
        printf("    %.1fmin | %d/%d | Elapsed: %.1fh\n",
               gp_time / 60.0, gi + 1, n_grid, (now_seconds() - total_start) / 3600.0);

        ext_save_checkpoint(output_dir, results_json, EXT_INC, EXT_TAU, total_start, false);
    }

    printf("\n======================================================================\n");
    printf("  EXTENDED TAU SWEEP COMPLETE\n");
    printf("  Total: %.1f hours | %d results\n",
           (now_seconds() - total_start) / 3600.0, (int)results_json.size());
    printf("======================================================================\n");

    ext_save_checkpoint(output_dir, results_json, EXT_INC, EXT_TAU, total_start, true);
    return 0;
}

// ============================================================
// GAP SWEEP: LOAD EXISTING CHECKPOINT FOR RESUME
// ============================================================
struct CompletedKey {
    int inc_idx, tau_idx;
    double gap_ms;
    bool operator<(const CompletedKey& o) const {
        if (inc_idx != o.inc_idx) return inc_idx < o.inc_idx;
        if (tau_idx != o.tau_idx) return tau_idx < o.tau_idx;
        return gap_ms < o.gap_ms;
    }
};

static std::pair<std::set<CompletedKey>, std::vector<std::string>>
load_gap_checkpoint(const std::string& dir) {
    std::set<CompletedKey> completed;
    std::vector<std::string> existing_json;

    // Try checkpoint first, then final
    for (auto& fname : {"gap_sweep_checkpoint.json", "gap_sweep.json"}) {
        std::string path = dir + "/" + fname;
        FILE* f = fopen(path.c_str(), "r");
        if (!f) continue;

        // Read entire file
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string content(sz, '\0');
        fread(&content[0], 1, sz, f);
        fclose(f);

        // Parse grid_results entries — extract inc_idx, tau_idx, gap_ms
        // Simple parsing: find each "inc_idx": N, "tau_idx": N, "gap_ms": N
        size_t pos = 0;
        while ((pos = content.find("\"inc_idx\":", pos)) != std::string::npos) {
            // Find the start of this JSON object (scan back for '{')
            size_t obj_start = content.rfind('{', pos);
            // Find the end of this JSON object
            size_t obj_end = content.find('}', pos);
            if (obj_start == std::string::npos || obj_end == std::string::npos) break;

            std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

            // Extract inc_idx
            auto get_int = [&](const std::string& s, const std::string& key) -> int {
                size_t p = s.find(key);
                if (p == std::string::npos) return -1;
                p += key.size();
                while (p < s.size() && (s[p] == ' ' || s[p] == ':')) p++;
                return std::atoi(s.c_str() + p);
            };
            auto get_dbl = [&](const std::string& s, const std::string& key) -> double {
                size_t p = s.find(key);
                if (p == std::string::npos) return -1;
                p += key.size();
                while (p < s.size() && (s[p] == ' ' || s[p] == ':')) p++;
                return std::atof(s.c_str() + p);
            };

            int ii = get_int(obj, "\"inc_idx\"");
            int ti = get_int(obj, "\"tau_idx\"");
            double gm = get_dbl(obj, "\"gap_ms\"");

            if (ii >= 0 && ti >= 0 && gm > 0) {
                completed.insert({ii, ti, gm});
                existing_json.push_back(obj);
            }

            pos = obj_end + 1;
        }

        if (!completed.empty()) {
            printf("  Loaded %d completed results from %s\n",
                   (int)completed.size(), fname);
            break;  // Use whichever file we found first
        }
    }
    return {completed, existing_json};
}

// ============================================================
// ENTRY POINT: VARIABLE GAP LENGTH SWEEP
// ============================================================
int run_gap_sweep(int argc, char** argv,
                   int n_workers, const std::string& output_dir,
                   const std::string& data_dir) {
    auto grid = build_ext_grid(EXT_INC, GAP_SWEEP_TAUS);
    int n_grid = (int)grid.size();
    int n_gaps = (int)GAP_LENGTHS_MS.size();
    int total_conditions = n_grid * n_gaps;
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  VARIABLE GAP LENGTH SWEEP (WM + XOR, Branch B only)\n");
    printf("  Grid: %d inc x %d tau x %d gaps = %d conditions\n",
           (int)EXT_INC.size(), (int)GAP_SWEEP_TAUS.size(), n_gaps, total_conditions);
    printf("  Tau values: ");
    for (double t : GAP_SWEEP_TAUS) printf("%.0f ", t);
    printf("\n  Gap lengths: ");
    for (double g : GAP_LENGTHS_MS) printf("%.0f ", g);
    printf("ms\n  Workers: %d\n", n_workers);
    if (!g_snapshot_path.empty()) printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig wm_sim;
    wm_sim.dt = 0.1;
    wm_sim.post_stimulus_ms = POST_STIM_MS;
    wm_sim.stimulus_current = INPUT_STIM_CURRENT;

    // 2. Load checkpoint for resume
    printf("\n[2] Checking for existing checkpoint...\n");
    auto [completed, existing_results] = load_gap_checkpoint(output_dir);
    int n_skipped = (int)completed.size();
    int n_remaining = total_conditions - n_skipped;
    printf("  %d/%d conditions already completed, %d remaining\n",
           n_skipped, total_conditions, n_remaining);

    // 3. Sweep: for each (inc, tau), calibrate once, then run all gap lengths
    printf("\n[3] Running remaining conditions (Branch B)...\n");

    std::vector<std::string> results_json = existing_results;
    std::vector<double> point_times;
    int condition_num = n_skipped;

    for (int gi = 0; gi < n_grid; gi++) {
        auto& pt = grid[gi];

        // Check if ALL gaps for this grid point are already done
        bool all_done = true;
        for (int gap_i = 0; gap_i < n_gaps; gap_i++) {
            if (completed.find({pt.inc_idx, pt.tau_idx, GAP_LENGTHS_MS[gap_i]}) == completed.end()) {
                all_done = false;
                break;
            }
        }
        if (all_done) continue;  // Skip entirely — no need to calibrate

        double gp_start = now_seconds();

        DynamicalOverrides dyn;
        dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn.adapt_inc = pt.adapt_inc;
        dyn.adapt_tau = pt.adapt_tau;
        dyn.nmda_tau = FIXED_NMDA_TAU;

        // Calibrate once using gap=150ms (the reference gap)
        auto ref_pairs = generate_pairs_with_gap(samples, CAL_N_PAIRS, PAIR_SEED + 1000, 150.0);

        double max_pair_ms = 0;
        for (auto& p : ref_pairs)
            if (p.total_stim_end_ms > max_pair_ms) max_pair_ms = p.total_stim_end_ms;
        SimConfig cal_sim = wm_sim;
        cal_sim.audio_duration_ms = max_pair_ms + 5.0;

        std::string eta = "...";
        if (!point_times.empty()) {
            int last_n = std::min((int)point_times.size(), 5);
            double avg = 0;
            for (int i = (int)point_times.size() - last_n; i < (int)point_times.size(); i++)
                avg += point_times[i];
            avg /= last_n;
            char buf[64]; snprintf(buf, sizeof(buf), "%.1fh", avg * (n_grid - gi) / 3600.0);
            eta = buf;
        }

        printf("\n  [%d/%d] inc=%.4f, tau=%.1f  |  ETA: %s\n",
               gi + 1, n_grid, pt.adapt_inc, pt.adapt_tau, eta.c_str());

        auto wm_measure = [&](double g, double r) -> double {
            return measure_rate_wm(base_cfg, ref_pairs, dyn, cal_sim, n_workers, g, r);
        };
        printf("    Calibrating (gap=150ms reference):\n");
        auto cal = calibrate_tonic(base_cfg, dyn, cal_sim, n_workers,
                                    RATE_TARGET_HZ, wm_measure);
        printf("    g_tonic=%.4f (%s) -> %.1f Hz\n",
               cal.g_tonic, cal.reversal < -1 ? "inh" : "exc", cal.rate_hz);

        // Run each gap length (skip already completed)
        for (int gap_i = 0; gap_i < n_gaps; gap_i++) {
            double gap_ms = GAP_LENGTHS_MS[gap_i];

            if (completed.find({pt.inc_idx, pt.tau_idx, gap_ms}) != completed.end()) {
                continue;  // Already in checkpoint
            }

            condition_num++;
            printf("    gap=%.0fms [%d/%d total]...\n", gap_ms,
                   condition_num, total_conditions);

            // Generate pairs with this gap
            auto pairs = generate_pairs_with_gap(samples, N_WM_PAIRS, PAIR_SEED, gap_ms);

            // Set sim duration for longest pair
            double max_ms = 0;
            for (auto& p : pairs)
                if (p.total_stim_end_ms > max_ms) max_ms = p.total_stim_end_ms;
            SimConfig gap_sim = wm_sim;
            gap_sim.audio_duration_ms = max_ms + 5.0;

            double t0 = now_seconds();
            auto res = run_wm_pairs(base_cfg, pairs, dyn, gap_sim, n_workers,
                                      cal.g_tonic, cal.reversal, true);
            double sim_time = now_seconds() - t0;

            auto [rate, rate_std] = compute_rate_stats(res.total_spikes, res.n_reservoir, pairs);

            auto wm_class = classify_wm(res.epoch_bins_list, res.digit_a_list,
                                          res.digit_b_list, res.n_reservoir);
            auto xor_class = classify_xor(res.epoch_bins_list, res.label_list,
                                            res.n_reservoir);

            double isi_cv = 0; int isi_n = 0;
            for (auto v : res.isi_cvs)
                if (!std::isnan(v)) { isi_cv += v; isi_n++; }
            if (isi_n > 0) isi_cv /= isi_n; else isi_cv = std::nan("");

            double adapt_mean = 0;
            for (auto v : res.adapt_b_onsets) adapt_mean += v;
            adapt_mean /= res.adapt_b_onsets.size();

            printf("      WM: %.1f%% | XOR: %.1f%% | Rate: %.1f Hz\n",
                   wm_class.accuracy * 100, xor_class.accuracy * 100, rate);

            auto json = build_gap_result_json(
                pt, gap_ms, cal.g_tonic, cal.reversal, cal.rate_hz,
                wm_class.accuracy, wm_class.accuracy_std, wm_class.per_repeat,
                xor_class.accuracy, xor_class.accuracy_std, xor_class.per_repeat,
                rate, rate_std, res.n_reservoir,
                sim_time, isi_cv, adapt_mean);
            results_json.push_back(json);
        }

        double gp_time = now_seconds() - gp_start;
        point_times.push_back(gp_time);
        printf("    %.1fmin | %d/%d grid points | Elapsed: %.1fh\n",
               gp_time / 60.0, gi + 1, n_grid, (now_seconds() - total_start) / 3600.0);

        gap_save_checkpoint(output_dir, results_json, EXT_INC, GAP_SWEEP_TAUS,
                             GAP_LENGTHS_MS, total_start, false);
    }

    printf("\n======================================================================\n");
    printf("  GAP SWEEP COMPLETE\n");
    printf("  Total: %.1f hours | %d results\n",
           (now_seconds() - total_start) / 3600.0, (int)results_json.size());
    printf("======================================================================\n");

    gap_save_checkpoint(output_dir, results_json, EXT_INC, GAP_SWEEP_TAUS,
                         GAP_LENGTHS_MS, total_start, true);
    return 0;
}

// ============================================================
// EXTENDED GAP SWEEP — focused test for WM-optimal SFA regime
// shift as a function of memory demand (long gaps).
//
// Fixed inc = 0.171, 4 tau values, 3 new gap lengths (1500, 2000, 3000 ms)
// = 12 grid points total.
// ============================================================

static const std::vector<double> GAP_EXT_INC = { 0.1710 };
static const std::vector<double> GAP_EXT_TAUS = { 3469.5, 5000.0, 7200.0, 10380.0 };
static const std::vector<double> GAP_EXT_LENGTHS_MS = { 1500.0, 2000.0, 3000.0 };

int run_gap_sweep_ext(int argc, char** argv,
                       int n_workers, const std::string& output_dir,
                       const std::string& data_dir) {
    auto grid = build_ext_grid(GAP_EXT_INC, GAP_EXT_TAUS);
    int n_grid = (int)grid.size();
    int n_gaps = (int)GAP_EXT_LENGTHS_MS.size();
    int total_conditions = n_grid * n_gaps;
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  EXTENDED GAP SWEEP (WM + XOR, Branch B only)\n");
    printf("  Fixed inc=0.171, 4 taus x 3 long gaps = %d conditions\n", total_conditions);
    printf("  Tau values: ");
    for (double t : GAP_EXT_TAUS) printf("%.1f ", t);
    printf("\n  Gap lengths: ");
    for (double g : GAP_EXT_LENGTHS_MS) printf("%.0f ", g);
    printf("ms\n  Workers: %d\n", n_workers);
    if (!g_snapshot_path.empty()) printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig wm_sim;
    wm_sim.dt = 0.1;
    wm_sim.post_stimulus_ms = POST_STIM_MS;
    wm_sim.stimulus_current = INPUT_STIM_CURRENT;

    // 2. Load checkpoint for resume
    printf("\n[2] Checking for existing checkpoint...\n");
    // Reuse gap checkpoint loader — it parses inc_idx/tau_idx/gap_ms generically
    auto [completed, existing_results] = load_gap_checkpoint(output_dir);
    int n_skipped = (int)completed.size();
    int n_remaining = total_conditions - n_skipped;
    printf("  %d/%d conditions already completed, %d remaining\n",
           n_skipped, total_conditions, n_remaining);

    // 3. Sweep
    printf("\n[3] Running conditions (Branch B)...\n");

    std::vector<std::string> results_json = existing_results;
    std::vector<double> point_times;
    int condition_num = n_skipped;

    for (int gi = 0; gi < n_grid; gi++) {
        auto& pt = grid[gi];

        // Check if ALL gaps for this grid point are already done
        bool all_done = true;
        for (int gap_i = 0; gap_i < n_gaps; gap_i++) {
            if (completed.find({pt.inc_idx, pt.tau_idx, GAP_EXT_LENGTHS_MS[gap_i]}) == completed.end()) {
                all_done = false;
                break;
            }
        }
        if (all_done) continue;

        double gp_start = now_seconds();

        DynamicalOverrides dyn;
        dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn.adapt_inc = pt.adapt_inc;
        dyn.adapt_tau = pt.adapt_tau;
        dyn.nmda_tau = FIXED_NMDA_TAU;

        // Calibrate once using gap=150ms (reference)
        auto ref_pairs = generate_pairs_with_gap(samples, CAL_N_PAIRS, PAIR_SEED + 1000, 150.0);

        double max_pair_ms = 0;
        for (auto& p : ref_pairs)
            if (p.total_stim_end_ms > max_pair_ms) max_pair_ms = p.total_stim_end_ms;
        SimConfig cal_sim = wm_sim;
        cal_sim.audio_duration_ms = max_pair_ms + 5.0;

        std::string eta = "...";
        if (!point_times.empty()) {
            int last_n = std::min((int)point_times.size(), 5);
            double avg = 0;
            for (int i = (int)point_times.size() - last_n; i < (int)point_times.size(); i++)
                avg += point_times[i];
            avg /= last_n;
            char buf[64]; snprintf(buf, sizeof(buf), "%.1fh", avg * (n_grid - gi) / 3600.0);
            eta = buf;
        }

        printf("\n  [%d/%d] inc=%.4f, tau=%.1f  |  ETA: %s\n",
               gi + 1, n_grid, pt.adapt_inc, pt.adapt_tau, eta.c_str());

        auto wm_measure = [&](double g, double r) -> double {
            return measure_rate_wm(base_cfg, ref_pairs, dyn, cal_sim, n_workers, g, r);
        };
        printf("    Calibrating (gap=150ms reference):\n");
        auto cal = calibrate_tonic(base_cfg, dyn, cal_sim, n_workers,
                                    RATE_TARGET_HZ, wm_measure);
        printf("    g_tonic=%.4f (%s) -> %.1f Hz\n",
               cal.g_tonic, cal.reversal < -1 ? "inh" : "exc", cal.rate_hz);

        // Run each gap length
        for (int gap_i = 0; gap_i < n_gaps; gap_i++) {
            double gap_ms = GAP_EXT_LENGTHS_MS[gap_i];

            if (completed.find({pt.inc_idx, pt.tau_idx, gap_ms}) != completed.end()) {
                continue;
            }

            condition_num++;
            printf("    gap=%.0fms [%d/%d total]...\n", gap_ms,
                   condition_num, total_conditions);

            auto pairs = generate_pairs_with_gap(samples, N_WM_PAIRS, PAIR_SEED, gap_ms);

            double max_ms = 0;
            for (auto& p : pairs)
                if (p.total_stim_end_ms > max_ms) max_ms = p.total_stim_end_ms;
            SimConfig gap_sim = wm_sim;
            gap_sim.audio_duration_ms = max_ms + 5.0;

            double t0 = now_seconds();
            auto res = run_wm_pairs(base_cfg, pairs, dyn, gap_sim, n_workers,
                                      cal.g_tonic, cal.reversal, true);
            double sim_time = now_seconds() - t0;

            auto [rate, rate_std] = compute_rate_stats(res.total_spikes, res.n_reservoir, pairs);

            auto wm_class = classify_wm(res.epoch_bins_list, res.digit_a_list,
                                          res.digit_b_list, res.n_reservoir);
            auto xor_class = classify_xor(res.epoch_bins_list, res.label_list,
                                            res.n_reservoir);

            double isi_cv = 0; int isi_n = 0;
            for (auto v : res.isi_cvs)
                if (!std::isnan(v)) { isi_cv += v; isi_n++; }
            if (isi_n > 0) isi_cv /= isi_n; else isi_cv = std::nan("");

            double adapt_mean = 0;
            for (auto v : res.adapt_b_onsets) adapt_mean += v;
            adapt_mean /= res.adapt_b_onsets.size();

            printf("      WM: %.1f%% | XOR: %.1f%% | Rate: %.1f Hz\n",
                   wm_class.accuracy * 100, xor_class.accuracy * 100, rate);

            auto json = build_gap_result_json(
                pt, gap_ms, cal.g_tonic, cal.reversal, cal.rate_hz,
                wm_class.accuracy, wm_class.accuracy_std, wm_class.per_repeat,
                xor_class.accuracy, xor_class.accuracy_std, xor_class.per_repeat,
                rate, rate_std, res.n_reservoir,
                sim_time, isi_cv, adapt_mean);
            results_json.push_back(json);
        }

        double gp_time = now_seconds() - gp_start;
        point_times.push_back(gp_time);
        printf("    %.1fmin | %d/%d grid points | Elapsed: %.1fh\n",
               gp_time / 60.0, gi + 1, n_grid, (now_seconds() - total_start) / 3600.0);

        gap_save_checkpoint(output_dir, results_json, GAP_EXT_INC, GAP_EXT_TAUS,
                             GAP_EXT_LENGTHS_MS, total_start, false);
    }

    printf("\n======================================================================\n");
    printf("  EXTENDED GAP SWEEP COMPLETE\n");
    printf("  Total: %.1f hours | %d results\n",
           (now_seconds() - total_start) / 3600.0, (int)results_json.size());
    printf("======================================================================\n");

    gap_save_checkpoint(output_dir, results_json, GAP_EXT_INC, GAP_EXT_TAUS,
                         GAP_EXT_LENGTHS_MS, total_start, true);
    return 0;
}

// ============================================================
// ABLATION STUDY — isolate SFA as the primary driver
//
// 5 conditions × 3 adaptation levels × 3 tasks = 45 measurements
// Conditions: baseline, -STD, -NMDA, -STD-NMDA ("SFA-only"), -SFA ("bare")
// Adaptation: inc=0 (no adapt), inc=0.071 (CLS opt), inc=0.171 (WM opt)
// All at tau=5000, Branch B (rate-matched), gap=150ms
// ============================================================

struct AblationCondition {
    std::string name;
    double std_u;        // 0.0 = disabled, STD_U = normal
    double nmda_ratio;   // 0.0 = disabled, -1.0 = normal (use network default)
};

static const std::vector<AblationCondition> ABLATION_CONDITIONS = {
    {"baseline",    STD_U, -1.0},
    {"no_std",      0.0,   -1.0},
    {"no_nmda",     STD_U,  0.0},
    {"sfa_only",    0.0,    0.0},
    {"bare",        0.0,    0.0},   // bare also sets adapt_inc=0 (handled in loop)
};

struct AblationPoint {
    std::string label;
    double adapt_inc;
};

static const std::vector<AblationPoint> ABLATION_POINTS = {
    {"no_adapt",    0.0},
    {"cls_opt",     0.0707},
    {"wm_opt",      0.1710},
};

static const double ABLATION_TAU = 5000.0;

int run_ablation(int argc, char** argv,
                  int n_workers, const std::string& output_dir,
                  const std::string& data_dir) {
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  ABLATION STUDY — Isolating SFA\n");
    printf("  %d conditions x %d adaptation levels x 3 tasks\n",
           (int)ABLATION_CONDITIONS.size(), (int)ABLATION_POINTS.size());
    printf("  All at tau=%.0f, Branch B (rate-matched), gap=150ms\n", ABLATION_TAU);
    printf("  Workers: %d\n", n_workers);
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // 2. Generate WM pairs (gap=150ms)
    printf("\n[2] Generating WM pairs (gap=150ms)...\n");
    auto pairs = generate_pairs_with_gap(samples, N_WM_PAIRS, PAIR_SEED, 150.0);
    printf("  %d pairs\n", (int)pairs.size());

    // Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    // CLS sim config
    double max_audio_ms = 0;
    for (auto& s : samples)
        if (!s.spike_times_ms.empty()) {
            double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
            if (mx > max_audio_ms) max_audio_ms = mx;
        }

    SimConfig cls_sim;
    cls_sim.dt = 0.1;
    cls_sim.post_stimulus_ms = POST_STIM_MS;
    cls_sim.audio_duration_ms = max_audio_ms + 5.0;
    cls_sim.stimulus_current = INPUT_STIM_CURRENT;

    // WM sim config
    SimConfig wm_sim;
    wm_sim.dt = 0.1;
    wm_sim.post_stimulus_ms = POST_STIM_MS;
    wm_sim.stimulus_current = INPUT_STIM_CURRENT;

    double max_pair_ms = 0;
    for (auto& p : pairs)
        if (p.total_stim_end_ms > max_pair_ms) max_pair_ms = p.total_stim_end_ms;
    wm_sim.audio_duration_ms = max_pair_ms + 5.0;

    // Calibration pairs (smaller set)
    auto cal_pairs = generate_pairs_with_gap(samples, CAL_N_PAIRS, PAIR_SEED + 1000, 150.0);
    SimConfig cal_sim = wm_sim;
    double max_cal_ms = 0;
    for (auto& p : cal_pairs)
        if (p.total_stim_end_ms > max_cal_ms) max_cal_ms = p.total_stim_end_ms;
    cal_sim.audio_duration_ms = max_cal_ms + 5.0;

    // 3. Run ablation
    printf("\n[3] Running ablation conditions...\n");

    // Output JSON
    std::string json_path = output_dir + "/ablation_results.json";
    FILE* out = fopen(json_path.c_str(), "w");
    fprintf(out, "{\n  \"experiment\": \"Ablation Study\",\n");
    fprintf(out, "  \"adapt_tau\": %.1f,\n", ABLATION_TAU);
    fprintf(out, "  \"gap_ms\": 150.0,\n");
    fprintf(out, "  \"results\": [\n");
    bool first_result = true;

    int total_conditions = (int)ABLATION_CONDITIONS.size() * (int)ABLATION_POINTS.size();
    int cond_num = 0;

    for (auto& cond : ABLATION_CONDITIONS) {
        for (auto& pt : ABLATION_POINTS) {
            cond_num++;
            double t0 = now_seconds();

            // For "bare" condition, force inc=0 regardless of pt
            double eff_inc = pt.adapt_inc;
            if (cond.name == "bare") eff_inc = 0.0;

            // Skip bare + non-zero adaptation (redundant with bare + no_adapt)
            if (cond.name == "bare" && pt.adapt_inc > 0.0) {
                printf("\n  [%d/%d] %s @ %s — skipped (bare forces inc=0)\n",
                       cond_num, total_conditions, cond.name.c_str(), pt.label.c_str());
                continue;
            }

            printf("\n  [%d/%d] %s @ %s (inc=%.4f)\n",
                   cond_num, total_conditions, cond.name.c_str(), pt.label.c_str(), eff_inc);

            DynamicalOverrides dyn;
            dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn.adapt_inc = eff_inc;
            dyn.adapt_tau = ABLATION_TAU;
            dyn.nmda_tau = FIXED_NMDA_TAU;

            // Calibrate (using WM paradigm)
            auto wm_measure = [&](double g, double r) -> double {
                return measure_rate_wm(base_cfg, cal_pairs, dyn, cal_sim, n_workers, g, r);
            };
            printf("    Calibrating...\n");
            auto cal = calibrate_tonic(base_cfg, dyn, cal_sim, n_workers,
                                        RATE_TARGET_HZ, wm_measure);
            printf("    g_tonic=%.4f (%s) -> %.1f Hz\n",
                   cal.g_tonic, cal.reversal < -1 ? "inh" : "exc", cal.rate_hz);

            // Run CLS
            printf("    Running CLS (%d samples)...\n", (int)samples.size());
            auto cls_res = run_cls_samples(base_cfg, samples, dyn, cls_sim, n_workers,
                                            cal.g_tonic, cal.reversal, false,
                                            cond.std_u, cond.nmda_ratio);
            auto cls_class = classify_cls(cls_res.bins_list, cls_res.digits,
                                           cls_res.n_reservoir, cls_res.n_bins);

            // Run WM + XOR
            printf("    Running WM/XOR (%d pairs)...\n", (int)pairs.size());
            auto wm_res = run_wm_pairs(base_cfg, pairs, dyn, wm_sim, n_workers,
                                         cal.g_tonic, cal.reversal, false,
                                         cond.std_u, cond.nmda_ratio);
            auto wm_class = classify_wm(wm_res.epoch_bins_list, wm_res.digit_a_list,
                                          wm_res.digit_b_list, wm_res.n_reservoir);
            auto xor_class = classify_xor(wm_res.epoch_bins_list, wm_res.label_list,
                                            wm_res.n_reservoir);

            double elapsed = now_seconds() - t0;
            printf("    CLS: %.1f%% | WM: %.1f%% | XOR: %.1f%% | %.1fs\n",
                   cls_class.accuracy * 100, wm_class.accuracy * 100,
                   xor_class.accuracy * 100, elapsed);

            // Write JSON
            if (!first_result) fprintf(out, ",\n");
            first_result = false;
            fprintf(out, "    {\"condition\": \"%s\", \"adapt_label\": \"%s\", ",
                    cond.name.c_str(), pt.label.c_str());
            fprintf(out, "\"adapt_inc\": %.4f, \"adapt_tau\": %.1f, ",
                    eff_inc, ABLATION_TAU);
            fprintf(out, "\"std_u\": %.4f, \"nmda_ratio\": %.2f, ",
                    cond.std_u, cond.nmda_ratio >= 0 ? cond.nmda_ratio : 0.5);
            fprintf(out, "\"g_tonic\": %.6f, \"cal_rate_hz\": %.2f, ",
                    cal.g_tonic, cal.rate_hz);
            fprintf(out, "\"cls_accuracy\": %.6f, \"cls_std\": %.6f, ",
                    cls_class.accuracy, cls_class.accuracy_std);
            fprintf(out, "\"cls_per_repeat\": [");
            for (size_t r = 0; r < cls_class.per_repeat.size(); r++) {
                if (r) fprintf(out, ", ");
                fprintf(out, "%.6f", cls_class.per_repeat[r]);
            }
            fprintf(out, "], ");
            fprintf(out, "\"wm_accuracy\": %.6f, \"wm_std\": %.6f, ",
                    wm_class.accuracy, wm_class.accuracy_std);
            fprintf(out, "\"wm_per_repeat\": [");
            for (size_t r = 0; r < wm_class.per_repeat.size(); r++) {
                if (r) fprintf(out, ", ");
                fprintf(out, "%.6f", wm_class.per_repeat[r]);
            }
            fprintf(out, "], ");
            fprintf(out, "\"xor_accuracy\": %.6f, \"xor_std\": %.6f, ",
                    xor_class.accuracy, xor_class.accuracy_std);
            fprintf(out, "\"xor_per_repeat\": [");
            for (size_t r = 0; r < xor_class.per_repeat.size(); r++) {
                if (r) fprintf(out, ", ");
                fprintf(out, "%.6f", xor_class.per_repeat[r]);
            }
            fprintf(out, "], ");
            fprintf(out, "\"sim_time_s\": %.1f}", elapsed);
        }
    }

    fprintf(out, "\n  ],\n");
    fprintf(out, "  \"total_time_s\": %.1f\n}\n", now_seconds() - total_start);
    fclose(out);

    printf("\n======================================================================\n");
    printf("  ABLATION STUDY COMPLETE\n");
    printf("  Total: %.1f minutes | Saved: %s\n",
           (now_seconds() - total_start) / 60.0, json_path.c_str());
    printf("======================================================================\n");

    return 0;
}

// ============================================================
// MULTI-SEED REPLICATION — test dissociation across random
// network topologies built from different RNG seeds.
//
// Pre-builds all networks on the main thread (deterministic),
// verifies uniqueness, then runs 3 inc levels x 3 tasks on each.
// ============================================================

static const int MULTI_SEED_N_SEEDS = 10;
static const int MULTI_SEED_SEEDS[] = {42, 137, 256, 314, 500, 628, 777, 888, 1000, 1234};

// Pre-built network cache
struct CachedNetwork {
    SphericalNetwork net;
    ZoneInfo zone_info;
    int seed;
    int n_reservoir;
};

// Run WM pairs on a pre-built network (workers copy from cached)
static WmResult run_wm_pairs_on(const CachedNetwork& cached,
                                  const std::vector<XorPairExt>& pairs,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers,
                                  double g_tonic, double tonic_rev) {
    int n_pairs = (int)pairs.size();
    WmResult out;
    out.epoch_bins_list.resize(n_pairs);
    out.digit_a_list.resize(n_pairs);
    out.digit_b_list.resize(n_pairs);
    out.label_list.resize(n_pairs);
    out.total_spikes.resize(n_pairs);
    out.isi_cvs.resize(n_pairs);
    out.adapt_b_onsets.resize(n_pairs);
    out.n_reservoir = cached.n_reservoir;

    #pragma omp parallel num_threads(n_workers)
    {
        // Each worker copies the cached network (overrides already baked in)
        SphericalNetwork net = cached.net;
        ZoneInfo zone_info = cached.zone_info;
        StdMasks masks = build_std_masks(net, zone_info);
        if (g_tonic != 0.0) set_tonic(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cached.seed * 10000ULL + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_pairs; i++) {
            SimConfig pair_sim = sim_cfg;
            pair_sim.audio_duration_ms = pairs[i].total_stim_end_ms + 5.0;

            auto result = run_sample_with_std(net, pairs[i].compound, zone_info, pair_sim,
                                               STD_U, STD_TAU_REC,
                                               masks, {pairs[i].stim_b_offset_ms});

            EpochBoundsExt eb = compute_epoch_bounds(pairs[i]);
            auto bins = bin_activity_epochs(result.activity_record,
                                             zone_info.reservoir_zone_indices,
                                             sim_cfg.dt, eb);

            int b_start_step = (int)(eb.start_ms[IDX_B_START] / sim_cfg.dt);
            int b_end_step = std::min((int)(eb.end_ms[IDX_B_START + N_BINS_B - 1] / sim_cfg.dt),
                                       (int)result.activity_record.size());
            std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                                   zone_info.reservoir_zone_indices.end());
            std::unordered_map<int, double> last_spike;
            std::vector<double> all_isis;
            for (int step = b_start_step; step < b_end_step; step++) {
                double t = step * sim_cfg.dt;
                for (int nid : result.activity_record[step]) {
                    if (res_set.count(nid)) {
                        auto it = last_spike.find(nid);
                        if (it != last_spike.end()) all_isis.push_back(t - it->second);
                        last_spike[nid] = t;
                    }
                }
            }
            double isi_cv = std::nan("");
            if (all_isis.size() >= 2) {
                double mn = 0;
                for (double v : all_isis) mn += v;
                mn /= all_isis.size();
                double var = 0;
                for (double v : all_isis) var += (v - mn) * (v - mn);
                var /= all_isis.size();
                isi_cv = std::sqrt(var) / std::max(mn, 1e-9);
            }

            int total = 0;
            for (auto& step_spikes : result.activity_record)
                for (int nid : step_spikes)
                    if (res_set.count(nid)) total++;

            out.epoch_bins_list[i] = std::move(bins);
            out.digit_a_list[i] = pairs[i].digit_a;
            out.digit_b_list[i] = pairs[i].digit_b;
            out.label_list[i] = pairs[i].label;
            out.total_spikes[i] = total;
            out.isi_cvs[i] = isi_cv;
            out.adapt_b_onsets[i] = 0.0;
        }
    }
    return out;
}

// Run CLS samples on a pre-built network
static ClsResult run_cls_samples_on(const CachedNetwork& cached,
                                      const std::vector<AudioSample>& samples,
                                      const DynamicalOverrides& dyn_ovr,
                                      const SimConfig& sim_cfg, int n_workers,
                                      double g_tonic, double tonic_rev) {
    int n_samples = (int)samples.size();
    ClsResult out;
    out.bins_list.resize(n_samples);
    out.digits.resize(n_samples);
    out.total_spikes.resize(n_samples);
    out.n_reservoir = cached.n_reservoir;
    out.n_bins = 0;

    #pragma omp parallel num_threads(n_workers)
    {
        SphericalNetwork net = cached.net;
        ZoneInfo zone_info = cached.zone_info;
        apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        StdMasks masks = build_std_masks(net, zone_info);
        if (g_tonic != 0.0) set_tonic(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cached.seed * 10000ULL + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_samples; i++) {
            auto result = run_sample_with_std(net, samples[i], zone_info, sim_cfg,
                                               STD_U, STD_TAU_REC, masks, {});
            double total_dur = sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms;
            int n_bins = std::max(1, (int)(total_dur / BIN_MS));
            auto bins = bin_activity_fixed(result.activity_record,
                                            zone_info.reservoir_zone_indices,
                                            sim_cfg.dt, BIN_MS, n_bins);
            std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                                   zone_info.reservoir_zone_indices.end());
            int total = 0;
            for (auto& step_spikes : result.activity_record)
                for (int nid : step_spikes)
                    if (res_set.count(nid)) total++;

            out.bins_list[i] = std::move(bins);
            out.digits[i] = samples[i].digit;
            out.total_spikes[i] = total;
            #pragma omp critical
            { if (n_bins > out.n_bins) out.n_bins = n_bins; }
        }
    }
    for (auto& b : out.bins_list)
        if (b.rows > out.n_bins) out.n_bins = b.rows;
    return out;
}

// Rate measurement on cached network
static double measure_rate_wm_on(const CachedNetwork& cached,
                                   const std::vector<XorPairExt>& pairs,
                                   const DynamicalOverrides& dyn_ovr,
                                   const SimConfig& sim_cfg, int n_workers,
                                   double g_tonic, double rev) {
    auto res = run_wm_pairs_on(cached, pairs, dyn_ovr, sim_cfg, n_workers, g_tonic, rev);
    double sum = 0;
    for (int i = 0; i < (int)pairs.size(); i++) {
        double dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * dur_s);
    }
    return sum / pairs.size();
}

int run_multi_seed(int argc, char** argv,
                    int n_workers, const std::string& output_dir,
                    const std::string& data_dir) {
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  MULTI-SEED REPLICATION\n");
    printf("  %d independent network topologies x 3 inc levels x 3 tasks\n",
           MULTI_SEED_N_SEEDS);
    printf("  All at tau=5000, Branch B (rate-matched), gap=150ms\n");
    printf("  Workers: %d\n", n_workers);
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // 2. Generate WM pairs
    printf("\n[2] Generating WM pairs (gap=150ms)...\n");
    auto pairs = generate_pairs_with_gap(samples, N_WM_PAIRS, PAIR_SEED, 150.0);
    printf("  %d pairs\n", (int)pairs.size());

    double max_audio_ms = 0;
    for (auto& s : samples)
        if (!s.spike_times_ms.empty()) {
            double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
            if (mx > max_audio_ms) max_audio_ms = mx;
        }

    SimConfig cls_sim;
    cls_sim.dt = 0.1;
    cls_sim.post_stimulus_ms = POST_STIM_MS;
    cls_sim.audio_duration_ms = max_audio_ms + 5.0;
    cls_sim.stimulus_current = INPUT_STIM_CURRENT;

    SimConfig wm_sim;
    wm_sim.dt = 0.1;
    wm_sim.post_stimulus_ms = POST_STIM_MS;
    wm_sim.stimulus_current = INPUT_STIM_CURRENT;
    double max_pair_ms = 0;
    for (auto& p : pairs)
        if (p.total_stim_end_ms > max_pair_ms) max_pair_ms = p.total_stim_end_ms;
    wm_sim.audio_duration_ms = max_pair_ms + 5.0;

    auto cal_pairs = generate_pairs_with_gap(samples, CAL_N_PAIRS, PAIR_SEED + 1000, 150.0);
    SimConfig cal_sim = wm_sim;
    double max_cal_ms = 0;
    for (auto& p : cal_pairs)
        if (p.total_stim_end_ms > max_cal_ms) max_cal_ms = p.total_stim_end_ms;
    cal_sim.audio_duration_ms = max_cal_ms + 5.0;

    // 3. Pre-build all networks with dynamical overrides (per seed x inc)
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    struct AdaptPt { std::string label; double inc; };
    std::vector<AdaptPt> adapt_pts = {
        {"no_adapt", 0.0}, {"cls_opt", 0.0707}, {"wm_opt", 0.1710}
    };

    int n_configs = MULTI_SEED_N_SEEDS * (int)adapt_pts.size();
    printf("\n[3] Pre-building %d networks (%d seeds x %d inc, main thread)...\n",
           n_configs, MULTI_SEED_N_SEEDS, (int)adapt_pts.size());

    // Build one network per (seed, inc) — store indexed by [seed_idx * 3 + inc_idx]
    std::vector<CachedNetwork> networks(n_configs);
    for (int si = 0; si < MULTI_SEED_N_SEEDS; si++) {
        int seed = MULTI_SEED_SEEDS[si];
        for (int ai = 0; ai < (int)adapt_pts.size(); ai++) {
            int idx = si * (int)adapt_pts.size() + ai;

            DynamicalOverrides dyn;
            dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn.adapt_inc = adapt_pts[ai].inc;
            dyn.adapt_tau = 5000.0;
            dyn.nmda_tau = FIXED_NMDA_TAU;

            NetworkConfig cfg_copy = base_cfg;
            build_full_network(networks[idx].net, networks[idx].zone_info,
                              cfg_copy, 0.1, true, &dyn, "default", true, seed);
            networks[idx].seed = seed;
            networks[idx].n_reservoir = (int)networks[idx].zone_info.reservoir_zone_indices.size();
        }
        printf("  Seed %4d: n_reservoir = %d\n", seed,
               networks[si * (int)adapt_pts.size()].n_reservoir);
    }

    // Verify uniqueness across seeds (compare no_adapt networks)
    printf("\n  Uniqueness check: ");
    bool all_unique = true;
    for (int i = 0; i < MULTI_SEED_N_SEEDS; i++)
        for (int j = i + 1; j < MULTI_SEED_N_SEEDS; j++) {
            int ni = networks[i * (int)adapt_pts.size()].n_reservoir;
            int nj = networks[j * (int)adapt_pts.size()].n_reservoir;
            if (ni == nj) {
                printf("WARNING: seeds %d and %d have same n_reservoir=%d\n",
                       MULTI_SEED_SEEDS[i], MULTI_SEED_SEEDS[j], ni);
                all_unique = false;
            }
        }
    if (all_unique) printf("PASS (all n_reservoir values distinct)\n");

    // 4. Run tasks on cached networks
    printf("\n[4] Running %d seeds x %d inc levels...\n",
           MULTI_SEED_N_SEEDS, (int)adapt_pts.size());

    std::string json_path = output_dir + "/multi_seed_results.json";
    FILE* fout = fopen(json_path.c_str(), "w");
    fprintf(fout, "{\n  \"experiment\": \"Multi-Seed Replication\",\n");
    fprintf(fout, "  \"n_seeds\": %d,\n", MULTI_SEED_N_SEEDS);
    fprintf(fout, "  \"adapt_tau\": 5000.0,\n");
    fprintf(fout, "  \"gap_ms\": 150.0,\n");
    fprintf(fout, "  \"results\": [\n");
    bool first_result = true;

    for (int si = 0; si < MULTI_SEED_N_SEEDS; si++) {
        int seed = MULTI_SEED_SEEDS[si];
        int n_adapt = (int)adapt_pts.size();
        auto& first_net = networks[si * n_adapt];
        double seed_start = now_seconds();

        printf("\n  === Seed %d (%d/%d, n_res=%d) ===\n",
               seed, si + 1, MULTI_SEED_N_SEEDS, first_net.n_reservoir);

        for (int ai = 0; ai < n_adapt; ai++) {
            auto& pt = adapt_pts[ai];
            auto& cached = networks[si * n_adapt + ai];
            double t0 = now_seconds();

            // Overrides already baked into cached network — create dummy for calibrate_tonic signature
            DynamicalOverrides dyn;
            dyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn.adapt_inc = pt.inc;
            dyn.adapt_tau = 5000.0;
            dyn.nmda_tau = FIXED_NMDA_TAU;

            // Calibrate on cached network (overrides already applied)
            auto wm_measure = [&](double g, double r) -> double {
                return measure_rate_wm_on(cached, cal_pairs, dyn, cal_sim, n_workers, g, r);
            };
            printf("    %s (inc=%.4f): calibrating...\n", pt.label.c_str(), pt.inc);
            auto cal = calibrate_tonic(base_cfg, dyn, cal_sim, n_workers,
                                        RATE_TARGET_HZ, wm_measure);
            printf("      g_tonic=%.4f -> %.1f Hz\n", cal.g_tonic, cal.rate_hz);

            // CLS on cached network
            printf("      CLS...\n");
            auto cls_res = run_cls_samples_on(cached, samples, dyn, cls_sim, n_workers,
                                               cal.g_tonic, cal.reversal);
            auto cls_class = classify_cls(cls_res.bins_list, cls_res.digits,
                                           cls_res.n_reservoir, cls_res.n_bins);

            // WM + XOR on cached network
            printf("      WM/XOR...\n");
            auto wm_res = run_wm_pairs_on(cached, pairs, dyn, wm_sim, n_workers,
                                            cal.g_tonic, cal.reversal);
            auto wm_class = classify_wm(wm_res.epoch_bins_list, wm_res.digit_a_list,
                                          wm_res.digit_b_list, wm_res.n_reservoir);
            auto xor_class = classify_xor(wm_res.epoch_bins_list, wm_res.label_list,
                                            wm_res.n_reservoir);

            double elapsed = now_seconds() - t0;
            printf("      CLS: %.1f%% | WM: %.1f%% | XOR: %.1f%% | %.1fs\n",
                   cls_class.accuracy * 100, wm_class.accuracy * 100,
                   xor_class.accuracy * 100, elapsed);

            if (!first_result) fprintf(fout, ",\n");
            first_result = false;
            fprintf(fout, "    {\"seed\": %d, \"adapt_label\": \"%s\", ",
                    cached.seed, pt.label.c_str());
            fprintf(fout, "\"adapt_inc\": %.4f, \"n_reservoir\": %d, ",
                    pt.inc, cached.n_reservoir);
            fprintf(fout, "\"g_tonic\": %.6f, \"cal_rate_hz\": %.2f, ",
                    cal.g_tonic, cal.rate_hz);
            fprintf(fout, "\"cls_accuracy\": %.6f, \"wm_accuracy\": %.6f, \"xor_accuracy\": %.6f, ",
                    cls_class.accuracy, wm_class.accuracy, xor_class.accuracy);
            fprintf(fout, "\"cls_std\": %.6f, \"wm_std\": %.6f, \"xor_std\": %.6f, ",
                    cls_class.accuracy_std, wm_class.accuracy_std, xor_class.accuracy_std);
            fprintf(fout, "\"sim_time_s\": %.1f}", elapsed);
        }

        printf("    Seed %d total: %.1f min\n", seed,
               (now_seconds() - seed_start) / 60.0);
    }

    fprintf(fout, "\n  ],\n");
    fprintf(fout, "  \"total_time_s\": %.1f\n}\n", now_seconds() - total_start);
    fclose(fout);

    printf("\n======================================================================\n");
    printf("  MULTI-SEED REPLICATION COMPLETE\n");
    printf("  Total: %.1f hours | Saved: %s\n",
           (now_seconds() - total_start) / 3600.0, json_path.c_str());
    printf("======================================================================\n");

    return 0;
}
