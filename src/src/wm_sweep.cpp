#include "experiments.h"
#include <atomic>

// ============================================================
// GRID AXES — identical to classification.cpp
// ============================================================
static const std::vector<double> WM_UNIFIED_INC = {
    0.0,
    0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000,
};
static const std::vector<double> WM_UNIFIED_TAU = {
    30.0, 43.2, 62.3, 89.8, 129.4,
    186.5, 268.7, 387.3, 558.1, 804.4,
    1159.2, 1670.6, 2407.5, 3469.5, 5000.0,
};

// ============================================================
// WM CONSTANTS
// ============================================================
static constexpr double GAP_MS = 150.0;
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
static constexpr int WM_CALIBRATION_N_PAIRS = 200;

// ============================================================
// GRID POINT
// ============================================================
struct WmGridPoint {
    double adapt_inc, adapt_tau;
    std::string point_id;
    int inc_idx, tau_idx;
};

static std::vector<WmGridPoint> build_wm_grid_points() {
    std::vector<WmGridPoint> points;
    points.reserve(WM_UNIFIED_INC.size() * WM_UNIFIED_TAU.size());
    for (int ii = 0; ii < (int)WM_UNIFIED_INC.size(); ii++) {
        for (int ti = 0; ti < (int)WM_UNIFIED_TAU.size(); ti++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "inc%.4f_tau%.1f",
                     WM_UNIFIED_INC[ii], WM_UNIFIED_TAU[ti]);
            points.push_back({WM_UNIFIED_INC[ii], WM_UNIFIED_TAU[ti], buf, ii, ti});
        }
    }
    return points;
}

// ============================================================
// XOR PAIR
// ============================================================
struct XorPair {
    AudioSample compound;       // concatenated A + gap + B spike trains
    int digit_a, digit_b;
    int label;                  // 1 = same, 0 = different
    std::string speaker_a, speaker_b;
    double stim_a_end_ms;       // max spike time in A
    double stim_b_offset_ms;    // stim_a_end_ms + GAP_MS
    double stim_b_end_ms;       // max spike time in B (absolute)
    double total_stim_end_ms;   // stim_b_end_ms (last BSA spike)
};

static std::vector<XorPair> generate_xor_pairs(
    const std::vector<AudioSample>& samples,
    int n_pairs, int seed)
{
    // Organize by digit
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

    std::vector<XorPair> pairs;
    pairs.reserve(n_pairs);

    // Same-digit pairs
    for (int p = 0; p < n_same; p++) {
        int d = digits[p % n_digits];
        auto& pool = by_digit[d];
        std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
        int ia = pool[dist(rng)];
        int ib = pool[dist(rng)];

        const auto& sa = samples[ia];
        const auto& sb = samples[ib];

        double a_end = *std::max_element(sa.spike_times_ms.begin(), sa.spike_times_ms.end());
        double b_offset = a_end + GAP_MS;

        AudioSample compound;
        compound.digit = sa.digit;
        compound.speaker = sa.speaker;
        compound.filename = sa.filename + "+" + sb.filename;

        // Copy A spikes
        compound.spike_times_ms = sa.spike_times_ms;
        compound.freq_bin_indices = sa.freq_bin_indices;

        // Append B spikes shifted
        for (size_t k = 0; k < sb.spike_times_ms.size(); k++) {
            compound.spike_times_ms.push_back(sb.spike_times_ms[k] + b_offset);
            compound.freq_bin_indices.push_back(sb.freq_bin_indices[k]);
        }

        double b_max = *std::max_element(sb.spike_times_ms.begin(), sb.spike_times_ms.end());
        double b_end_abs = b_offset + b_max;

        pairs.push_back({
            std::move(compound), sa.digit, sb.digit, 1,
            sa.speaker, sb.speaker,
            a_end, b_offset, b_end_abs, b_end_abs
        });
    }

    // Different-digit pairs
    for (int p = 0; p < n_diff; p++) {
        // Pick two different digits
        int di_a = p % n_digits;
        int di_b = (di_a + 1 + (p / n_digits) % (n_digits - 1)) % n_digits;
        if (di_b == di_a) di_b = (di_a + 1) % n_digits;
        int da = digits[di_a];
        int db = digits[di_b];

        auto& pool_a = by_digit[da];
        auto& pool_b = by_digit[db];
        std::uniform_int_distribution<int> dist_a(0, (int)pool_a.size() - 1);
        std::uniform_int_distribution<int> dist_b(0, (int)pool_b.size() - 1);
        int ia = pool_a[dist_a(rng)];
        int ib = pool_b[dist_b(rng)];

        const auto& sa = samples[ia];
        const auto& sb = samples[ib];

        double a_end = *std::max_element(sa.spike_times_ms.begin(), sa.spike_times_ms.end());
        double b_offset = a_end + GAP_MS;

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

        double b_max = *std::max_element(sb.spike_times_ms.begin(), sb.spike_times_ms.end());
        double b_end_abs = b_offset + b_max;

        pairs.push_back({
            std::move(compound), sa.digit, sb.digit, 0,
            sa.speaker, sb.speaker,
            a_end, b_offset, b_end_abs, b_end_abs
        });
    }

    // Shuffle
    std::shuffle(pairs.begin(), pairs.end(), rng);
    return pairs;
}

// ============================================================
// EPOCH BOUNDARY COMPUTATION
// ============================================================
struct EpochBounds {
    double start_ms[N_EPOCH_BINS];
    double end_ms[N_EPOCH_BINS];
    double dur_ms[N_EPOCH_BINS];
};

static EpochBounds compute_epoch_bounds(const XorPair& pair) {
    EpochBounds eb;
    int b = 0;

    // A bins (5): subdivide [0, stim_a_end_ms]
    double a_dur = pair.stim_a_end_ms;
    double a_bin = a_dur / N_BINS_A;
    for (int i = 0; i < N_BINS_A; i++, b++) {
        eb.start_ms[b] = i * a_bin;
        eb.end_ms[b] = (i + 1) * a_bin;
    }

    // Gap bin (1): [stim_a_end_ms, stim_b_offset_ms]
    eb.start_ms[b] = pair.stim_a_end_ms;
    eb.end_ms[b] = pair.stim_b_offset_ms;
    b++;

    // B bins (5): subdivide [stim_b_offset_ms, stim_b_end_ms]
    double b_dur = pair.stim_b_end_ms - pair.stim_b_offset_ms;
    double b_bin = b_dur / N_BINS_B;
    for (int i = 0; i < N_BINS_B; i++, b++) {
        eb.start_ms[b] = pair.stim_b_offset_ms + i * b_bin;
        eb.end_ms[b] = pair.stim_b_offset_ms + (i + 1) * b_bin;
    }

    // Post bins (2): subdivide [stim_b_end_ms, stim_b_end_ms + POST_STIM_MS]
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
struct EpochBinResult {
    Mat bins;           // (N_EPOCH_BINS, n_channels) — rate-normalized
    int total_spikes;
};

static EpochBinResult bin_activity_epochs(
    const std::vector<std::vector<int>>& activity_record,
    const std::vector<int>& neuron_indices,
    double dt, const EpochBounds& eb)
{
    int n_ch = (int)neuron_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_ch; p++) idx_to_pos[neuron_indices[p]] = p;

    Mat bins(N_EPOCH_BINS, n_ch, 0.0);
    int total = 0;
    int n_steps = (int)activity_record.size();

    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        // Find which epoch bin this timestep belongs to
        int b = -1;
        for (int e = 0; e < N_EPOCH_BINS; e++) {
            if (t >= eb.start_ms[e] && t < eb.end_ms[e]) { b = e; break; }
        }
        if (b < 0) {
            // Could be past the last bin — assign to last bin
            if (t >= eb.start_ms[N_EPOCH_BINS - 1]) b = N_EPOCH_BINS - 1;
            else continue;
        }

        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end()) {
                bins(b, it->second) += 1.0;
                total++;
            }
        }
    }

    // Rate-normalize: divide by bin duration in ms
    for (int e = 0; e < N_EPOCH_BINS; e++) {
        if (eb.dur_ms[e] > 1e-6) {
            for (int c = 0; c < n_ch; c++)
                bins(e, c) /= eb.dur_ms[e];
        }
    }

    return {bins, total};
}

// ============================================================
// BSA EPOCH BINNING (128 freq channels)
// ============================================================
static Mat extract_bsa_epoch_bins(const XorPair& pair, const EpochBounds& eb) {
    int n_channels = 128;
    Mat bins(N_EPOCH_BINS, n_channels, 0.0);

    for (size_t k = 0; k < pair.compound.spike_times_ms.size(); k++) {
        double t = pair.compound.spike_times_ms[k];
        int ch = pair.compound.freq_bin_indices[k];
        if (ch < 0 || ch >= n_channels) continue;

        int b = -1;
        for (int e = 0; e < N_EPOCH_BINS; e++) {
            if (t >= eb.start_ms[e] && t < eb.end_ms[e]) { b = e; break; }
        }
        if (b < 0) {
            if (t >= eb.start_ms[N_EPOCH_BINS - 1]) b = N_EPOCH_BINS - 1;
            else continue;
        }
        bins(b, ch) += 1.0;
    }

    // Rate-normalize
    for (int e = 0; e < N_EPOCH_BINS; e++) {
        if (eb.dur_ms[e] > 1e-6) {
            for (int c = 0; c < n_channels; c++)
                bins(e, c) /= eb.dur_ms[e];
        }
    }
    return bins;
}

// ============================================================
// SIMULATION WORKER — for XOR pairs
// ============================================================
struct WmWorkerResult {
    Mat epoch_bins;           // (N_EPOCH_BINS, n_reservoir)
    int total_res_spikes;
    int digit_a, digit_b;
    int label;
    int pair_idx;
    double isi_cv;            // ISI CV during B epoch
    double mean_adapt_b_onset;
};

static WmWorkerResult wm_sim_worker(SphericalNetwork& net, const XorPair& pair,
                                     int pair_idx, const ZoneInfo& zone_info,
                                     const SimConfig& sim_cfg, const StdMasks& masks) {
    double dt = sim_cfg.dt;

    // Run the compound sample through the network
    auto result = run_sample_with_std(net, pair.compound, zone_info, sim_cfg,
                                       STD_U, STD_TAU_REC, masks,
                                       {pair.stim_b_offset_ms});  // record adapt at B onset

    // Epoch binning
    EpochBounds eb = compute_epoch_bounds(pair);
    auto br = bin_activity_epochs(result.activity_record, zone_info.reservoir_zone_indices,
                                   dt, eb);

    // ISI CV during B epoch
    int b_start_step = (int)(eb.start_ms[IDX_B_START] / dt);
    int b_end_step = std::min((int)(eb.end_ms[IDX_B_START + N_BINS_B - 1] / dt),
                               (int)result.activity_record.size());

    std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                           zone_info.reservoir_zone_indices.end());

    std::unordered_map<int, double> last_spike;
    std::vector<double> all_isis;

    for (int step = b_start_step; step < b_end_step; step++) {
        double t = step * dt;
        for (int nid : result.activity_record[step]) {
            if (res_set.count(nid)) {
                auto it = last_spike.find(nid);
                if (it != last_spike.end())
                    all_isis.push_back(t - it->second);
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

    // Mean adaptation at B onset
    double adapt_mean = 0;
    if (!result.adapt_snapshots.empty() && !result.adapt_snapshots[0].empty()) {
        for (int nid : zone_info.reservoir_zone_indices)
            adapt_mean += result.adapt_snapshots[0][nid];
        adapt_mean /= zone_info.reservoir_zone_indices.size();
    }

    return {br.bins, br.total_spikes, pair.digit_a, pair.digit_b,
            pair.label, pair_idx, isi_cv, adapt_mean};
}

// ============================================================
// RUN ALL PAIRS (OpenMP parallel)
// ============================================================
struct WmAllResult {
    std::vector<Mat> epoch_bins_list;
    std::vector<int> digit_a_list;
    std::vector<int> digit_b_list;
    std::vector<int> label_list;
    std::vector<double> total_spikes;
    int n_reservoir;
    std::vector<double> isi_cvs;
    std::vector<double> adapt_b_onsets;
};

static void set_wm_tonic_conductance(SphericalNetwork& net,
                                      const ZoneInfo& zone_info,
                                      double g_tonic, double reversal) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zone_info.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g_tonic;
        net.tonic_reversal[nid] = reversal;
    }
}

static WmAllResult run_all_pairs(const NetworkConfig& cfg,
                                  const std::vector<XorPair>& pairs,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg,
                                  int n_workers,
                                  bool verbose = true,
                                  double g_tonic = 0.0,
                                  double tonic_rev = -80.0) {
    int n_pairs = (int)pairs.size();
    WmAllResult out;
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
            NetworkConfig cfg_copy = cfg;
            build_full_network(net, zone_info, cfg_copy, sim_cfg.dt, true,
                              &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (g_tonic != 0.0)
            set_wm_tonic_conductance(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp single
        {
            out.n_reservoir = (int)zone_info.reservoir_zone_indices.size();
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_pairs; i++) {
            // Each pair has its own sim_cfg with correct audio_duration_ms
            SimConfig pair_sim = sim_cfg;
            pair_sim.audio_duration_ms = pairs[i].total_stim_end_ms + 5.0;

            auto wr = wm_sim_worker(net, pairs[i], i, zone_info, pair_sim, masks);
            out.epoch_bins_list[i] = std::move(wr.epoch_bins);
            out.digit_a_list[i] = wr.digit_a;
            out.digit_b_list[i] = wr.digit_b;
            out.label_list[i] = wr.label;
            out.total_spikes[i] = wr.total_res_spikes;
            out.isi_cvs[i] = wr.isi_cv;
            out.adapt_b_onsets[i] = wr.mean_adapt_b_onset;

            int c = ++completed;
            if (verbose && (c % 100 == 0 || c == n_pairs)) {
                #pragma omp critical
                printf("        %5d/%d\n", c, n_pairs);
            }
        }
    }

    return out;
}

// ============================================================
// RATE MEASUREMENT (for calibration)
// ============================================================
static double wm_measure_rate(const NetworkConfig& cfg,
                               const std::vector<XorPair>& pairs_subset,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& sim_cfg, int n_workers,
                               double g_tonic = 0.0, double tonic_rev = -80.0) {
    auto res = run_all_pairs(cfg, pairs_subset, dyn_ovr, sim_cfg, n_workers, false, g_tonic, tonic_rev);
    double sum = 0;
    for (int i = 0; i < (int)pairs_subset.size(); i++) {
        double trial_dur_s = (pairs_subset[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
    }
    return sum / pairs_subset.size();
}

// ============================================================
// TONIC CONDUCTANCE CALIBRATION (reused from classification.cpp pattern)
// ============================================================
static constexpr double WM_G_TONIC_HI = 5.0;

struct WmCalResult {
    double g_tonic;
    double reversal;
    double rate_hz;
};

static WmCalResult
wm_calibrate_tonic(const NetworkConfig& cfg,
                    const DynamicalOverrides& dyn_ovr,
                    const std::vector<XorPair>& cal_pairs,
                    const SimConfig& sim_cfg, int n_workers,
                    double target_rate,
                    double initial_guess = 0.0,
                    double prev_reversal = -80.0) {
    // Measure natural rate
    double natural_rate = wm_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers, 0.0);
    printf("    cal[0] g=0 (natural) -> %.1f Hz (target=%.1f)\n", natural_rate, target_rate);

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate};

    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    const char* mode = (reversal < -1.0) ? "inhibitory" : "excitatory";
    printf("    cal: natural=%.1f Hz, target=%.1f Hz -> %s mode\n",
           natural_rate, target_rate, mode);

    double lo = 0.0, hi = WM_G_TONIC_HI;
    int iteration = 1;

    if (initial_guess > 0.0 && prev_reversal == reversal) {
        double rate = wm_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers,
                                       initial_guess, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, initial_guess, rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, reversal, rate};
        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = initial_guess; else lo = initial_guess;
    }

    for (;; iteration++) {
        double mid = (lo + hi) / 2.0;
        double rate = wm_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers, mid, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, mid, rate);

        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {mid, reversal, rate};

        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = mid; else lo = mid;

        if (hi - lo < 1e-6) {
            double span = std::max(mid, 1.0);
            hi = mid + span;
            printf("    cal: interval collapsed, widening hi to %.4f\n", hi);
        }
    }
}

// ============================================================
// WM CLASSIFICATION — B+Post window, different-digit pairs only
// ============================================================
struct WmClassifyResult {
    double accuracy;
    double accuracy_std;
    std::vector<double> per_repeat_accuracy;
};

static WmClassifyResult classify_wm_bpost(const std::vector<Mat>& epoch_bins_list,
                                            const std::vector<int>& digit_a_list,
                                            const std::vector<int>& digit_b_list,
                                            int n_features_per_bin) {
    // Filter to different-digit pairs
    std::vector<int> diff_indices;
    for (int i = 0; i < (int)digit_a_list.size(); i++) {
        if (digit_a_list[i] != digit_b_list[i])
            diff_indices.push_back(i);
    }

    int n_diff = (int)diff_indices.size();
    int n_feat = N_BPOST_BINS * n_features_per_bin;

    // Build feature matrix from B+Post bins
    Mat X(n_diff, n_feat, 0.0);
    std::vector<int> y(n_diff);
    for (int di = 0; di < n_diff; di++) {
        int idx = diff_indices[di];
        y[di] = digit_a_list[idx];
        for (int b = 0; b < N_BPOST_BINS; b++) {
            int epoch_b = IDX_B_START + b;
            if (epoch_b < epoch_bins_list[idx].rows) {
                for (int r = 0; r < std::min(n_features_per_bin, epoch_bins_list[idx].cols); r++) {
                    X(di, b * n_features_per_bin + r) = epoch_bins_list[idx](epoch_b, r);
                }
            }
        }
    }

    // Stratified KFold x repeats (same as classification.cpp)
    std::vector<double> repeat_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double rep_correct = 0;
        int rep_total = 0;

        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& split = folds[f];

            Mat X_train((int)split.train.size(), n_feat);
            std::vector<int> y_train(split.train.size());
            for (int i = 0; i < (int)split.train.size(); i++) {
                for (int j = 0; j < n_feat; j++)
                    X_train(i, j) = X(split.train[i], j);
                y_train[i] = y[split.train[i]];
            }

            Mat X_test((int)split.test.size(), n_feat);
            std::vector<int> y_test(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_feat; j++)
                    X_test(i, j) = X(split.test[i], j);
                y_test[i] = y[split.test[i]];
            }

            cls::StandardScaler scaler;
            X_train = scaler.fit_transform(X_train);
            X_test = scaler.transform(X_test);
            cls::nan_to_num(X_train);
            cls::nan_to_num(X_test);

            auto fold_ctx = cls::ridge_fold_prepare(X_train, y_train, X_test, y_test,
                                                     DEFAULT_DIGITS);
            double best_acc = -1;
            std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(fold_ctx, X_test, y_test, alpha);
                if (rr.accuracy > best_acc) {
                    best_acc = rr.accuracy;
                    best_preds = rr.predictions;
                }
            }

            int n_test = (int)split.test.size();
            for (int i = 0; i < n_test; i++) {
                if (best_preds[i] == y[split.test[i]]) rep_correct++;
            }
            rep_total += n_test;
        }
        repeat_accs.push_back(rep_correct / rep_total);
    }

    double mean = 0;
    for (double a : repeat_accs) mean += a;
    mean /= repeat_accs.size();
    double var = 0;
    for (double a : repeat_accs) var += (a - mean) * (a - mean);
    var /= repeat_accs.size();

    return {mean, std::sqrt(var), repeat_accs};
}

// ============================================================
// XOR CLASSIFICATION — same/different on all pairs, B+Post window
// ============================================================
static WmClassifyResult classify_xor_bpost(const std::vector<Mat>& epoch_bins_list,
                                             const std::vector<int>& label_list,
                                             int n_features_per_bin) {
    int n_pairs = (int)epoch_bins_list.size();
    int n_feat = N_BPOST_BINS * n_features_per_bin;

    Mat X(n_pairs, n_feat, 0.0);
    std::vector<int> y(n_pairs);
    for (int i = 0; i < n_pairs; i++) {
        y[i] = label_list[i];
        for (int b = 0; b < N_BPOST_BINS; b++) {
            int epoch_b = IDX_B_START + b;
            if (epoch_b < epoch_bins_list[i].rows) {
                for (int r = 0; r < std::min(n_features_per_bin, epoch_bins_list[i].cols); r++) {
                    X(i, b * n_features_per_bin + r) = epoch_bins_list[i](epoch_b, r);
                }
            }
        }
    }

    // XOR is 2-class: labels 0/1
    std::vector<int> xor_classes = {0, 1};

    std::vector<double> repeat_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double rep_correct = 0;
        int rep_total = 0;

        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& split = folds[f];

            Mat X_train((int)split.train.size(), n_feat);
            std::vector<int> y_train(split.train.size());
            for (int i = 0; i < (int)split.train.size(); i++) {
                for (int j = 0; j < n_feat; j++)
                    X_train(i, j) = X(split.train[i], j);
                y_train[i] = y[split.train[i]];
            }

            Mat X_test((int)split.test.size(), n_feat);
            std::vector<int> y_test(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_feat; j++)
                    X_test(i, j) = X(split.test[i], j);
                y_test[i] = y[split.test[i]];
            }

            cls::StandardScaler scaler;
            X_train = scaler.fit_transform(X_train);
            X_test = scaler.transform(X_test);
            cls::nan_to_num(X_train);
            cls::nan_to_num(X_test);

            auto fold_ctx = cls::ridge_fold_prepare(X_train, y_train, X_test, y_test,
                                                     xor_classes);
            double best_acc = -1;
            std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(fold_ctx, X_test, y_test, alpha);
                if (rr.accuracy > best_acc) {
                    best_acc = rr.accuracy;
                    best_preds = rr.predictions;
                }
            }

            int n_test = (int)split.test.size();
            for (int i = 0; i < n_test; i++) {
                if (best_preds[i] == y[split.test[i]]) rep_correct++;
            }
            rep_total += n_test;
        }
        repeat_accs.push_back(rep_correct / rep_total);
    }

    double mean = 0;
    for (double a : repeat_accs) mean += a;
    mean /= repeat_accs.size();
    double var = 0;
    for (double a : repeat_accs) var += (a - mean) * (a - mean);
    var /= repeat_accs.size();

    return {mean, std::sqrt(var), repeat_accs};
}

// ============================================================
// PER-BIN ACCURACY (WM task, all 13 epoch bins)
// ============================================================
static std::vector<double> wm_per_bin_accuracy(const std::vector<Mat>& epoch_bins_list,
                                                const std::vector<int>& digit_a_list,
                                                const std::vector<int>& digit_b_list,
                                                int n_features_per_bin) {
    // Different-digit pairs only
    std::vector<int> diff_indices;
    for (int i = 0; i < (int)digit_a_list.size(); i++) {
        if (digit_a_list[i] != digit_b_list[i])
            diff_indices.push_back(i);
    }
    int n_diff = (int)diff_indices.size();

    std::vector<double> bin_accs(N_EPOCH_BINS, 0.0);

    for (int b = 0; b < N_EPOCH_BINS; b++) {
        Mat X(n_diff, n_features_per_bin, 0.0);
        std::vector<int> y(n_diff);
        for (int di = 0; di < n_diff; di++) {
            int idx = diff_indices[di];
            y[di] = digit_a_list[idx];
            if (b < epoch_bins_list[idx].rows) {
                for (int r = 0; r < std::min(n_features_per_bin, epoch_bins_list[idx].cols); r++)
                    X(di, r) = epoch_bins_list[idx](b, r);
            }
        }

        std::vector<double> rep_accs;
        for (int rep = 0; rep < N_CV_REPEATS; rep++) {
            auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
            double rep_correct = 0;
            int rep_total = 0;

            for (int f = 0; f < N_CV_FOLDS; f++) {
                auto& split = folds[f];
                Mat Xtr((int)split.train.size(), n_features_per_bin);
                std::vector<int> ytr(split.train.size());
                for (int i = 0; i < (int)split.train.size(); i++) {
                    for (int j = 0; j < n_features_per_bin; j++) Xtr(i, j) = X(split.train[i], j);
                    ytr[i] = y[split.train[i]];
                }
                Mat Xte((int)split.test.size(), n_features_per_bin);
                std::vector<int> yte(split.test.size());
                for (int i = 0; i < (int)split.test.size(); i++) {
                    for (int j = 0; j < n_features_per_bin; j++) Xte(i, j) = X(split.test[i], j);
                    yte[i] = y[split.test[i]];
                }
                cls::StandardScaler sc;
                Xtr = sc.fit_transform(Xtr);
                Xte = sc.transform(Xte);
                cls::nan_to_num(Xtr);
                cls::nan_to_num(Xte);

                auto fold_ctx = cls::ridge_fold_prepare(Xtr, ytr, Xte, yte, DEFAULT_DIGITS);
                double best_acc = -1;
                std::vector<int> best_preds;
                for (double alpha : RIDGE_ALPHAS) {
                    auto rr = cls::ridge_fold_solve(fold_ctx, Xte, yte, alpha);
                    if (rr.accuracy > best_acc) {
                        best_acc = rr.accuracy;
                        best_preds = rr.predictions;
                    }
                }
                int n_test = (int)split.test.size();
                for (int i = 0; i < n_test; i++) {
                    if (best_preds[i] == y[split.test[i]]) rep_correct++;
                }
                rep_total += n_test;
            }
            rep_accs.push_back(rep_correct / rep_total);
        }
        double s = 0;
        for (double a : rep_accs) s += a;
        bin_accs[b] = s / rep_accs.size();
    }
    return bin_accs;
}

// ============================================================
// PARTICIPATION RATIO (B+Post bins, different-digit pairs)
// ============================================================
static std::pair<double, std::vector<double>>
wm_participation_ratio(const std::vector<Mat>& epoch_bins_list,
                        const std::vector<int>& digit_a_list,
                        const std::vector<int>& digit_b_list,
                        int n_reservoir) {
    std::vector<int> diff_indices;
    for (int i = 0; i < (int)digit_a_list.size(); i++) {
        if (digit_a_list[i] != digit_b_list[i])
            diff_indices.push_back(i);
    }
    int n_diff = (int)diff_indices.size();
    int n_eff = std::min(n_diff, n_reservoir);
    std::vector<double> pr_per_bin(N_BPOST_BINS, 0.0);

    for (int b = 0; b < N_BPOST_BINS; b++) {
        int epoch_b = IDX_B_START + b;
        Mat X(n_diff, n_reservoir, 0.0);
        for (int di = 0; di < n_diff; di++) {
            int idx = diff_indices[di];
            if (epoch_b < epoch_bins_list[idx].rows) {
                for (int r = 0; r < std::min(n_reservoir, epoch_bins_list[idx].cols); r++)
                    X(di, r) = epoch_bins_list[idx](epoch_b, r);
            }
        }

        std::vector<double> col_mean(n_reservoir, 0.0);
        for (int j = 0; j < n_reservoir; j++) {
            for (int i = 0; i < n_diff; i++) col_mean[j] += X(i, j);
            col_mean[j] /= n_diff;
        }
        for (int i = 0; i < n_diff; i++)
            for (int j = 0; j < n_reservoir; j++)
                X(i, j) -= col_mean[j];

        double total_var = 0;
        for (auto& v : X.data) total_var += v * v;
        if (total_var < 1e-10) continue;

        std::vector<double> S;
        Mat U, Vt;
        svd_econ(X, S, U, Vt);

        double sum_lam = 0, sum_lam2 = 0;
        for (double s : S) {
            double lam = s * s / std::max(n_diff - 1, 1);
            sum_lam += lam;
            sum_lam2 += lam * lam;
        }
        if (sum_lam2 > 1e-20) {
            pr_per_bin[b] = (sum_lam * sum_lam) / (n_eff * sum_lam2);
        }
    }

    double mean_pr = 0;
    for (double v : pr_per_bin) mean_pr += v;
    mean_pr /= N_BPOST_BINS;
    return {mean_pr, pr_per_bin};
}

// ============================================================
// JSON OUTPUT
// ============================================================
static void wm_save_checkpoint(const std::string& results_dir,
                                 const std::vector<std::map<std::string, std::string>>& grid_results_json,
                                 const std::string& bsa_wm_json,
                                 const std::string& bsa_xor_json,
                                 const std::string& baseline_json,
                                 double target_rate_hz,
                                 int n_pairs,
                                 double total_start_time,
                                 bool final_save) {
    std::string fname = final_save ? "wm_adaptation_sweep.json"
                                   : "wm_adaptation_sweep_checkpoint.json";
    std::string path = results_dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    double elapsed = now_seconds() - total_start_time;

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Working Memory Adaptation Parameter Sweep (C++ Port)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", elapsed);
    fprintf(f, "  \"task\": \"Working memory: classify digit_a from B+Post (different-digit pairs)\",\n");
    fprintf(f, "  \"digits\": [0, 1, 2, 3, 4],\n");
    fprintf(f, "  \"n_pairs\": %d,\n", n_pairs);
    fprintf(f, "  \"wm_chance_level\": 0.20,\n");
    fprintf(f, "  \"xor_chance_level\": 0.50,\n");
    fprintf(f, "  \"epoch_bins\": {\"n_bins_a\": %d, \"n_bins_gap\": %d, \"n_bins_b\": %d, \"n_bins_post\": %d, \"total\": %d},\n",
            N_BINS_A, N_BINS_GAP, N_BINS_B, N_BINS_POST, N_EPOCH_BINS);
    fprintf(f, "  \"readout_window\": \"B+Post (bins %d-%d)\",\n", IDX_B_START, IDX_BPOST_END - 1);
    fprintf(f, "  \"gap_ms\": %.1f,\n", GAP_MS);
    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.2f, \"tolerance_hz\": %.1f},\n",
            target_rate_hz, RATE_TOLERANCE_HZ);

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); json_write_double_array(f, WM_UNIFIED_INC); fprintf(f, ",\n");
    fprintf(f, "    \"unified_tau\": "); json_write_double_array(f, WM_UNIFIED_TAU); fprintf(f, ",\n");
    fprintf(f, "    \"n_inc\": %d, \"n_tau\": %d\n", (int)WM_UNIFIED_INC.size(), (int)WM_UNIFIED_TAU.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"bsa_wm_baseline\": %s,\n", bsa_wm_json.c_str());
    fprintf(f, "  \"bsa_xor_baseline\": %s,\n", bsa_xor_json.c_str());
    fprintf(f, "  \"lhs021_baseline\": %s,\n", baseline_json.c_str());

    fprintf(f, "  \"grid_results\": [\n");
    for (size_t i = 0; i < grid_results_json.size(); i++) {
        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    %s", grid_results_json[i].at("json").c_str());
    }
    fprintf(f, "\n  ]\n");
    fprintf(f, "}\n");
    fclose(f);

    if (final_save) printf("\n  Saved: %s\n", path.c_str());
}

// ============================================================
// MAIN WM SWEEP ENTRY POINT
// ============================================================
int run_wm_sweep(int argc, char** argv,
                  int n_workers, const std::string& output_dir,
                  const std::string& data_dir) {
    auto grid_points = build_wm_grid_points();
    int n_grid = (int)grid_points.size();

    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  WORKING MEMORY ADAPTATION PARAMETER SWEEP (C++ PORT)\n");
    printf("  Grid: %d x %d = %d points x 2 branches\n",
           (int)WM_UNIFIED_INC.size(), (int)WM_UNIFIED_TAU.size(), n_grid);
    printf("  Task: Working memory (digit_a from B+Post, diff-pairs only)\n");
    printf("  Pairs: %d per grid point (balanced same/different)\n", N_WM_PAIRS);
    printf("  Epochs: %d bins (A:%d + Gap:%d + B:%d + Post:%d)\n",
           N_EPOCH_BINS, N_BINS_A, N_BINS_GAP, N_BINS_B, N_BINS_POST);
    printf("  Readout: B+Post (bins %d-%d), rate-normalized\n", IDX_B_START, IDX_BPOST_END - 1);
    printf("  WM: 5-class (digit_a, chance=20%%), diff-pairs only\n");
    printf("  XOR: 2-class (same/diff, chance=50%%), all pairs\n");
    printf("  Workers: %d\n", n_workers);
    printf("  CV: StratifiedKFold(%d) x %d repeats\n", N_CV_FOLDS, N_CV_REPEATS);
    if (!g_snapshot_path.empty()) {
        printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    }
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded\n", n_samples);

    // 2. Generate XOR pairs
    printf("\n[2] Generating XOR pairs...\n");
    auto pairs = generate_xor_pairs(samples, N_WM_PAIRS, PAIR_SEED);
    int n_pairs = (int)pairs.size();
    int n_same = 0, n_diff = 0;
    for (auto& p : pairs) { if (p.label == 1) n_same++; else n_diff++; }
    printf("  %d pairs (%d same, %d different)\n", n_pairs, n_same, n_diff);

    // Find max audio duration for sim config
    double max_total_stim_ms = 0;
    for (auto& p : pairs) {
        if (p.total_stim_end_ms > max_total_stim_ms)
            max_total_stim_ms = p.total_stim_end_ms;
    }
    max_total_stim_ms += 5.0;

    // 3. BSA baselines
    printf("\n[3] Computing BSA baselines...\n");
    {
        std::vector<Mat> bsa_epoch_list(n_pairs);
        std::vector<int> digit_a_list(n_pairs), digit_b_list(n_pairs), label_list(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            EpochBounds eb = compute_epoch_bounds(pairs[i]);
            bsa_epoch_list[i] = extract_bsa_epoch_bins(pairs[i], eb);
            digit_a_list[i] = pairs[i].digit_a;
            digit_b_list[i] = pairs[i].digit_b;
            label_list[i] = pairs[i].label;
        }
        auto bsa_wm = classify_wm_bpost(bsa_epoch_list, digit_a_list, digit_b_list, 128);
        auto bsa_xor = classify_xor_bpost(bsa_epoch_list, label_list, 128);
        printf("  BSA WM: %.1f%% (chance=20%%)\n", bsa_wm.accuracy * 100);
        printf("  BSA XOR: %.1f%% (chance=50%%)\n", bsa_xor.accuracy * 100);
    }

    // Re-compute BSA for JSON (separate scope for clarity)
    std::string bsa_wm_json, bsa_xor_json;
    {
        std::vector<Mat> bsa_epoch_list(n_pairs);
        std::vector<int> digit_a_list(n_pairs), digit_b_list(n_pairs), label_list(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            EpochBounds eb = compute_epoch_bounds(pairs[i]);
            bsa_epoch_list[i] = extract_bsa_epoch_bins(pairs[i], eb);
            digit_a_list[i] = pairs[i].digit_a;
            digit_b_list[i] = pairs[i].digit_b;
            label_list[i] = pairs[i].label;
        }
        auto bsa_wm = classify_wm_bpost(bsa_epoch_list, digit_a_list, digit_b_list, 128);
        auto bsa_xor = classify_xor_bpost(bsa_epoch_list, label_list, 128);

        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"wm_accuracy\": %.10f, \"wm_accuracy_std\": %.10f, "
                 "\"xor_accuracy\": %.10f, \"xor_accuracy_std\": %.10f}",
                 bsa_wm.accuracy, bsa_wm.accuracy_std,
                 bsa_xor.accuracy, bsa_xor.accuracy_std);
        bsa_wm_json = buf;
        // Also store xor separately for clarity
        snprintf(buf, sizeof(buf),
                 "{\"accuracy\": %.10f, \"accuracy_std\": %.10f}",
                 bsa_xor.accuracy, bsa_xor.accuracy_std);
        bsa_xor_json = buf;
        // Fix: make bsa_wm_json just the WM part
        snprintf(buf, sizeof(buf),
                 "{\"accuracy\": %.10f, \"accuracy_std\": %.10f}",
                 bsa_wm.accuracy, bsa_wm.accuracy_std);
        bsa_wm_json = buf;
    }

    // 4. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_total_stim_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    // Calibration subset
    std::vector<XorPair> cal_pairs(pairs.begin(),
                                    pairs.begin() + std::min(WM_CALIBRATION_N_PAIRS, n_pairs));

    // Branch B calibration target
    double target_rate_hz = RATE_TARGET_HZ;

    // 5. LHS-021 baseline
    printf("\n[4] Running LHS-021 baseline...\n");
    DynamicalOverrides lhs021_ovr;
    lhs021_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
    lhs021_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
    lhs021_ovr.adapt_inc = LHS021_ADAPT_INC;
    lhs021_ovr.nmda_tau = FIXED_NMDA_TAU;

    printf("  Simulating %d pairs (%d workers)...\n", n_pairs, n_workers);
    double bl_t0 = now_seconds();
    auto bl_res = run_all_pairs(base_cfg, pairs, lhs021_ovr, sim_cfg, n_workers);
    double bl_sim_time = now_seconds() - bl_t0;

    // Rate
    double bl_rate_hz = 0;
    for (int i = 0; i < n_pairs; i++) {
        double trial_dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        bl_rate_hz += bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s);
    }
    bl_rate_hz /= n_pairs;

    // WM and XOR classification
    auto bl_wm = classify_wm_bpost(bl_res.epoch_bins_list, bl_res.digit_a_list,
                                    bl_res.digit_b_list, bl_res.n_reservoir);
    auto bl_xor = classify_xor_bpost(bl_res.epoch_bins_list, bl_res.label_list,
                                      bl_res.n_reservoir);

    printf("  LHS-021: %.0fs, Rate: %.1f Hz\n", bl_sim_time, bl_rate_hz);
    printf("  WM: %.1f%% | XOR: %.1f%%\n", bl_wm.accuracy * 100, bl_xor.accuracy * 100);

    // Baseline JSON
    char bl_json_buf[2048];
    snprintf(bl_json_buf, sizeof(bl_json_buf),
             "{\"wm_accuracy\": %.10f, \"wm_accuracy_std\": %.10f, "
             "\"xor_accuracy\": %.10f, \"xor_accuracy_std\": %.10f, "
             "\"firing_rate_hz\": %.4f, \"sim_time_s\": %.1f, \"n_reservoir\": %d}",
             bl_wm.accuracy, bl_wm.accuracy_std,
             bl_xor.accuracy, bl_xor.accuracy_std,
             bl_rate_hz, bl_sim_time, bl_res.n_reservoir);
    std::string baseline_json = bl_json_buf;

    // ============================================================
    // EVALUATE ONE BRANCH — shared helper
    // ============================================================
    auto evaluate_branch = [&](const WmGridPoint& pt,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& eval_sim,
                               const std::string& branch,
                               double cal_value,
                               double cal_rate,
                               double tonic_rev = -80.0) -> std::string {
        double t0 = now_seconds();
        double gt = (branch == "B_matched") ? cal_value : 0.0;
        auto res = run_all_pairs(base_cfg, pairs, dyn_ovr, eval_sim, n_workers, true, gt, tonic_rev);
        double sim_time = now_seconds() - t0;

        // Rate
        double rate_mean = 0, rate_std = 0;
        for (int i = 0; i < n_pairs; i++) {
            double trial_dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
            rate_mean += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
        }
        rate_mean /= n_pairs;
        for (int i = 0; i < n_pairs; i++) {
            double trial_dur_s = (pairs[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
            double r = res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
            rate_std += (r - rate_mean) * (r - rate_mean);
        }
        rate_std = std::sqrt(rate_std / n_pairs);

        // WM + XOR classification
        auto wm_res = classify_wm_bpost(res.epoch_bins_list, res.digit_a_list,
                                          res.digit_b_list, res.n_reservoir);
        auto xor_res = classify_xor_bpost(res.epoch_bins_list, res.label_list,
                                            res.n_reservoir);

        // ISI CV
        double isi_cv_mean = 0;
        int isi_valid = 0;
        for (int i = 0; i < n_pairs; i++) {
            if (!std::isnan(res.isi_cvs[i])) { isi_cv_mean += res.isi_cvs[i]; isi_valid++; }
        }
        if (isi_valid > 0) isi_cv_mean /= isi_valid;

        // Adaptation at B onset
        double adapt_mean = 0;
        for (int i = 0; i < n_pairs; i++) adapt_mean += res.adapt_b_onsets[i];
        adapt_mean /= n_pairs;

        // Participation ratio
        auto [pr_mean, pr_per_bin] = wm_participation_ratio(
            res.epoch_bins_list, res.digit_a_list, res.digit_b_list, res.n_reservoir);

        // Per-bin accuracy
        auto per_bin_acc = wm_per_bin_accuracy(res.epoch_bins_list, res.digit_a_list,
                                                res.digit_b_list, res.n_reservoir);

        printf("    [%s] Rate: %.1f Hz | WM: %.1f%% | XOR: %.1f%% | ISI CV: %.3f | PR: %.4f\n",
               branch.c_str(), rate_mean, wm_res.accuracy * 100, xor_res.accuracy * 100,
               isi_cv_mean, pr_mean);

        // Build JSON
        std::ostringstream oss;
        oss << "{";
        oss << "\"branch\": \"" << branch << "\", ";
        oss << "\"point_id\": \"" << pt.point_id << "\", ";
        oss << "\"inc_idx\": " << pt.inc_idx << ", ";
        oss << "\"tau_idx\": " << pt.tau_idx << ", ";
        oss << std::fixed;
        oss.precision(10);
        oss << "\"adapt_inc\": " << pt.adapt_inc << ", ";
        oss << "\"adapt_tau\": " << pt.adapt_tau << ", ";
        oss << "\"stimulus_current\": " << eval_sim.stimulus_current << ", ";
        oss << "\"tonic_conductance\": " << gt << ", ";
        oss << "\"tonic_reversal\": " << tonic_rev << ", ";
        oss << "\"calibration_rate_hz\": " << cal_rate << ", ";
        oss << "\"wm_accuracy\": " << wm_res.accuracy << ", ";
        oss.precision(6);
        oss << "\"wm_accuracy_std\": " << wm_res.accuracy_std << ", ";
        oss.precision(10);
        oss << "\"xor_accuracy\": " << xor_res.accuracy << ", ";
        oss.precision(6);
        oss << "\"xor_accuracy_std\": " << xor_res.accuracy_std << ", ";
        oss << "\"firing_rate_hz\": " << rate_mean << ", ";
        oss << "\"firing_rate_std\": " << rate_std << ", ";
        oss << "\"n_reservoir\": " << res.n_reservoir << ", ";
        oss << "\"sim_time_s\": " << sim_time << ", ";
        if (isi_valid > 0) oss << "\"isi_cv_mean\": " << isi_cv_mean << ", ";
        else oss << "\"isi_cv_mean\": null, ";
        oss << "\"adapt_at_b_onset_mean\": " << adapt_mean << ", ";
        oss << "\"participation_ratio_mean\": " << pr_mean << ", ";

        oss << "\"per_bin_accuracy\": [";
        for (size_t i = 0; i < per_bin_acc.size(); i++) {
            if (i > 0) oss << ", ";
            oss << per_bin_acc[i];
        }
        oss << "], ";

        oss << "\"wm_per_repeat_accuracy\": [";
        for (size_t i = 0; i < wm_res.per_repeat_accuracy.size(); i++) {
            if (i > 0) oss << ", ";
            oss << wm_res.per_repeat_accuracy[i];
        }
        oss << "], ";

        oss << "\"xor_per_repeat_accuracy\": [";
        for (size_t i = 0; i < xor_res.per_repeat_accuracy.size(); i++) {
            if (i > 0) oss << ", ";
            oss << xor_res.per_repeat_accuracy[i];
        }
        oss << "]";

        oss << "}";
        return oss.str();
    };

    // 6. Sweep — two branches per grid point
    printf("\n[5] Running %d grid points x 2 branches...\n", n_grid);
    printf("    Branch A: Unmatched (natural rate, fixed stim=%.4f)\n", sim_cfg.stimulus_current);
    printf("    Branch B: Tonic-conductance-matched (target=%.1f +/- %.0f Hz)\n",
           target_rate_hz, RATE_TOLERANCE_HZ);

    std::vector<std::map<std::string, std::string>> grid_results_json;
    std::vector<double> grid_point_times;
    double sweep_start = now_seconds();
    std::map<int, std::pair<double,double>> last_cal_by_tau;

    for (int pt_num = 0; pt_num < n_grid; pt_num++) {
        auto& pt = grid_points[pt_num];
        double inc_val = pt.adapt_inc;
        double tau_val = pt.adapt_tau;

        double gp_start = now_seconds();

        std::string eta_str = "calculating...";
        if (!grid_point_times.empty()) {
            int last_n = std::min((int)grid_point_times.size(), 10);
            double avg_t = 0;
            for (int i = (int)grid_point_times.size() - last_n; i < (int)grid_point_times.size(); i++)
                avg_t += grid_point_times[i];
            avg_t /= last_n;
            double eta_h = avg_t * (n_grid - pt_num) / 3600.0;
            char buf[64];
            snprintf(buf, sizeof(buf), "%.1fh", eta_h);
            eta_str = buf;
        }

        printf("\n  [%d/%d] inc=%.4f, tau=%.1fms  |  ETA: %s\n",
               pt_num + 1, n_grid, inc_val, tau_val, eta_str.c_str());

        DynamicalOverrides dyn_ovr;
        dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn_ovr.adapt_inc = inc_val;
        dyn_ovr.adapt_tau = tau_val;
        dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

        // --- Branch A: Unmatched ---
        {
            SimConfig eval_a = sim_cfg;
            double natural_rate = wm_measure_rate(base_cfg, cal_pairs, dyn_ovr, eval_a, n_workers);
            printf("    [A] Natural rate: %.1f Hz\n", natural_rate);

            auto json = evaluate_branch(pt, dyn_ovr, eval_a, "A_unmatched",
                                         eval_a.stimulus_current, natural_rate);
            grid_results_json.push_back({{"json", json}});
        }

        // --- Branch B: Tonic-conductance-matched ---
        {
            SimConfig eval_b = sim_cfg;

            double prev_gt = 0.0, prev_rev = -80.0;
            if (pt.tau_idx >= 0 && last_cal_by_tau.count(pt.tau_idx)) {
                prev_gt = last_cal_by_tau[pt.tau_idx].first;
                prev_rev = last_cal_by_tau[pt.tau_idx].second;
            }

            auto cal = wm_calibrate_tonic(base_cfg, dyn_ovr, cal_pairs, eval_b,
                                            n_workers, target_rate_hz, prev_gt, prev_rev);

            if (pt.tau_idx >= 0) last_cal_by_tau[pt.tau_idx] = {cal.g_tonic, cal.reversal};
            const char* mode = (cal.reversal < -1.0) ? "inh" : "exc";
            printf("    [B] Calibrated: g_tonic=%.4f (%s) -> %.1f Hz\n",
                   cal.g_tonic, mode, cal.rate_hz);

            auto json = evaluate_branch(pt, dyn_ovr, eval_b, "B_matched",
                                         cal.g_tonic, cal.rate_hz, cal.reversal);
            grid_results_json.push_back({{"json", json}});
        }

        double gp_elapsed = now_seconds() - gp_start;
        grid_point_times.push_back(gp_elapsed);
        printf("    %.1fmin  |  %d/%d  |  Elapsed: %.1fh\n",
               gp_elapsed / 60.0, pt_num + 1, n_grid,
               (now_seconds() - sweep_start) / 3600.0);

        wm_save_checkpoint(output_dir, grid_results_json, bsa_wm_json, bsa_xor_json,
                            baseline_json, target_rate_hz, n_pairs, total_start, false);
    }

    double total_time = now_seconds() - total_start;

    printf("\n======================================================================\n");
    printf("  WM SWEEP COMPLETE\n");
    printf("  Total time: %.0fs (%.1f hours)\n", total_time, total_time / 3600.0);
    printf("  Grid points: %d results (%d grid x 2 branches)\n",
           (int)grid_results_json.size(), n_grid);
    printf("======================================================================\n");

    wm_save_checkpoint(output_dir, grid_results_json, bsa_wm_json, bsa_xor_json,
                        baseline_json, target_rate_hz, n_pairs, total_start, true);

    printf("\nDone.\n");
    return 0;
}
