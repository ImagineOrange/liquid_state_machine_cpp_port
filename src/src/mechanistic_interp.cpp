#include "experiments.h"
#include <atomic>
#include <set>

// ============================================================
// GRID POINTS — 4 diagnostic points, all at tau=5000
// ============================================================
struct MechPoint {
    double adapt_inc, adapt_tau;
    std::string label;
    int inc_idx, tau_idx;  // indices into UNIFIED grid
};

static const std::vector<MechPoint> MECH_POINTS = {
    {0.0,    5000.0, "no_adapt",       0, 14},
    {0.0707, 5000.0, "cls_optimum",   10, 14},
    {0.1274, 5000.0, "perbin_optimum", 12, 14},
    {0.1710, 5000.0, "wm_optimum",    13, 14},
};

// ============================================================
// WM CONSTANTS (same as wm_sweep.cpp)
// ============================================================
static constexpr double MECH_GAP_MS = 150.0;
static constexpr int MECH_N_BINS_A = 5;
static constexpr int MECH_N_BINS_GAP = 1;
static constexpr int MECH_N_BINS_B = 5;
static constexpr int MECH_N_BINS_POST = 2;
static constexpr int MECH_N_EPOCH_BINS = MECH_N_BINS_A + MECH_N_BINS_GAP + MECH_N_BINS_B + MECH_N_BINS_POST;  // 13
static constexpr int MECH_IDX_B_START = MECH_N_BINS_A + MECH_N_BINS_GAP;  // 6
static constexpr int MECH_IDX_BPOST_END = MECH_N_EPOCH_BINS;              // 13
static constexpr int MECH_N_BPOST_BINS = MECH_IDX_BPOST_END - MECH_IDX_B_START;  // 7

static constexpr int MECH_N_WM_PAIRS = 2500;
static constexpr int MECH_PAIR_SEED = 99;
static constexpr int MECH_WM_CAL_N_PAIRS = 200;

// ============================================================
// XOR PAIR (identical structure to wm_sweep.cpp)
// ============================================================
struct MechXorPair {
    AudioSample compound;
    int digit_a, digit_b;
    int label;
    std::string speaker_a, speaker_b;
    double stim_a_end_ms;
    double stim_b_offset_ms;
    double stim_b_end_ms;
    double total_stim_end_ms;
};

static std::vector<MechXorPair> mech_generate_xor_pairs(
    const std::vector<AudioSample>& samples, int n_pairs, int seed)
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
    std::vector<MechXorPair> pairs;
    pairs.reserve(n_pairs);

    for (int p = 0; p < n_same; p++) {
        int d = digits[p % n_digits];
        auto& pool = by_digit[d];
        std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
        int ia = pool[dist(rng)], ib = pool[dist(rng)];
        const auto& sa = samples[ia];
        const auto& sb = samples[ib];
        double a_end = *std::max_element(sa.spike_times_ms.begin(), sa.spike_times_ms.end());
        double b_offset = a_end + MECH_GAP_MS;

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
        pairs.push_back({std::move(compound), sa.digit, sb.digit, 1,
                          sa.speaker, sb.speaker, a_end, b_offset, b_end_abs, b_end_abs});
    }

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
        double b_offset = a_end + MECH_GAP_MS;

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
        pairs.push_back({std::move(compound), sa.digit, sb.digit, 0,
                          sa.speaker, sb.speaker, a_end, b_offset, b_end_abs, b_end_abs});
    }

    std::shuffle(pairs.begin(), pairs.end(), rng);
    return pairs;
}

// ============================================================
// EPOCH BOUNDARIES
// ============================================================
struct MechEpochBounds {
    double start_ms[MECH_N_EPOCH_BINS];
    double end_ms[MECH_N_EPOCH_BINS];
    double dur_ms[MECH_N_EPOCH_BINS];
};

static MechEpochBounds mech_epoch_bounds(const MechXorPair& pair) {
    MechEpochBounds eb;
    int b = 0;
    double a_bin = pair.stim_a_end_ms / MECH_N_BINS_A;
    for (int i = 0; i < MECH_N_BINS_A; i++, b++) {
        eb.start_ms[b] = i * a_bin;
        eb.end_ms[b] = (i + 1) * a_bin;
    }
    eb.start_ms[b] = pair.stim_a_end_ms;
    eb.end_ms[b] = pair.stim_b_offset_ms;
    b++;
    double b_dur = pair.stim_b_end_ms - pair.stim_b_offset_ms;
    double b_bin = b_dur / MECH_N_BINS_B;
    for (int i = 0; i < MECH_N_BINS_B; i++, b++) {
        eb.start_ms[b] = pair.stim_b_offset_ms + i * b_bin;
        eb.end_ms[b] = pair.stim_b_offset_ms + (i + 1) * b_bin;
    }
    double post_bin = POST_STIM_MS / MECH_N_BINS_POST;
    for (int i = 0; i < MECH_N_BINS_POST; i++, b++) {
        eb.start_ms[b] = pair.stim_b_end_ms + i * post_bin;
        eb.end_ms[b] = pair.stim_b_end_ms + (i + 1) * post_bin;
    }
    for (int i = 0; i < MECH_N_EPOCH_BINS; i++)
        eb.dur_ms[i] = eb.end_ms[i] - eb.start_ms[i];
    return eb;
}

// ============================================================
// EPOCH BINNING — rate-normalized (same as wm_sweep)
// ============================================================
static Mat mech_bin_activity_epochs(
    const std::vector<std::vector<int>>& activity_record,
    const std::vector<int>& neuron_indices,
    double dt, const MechEpochBounds& eb)
{
    int n_ch = (int)neuron_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_ch; p++) idx_to_pos[neuron_indices[p]] = p;

    Mat bins(MECH_N_EPOCH_BINS, n_ch, 0.0);
    int n_steps = (int)activity_record.size();
    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        int b = -1;
        for (int e = 0; e < MECH_N_EPOCH_BINS; e++) {
            if (t >= eb.start_ms[e] && t < eb.end_ms[e]) { b = e; break; }
        }
        if (b < 0) {
            if (t >= eb.start_ms[MECH_N_EPOCH_BINS - 1]) b = MECH_N_EPOCH_BINS - 1;
            else continue;
        }
        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end()) bins(b, it->second) += 1.0;
        }
    }
    // Rate-normalize
    for (int e = 0; e < MECH_N_EPOCH_BINS; e++) {
        if (eb.dur_ms[e] > 1e-6) {
            for (int c = 0; c < n_ch; c++) bins(e, c) /= eb.dur_ms[e];
        }
    }
    return bins;
}

// ============================================================
// EXTENDED WORKER RESULT — captures per-neuron diagnostics
// ============================================================
struct MechWorkerResult {
    Mat epoch_bins;                      // (13 x n_reservoir) rate-normalized
    std::vector<double> a_spikes;        // (n_reservoir,) total spike count during A epoch
    std::vector<double> adapt_at_b;      // (n_reservoir,) adaptation at B onset
    int digit_a, digit_b, label, pair_idx;
    int total_spikes;                    // raw spike count (reservoir neurons only)
    double isi_cv;
};

static MechWorkerResult mech_sim_worker(SphericalNetwork& net, const MechXorPair& pair,
                                         int pair_idx, const ZoneInfo& zone_info,
                                         const SimConfig& sim_cfg, const StdMasks& masks) {
    double dt = sim_cfg.dt;
    auto result = run_sample_with_std(net, pair.compound, zone_info, sim_cfg,
                                       STD_U, STD_TAU_REC, masks,
                                       {pair.stim_b_offset_ms});

    MechEpochBounds eb = mech_epoch_bounds(pair);
    Mat bins = mech_bin_activity_epochs(result.activity_record, zone_info.reservoir_zone_indices,
                                         dt, eb);

    int n_res = (int)zone_info.reservoir_zone_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_res; p++) idx_to_pos[zone_info.reservoir_zone_indices[p]] = p;

    // Raw total spike count across all timesteps (reservoir neurons only)
    int raw_total_spikes = 0;

    // A-epoch spike counts: total spikes per reservoir neuron during A epoch (bins 0..4)
    std::vector<double> a_spikes(n_res, 0.0);
    int a_end_step = std::min((int)(pair.stim_a_end_ms / dt), (int)result.activity_record.size());
    for (int step = 0; step < a_end_step; step++) {
        for (int nid : result.activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end()) a_spikes[it->second] += 1.0;
        }
    }

    // Count all reservoir spikes across entire trial
    for (int step = 0; step < (int)result.activity_record.size(); step++) {
        for (int nid : result.activity_record[step]) {
            if (idx_to_pos.count(nid)) raw_total_spikes++;
        }
    }

    // Adaptation at B onset (full per-neuron vector)
    std::vector<double> adapt_at_b(n_res, 0.0);
    if (!result.adapt_snapshots.empty() && !result.adapt_snapshots[0].empty()) {
        for (int p = 0; p < n_res; p++) {
            int nid = zone_info.reservoir_zone_indices[p];
            adapt_at_b[p] = result.adapt_snapshots[0][nid];
        }
    }

    // ISI CV during B epoch
    int b_start_step = (int)(eb.start_ms[MECH_IDX_B_START] / dt);
    int b_end_step = std::min((int)(eb.end_ms[MECH_IDX_B_START + MECH_N_BINS_B - 1] / dt),
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

    return {bins, a_spikes, adapt_at_b, pair.digit_a, pair.digit_b,
            pair.label, pair_idx, raw_total_spikes, isi_cv};
}

// ============================================================
// RUN ALL PAIRS (OpenMP parallel)
// ============================================================
struct MechAllResult {
    std::vector<Mat> epoch_bins_list;
    std::vector<std::vector<double>> a_spikes_list;  // per-trial per-neuron A spike counts
    std::vector<std::vector<double>> adapt_at_b_list; // per-trial per-neuron adaptation
    std::vector<int> digit_a_list, digit_b_list, label_list;
    std::vector<double> total_spikes;
    int n_reservoir;
    std::vector<bool> neuron_is_shell;  // (n_reservoir,) shell = receives input connections
};

static void mech_set_tonic(SphericalNetwork& net, const ZoneInfo& zone_info,
                            double g_tonic, double reversal) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zone_info.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g_tonic;
        net.tonic_reversal[nid] = reversal;
    }
}

static MechAllResult mech_run_all_pairs(const NetworkConfig& cfg,
                                          const std::vector<MechXorPair>& pairs,
                                          const DynamicalOverrides& dyn_ovr,
                                          const SimConfig& sim_cfg,
                                          int n_workers,
                                          double g_tonic = 0.0,
                                          double tonic_rev = -80.0) {
    int n_pairs = (int)pairs.size();
    MechAllResult out;
    out.epoch_bins_list.resize(n_pairs);
    out.a_spikes_list.resize(n_pairs);
    out.adapt_at_b_list.resize(n_pairs);
    out.digit_a_list.resize(n_pairs);
    out.digit_b_list.resize(n_pairs);
    out.label_list.resize(n_pairs);
    out.total_spikes.resize(n_pairs);
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
            mech_set_tonic(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp single
        {
            out.n_reservoir = (int)zone_info.reservoir_zone_indices.size();

            // Identify shell neurons: reservoir neurons that receive connections from input neurons
            out.neuron_is_shell.assign(out.n_reservoir, false);
            std::set<int> input_set(zone_info.input_neuron_indices.begin(),
                                     zone_info.input_neuron_indices.end());
            std::unordered_map<int, int> res_to_pos;
            for (int p = 0; p < out.n_reservoir; p++)
                res_to_pos[zone_info.reservoir_zone_indices[p]] = p;

            for (int src : zone_info.input_neuron_indices) {
                int64_t start = net.csr_indptr[src];
                int64_t end = net.csr_indptr[src + 1];
                for (int64_t j = start; j < end; j++) {
                    int tgt = net.csr_targets[j];
                    auto it = res_to_pos.find(tgt);
                    if (it != res_to_pos.end()) {
                        out.neuron_is_shell[it->second] = true;
                    }
                }
            }
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_pairs; i++) {
            SimConfig pair_sim = sim_cfg;
            pair_sim.audio_duration_ms = pairs[i].total_stim_end_ms + 5.0;

            auto wr = mech_sim_worker(net, pairs[i], i, zone_info, pair_sim, masks);
            out.epoch_bins_list[i] = std::move(wr.epoch_bins);
            out.a_spikes_list[i] = std::move(wr.a_spikes);
            out.adapt_at_b_list[i] = std::move(wr.adapt_at_b);
            out.digit_a_list[i] = wr.digit_a;
            out.digit_b_list[i] = wr.digit_b;
            out.label_list[i] = wr.label;
            out.total_spikes[i] = wr.total_spikes;

            int c = ++completed;
            if (c % 100 == 0 || c == n_pairs) {
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
static double mech_measure_rate(const NetworkConfig& cfg,
                                  const std::vector<MechXorPair>& pairs_subset,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers,
                                  double g_tonic = 0.0, double tonic_rev = -80.0) {
    auto res = mech_run_all_pairs(cfg, pairs_subset, dyn_ovr, sim_cfg, n_workers, g_tonic, tonic_rev);
    double sum = 0;
    for (int i = 0; i < (int)pairs_subset.size(); i++) {
        double trial_dur_s = (pairs_subset[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
    }
    return sum / pairs_subset.size();
}

// ============================================================
// CALIBRATION
// ============================================================
struct MechCalResult {
    double g_tonic, reversal, rate_hz;
};

static MechCalResult mech_calibrate_tonic(const NetworkConfig& cfg,
                                            const DynamicalOverrides& dyn_ovr,
                                            const std::vector<MechXorPair>& cal_pairs,
                                            const SimConfig& sim_cfg, int n_workers,
                                            double target_rate,
                                            double initial_guess = 0.0,
                                            double prev_reversal = -80.0) {
    double natural_rate = mech_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers, 0.0);
    printf("    cal[0] g=0 (natural) -> %.1f Hz (target=%.1f)\n", natural_rate, target_rate);

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate};

    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    printf("    cal: natural=%.1f Hz -> %s mode\n", natural_rate,
           (reversal < -1.0) ? "inhibitory" : "excitatory");

    double lo = 0.0, hi = 5.0;
    int iteration = 1;

    if (initial_guess > 0.0 && prev_reversal == reversal) {
        double rate = mech_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers,
                                         initial_guess, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, initial_guess, rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, reversal, rate};
        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = initial_guess; else lo = initial_guess;
    }

    double best_g = 0.0, best_rate = natural_rate;
    double best_dist = std::abs(natural_rate - target_rate);

    for (; iteration <= 20; iteration++) {
        double mid = (lo + hi) / 2.0;
        double rate = mech_measure_rate(cfg, cal_pairs, dyn_ovr, sim_cfg, n_workers, mid, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, mid, rate);

        double dist = std::abs(rate - target_rate);
        if (dist < best_dist) { best_dist = dist; best_g = mid; best_rate = rate; }

        if (dist <= RATE_TOLERANCE_HZ)
            return {mid, reversal, rate};

        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = mid; else lo = mid;

        if (hi - lo < 1e-6) {
            hi = mid + std::max(mid, 1.0);
            printf("    cal: widening hi to %.4f\n", hi);
        }
    }

    printf("    cal: max iterations reached, using best g=%.4f -> %.1f Hz (target=%.1f)\n",
           best_g, best_rate, target_rate);
    return {best_g, reversal, best_rate};
}

// ============================================================
// WM CLASSIFICATION — with best-alpha extraction
// ============================================================
struct MechWmClassifyResult {
    double accuracy, accuracy_std;
    std::vector<double> per_repeat_accuracy;
    double best_alpha;
};

static MechWmClassifyResult mech_classify_wm_bpost(
    const std::vector<Mat>& epoch_bins_list,
    const std::vector<int>& digit_a_list,
    const std::vector<int>& digit_b_list,
    int n_features_per_bin)
{
    // Filter to different-digit pairs
    std::vector<int> diff_indices;
    for (int i = 0; i < (int)digit_a_list.size(); i++) {
        if (digit_a_list[i] != digit_b_list[i])
            diff_indices.push_back(i);
    }
    int n_diff = (int)diff_indices.size();
    int n_feat = MECH_N_BPOST_BINS * n_features_per_bin;

    Mat X(n_diff, n_feat, 0.0);
    std::vector<int> y(n_diff);
    for (int di = 0; di < n_diff; di++) {
        int idx = diff_indices[di];
        y[di] = digit_a_list[idx];
        for (int b = 0; b < MECH_N_BPOST_BINS; b++) {
            int epoch_b = MECH_IDX_B_START + b;
            if (epoch_b < epoch_bins_list[idx].rows) {
                for (int r = 0; r < std::min(n_features_per_bin, epoch_bins_list[idx].cols); r++)
                    X(di, b * n_features_per_bin + r) = epoch_bins_list[idx](epoch_b, r);
            }
        }
    }

    // Track best alpha across all folds/repeats
    std::map<double, double> alpha_total_acc;
    for (double a : RIDGE_ALPHAS) alpha_total_acc[a] = 0.0;

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
                for (int j = 0; j < n_feat; j++) X_train(i, j) = X(split.train[i], j);
                y_train[i] = y[split.train[i]];
            }
            Mat X_test((int)split.test.size(), n_feat);
            std::vector<int> y_test(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_feat; j++) X_test(i, j) = X(split.test[i], j);
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
                alpha_total_acc[alpha] += rr.accuracy;
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

    // Find globally best alpha
    double best_alpha = RIDGE_ALPHAS[0];
    double best_total = -1;
    for (auto& [a, total] : alpha_total_acc) {
        if (total > best_total) { best_total = total; best_alpha = a; }
    }

    return {mean, std::sqrt(var), repeat_accs, best_alpha};
}

// ============================================================
// ADAPTATION-STATE-ONLY CLASSIFIER (Analysis 6)
// ============================================================
static MechWmClassifyResult mech_classify_adapt_state(
    const std::vector<std::vector<double>>& adapt_at_b_list,
    const std::vector<int>& digit_a_list,
    const std::vector<int>& digit_b_list,
    int n_reservoir)
{
    std::vector<int> diff_indices;
    for (int i = 0; i < (int)digit_a_list.size(); i++) {
        if (digit_a_list[i] != digit_b_list[i])
            diff_indices.push_back(i);
    }
    int n_diff = (int)diff_indices.size();

    Mat X(n_diff, n_reservoir, 0.0);
    std::vector<int> y(n_diff);
    for (int di = 0; di < n_diff; di++) {
        int idx = diff_indices[di];
        y[di] = digit_a_list[idx];
        for (int r = 0; r < n_reservoir; r++)
            X(di, r) = adapt_at_b_list[idx][r];
    }

    std::vector<double> repeat_accs;
    std::map<double, double> alpha_total_acc;
    for (double a : RIDGE_ALPHAS) alpha_total_acc[a] = 0.0;

    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double rep_correct = 0;
        int rep_total = 0;

        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& split = folds[f];
            Mat Xtr((int)split.train.size(), n_reservoir);
            std::vector<int> ytr(split.train.size());
            for (int i = 0; i < (int)split.train.size(); i++) {
                for (int j = 0; j < n_reservoir; j++) Xtr(i, j) = X(split.train[i], j);
                ytr[i] = y[split.train[i]];
            }
            Mat Xte((int)split.test.size(), n_reservoir);
            std::vector<int> yte(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_reservoir; j++) Xte(i, j) = X(split.test[i], j);
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
                alpha_total_acc[alpha] += rr.accuracy;
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

    double best_alpha = RIDGE_ALPHAS[0];
    double best_total = -1;
    for (auto& [a, total] : alpha_total_acc) {
        if (total > best_total) { best_total = total; best_alpha = a; }
    }

    return {mean, std::sqrt(var), repeat_accs, best_alpha};
}

// ============================================================
// ANALYSIS COMPUTATIONS
// ============================================================

// Per-digit mean selectivity and adaptation vectors
struct PerDigitVectors {
    std::vector<std::vector<double>> selectivity;   // [5][n_res]
    std::vector<std::vector<double>> adapt_mean;    // [5][n_res]
    std::vector<std::vector<double>> adapt_std;     // [5][n_res]
    std::vector<int> digit_counts;                  // [5]
};

static PerDigitVectors compute_per_digit_vectors(
    const MechAllResult& res,
    const std::vector<MechXorPair>& pairs)
{
    int n_res = res.n_reservoir;
    PerDigitVectors pdv;
    pdv.selectivity.assign(N_DIGITS, std::vector<double>(n_res, 0.0));
    pdv.adapt_mean.assign(N_DIGITS, std::vector<double>(n_res, 0.0));
    pdv.adapt_std.assign(N_DIGITS, std::vector<double>(n_res, 0.0));
    pdv.digit_counts.assign(N_DIGITS, 0);

    // Accumulate sums for mean
    for (int i = 0; i < (int)pairs.size(); i++) {
        int da = res.digit_a_list[i];
        pdv.digit_counts[da]++;
        for (int r = 0; r < n_res; r++) {
            pdv.selectivity[da][r] += res.a_spikes_list[i][r];
            pdv.adapt_mean[da][r] += res.adapt_at_b_list[i][r];
        }
    }
    for (int d = 0; d < N_DIGITS; d++) {
        if (pdv.digit_counts[d] > 0) {
            for (int r = 0; r < n_res; r++) {
                pdv.selectivity[d][r] /= pdv.digit_counts[d];
                pdv.adapt_mean[d][r] /= pdv.digit_counts[d];
            }
        }
    }

    // Compute std
    for (int i = 0; i < (int)pairs.size(); i++) {
        int da = res.digit_a_list[i];
        for (int r = 0; r < n_res; r++) {
            double diff = res.adapt_at_b_list[i][r] - pdv.adapt_mean[da][r];
            pdv.adapt_std[da][r] += diff * diff;
        }
    }
    for (int d = 0; d < N_DIGITS; d++) {
        if (pdv.digit_counts[d] > 1) {
            for (int r = 0; r < n_res; r++)
                pdv.adapt_std[d][r] = std::sqrt(pdv.adapt_std[d][r] / (pdv.digit_counts[d] - 1));
        }
    }
    return pdv;
}

// Pearson r between two vectors
static double vec_pearson_r(const std::vector<double>& x, const std::vector<double>& y) {
    return pearson_r_vec(x, y);  // from experiments.h
}

// Analysis 1: 5x5 correlation matrix of digit selectivity vectors
static std::vector<std::vector<double>> compute_selectivity_corr_matrix(const PerDigitVectors& pdv) {
    std::vector<std::vector<double>> corr(N_DIGITS, std::vector<double>(N_DIGITS, 0.0));
    for (int i = 0; i < N_DIGITS; i++)
        for (int j = 0; j < N_DIGITS; j++)
            corr[i][j] = vec_pearson_r(pdv.selectivity[i], pdv.selectivity[j]);
    return corr;
}

// Analysis 2: mean trial-level r(A_spikes, g_adapt_at_B) per digit
static std::vector<double> compute_stimulus_adapt_corr(
    const MechAllResult& res,
    const std::vector<MechXorPair>& /*pairs*/)
{
    std::vector<double> per_digit_r(N_DIGITS, 0.0);
    std::vector<int> per_digit_n(N_DIGITS, 0);

    for (int i = 0; i < (int)res.digit_a_list.size(); i++) {
        double r = vec_pearson_r(res.a_spikes_list[i], res.adapt_at_b_list[i]);
        int da = res.digit_a_list[i];
        per_digit_r[da] += r;
        per_digit_n[da]++;
    }
    for (int d = 0; d < N_DIGITS; d++) {
        if (per_digit_n[d] > 0) per_digit_r[d] /= per_digit_n[d];
    }
    return per_digit_r;
}

// Analysis 3 & 4: correlations between ridge weights and selectivity/adaptation
struct WeightCorrelations {
    std::vector<std::vector<double>> within_class;  // [n_classes][n_analyses]
    std::vector<std::vector<double>> cross_class;
    double within_mean, cross_mean;
};

static WeightCorrelations compute_weight_adapt_corr(
    const cls::RidgePrimalWeights& rw,
    const PerDigitVectors& pdv,
    int n_res, int n_bins,
    bool use_adapt)  // true = Analysis 3, false = Analysis 4
{
    int n_classes = N_DIGITS;
    WeightCorrelations wc;
    wc.within_class.resize(n_classes);
    wc.cross_class.resize(n_classes);

    // Compute neuron-marginal weight magnitude per class: mean(|w|, axis=bins)
    // W shape: (n_features x n_classes) = (n_bins * n_res x n_classes)
    std::vector<std::vector<double>> W_neuron(n_classes, std::vector<double>(n_res, 0.0));
    for (int c = 0; c < n_classes; c++) {
        for (int b = 0; b < n_bins; b++) {
            for (int r = 0; r < n_res; r++) {
                W_neuron[c][r] += std::abs(rw.weights(b * n_res + r, c));
            }
        }
        for (int r = 0; r < n_res; r++) W_neuron[c][r] /= n_bins;
    }

    double within_sum = 0, cross_sum = 0;
    int within_n = 0, cross_n = 0;

    for (int d = 0; d < n_classes; d++) {
        for (int e = 0; e < n_classes; e++) {
            const auto& target = use_adapt ? pdv.adapt_mean[e] : pdv.selectivity[e];
            double r = vec_pearson_r(W_neuron[d], target);
            if (d == e) {
                wc.within_class[d].push_back(r);
                within_sum += r;
                within_n++;
            } else {
                wc.cross_class[d].push_back(r);
                cross_sum += r;
                cross_n++;
            }
        }
    }
    wc.within_mean = within_n > 0 ? within_sum / within_n : 0.0;
    wc.cross_mean = cross_n > 0 ? cross_sum / cross_n : 0.0;
    return wc;
}

// Analysis 7: shell vs core adaptation and discriminability
struct ShellCoreStats {
    double shell_adapt_mean, core_adapt_mean;
    double shell_f_stat, core_f_stat;
    int n_shell, n_core;
};

static ShellCoreStats compute_shell_core_stats(
    const MechAllResult& res,
    const PerDigitVectors& pdv)
{
    int n_res = res.n_reservoir;
    ShellCoreStats sc;

    std::vector<int> shell_idx, core_idx;
    for (int r = 0; r < n_res; r++) {
        if (res.neuron_is_shell[r]) shell_idx.push_back(r);
        else core_idx.push_back(r);
    }
    sc.n_shell = (int)shell_idx.size();
    sc.n_core = (int)core_idx.size();

    // Mean adaptation state
    sc.shell_adapt_mean = 0;
    sc.core_adapt_mean = 0;
    for (int d = 0; d < N_DIGITS; d++) {
        for (int r : shell_idx) sc.shell_adapt_mean += pdv.adapt_mean[d][r];
        for (int r : core_idx) sc.core_adapt_mean += pdv.adapt_mean[d][r];
    }
    if (sc.n_shell > 0) sc.shell_adapt_mean /= (N_DIGITS * sc.n_shell);
    if (sc.n_core > 0) sc.core_adapt_mean /= (N_DIGITS * sc.n_core);

    // F-statistic (one-way ANOVA across 5 digits) for shell and core neurons
    auto compute_f = [&](const std::vector<int>& neuron_idx) -> double {
        if (neuron_idx.empty()) return 0.0;
        // For each neuron, compute between-class variance / within-class variance
        double f_sum = 0;
        int f_n = 0;
        for (int r : neuron_idx) {
            double grand_mean = 0;
            for (int d = 0; d < N_DIGITS; d++) grand_mean += pdv.adapt_mean[d][r];
            grand_mean /= N_DIGITS;

            double ss_between = 0;
            for (int d = 0; d < N_DIGITS; d++) {
                double diff = pdv.adapt_mean[d][r] - grand_mean;
                ss_between += pdv.digit_counts[d] * diff * diff;
            }

            double ss_within = 0;
            for (int d = 0; d < N_DIGITS; d++) {
                ss_within += pdv.adapt_std[d][r] * pdv.adapt_std[d][r] * (pdv.digit_counts[d] - 1);
            }

            int total_n = 0;
            for (int d = 0; d < N_DIGITS; d++) total_n += pdv.digit_counts[d];

            double ms_between = ss_between / (N_DIGITS - 1);
            double ms_within = ss_within / std::max(total_n - N_DIGITS, 1);
            if (ms_within > 1e-12) {
                f_sum += ms_between / ms_within;
                f_n++;
            }
        }
        return f_n > 0 ? f_sum / f_n : 0.0;
    };

    sc.shell_f_stat = compute_f(shell_idx);
    sc.core_f_stat = compute_f(core_idx);
    return sc;
}

// ============================================================
// JSON HELPERS
// ============================================================
static void json_write_vec(FILE* f, const std::vector<double>& v, int precision = 6) {
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.*f", precision, v[i]);
    }
    fprintf(f, "]");
}

static void json_write_bool_vec(FILE* f, const std::vector<bool>& v) {
    fprintf(f, "[");
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%s", v[i] ? "true" : "false");
    }
    fprintf(f, "]");
}

static void json_write_vec2d(FILE* f, const std::vector<std::vector<double>>& m, int precision = 6) {
    fprintf(f, "[");
    for (size_t i = 0; i < m.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        json_write_vec(f, m[i], precision);
    }
    fprintf(f, "]");
}

static void json_write_mat(FILE* f, const Mat& m, int precision = 6) {
    fprintf(f, "[");
    for (int r = 0; r < m.rows; r++) {
        if (r > 0) fprintf(f, ", ");
        fprintf(f, "[");
        for (int c = 0; c < m.cols; c++) {
            if (c > 0) fprintf(f, ", ");
            fprintf(f, "%.*f", precision, m(r, c));
        }
        fprintf(f, "]");
    }
    fprintf(f, "]");
}

// ============================================================
// CLS SIMULATION (single-digit, standard 20ms bins) for Analysis 5
// ============================================================
static void mech_run_cls_samples(
    const NetworkConfig& cfg,
    const std::vector<AudioSample>& samples,
    const DynamicalOverrides& dyn_ovr,
    const SimConfig& sim_cfg,
    int n_workers,
    double g_tonic, double tonic_rev,
    std::vector<Mat>& bins_list_out,
    std::vector<int>& digits_out,
    int& n_reservoir_out,
    int n_bins)
{
    int n_samples = (int)samples.size();
    bins_list_out.resize(n_samples);
    digits_out.resize(n_samples);
    n_reservoir_out = 0;

    std::atomic<int> completed(0);

    #pragma omp parallel num_threads(n_workers)
    {
        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
            apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        } else {
            build_full_network(net, zone_info, cfg, sim_cfg.dt, true, &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (g_tonic != 0.0) {
            net.tonic_conductance.assign(net.n_neurons, 0.0);
            net.tonic_reversal.assign(net.n_neurons, 0.0);
            for (int nid : zone_info.reservoir_zone_indices) {
                net.tonic_conductance[nid] = g_tonic;
                net.tonic_reversal[nid] = tonic_rev;
            }
        }
        rng_seed(cfg.n_neurons + 99 + (uint64_t)omp_get_thread_num() * 1000 + (uint64_t)getpid());

        int n_res = (int)zone_info.reservoir_zone_indices.size();
        #pragma omp single
        { n_reservoir_out = n_res; }

        std::unordered_map<int, int> idx_to_pos;
        for (int p = 0; p < n_res; p++)
            idx_to_pos[zone_info.reservoir_zone_indices[p]] = p;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_samples; i++) {
            auto result = run_sample_with_std(net, samples[i], zone_info, sim_cfg,
                                               STD_U, STD_TAU_REC, masks);

            Mat bins(n_bins, n_res, 0.0);
            for (int step = 0; step < (int)result.activity_record.size(); step++) {
                double t = step * sim_cfg.dt;
                int b = (int)(t / BIN_MS);
                if (b >= n_bins) b = n_bins - 1;
                for (int nid : result.activity_record[step]) {
                    auto it = idx_to_pos.find(nid);
                    if (it != idx_to_pos.end()) bins(b, it->second) += 1.0;
                }
            }

            bins_list_out[i] = std::move(bins);
            digits_out[i] = samples[i].digit;

            int c = ++completed;
            if (c % 100 == 0 || c == n_samples) {
                #pragma omp critical
                printf("        CLS %5d/%d\n", c, n_samples);
            }
        }
    }
}

// CLS classification with best-alpha tracking (reuses MechWmClassifyResult)
static MechWmClassifyResult mech_classify_cls_flat(
    const std::vector<Mat>& bins_list,
    const std::vector<int>& y,
    int n_bins, int n_reservoir)
{
    int n_samples = (int)bins_list.size();
    int n_features = n_bins * n_reservoir;

    Mat X_flat(n_samples, n_features, 0.0);
    for (int i = 0; i < n_samples; i++) {
        for (int b = 0; b < std::min(n_bins, bins_list[i].rows); b++)
            for (int r = 0; r < std::min(n_reservoir, bins_list[i].cols); r++)
                X_flat(i, b * n_reservoir + r) = bins_list[i](b, r);
    }

    std::map<double, double> alpha_total_acc;
    for (double a : RIDGE_ALPHAS) alpha_total_acc[a] = 0.0;

    std::vector<double> repeat_accs;
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
        double rep_correct = 0;
        int rep_total = 0;

        for (int f = 0; f < N_CV_FOLDS; f++) {
            auto& split = folds[f];
            Mat X_train((int)split.train.size(), n_features);
            std::vector<int> y_train(split.train.size());
            for (int i = 0; i < (int)split.train.size(); i++) {
                for (int j = 0; j < n_features; j++)
                    X_train(i, j) = X_flat(split.train[i], j);
                y_train[i] = y[split.train[i]];
            }
            Mat X_test((int)split.test.size(), n_features);
            std::vector<int> y_test(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_features; j++)
                    X_test(i, j) = X_flat(split.test[i], j);
                y_test[i] = y[split.test[i]];
            }
            cls::StandardScaler scaler;
            X_train = scaler.fit_transform(X_train);
            X_test = scaler.transform(X_test);
            cls::nan_to_num(X_train);
            cls::nan_to_num(X_test);

            auto fold_ctx = cls::ridge_fold_prepare(X_train, y_train, X_test, y_test, DEFAULT_DIGITS);
            double best_acc = -1;
            std::vector<int> best_preds;
            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(fold_ctx, X_test, y_test, alpha);
                alpha_total_acc[alpha] += rr.accuracy;
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

    double best_alpha = RIDGE_ALPHAS[0];
    double best_total = -1;
    for (auto& [a, total] : alpha_total_acc) {
        if (total > best_total) { best_total = total; best_alpha = a; }
    }

    return {mean, std::sqrt(var), repeat_accs, best_alpha};
}

// Analysis 5: Weight concentration — temporal vs neural
struct WeightConcentration {
    double temporal;  // mean over classes of mean_bins(std_neurons(|W|))
    double neural;    // mean over classes of mean_neurons(std_bins(|W|))
};

static WeightConcentration compute_weight_concentration(
    const Mat& weights,  // (n_bins * n_res x n_classes)
    int n_bins, int n_res)
{
    int n_classes = weights.cols;
    double temporal_sum = 0, neural_sum = 0;

    for (int c = 0; c < n_classes; c++) {
        // Temporal: mean over bins of std(|W[b,:]|) across neurons
        double temp_accum = 0;
        for (int b = 0; b < n_bins; b++) {
            double mean = 0;
            for (int r = 0; r < n_res; r++)
                mean += std::abs(weights(b * n_res + r, c));
            mean /= n_res;
            double var = 0;
            for (int r = 0; r < n_res; r++) {
                double diff = std::abs(weights(b * n_res + r, c)) - mean;
                var += diff * diff;
            }
            temp_accum += std::sqrt(var / std::max(n_res - 1, 1));
        }
        temporal_sum += temp_accum / n_bins;

        // Neural: mean over neurons of std(|W[:,r]|) across bins
        double neur_accum = 0;
        for (int r = 0; r < n_res; r++) {
            double mean = 0;
            for (int b = 0; b < n_bins; b++)
                mean += std::abs(weights(b * n_res + r, c));
            mean /= n_bins;
            double var = 0;
            for (int b = 0; b < n_bins; b++) {
                double diff = std::abs(weights(b * n_res + r, c)) - mean;
                var += diff * diff;
            }
            neur_accum += std::sqrt(var / std::max(n_bins - 1, 1));
        }
        neural_sum += neur_accum / n_res;
    }

    return {temporal_sum / n_classes, neural_sum / n_classes};
}

// ============================================================
// MAIN ENTRY POINT
// ============================================================
int run_mechanistic_interp(int argc, char** argv,
                            int n_workers, const std::string& output_dir,
                            const std::string& data_dir) {
    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  MECHANISTIC INTERPRETABILITY EXPERIMENT\n");
    printf("  Grid points: %d (all at tau=5000, Branch B rate-matched)\n", (int)MECH_POINTS.size());
    printf("  Task: WM (digit_a from B+Post, diff-pairs only)\n");
    printf("  Pairs: %d per grid point\n", MECH_N_WM_PAIRS);
    printf("  Workers: %d\n", n_workers);
    printf("  CV: StratifiedKFold(%d) x %d repeats\n", N_CV_FOLDS, N_CV_REPEATS);
    if (!g_snapshot_path.empty())
        printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    printf("======================================================================\n");

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    printf("  %d samples loaded\n", (int)samples.size());

    // 2. Generate XOR pairs
    printf("\n[2] Generating XOR pairs...\n");
    auto pairs = mech_generate_xor_pairs(samples, MECH_N_WM_PAIRS, MECH_PAIR_SEED);
    int n_pairs = (int)pairs.size();
    int n_same = 0, n_diff = 0;
    for (auto& p : pairs) { if (p.label == 1) n_same++; else n_diff++; }
    printf("  %d pairs (%d same, %d different)\n", n_pairs, n_same, n_diff);

    double max_total_stim_ms = 0;
    for (auto& p : pairs) {
        if (p.total_stim_end_ms > max_total_stim_ms) max_total_stim_ms = p.total_stim_end_ms;
    }
    max_total_stim_ms += 5.0;

    // 3. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_total_stim_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    // Calibration subset
    std::vector<MechXorPair> cal_pairs(pairs.begin(),
                                        pairs.begin() + std::min(MECH_WM_CAL_N_PAIRS, n_pairs));

    double target_rate_hz = RATE_TARGET_HZ;

    // CLS sim config (single-digit samples, shorter duration than WM compound stimuli)
    double max_single_audio_ms = 0;
    for (auto& s : samples) {
        double end = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (end > max_single_audio_ms) max_single_audio_ms = end;
    }
    max_single_audio_ms += 5.0;

    SimConfig cls_sim_cfg = sim_cfg;
    cls_sim_cfg.audio_duration_ms = max_single_audio_ms;
    int cls_n_bins = (int)std::ceil((max_single_audio_ms + POST_STIM_MS) / BIN_MS);
    printf("  CLS sim config: audio=%.1fms, %d bins (for Analysis 5)\n",
           max_single_audio_ms, cls_n_bins);

    // Prepare digit labels for CLS samples
    std::vector<int> cls_y(samples.size());
    for (int i = 0; i < (int)samples.size(); i++) cls_y[i] = samples[i].digit;

    // Open output JSON
    std::string json_path = output_dir + "/mechanistic_interp.json";
    FILE* fout = fopen(json_path.c_str(), "w");
    if (!fout) { fprintf(stderr, "Cannot write %s\n", json_path.c_str()); return 1; }

    fprintf(fout, "{\n");
    fprintf(fout, "  \"experiment\": \"Mechanistic Interpretability: Stimulus-to-Readout Causal Chain\",\n");
    fprintf(fout, "  \"n_pairs\": %d,\n", n_pairs);
    fprintf(fout, "  \"n_diff_pairs\": %d,\n", n_diff);
    fprintf(fout, "  \"n_digits\": %d,\n", N_DIGITS);
    fprintf(fout, "  \"gap_ms\": %.1f,\n", MECH_GAP_MS);
    fprintf(fout, "  \"epoch_bins\": {\"n_bins_a\": %d, \"n_bins_gap\": %d, \"n_bins_b\": %d, \"n_bins_post\": %d},\n",
            MECH_N_BINS_A, MECH_N_BINS_GAP, MECH_N_BINS_B, MECH_N_BINS_POST);
    fprintf(fout, "  \"readout_window\": \"B+Post (bins %d-%d)\",\n", MECH_IDX_B_START, MECH_IDX_BPOST_END - 1);
    fprintf(fout, "  \"rate_matching\": {\"target_rate_hz\": %.2f, \"tolerance_hz\": %.1f},\n",
            target_rate_hz, RATE_TOLERANCE_HZ);

    fprintf(fout, "  \"grid_points\": [\n");
    for (size_t i = 0; i < MECH_POINTS.size(); i++) {
        if (i > 0) fprintf(fout, ",\n");
        fprintf(fout, "    {\"label\": \"%s\", \"adapt_inc\": %.4f, \"adapt_tau\": %.1f}",
                MECH_POINTS[i].label.c_str(), MECH_POINTS[i].adapt_inc, MECH_POINTS[i].adapt_tau);
    }
    fprintf(fout, "\n  ],\n");
    fprintf(fout, "  \"results\": [\n");

    // 4. Run each grid point
    for (int pt_num = 0; pt_num < (int)MECH_POINTS.size(); pt_num++) {
        auto& pt = MECH_POINTS[pt_num];
        double pt_start = now_seconds();

        printf("\n======================================================================\n");
        printf("  [%d/%d] %s: inc=%.4f, tau=%.1f\n",
               pt_num + 1, (int)MECH_POINTS.size(), pt.label.c_str(), pt.adapt_inc, pt.adapt_tau);
        printf("======================================================================\n");

        DynamicalOverrides dyn_ovr;
        dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn_ovr.adapt_inc = pt.adapt_inc;
        dyn_ovr.adapt_tau = pt.adapt_tau;
        dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

        // Calibrate
        printf("  [Calibrating]...\n");
        auto cal = mech_calibrate_tonic(base_cfg, dyn_ovr, cal_pairs, sim_cfg,
                                          n_workers, target_rate_hz);
        printf("  Calibrated: g_tonic=%.4f (%s) -> %.1f Hz\n",
               cal.g_tonic, (cal.reversal < -1.0) ? "inh" : "exc", cal.rate_hz);

        // Run all pairs
        printf("  [Simulating %d pairs]...\n", n_pairs);
        auto res = mech_run_all_pairs(base_cfg, pairs, dyn_ovr, sim_cfg, n_workers,
                                        cal.g_tonic, cal.reversal);

        int n_res = res.n_reservoir;
        printf("  n_reservoir=%d\n", n_res);

        // Per-digit vectors
        auto pdv = compute_per_digit_vectors(res, pairs);

        // Analysis 1: Selectivity correlation matrix
        printf("  [Analysis 1] Digit selectivity correlations...\n");
        auto sel_corr = compute_selectivity_corr_matrix(pdv);
        double within_sel = 0, between_sel = 0;
        int n_within = 0, n_between = 0;
        for (int i = 0; i < N_DIGITS; i++) {
            for (int j = 0; j < N_DIGITS; j++) {
                if (i == j) { within_sel += sel_corr[i][j]; n_within++; }
                else { between_sel += sel_corr[i][j]; n_between++; }
            }
        }
        within_sel /= n_within;
        between_sel /= n_between;
        printf("    Within-class r=%.4f, Between-class r=%.4f\n", within_sel, between_sel);

        // Analysis 2: Stimulus → adaptation correlation
        printf("  [Analysis 2] Stimulus-drive → adaptation correlation...\n");
        auto stim_adapt_r = compute_stimulus_adapt_corr(res, pairs);
        double stim_adapt_mean = 0;
        for (double r : stim_adapt_r) stim_adapt_mean += r;
        stim_adapt_mean /= N_DIGITS;
        printf("    Mean r(A_spikes, g_adapt_at_B) = %.4f\n", stim_adapt_mean);
        for (int d = 0; d < N_DIGITS; d++)
            printf("      digit %d: r=%.4f\n", d, stim_adapt_r[d]);

        // WM classification
        printf("  [WM Classification]...\n");
        auto wm_cls = mech_classify_wm_bpost(res.epoch_bins_list, res.digit_a_list,
                                               res.digit_b_list, n_res);
        printf("    WM accuracy: %.1f%% (std=%.3f, best_alpha=%.1f)\n",
               wm_cls.accuracy * 100, wm_cls.accuracy_std, wm_cls.best_alpha);

        // Extract ridge weights for WM (refit on all data with best alpha)
        printf("  [Ridge weight extraction]...\n");
        int n_feat = MECH_N_BPOST_BINS * n_res;
        std::vector<int> diff_indices;
        for (int i = 0; i < n_pairs; i++) {
            if (res.digit_a_list[i] != res.digit_b_list[i])
                diff_indices.push_back(i);
        }
        int n_diff_actual = (int)diff_indices.size();
        Mat X_wm(n_diff_actual, n_feat, 0.0);
        std::vector<int> y_wm(n_diff_actual);
        for (int di = 0; di < n_diff_actual; di++) {
            int idx = diff_indices[di];
            y_wm[di] = res.digit_a_list[idx];
            for (int b = 0; b < MECH_N_BPOST_BINS; b++) {
                int epoch_b = MECH_IDX_B_START + b;
                if (epoch_b < res.epoch_bins_list[idx].rows) {
                    for (int r = 0; r < std::min(n_res, res.epoch_bins_list[idx].cols); r++)
                        X_wm(di, b * n_res + r) = res.epoch_bins_list[idx](epoch_b, r);
                }
            }
        }
        auto wm_weights = cls::ridge_extract_weights(X_wm, y_wm, wm_cls.best_alpha, DEFAULT_DIGITS);
        printf("    Ridge weight matrix: %d x %d (training acc=%.1f%%)\n",
               wm_weights.weights.rows, wm_weights.weights.cols, wm_weights.accuracy * 100);

        // Analysis 3: Adaptation → ridge weights
        printf("  [Analysis 3] Adaptation → ridge weight correlations...\n");
        auto adapt_weight_corr = compute_weight_adapt_corr(wm_weights, pdv, n_res,
                                                             MECH_N_BPOST_BINS, true);
        printf("    Within-class r=%.4f, Cross-class r=%.4f\n",
               adapt_weight_corr.within_mean, adapt_weight_corr.cross_mean);

        // Analysis 4: Stimulus → ridge weights (closing the loop)
        printf("  [Analysis 4] Stimulus → ridge weight correlations...\n");
        auto stim_weight_corr = compute_weight_adapt_corr(wm_weights, pdv, n_res,
                                                            MECH_N_BPOST_BINS, false);
        printf("    Within-class r=%.4f, Cross-class r=%.4f\n",
               stim_weight_corr.within_mean, stim_weight_corr.cross_mean);

        // Analysis 6: Adaptation-state-only classifier
        printf("  [Analysis 6] Adaptation-state-only WM classifier...\n");
        auto adapt_cls = mech_classify_adapt_state(res.adapt_at_b_list, res.digit_a_list,
                                                     res.digit_b_list, n_res);
        printf("    Adapt-only accuracy: %.1f%% (vs spike-count: %.1f%%)\n",
               adapt_cls.accuracy * 100, wm_cls.accuracy * 100);

        // Analysis 7: Shell vs core
        printf("  [Analysis 7] Shell vs core stats...\n");
        auto sc_stats = compute_shell_core_stats(res, pdv);
        printf("    Shell: n=%d, adapt=%.6f, F=%.2f\n",
               sc_stats.n_shell, sc_stats.shell_adapt_mean, sc_stats.shell_f_stat);
        printf("    Core:  n=%d, adapt=%.6f, F=%.2f\n",
               sc_stats.n_core, sc_stats.core_adapt_mean, sc_stats.core_f_stat);

        // Analysis 5: CLS vs WM weight structure comparison (only at point 2 = cls_optimum)
        cls::RidgePrimalWeights cls_weights_result;
        WeightConcentration cls_conc = {0, 0}, wm_conc = {0, 0};
        MechWmClassifyResult cls_cls_result = {0, 0, {}, 0};
        bool has_analysis5 = (pt_num == 1);  // cls_optimum

        if (has_analysis5) {
            printf("  [Analysis 5] CLS vs WM weight structure...\n");

            // Run CLS simulation at this point's params + calibrated tonic conductance
            printf("    Running CLS simulation (%d samples)...\n", (int)samples.size());
            std::vector<Mat> cls_bins_list;
            std::vector<int> cls_digits;
            int cls_n_res = 0;
            mech_run_cls_samples(base_cfg, samples, dyn_ovr, cls_sim_cfg, n_workers,
                                  cal.g_tonic, cal.reversal,
                                  cls_bins_list, cls_digits, cls_n_res, cls_n_bins);

            // CLS classification
            printf("    CLS classification (%d bins x %d neurons)...\n", cls_n_bins, cls_n_res);
            cls_cls_result = mech_classify_cls_flat(cls_bins_list, cls_y, cls_n_bins, cls_n_res);
            printf("    CLS accuracy: %.1f%% (std=%.3f, best_alpha=%.1f)\n",
                   cls_cls_result.accuracy * 100, cls_cls_result.accuracy_std, cls_cls_result.best_alpha);

            // Extract CLS ridge weights (refit on all data with best alpha)
            int cls_n_feat = cls_n_bins * cls_n_res;
            Mat X_cls(int(cls_bins_list.size()), cls_n_feat, 0.0);
            for (int i = 0; i < (int)cls_bins_list.size(); i++) {
                for (int b = 0; b < std::min(cls_n_bins, cls_bins_list[i].rows); b++)
                    for (int r = 0; r < std::min(cls_n_res, cls_bins_list[i].cols); r++)
                        X_cls(i, b * cls_n_res + r) = cls_bins_list[i](b, r);
            }
            cls_weights_result = cls::ridge_extract_weights(X_cls, cls_y,
                                                              cls_cls_result.best_alpha, DEFAULT_DIGITS);
            printf("    CLS weight matrix: %d x %d\n",
                   cls_weights_result.weights.rows, cls_weights_result.weights.cols);

            // Compute concentration for both CLS and WM
            cls_conc = compute_weight_concentration(cls_weights_result.weights, cls_n_bins, cls_n_res);
            wm_conc = compute_weight_concentration(wm_weights.weights, MECH_N_BPOST_BINS, n_res);

            printf("    CLS concentration: temporal=%.6f, neural=%.6f\n",
                   cls_conc.temporal, cls_conc.neural);
            printf("    WM  concentration: temporal=%.6f, neural=%.6f\n",
                   wm_conc.temporal, wm_conc.neural);
        }

        double pt_elapsed = now_seconds() - pt_start;
        printf("  Point %d done in %.1f min\n", pt_num + 1, pt_elapsed / 60.0);

        // ============================================================
        // Write JSON for this grid point
        // ============================================================
        if (pt_num > 0) fprintf(fout, ",\n");
        fprintf(fout, "    {\n");
        fprintf(fout, "      \"label\": \"%s\",\n", pt.label.c_str());
        fprintf(fout, "      \"adapt_inc\": %.4f,\n", pt.adapt_inc);
        fprintf(fout, "      \"adapt_tau\": %.1f,\n", pt.adapt_tau);
        fprintf(fout, "      \"tonic_conductance\": %.6f,\n", cal.g_tonic);
        fprintf(fout, "      \"tonic_reversal\": %.1f,\n", cal.reversal);
        fprintf(fout, "      \"calibration_rate_hz\": %.1f,\n", cal.rate_hz);
        fprintf(fout, "      \"n_reservoir\": %d,\n", n_res);
        fprintf(fout, "      \"time_s\": %.1f,\n", pt_elapsed);

        // WM accuracy
        fprintf(fout, "      \"wm_accuracy\": %.10f,\n", wm_cls.accuracy);
        fprintf(fout, "      \"wm_accuracy_std\": %.6f,\n", wm_cls.accuracy_std);
        fprintf(fout, "      \"wm_best_alpha\": %.1f,\n", wm_cls.best_alpha);
        fprintf(fout, "      \"wm_per_repeat_accuracy\": ");
        json_write_vec(fout, wm_cls.per_repeat_accuracy, 10);
        fprintf(fout, ",\n");

        // Adapt-only accuracy
        fprintf(fout, "      \"adapt_state_wm_accuracy\": %.10f,\n", adapt_cls.accuracy);
        fprintf(fout, "      \"adapt_state_wm_accuracy_std\": %.6f,\n", adapt_cls.accuracy_std);
        fprintf(fout, "      \"adapt_state_per_repeat_accuracy\": ");
        json_write_vec(fout, adapt_cls.per_repeat_accuracy, 10);
        fprintf(fout, ",\n");

        // Shell/core
        fprintf(fout, "      \"n_shell\": %d,\n", sc_stats.n_shell);
        fprintf(fout, "      \"n_core\": %d,\n", sc_stats.n_core);
        fprintf(fout, "      \"shell_adapt_mean\": %.8f,\n", sc_stats.shell_adapt_mean);
        fprintf(fout, "      \"core_adapt_mean\": %.8f,\n", sc_stats.core_adapt_mean);
        fprintf(fout, "      \"shell_f_stat\": %.4f,\n", sc_stats.shell_f_stat);
        fprintf(fout, "      \"core_f_stat\": %.4f,\n", sc_stats.core_f_stat);

        // Analysis 1: selectivity correlation matrix
        fprintf(fout, "      \"selectivity_corr_matrix\": ");
        json_write_vec2d(fout, sel_corr, 4);
        fprintf(fout, ",\n");
        fprintf(fout, "      \"selectivity_within_mean\": %.6f,\n", within_sel);
        fprintf(fout, "      \"selectivity_between_mean\": %.6f,\n", between_sel);

        // Analysis 2: stimulus-adapt correlations per digit
        fprintf(fout, "      \"stim_adapt_r_per_digit\": ");
        json_write_vec(fout, stim_adapt_r, 6);
        fprintf(fout, ",\n");
        fprintf(fout, "      \"stim_adapt_r_mean\": %.6f,\n", stim_adapt_mean);

        // Analysis 3: adapt-weight correlations
        fprintf(fout, "      \"adapt_weight_within_mean\": %.6f,\n", adapt_weight_corr.within_mean);
        fprintf(fout, "      \"adapt_weight_cross_mean\": %.6f,\n", adapt_weight_corr.cross_mean);

        // Analysis 4: stim-weight correlations
        fprintf(fout, "      \"stim_weight_within_mean\": %.6f,\n", stim_weight_corr.within_mean);
        fprintf(fout, "      \"stim_weight_cross_mean\": %.6f,\n", stim_weight_corr.cross_mean);

        // Per-digit selectivity vectors (5 x n_res)
        fprintf(fout, "      \"digit_selectivity_vectors\": ");
        json_write_vec2d(fout, pdv.selectivity, 4);
        fprintf(fout, ",\n");

        // Per-digit adaptation at B onset (5 x n_res)
        fprintf(fout, "      \"mean_adapt_at_b_onset_by_digit\": ");
        json_write_vec2d(fout, pdv.adapt_mean, 8);
        fprintf(fout, ",\n");

        fprintf(fout, "      \"std_adapt_at_b_onset_by_digit\": ");
        json_write_vec2d(fout, pdv.adapt_std, 8);
        fprintf(fout, ",\n");

        // Ridge weights WM: (n_features x n_classes)
        fprintf(fout, "      \"ridge_weights_wm\": ");
        json_write_mat(fout, wm_weights.weights, 8);
        fprintf(fout, ",\n");

        // Analysis 5: CLS vs WM weight structure (only at cls_optimum)
        if (has_analysis5) {
            fprintf(fout, "      \"cls_accuracy\": %.10f,\n", cls_cls_result.accuracy);
            fprintf(fout, "      \"cls_accuracy_std\": %.6f,\n", cls_cls_result.accuracy_std);
            fprintf(fout, "      \"cls_best_alpha\": %.1f,\n", cls_cls_result.best_alpha);
            fprintf(fout, "      \"cls_per_repeat_accuracy\": ");
            json_write_vec(fout, cls_cls_result.per_repeat_accuracy, 10);
            fprintf(fout, ",\n");
            fprintf(fout, "      \"cls_n_bins\": %d,\n", cls_n_bins);
            fprintf(fout, "      \"cls_temporal_concentration\": %.8f,\n", cls_conc.temporal);
            fprintf(fout, "      \"cls_neural_concentration\": %.8f,\n", cls_conc.neural);
            fprintf(fout, "      \"wm_temporal_concentration\": %.8f,\n", wm_conc.temporal);
            fprintf(fout, "      \"wm_neural_concentration\": %.8f,\n", wm_conc.neural);
            fprintf(fout, "      \"ridge_weights_cls\": ");
            json_write_mat(fout, cls_weights_result.weights, 8);
            fprintf(fout, ",\n");
        }

        // Shell mask
        fprintf(fout, "      \"neuron_is_shell\": ");
        json_write_bool_vec(fout, res.neuron_is_shell);
        fprintf(fout, "\n");

        fprintf(fout, "    }");
    }

    double total_time = now_seconds() - total_start;

    fprintf(fout, "\n  ],\n");
    fprintf(fout, "  \"total_time_s\": %.1f\n", total_time);
    fprintf(fout, "}\n");
    fclose(fout);

    printf("\n======================================================================\n");
    printf("  MECHANISTIC INTERPRETABILITY COMPLETE\n");
    printf("  Total time: %.0fs (%.1f min)\n", total_time, total_time / 60.0);
    printf("  Output: %s\n", json_path.c_str());
    printf("======================================================================\n");

    return 0;
}

// ============================================================
// STANDALONE RASTER DUMP — runs 2 quick trials (no_adapt + wm_opt)
// ============================================================
int run_mech_raster(int n_workers, const std::string& output_dir,
                     const std::string& data_dir) {
    printf("======================================================================\n");
    printf("  MECHANISTIC INTERP — RASTER DUMP\n");
    printf("  Running 2 example trials (no_adapt + wm_optimum)\n");
    printf("======================================================================\n");

    // Load audio
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);

    // Generate pairs (same seed as main experiment)
    auto pairs = mech_generate_xor_pairs(samples, MECH_N_WM_PAIRS, MECH_PAIR_SEED);

    double max_total_stim_ms = 0;
    for (auto& p : pairs)
        if (p.total_stim_end_ms > max_total_stim_ms) max_total_stim_ms = p.total_stim_end_ms;
    max_total_stim_ms += 5.0;

    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_total_stim_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    // Calibration subset
    std::vector<MechXorPair> cal_pairs(pairs.begin(),
                                        pairs.begin() + std::min(MECH_WM_CAL_N_PAIRS, (int)pairs.size()));

    // Pick the first diff-pair as the example
    int example_pair_idx = -1;
    for (int i = 0; i < (int)pairs.size(); i++) {
        if (pairs[i].digit_a != pairs[i].digit_b) { example_pair_idx = i; break; }
    }
    if (example_pair_idx < 0) example_pair_idx = 0;
    auto& ep = pairs[example_pair_idx];

    printf("  Example pair: digit_a=%d, digit_b=%d, a_end=%.1fms, b_offset=%.1fms, b_end=%.1fms\n",
           ep.digit_a, ep.digit_b, ep.stim_a_end_ms, ep.stim_b_offset_ms, ep.stim_b_end_ms);

    fs::create_directories(output_dir);
    std::string raster_path = output_dir + "/example_rasters.json";
    FILE* rf = fopen(raster_path.c_str(), "w");
    if (!rf) { fprintf(stderr, "Cannot write %s\n", raster_path.c_str()); return 1; }

    fprintf(rf, "{\n");
    fprintf(rf, "  \"pair_idx\": %d,\n", example_pair_idx);
    fprintf(rf, "  \"digit_a\": %d,\n", ep.digit_a);
    fprintf(rf, "  \"digit_b\": %d,\n", ep.digit_b);
    fprintf(rf, "  \"stim_a_end_ms\": %.2f,\n", ep.stim_a_end_ms);
    fprintf(rf, "  \"stim_b_offset_ms\": %.2f,\n", ep.stim_b_offset_ms);
    fprintf(rf, "  \"stim_b_end_ms\": %.2f,\n", ep.stim_b_end_ms);
    fprintf(rf, "  \"total_stim_end_ms\": %.2f,\n", ep.total_stim_end_ms);
    fprintf(rf, "  \"dt\": %.2f,\n", sim_cfg.dt);
    fprintf(rf, "  \"post_stim_ms\": %.1f,\n", POST_STIM_MS);

    int raster_pts[] = {0, 3};
    const char* raster_labels[] = {"no_adapt", "wm_optimum"};

    fprintf(rf, "  \"rasters\": [\n");
    for (int ri = 0; ri < 2; ri++) {
        auto& rpt = MECH_POINTS[raster_pts[ri]];

        DynamicalOverrides rdyn;
        rdyn.shell_core_mult = LHS021_SHELL_CORE_MULT;
        rdyn.core_core_mult = LHS021_CORE_CORE_MULT;
        rdyn.adapt_inc = rpt.adapt_inc;
        rdyn.adapt_tau = rpt.adapt_tau;
        rdyn.nmda_tau = FIXED_NMDA_TAU;

        printf("  [%d/2] %s: calibrating...\n", ri + 1, raster_labels[ri]);
        auto rcal = mech_calibrate_tonic(base_cfg, rdyn, cal_pairs, sim_cfg,
                                           n_workers, RATE_TARGET_HZ);
        printf("    g_tonic=%.4f -> %.1f Hz\n", rcal.g_tonic, rcal.rate_hz);

        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
            apply_dynamical_overrides(net, zone_info, sim_cfg.dt, rdyn);
        } else {
            NetworkConfig cfg_copy = base_cfg;
            build_full_network(net, zone_info, cfg_copy, sim_cfg.dt, true,
                              &rdyn, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (rcal.g_tonic != 0.0)
            mech_set_tonic(net, zone_info, rcal.g_tonic, rcal.reversal);

        rng_seed(base_cfg.n_neurons + 999 + ri);

        SimConfig rsim = sim_cfg;
        rsim.audio_duration_ms = ep.total_stim_end_ms + 5.0;

        // Build adaptation snapshot times every 1 ms
        double total_trial_ms = rsim.audio_duration_ms + rsim.post_stimulus_ms;
        int n_adapt_snaps = (int)std::ceil(total_trial_ms);
        std::vector<double> adapt_snap_times;
        adapt_snap_times.reserve(n_adapt_snaps);
        for (int ms = 0; ms < n_adapt_snaps; ms++)
            adapt_snap_times.push_back(ms + 0.95);  // just before each ms boundary

        printf("    Simulating (%d adapt snapshots)...\n", n_adapt_snaps);
        auto result = run_sample_with_std(net, ep.compound, zone_info, rsim,
                                            STD_U, STD_TAU_REC, masks,
                                            adapt_snap_times);

        int n_res_local = (int)zone_info.reservoir_zone_indices.size();
        std::unordered_map<int, int> idx_to_pos;
        for (int p = 0; p < n_res_local; p++)
            idx_to_pos[zone_info.reservoir_zone_indices[p]] = p;

        // Reservoir-only spikes (backward compat: pos is reservoir-local 0..n_res-1)
        std::vector<std::pair<double, int>> spike_events;
        // All-neuron spikes (global neuron IDs for the 3D vis)
        std::vector<std::pair<double, int>> all_spike_events;
        double dt_val = sim_cfg.dt;
        for (int step = 0; step < (int)result.activity_record.size(); step++) {
            double t = step * dt_val;
            for (int nid : result.activity_record[step]) {
                all_spike_events.push_back({t, nid});
                auto it = idx_to_pos.find(nid);
                if (it != idx_to_pos.end())
                    spike_events.push_back({t, it->second});
            }
        }

        // Extract adaptation snapshots into time x neuron matrix
        int n_snaps_actual = std::min(n_adapt_snaps, (int)result.adapt_snapshots.size());

        if (ri > 0) fprintf(rf, ",\n");
        fprintf(rf, "    {\n");
        fprintf(rf, "      \"label\": \"%s\",\n", raster_labels[ri]);
        fprintf(rf, "      \"adapt_inc\": %.4f,\n", rpt.adapt_inc);
        fprintf(rf, "      \"adapt_tau\": %.1f,\n", rpt.adapt_tau);
        fprintf(rf, "      \"g_tonic\": %.6f,\n", rcal.g_tonic);
        fprintf(rf, "      \"rate_hz\": %.1f,\n", rcal.rate_hz);
        fprintf(rf, "      \"n_reservoir\": %d,\n", n_res_local);
        fprintf(rf, "      \"n_spikes\": %d,\n", (int)spike_events.size());
        fprintf(rf, "      \"n_adapt_snapshots\": %d,\n", n_snaps_actual);

        fprintf(rf, "      \"spike_times_ms\": [");
        for (size_t s = 0; s < spike_events.size(); s++) {
            if (s > 0) fprintf(rf, ",");
            if (s % 500 == 0 && s > 0) fprintf(rf, "\n        ");
            fprintf(rf, "%.1f", spike_events[s].first);
        }
        fprintf(rf, "],\n");

        fprintf(rf, "      \"spike_neuron_pos\": [");
        for (size_t s = 0; s < spike_events.size(); s++) {
            if (s > 0) fprintf(rf, ",");
            if (s % 500 == 0 && s > 0) fprintf(rf, "\n        ");
            fprintf(rf, "%d", spike_events[s].second);
        }
        fprintf(rf, "],\n");

        // All-neuron spikes (global IDs, includes input neurons)
        fprintf(rf, "      \"n_total\": %d,\n", net.n_neurons);
        fprintf(rf, "      \"all_spike_times_ms\": [");
        for (size_t s = 0; s < all_spike_events.size(); s++) {
            if (s > 0) fprintf(rf, ",");
            if (s % 500 == 0 && s > 0) fprintf(rf, "\n        ");
            fprintf(rf, "%.1f", all_spike_events[s].first);
        }
        fprintf(rf, "],\n");

        fprintf(rf, "      \"all_spike_neuron_ids\": [");
        for (size_t s = 0; s < all_spike_events.size(); s++) {
            if (s > 0) fprintf(rf, ",");
            if (s % 500 == 0 && s > 0) fprintf(rf, "\n        ");
            fprintf(rf, "%d", all_spike_events[s].second);
        }
        fprintf(rf, "],\n");

        // Dump adaptation snapshots as flat row-major array [time][neuron]
        fprintf(rf, "      \"adapt_snapshots\": [");
        bool first_val = true;
        for (int t = 0; t < n_snaps_actual; t++) {
            for (int p = 0; p < n_res_local; p++) {
                int nid = zone_info.reservoir_zone_indices[p];
                double val = (nid < (int)result.adapt_snapshots[t].size())
                             ? result.adapt_snapshots[t][nid] : 0.0;
                if (!first_val) fprintf(rf, ",");
                if ((t * n_res_local + p) % 1000 == 0 && !first_val)
                    fprintf(rf, "\n        ");
                fprintf(rf, "%.4f", val);
                first_val = false;
            }
        }
        fprintf(rf, "]\n");

        fprintf(rf, "    }");
        printf("    %s: %d spikes, %d adapt snapshots\n",
               raster_labels[ri], (int)spike_events.size(), n_snaps_actual);
    }

    fprintf(rf, "\n  ]\n");
    fprintf(rf, "}\n");
    fclose(rf);

    printf("\n  Rasters saved to %s\n", raster_path.c_str());
    return 0;
}
