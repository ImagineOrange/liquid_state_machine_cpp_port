#include "experiments.h"
#include <atomic>
#include <random>

// ============================================================
// GRID AXES — same inc values as classification.cpp
// ============================================================
static const std::vector<double> NOISY_INC = {
    0.0,
    0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000,
};

// Pilot: tau=5000 only. Full: all 15 values.
static const std::vector<double> NOISY_TAU_PILOT = {5000.0};
static const std::vector<double> NOISY_TAU_FULL = {
    30.0, 43.2, 62.3, 89.8, 129.4,
    186.5, 268.7, 387.3, 558.1, 804.4,
    1159.2, 1670.6, 2407.5, 3469.5, 5000.0,
};

static const std::vector<double> NOISE_LEVELS = {0.0, 0.25, 0.5, 0.75, 1.0};

// Deterministic noise seeds
static constexpr int NOISE_SEED_BASE = 777;

// ============================================================
// GRID POINT
// ============================================================
struct NoisyGridPoint {
    double adapt_inc, adapt_tau;
    std::string point_id;
    int inc_idx, tau_idx;
};

static std::vector<NoisyGridPoint> build_noisy_grid(const std::vector<double>& taus) {
    std::vector<NoisyGridPoint> points;
    points.reserve(NOISY_INC.size() * taus.size());
    for (int ii = 0; ii < (int)NOISY_INC.size(); ii++) {
        for (int ti = 0; ti < (int)taus.size(); ti++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "inc%.4f_tau%.1f",
                     NOISY_INC[ii], taus[ti]);
            points.push_back({NOISY_INC[ii], taus[ti], buf, ii, ti});
        }
    }
    return points;
}

// ============================================================
// NOISE CONSTRUCTION
// ============================================================

// Structured distractor: superimpose another digit's BSA spike train.
// Returns noisy samples and per-trial distractor digit for confusability analysis.
// Short distractors are looped (time-shifted copies) to fill the target duration,
// ensuring uniform noise coverage regardless of relative stimulus lengths.
static std::vector<AudioSample> make_structured_noise(
    const std::vector<AudioSample>& samples,
    double power_ratio,
    int seed,
    std::vector<int>& distractor_digits)
{
    distractor_digits.resize(samples.size(), -1);
    if (power_ratio <= 0.0) return samples;

    std::mt19937 rng(seed);

    // Group samples by digit
    std::map<int, std::vector<int>> by_digit;
    for (int i = 0; i < (int)samples.size(); i++)
        by_digit[samples[i].digit].push_back(i);

    std::vector<int> digits;
    for (auto& [d, _] : by_digit) digits.push_back(d);

    std::vector<AudioSample> noisy(samples.size());

    for (int i = 0; i < (int)samples.size(); i++) {
        const auto& target = samples[i];

        // Pick random distractor digit != target
        std::vector<int> other_digits;
        for (int d : digits) {
            if (d != target.digit) other_digits.push_back(d);
        }
        std::uniform_int_distribution<int> digit_dist(0, (int)other_digits.size() - 1);
        int dist_digit = other_digits[digit_dist(rng)];
        auto& dist_pool = by_digit[dist_digit];
        std::uniform_int_distribution<int> pool_dist(0, (int)dist_pool.size() - 1);
        const auto& distractor = samples[dist_pool[pool_dist(rng)]];

        distractor_digits[i] = dist_digit;

        // Start with all target spikes
        AudioSample compound;
        compound.digit = target.digit;
        compound.speaker = target.speaker;
        compound.filename = target.filename;
        compound.spike_times_ms = target.spike_times_ms;
        compound.freq_bin_indices = target.freq_bin_indices;

        // Target duration
        double target_dur = *std::max_element(target.spike_times_ms.begin(),
                                               target.spike_times_ms.end());

        // Distractor duration
        double dist_dur = *std::max_element(distractor.spike_times_ms.begin(),
                                              distractor.spike_times_ms.end());

        // Add distractor spikes, looping to fill target duration.
        // Each loop iteration shifts distractor by dist_dur to cover the full window.
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        double offset = 0.0;
        while (offset < target_dur) {
            for (size_t k = 0; k < distractor.spike_times_ms.size(); k++) {
                double t = distractor.spike_times_ms[k] + offset;
                if (t > target_dur) continue;
                if (unif(rng) < power_ratio) {
                    compound.spike_times_ms.push_back(t);
                    compound.freq_bin_indices.push_back(distractor.freq_bin_indices[k]);
                }
            }
            offset += dist_dur;
        }

        noisy[i] = std::move(compound);
    }

    return noisy;
}

// Unstructured noise: random spikes uniformly across frequency channels
static std::vector<AudioSample> make_unstructured_noise(
    const std::vector<AudioSample>& samples,
    double noise_ratio,
    double mean_bsa_count,
    int seed)
{
    if (noise_ratio <= 0.0) return samples;

    std::mt19937 rng(seed);
    std::vector<AudioSample> noisy(samples.size());

    for (int i = 0; i < (int)samples.size(); i++) {
        noisy[i] = samples[i];

        double stim_dur = *std::max_element(samples[i].spike_times_ms.begin(),
                                             samples[i].spike_times_ms.end());

        // Uniform random spikes need much higher power to degrade classification
        // (no digit-correlated structure to exploit)
        constexpr double UNSTRUCTURED_POWER_MULT = 10.0;
        int n_noise = (int)(mean_bsa_count * noise_ratio * UNSTRUCTURED_POWER_MULT);

        std::uniform_real_distribution<double> time_dist(0.0, stim_dur);
        std::uniform_int_distribution<int> chan_dist(0, 127);

        for (int j = 0; j < n_noise; j++) {
            noisy[i].spike_times_ms.push_back(time_dist(rng));
            noisy[i].freq_bin_indices.push_back(chan_dist(rng));
        }
    }

    return noisy;
}

// ============================================================
// ACTIVITY BINNING
// ============================================================
struct NoisyBinResult {
    Mat bins;
    int total_spikes;
};

static NoisyBinResult noisy_bin_activity(const std::vector<std::vector<int>>& activity_record,
                                          const std::vector<int>& reservoir_indices,
                                          double dt, double bin_ms, int n_bins) {
    int n_res = (int)reservoir_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_res; p++) idx_to_pos[reservoir_indices[p]] = p;

    Mat bins(n_bins, n_res, 0.0);
    int total = 0;
    int n_steps = (int)activity_record.size();

    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        int b = (int)(t / bin_ms);
        if (b >= n_bins) b = n_bins - 1;
        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end()) {
                bins(b, it->second) += 1.0;
                total++;
            }
        }
    }
    return {bins, total};
}

// ============================================================
// SIMULATION WORKER
// ============================================================
struct NoisyWorkerResult {
    Mat res_bins;
    int total_res_spikes;
    int digit;
    int sample_idx;
    double isi_cv;
    double mean_adapt_stim_end;
};

static NoisyWorkerResult noisy_sim_worker(SphericalNetwork& net, const AudioSample& sample,
                                           int sample_idx, const ZoneInfo& zone_info,
                                           const SimConfig& sim_cfg, const StdMasks& masks) {
    double dt = sim_cfg.dt;
    double total_ms = sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms;
    int n_bins = (int)std::ceil(total_ms / BIN_MS);

    double stim_end_ms = *std::max_element(sample.spike_times_ms.begin(),
                                            sample.spike_times_ms.end());

    auto result = run_sample_with_std(net, sample, zone_info, sim_cfg,
                                       STD_U, STD_TAU_REC, masks, {stim_end_ms});

    auto br = noisy_bin_activity(result.activity_record, zone_info.reservoir_zone_indices,
                                  dt, BIN_MS, n_bins);

    // ISI CV during stimulus
    int stim_end_step = std::min((int)(stim_end_ms / dt), (int)result.activity_record.size());
    std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                           zone_info.reservoir_zone_indices.end());

    std::unordered_map<int, std::vector<double>> spike_times;
    for (int step = 0; step < stim_end_step; step++) {
        double t = step * dt;
        for (int nid : result.activity_record[step]) {
            if (res_set.count(nid)) spike_times[nid].push_back(t);
        }
    }

    std::vector<double> all_isis;
    for (auto& [nid, times] : spike_times) {
        for (size_t i = 1; i < times.size(); i++)
            all_isis.push_back(times[i] - times[i-1]);
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

    // Mean adaptation at stim end
    double adapt_mean = 0;
    if (!result.adapt_snapshots.empty() && !result.adapt_snapshots[0].empty()) {
        for (int nid : zone_info.reservoir_zone_indices)
            adapt_mean += result.adapt_snapshots[0][nid];
        adapt_mean /= zone_info.reservoir_zone_indices.size();
    }

    return {br.bins, br.total_spikes, sample.digit, sample_idx, isi_cv, adapt_mean};
}

// ============================================================
// TONIC CONDUCTANCE — reservoir only
// ============================================================
static void noisy_set_tonic(SphericalNetwork& net, const ZoneInfo& zone_info,
                             double g_tonic, double reversal) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zone_info.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g_tonic;
        net.tonic_reversal[nid] = reversal;
    }
}

// ============================================================
// RUN ALL SAMPLES (OpenMP parallel)
// ============================================================
struct NoisyAllResult {
    std::vector<Mat> res_bins_list;
    std::vector<int> digits;
    std::vector<double> total_spikes;
    int n_reservoir;
    std::vector<double> isi_cvs;
    std::vector<double> adapt_stim_ends;
};

static NoisyAllResult noisy_run_all(const NetworkConfig& cfg,
                                     const std::vector<AudioSample>& samples,
                                     const DynamicalOverrides& dyn_ovr,
                                     const SimConfig& sim_cfg,
                                     int n_workers,
                                     bool verbose = true,
                                     double g_tonic = 0.0,
                                     double tonic_rev = -80.0) {
    int n_samples = (int)samples.size();
    NoisyAllResult out;
    out.res_bins_list.resize(n_samples);
    out.digits.resize(n_samples);
    out.total_spikes.resize(n_samples);
    out.isi_cvs.resize(n_samples);
    out.adapt_stim_ends.resize(n_samples);
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
            noisy_set_tonic(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp single
        {
            out.n_reservoir = (int)zone_info.reservoir_zone_indices.size();
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_samples; i++) {
            auto wr = noisy_sim_worker(net, samples[i], i, zone_info, sim_cfg, masks);
            out.res_bins_list[i] = std::move(wr.res_bins);
            out.digits[i] = wr.digit;
            out.total_spikes[i] = wr.total_res_spikes;
            out.isi_cvs[i] = wr.isi_cv;
            out.adapt_stim_ends[i] = wr.mean_adapt_stim_end;

            int c = ++completed;
            if (verbose && (c % 100 == 0 || c == n_samples)) {
                #pragma omp critical
                printf("        %5d/%d\n", c, n_samples);
            }
        }
    }

    return out;
}

// ============================================================
// RATE MEASUREMENT
// ============================================================
static double noisy_measure_rate(const NetworkConfig& cfg,
                                  const std::vector<AudioSample>& samples_subset,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers,
                                  double g_tonic = 0.0, double tonic_rev = -80.0) {
    auto res = noisy_run_all(cfg, samples_subset, dyn_ovr, sim_cfg, n_workers, false, g_tonic, tonic_rev);
    double trial_dur_s = (sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms) / 1000.0;
    double sum = 0;
    for (int i = 0; i < (int)samples_subset.size(); i++) {
        sum += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
    }
    return sum / samples_subset.size();
}

// ============================================================
// CALIBRATION — TONIC CONDUCTANCE
// ============================================================
static constexpr double NOISY_G_TONIC_HI = 10.0;

struct NoisyCalResult {
    double g_tonic;
    double reversal;
    double rate_hz;
};

static NoisyCalResult
noisy_calibrate_tonic(const NetworkConfig& cfg,
                       const DynamicalOverrides& dyn_ovr,
                       const std::vector<AudioSample>& cal_samples,
                       const SimConfig& sim_cfg, int n_workers,
                       double target_rate,
                       double initial_guess = 0.0,
                       double prev_reversal = -80.0) {
    // Measure natural rate
    double natural_rate = noisy_measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers, 0.0);
    printf("    cal[0] g=0 (natural) -> %.1f Hz (target=%.1f)\n", natural_rate, target_rate);

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate};

    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    const char* mode = (reversal < -1.0) ? "inhibitory" : "excitatory";
    printf("    cal: natural=%.1f Hz, target=%.1f Hz -> %s mode\n",
           natural_rate, target_rate, mode);

    double lo = 0.0, hi = NOISY_G_TONIC_HI;
    int iteration = 1;

    if (initial_guess > 0.0 && prev_reversal == reversal) {
        double rate = noisy_measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers,
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
        double rate = noisy_measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers, mid, reversal);
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
// CLASSIFICATION — flat ridge (all bins concatenated)
// ============================================================
struct NoisyClassifyResult {
    double accuracy;
    double accuracy_std;
    std::vector<double> per_repeat_accuracy;
};

static NoisyClassifyResult noisy_classify_flat(const std::vector<Mat>& bins_list,
                                                const std::vector<int>& y,
                                                int n_bins, int n_features_per_bin) {
    int n_samples = (int)bins_list.size();
    int n_features = n_bins * n_features_per_bin;

    Mat X_flat(n_samples, n_features, 0.0);
    for (int i = 0; i < n_samples; i++) {
        int actual_bins = bins_list[i].rows;
        int actual_feat = bins_list[i].cols;
        for (int b = 0; b < std::min(n_bins, actual_bins); b++) {
            for (int r = 0; r < std::min(n_features_per_bin, actual_feat); r++) {
                X_flat(i, b * n_features_per_bin + r) = bins_list[i](b, r);
            }
        }
    }

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
            std::vector<int> y_test_vec(split.test.size());
            for (int i = 0; i < (int)split.test.size(); i++) {
                for (int j = 0; j < n_features; j++)
                    X_test(i, j) = X_flat(split.test[i], j);
                y_test_vec[i] = y[split.test[i]];
            }

            cls::StandardScaler scaler;
            X_train = scaler.fit_transform(X_train);
            X_test = scaler.transform(X_test);
            cls::nan_to_num(X_train);
            cls::nan_to_num(X_test);

            auto fold_ctx = cls::ridge_fold_prepare(X_train, y_train, X_test, y_test_vec,
                                                     DEFAULT_DIGITS);

            double best_acc = -1;
            std::vector<int> best_preds;

            for (double alpha : RIDGE_ALPHAS) {
                auto rr = cls::ridge_fold_solve(fold_ctx, X_test, y_test_vec, alpha);
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
// CLASSIFICATION — per-bin accuracy
// ============================================================
static std::vector<double> noisy_classify_per_bin(const std::vector<Mat>& bins_list,
                                                   const std::vector<int>& y,
                                                   int n_bins, int n_reservoir) {
    int n_samples = (int)bins_list.size();
    std::vector<double> bin_accs(n_bins, 0.0);

    for (int b = 0; b < n_bins; b++) {
        Mat X_bin(n_samples, n_reservoir, 0.0);
        for (int i = 0; i < n_samples; i++) {
            if (b < bins_list[i].rows) {
                for (int r = 0; r < std::min(n_reservoir, bins_list[i].cols); r++) {
                    X_bin(i, r) = bins_list[i](b, r);
                }
            }
        }

        std::vector<double> rep_accs;
        for (int rep = 0; rep < N_CV_REPEATS; rep++) {
            auto folds = cls::stratified_kfold(y, N_CV_FOLDS, SEED + rep);
            double rep_correct = 0;
            int rep_total = 0;

            for (int f = 0; f < N_CV_FOLDS; f++) {
                auto& split = folds[f];

                Mat Xtr((int)split.train.size(), n_reservoir);
                std::vector<int> ytr(split.train.size());
                for (int i = 0; i < (int)split.train.size(); i++) {
                    for (int j = 0; j < n_reservoir; j++) Xtr(i, j) = X_bin(split.train[i], j);
                    ytr[i] = y[split.train[i]];
                }
                Mat Xte((int)split.test.size(), n_reservoir);
                std::vector<int> yte(split.test.size());
                for (int i = 0; i < (int)split.test.size(); i++) {
                    for (int j = 0; j < n_reservoir; j++) Xte(i, j) = X_bin(split.test[i], j);
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
// CLASSIFICATION — last stimulus bin only (per-sample)
// ============================================================
// Per-sample last stimulus bin: each sample uses its own last bin based on
// actual stimulus duration, avoiding the confound of reading post-stimulus
// activity for short stimuli when using a global max_audio_ms bin index.
static NoisyClassifyResult noisy_classify_last_bin(const std::vector<Mat>& bins_list,
                                                    const std::vector<int>& y,
                                                    const std::vector<int>& per_sample_last_bin,
                                                    int n_reservoir) {
    std::vector<Mat> last_bins(bins_list.size());
    for (int i = 0; i < (int)bins_list.size(); i++) {
        last_bins[i] = Mat(1, n_reservoir, 0.0);
        int b = std::min(per_sample_last_bin[i], bins_list[i].rows - 1);
        if (b >= 0) {
            for (int r = 0; r < std::min(n_reservoir, bins_list[i].cols); r++) {
                last_bins[i](0, r) = bins_list[i](b, r);
            }
        }
    }
    return noisy_classify_flat(last_bins, y, 1, n_reservoir);
}

// ============================================================
// BSA FEATURE EXTRACTION
// ============================================================
static Mat noisy_extract_bsa_bins(const AudioSample& sample, double total_duration_ms) {
    int n_channels = 128;
    int n_bins = (int)std::ceil(total_duration_ms / BIN_MS);
    Mat bins(n_bins, n_channels, 0.0);
    for (size_t k = 0; k < sample.spike_times_ms.size(); k++) {
        int b = (int)(sample.spike_times_ms[k] / BIN_MS);
        int ch = sample.freq_bin_indices[k];
        if (b >= 0 && b < n_bins && ch >= 0 && ch < n_channels) {
            bins(b, ch) += 1.0;
        }
    }
    return bins;
}

// ============================================================
// JSON CHECKPOINT
// ============================================================
static void noisy_save_checkpoint(const std::string& results_dir,
                                   const std::vector<std::string>& grid_results_json,
                                   const std::vector<std::string>& bsa_baselines_json,
                                   const std::vector<double>& taus,
                                   double target_rate_hz,
                                   double total_start_time,
                                   bool final_save) {
    std::string fname = final_save ? "noisy_classification_sweep.json"
                                   : "noisy_classification_sweep_checkpoint.json";
    std::string path = results_dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    double elapsed = now_seconds() - total_start_time;

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Noisy Classification Adaptation Sweep (C++ Port)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", elapsed);
    fprintf(f, "  \"task\": \"5-class digit classification under noise\",\n");
    fprintf(f, "  \"digits\": [0, 1, 2, 3, 4],\n");
    fprintf(f, "  \"n_samples\": %d,\n", SAMPLES_PER_DIGIT * N_DIGITS);
    fprintf(f, "  \"noise_levels\": "); json_write_double_array(f, NOISE_LEVELS); fprintf(f, ",\n");
    fprintf(f, "  \"noise_types\": [\"structured\"],\n");
    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.2f, \"tolerance_hz\": %.1f},\n",
            target_rate_hz, RATE_TOLERANCE_HZ);

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); json_write_double_array(f, NOISY_INC); fprintf(f, ",\n");
    fprintf(f, "    \"unified_tau\": "); json_write_double_array(f, taus); fprintf(f, ",\n");
    fprintf(f, "    \"n_inc\": %d, \"n_tau\": %d\n", (int)NOISY_INC.size(), (int)taus.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"bsa_baselines\": [\n");
    for (size_t i = 0; i < bsa_baselines_json.size(); i++) {
        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    %s", bsa_baselines_json[i].c_str());
    }
    fprintf(f, "\n  ],\n");

    fprintf(f, "  \"grid_results\": [\n");
    for (size_t i = 0; i < grid_results_json.size(); i++) {
        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    %s", grid_results_json[i].c_str());
    }
    fprintf(f, "\n  ]\n");
    fprintf(f, "}\n");
    fclose(f);

    if (final_save) printf("\n  Saved: %s\n", path.c_str());
}

// ============================================================
// MAIN ENTRY POINT
// ============================================================
int run_noisy_sweep(int argc, char** argv,
                     int n_workers, const std::string& output_dir,
                     const std::string& data_dir,
                     const std::vector<double>& tau_values,
                     bool do_per_bin) {
    const auto& taus = tau_values;
    auto grid_points = build_noisy_grid(taus);
    int n_grid = (int)grid_points.size();

    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  NOISY CLASSIFICATION ADAPTATION SWEEP (C++ PORT)\n");
    printf("  Grid: %d inc x %d tau = %d SFA points x 2 branches\n",
           (int)NOISY_INC.size(), (int)taus.size(), n_grid);
    printf("  Noise conditions: structured distractor\n");
    printf("  Noise levels: %d values per condition\n", (int)NOISE_LEVELS.size());
    printf("  Task: 5-class digit classification (digits 0-4)\n");
    printf("  Samples: %d per digit = %d total\n", SAMPLES_PER_DIGIT, SAMPLES_PER_DIGIT * N_DIGITS);
    printf("  Readouts: All-bins CLS, Last-1-bin (per-sample)%s\n",
           do_per_bin ? ", Per-bin" : "");
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

    std::vector<int> y(n_samples);
    for (int i = 0; i < n_samples; i++) y[i] = samples[i].digit;

    // Compute mean BSA spike count (for unstructured noise calibration)
    double mean_bsa_count = 0;
    for (auto& s : samples)
        mean_bsa_count += s.spike_times_ms.size();
    mean_bsa_count /= n_samples;
    printf("  Mean BSA spike count: %.1f\n", mean_bsa_count);

    // Max audio duration
    double max_audio_ms = 0;
    for (auto& s : samples) {
        double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (mx > max_audio_ms) max_audio_ms = mx;
    }
    max_audio_ms += 5.0;

    // Bin indices for readouts
    double total_sim_ms = max_audio_ms + POST_STIM_MS;
    int total_bins = (int)std::ceil(total_sim_ms / BIN_MS);
    int cls_n_bins = total_bins;

    printf("  Max audio: %.1fms | Total sim: %.1fms | Bins: %d\n",
           max_audio_ms, total_sim_ms, total_bins);

    // 2. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_audio_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    double target_rate_hz = RATE_TARGET_HZ;

    // Noise conditions
    const std::vector<std::string> noise_types = {"structured"};

    std::vector<std::string> grid_results_json;
    std::vector<std::string> bsa_baselines_json;
    std::vector<double> grid_point_times;
    double sweep_start = now_seconds();
    int total_evaluations = 0;
    int total_evals_expected = (int)noise_types.size() * (int)NOISE_LEVELS.size() * n_grid * 2;

    printf("\n[2] Starting sweep: %d noise conditions x %d levels x %d grid x 2 branches = %d evaluations\n",
           (int)noise_types.size(), (int)NOISE_LEVELS.size(), n_grid, total_evals_expected);

    // --- Phase 1: Precompute BSA baselines for ALL noise conditions ---
    struct NoiseConditionData {
        std::vector<AudioSample> noisy_samples;
        std::vector<int> distractor_digits;
        std::vector<int> noisy_per_sample_last_bin;
        NoisyClassifyResult bsa_cls;
        NoisyClassifyResult bsa_last;
        double noisy_mean_spikes;
    };

    // Indexed as precomputed[nt][nl]
    std::vector<std::vector<NoiseConditionData>> precomputed(
        noise_types.size(), std::vector<NoiseConditionData>(NOISE_LEVELS.size()));

    printf("\n[2a] Computing BSA baselines for all %d noise conditions...\n",
           (int)(noise_types.size() * NOISE_LEVELS.size()));

    for (int nt = 0; nt < (int)noise_types.size(); nt++) {
        const std::string& noise_type = noise_types[nt];
        for (int nl = 0; nl < (int)NOISE_LEVELS.size(); nl++) {
            double noise_level = NOISE_LEVELS[nl];
            int noise_seed = NOISE_SEED_BASE + nt * 100 + nl * 10;

            auto& cond = precomputed[nt][nl];

            // Create noisy samples (structured distractor only)
            cond.noisy_samples = make_structured_noise(samples, noise_level, noise_seed, cond.distractor_digits);

            // Mean spike count
            cond.noisy_mean_spikes = 0;
            for (auto& s : cond.noisy_samples) cond.noisy_mean_spikes += s.spike_times_ms.size();
            cond.noisy_mean_spikes /= cond.noisy_samples.size();

            // Per-sample last stim bin
            double bsa_total_dur = max_audio_ms + POST_STIM_MS;
            cond.noisy_per_sample_last_bin.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                double stim_end = *std::max_element(cond.noisy_samples[i].spike_times_ms.begin(),
                                                     cond.noisy_samples[i].spike_times_ms.end());
                int lb = (int)(stim_end / BIN_MS);
                cond.noisy_per_sample_last_bin[i] = std::max(0, std::min(lb, total_bins - 1));
            }

            // BSA binning and classification
            std::vector<Mat> bsa_bins_list(n_samples);
            for (int i = 0; i < n_samples; i++)
                bsa_bins_list[i] = noisy_extract_bsa_bins(cond.noisy_samples[i], bsa_total_dur);

            cond.bsa_cls = noisy_classify_flat(bsa_bins_list, y, cls_n_bins, 128);
            cond.bsa_last = noisy_classify_last_bin(bsa_bins_list, y, cond.noisy_per_sample_last_bin, 128);

            // Save BSA baseline JSON
            {
                std::ostringstream oss;
                oss << std::fixed;
                oss.precision(10);
                oss << "{\"noise_type\": \"" << noise_type << "\", ";
                oss << "\"noise_level\": " << noise_level << ", ";
                oss << "\"bsa_cls_accuracy\": " << cond.bsa_cls.accuracy << ", ";
                oss.precision(6);
                oss << "\"bsa_cls_accuracy_std\": " << cond.bsa_cls.accuracy_std << ", ";
                oss.precision(10);
                oss << "\"bsa_last_bin_accuracy\": " << cond.bsa_last.accuracy << ", ";
                oss.precision(6);
                oss << "\"bsa_last_bin_accuracy_std\": " << cond.bsa_last.accuracy_std << ", ";
                oss << "\"mean_spike_count\": " << cond.noisy_mean_spikes << "}";
                bsa_baselines_json.push_back(oss.str());
            }
        }
    }

    // Print BSA baseline summary table
    printf("\n  %-14s  %5s  %9s  %7s  %9s  %7s  %10s\n",
           "Noise Type", "Level", "CLS Acc", "  +/-", "Last Acc", "  +/-", "Mean Spks");
    printf("  %-14s  %5s  %9s  %7s  %9s  %7s  %10s\n",
           "--------------", "-----", "---------", "-------", "---------", "-------", "----------");
    for (int nt = 0; nt < (int)noise_types.size(); nt++) {
        for (int nl = 0; nl < (int)NOISE_LEVELS.size(); nl++) {
            auto& c = precomputed[nt][nl];
            printf("  %-14s  %5.2f  %8.1f%%  %6.1f%%  %8.1f%%  %6.1f%%  %10.1f\n",
                   noise_types[nt].c_str(), NOISE_LEVELS[nl],
                   c.bsa_cls.accuracy * 100, c.bsa_cls.accuracy_std * 100,
                   c.bsa_last.accuracy * 100, c.bsa_last.accuracy_std * 100,
                   c.noisy_mean_spikes);
        }
    }
    printf("\n[2b] BSA baselines complete. Starting grid sweep...\n");

    // --- Phase 2: Grid sweep using precomputed data ---
    for (int nt = 0; nt < (int)noise_types.size(); nt++) {
        const std::string& noise_type = noise_types[nt];

        printf("\n======== NOISE CONDITION: %s ========\n", noise_type.c_str());

        for (int nl = 0; nl < (int)NOISE_LEVELS.size(); nl++) {
            double noise_level = NOISE_LEVELS[nl];

            printf("\n  ---- %s noise_level=%.2f ----\n", noise_type.c_str(), noise_level);

            auto& cond = precomputed[nt][nl];
            auto& noisy_samples = cond.noisy_samples;
            auto& distractor_digits = cond.distractor_digits;
            auto& noisy_per_sample_last_bin = cond.noisy_per_sample_last_bin;
            auto& bsa_cls = cond.bsa_cls;

            printf("  Mean spike count: %.1f (clean: %.1f, ratio: %.2f)\n",
                   cond.noisy_mean_spikes, mean_bsa_count, cond.noisy_mean_spikes / mean_bsa_count);

            // Calibration subset (stratified, from noisy samples)
            std::vector<AudioSample> cal_samples;
            {
                int per_digit = std::max(1, CALIBRATION_N_SAMPLES / N_DIGITS);
                std::map<int, int> digit_count;
                for (auto& s : noisy_samples) {
                    if ((int)cal_samples.size() >= CALIBRATION_N_SAMPLES) break;
                    if (digit_count[s.digit] < per_digit) {
                        cal_samples.push_back(s);
                        digit_count[s.digit]++;
                    }
                }
            }

            // Warm-start calibration cache (per tau)
            std::map<int, std::pair<double,double>> last_cal_by_tau;

            // 2d. Grid sweep at this noise level
            for (int pt_num = 0; pt_num < n_grid; pt_num++) {
                auto& pt = grid_points[pt_num];

                double gp_start = now_seconds();

                std::string eta_str = "calculating...";
                if (!grid_point_times.empty()) {
                    int last_n = std::min((int)grid_point_times.size(), 10);
                    double avg_t = 0;
                    for (int i = (int)grid_point_times.size() - last_n; i < (int)grid_point_times.size(); i++)
                        avg_t += grid_point_times[i];
                    avg_t /= last_n;
                    double remaining = total_evals_expected - total_evaluations;
                    double eta_h = avg_t * remaining / 2.0 / 3600.0;  // /2 because 2 branches per point time
                    char buf[64];
                    snprintf(buf, sizeof(buf), "%.1fh", eta_h);
                    eta_str = buf;
                }

                printf("\n  [%s nl=%.2f] [%d/%d] inc=%.4f, tau=%.1fms  |  ETA: %s\n",
                       noise_type.c_str(), noise_level,
                       pt_num + 1, n_grid, pt.adapt_inc, pt.adapt_tau, eta_str.c_str());

                DynamicalOverrides dyn_ovr;
                dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
                dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
                dyn_ovr.adapt_inc = pt.adapt_inc;
                dyn_ovr.adapt_tau = pt.adapt_tau;
                dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

                double trial_dur_s = (sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms) / 1000.0;

                // Lambda: evaluate one branch
                auto evaluate_branch = [&](const std::string& branch,
                                            double cal_value,
                                            double cal_rate,
                                            double tonic_rev = -80.0) -> std::string {
                    double t0 = now_seconds();
                    double gt = (branch == "B_matched") ? cal_value : 0.0;
                    auto res = noisy_run_all(base_cfg, noisy_samples, dyn_ovr, sim_cfg,
                                             n_workers, true, gt, tonic_rev);
                    double sim_time = now_seconds() - t0;

                    // Rate stats
                    double rate_mean = 0, rate_std = 0;
                    for (int i = 0; i < n_samples; i++) {
                        double r = res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
                        rate_mean += r;
                    }
                    rate_mean /= n_samples;
                    for (int i = 0; i < n_samples; i++) {
                        double r = res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
                        rate_std += (r - rate_mean) * (r - rate_mean);
                    }
                    rate_std = std::sqrt(rate_std / n_samples);

                    // All-bins CLS (standard window)
                    auto cls_res = noisy_classify_flat(res.res_bins_list, y, cls_n_bins, res.n_reservoir);
                    auto cls_stats = compute_paired_stats(cls_res.per_repeat_accuracy,
                                                          bsa_cls.per_repeat_accuracy,
                                                          cls_res.accuracy, bsa_cls.accuracy);

                    // Last-1-bin CLS (per-sample last stimulus bin)
                    auto last_res = noisy_classify_last_bin(res.res_bins_list, y,
                                                             noisy_per_sample_last_bin, res.n_reservoir);

                    // Per-bin accuracy (expensive — gated behind do_per_bin)
                    std::vector<double> per_bin_acc;
                    if (do_per_bin) {
                        per_bin_acc = noisy_classify_per_bin(res.res_bins_list, y,
                                                             total_bins, res.n_reservoir);
                    }

                    // ISI CV
                    double isi_cv_mean = 0;
                    int isi_valid = 0;
                    for (int i = 0; i < n_samples; i++) {
                        if (!std::isnan(res.isi_cvs[i])) { isi_cv_mean += res.isi_cvs[i]; isi_valid++; }
                    }
                    if (isi_valid > 0) isi_cv_mean /= isi_valid;

                    // Adaptation at stim end
                    double adapt_mean = 0;
                    for (int i = 0; i < n_samples; i++) adapt_mean += res.adapt_stim_ends[i];
                    adapt_mean /= n_samples;

                    printf("    [%s] Rate: %.1f Hz | CLS: %.1f%% (gap=%+.1fpp) | Last: %.1f%%\n",
                           branch.c_str(), rate_mean,
                           cls_res.accuracy * 100, cls_stats.gap_pp,
                           last_res.accuracy * 100);

                    // Build JSON
                    std::ostringstream oss;
                    oss << "{";
                    oss << "\"noise_type\": \"" << noise_type << "\", ";
                    oss << std::fixed;
                    oss.precision(4);
                    oss << "\"noise_level\": " << noise_level << ", ";
                    oss << "\"branch\": \"" << branch << "\", ";
                    oss << "\"point_id\": \"" << pt.point_id << "\", ";
                    oss << "\"inc_idx\": " << pt.inc_idx << ", ";
                    oss << "\"tau_idx\": " << pt.tau_idx << ", ";
                    oss.precision(10);
                    oss << "\"adapt_inc\": " << pt.adapt_inc << ", ";
                    oss << "\"adapt_tau\": " << pt.adapt_tau << ", ";
                    oss << "\"stimulus_current\": " << sim_cfg.stimulus_current << ", ";
                    oss << "\"tonic_conductance\": " << gt << ", ";
                    oss << "\"tonic_reversal\": " << tonic_rev << ", ";
                    oss << "\"calibration_rate_hz\": " << cal_rate << ", ";

                    // CLS readout
                    oss << "\"classification_accuracy\": " << cls_res.accuracy << ", ";
                    oss.precision(6);
                    oss << "\"classification_accuracy_std\": " << cls_res.accuracy_std << ", ";
                    oss << "\"cls_bsa_gap_pp\": " << cls_stats.gap_pp << ", ";
                    oss << "\"cls_bsa_p_value\": " << cls_stats.p_value << ", ";
                    oss << "\"cls_bsa_stars\": \"" << cls_stats.stars << "\", ";

                    // Last-1-bin readout
                    oss.precision(10);
                    oss << "\"last_bin_accuracy\": " << last_res.accuracy << ", ";
                    oss.precision(6);
                    oss << "\"last_bin_accuracy_std\": " << last_res.accuracy_std << ", ";

                    // Dynamics
                    oss << "\"firing_rate_hz\": " << rate_mean << ", ";
                    oss << "\"firing_rate_std\": " << rate_std << ", ";
                    oss << "\"n_reservoir\": " << res.n_reservoir << ", ";
                    oss << "\"sim_time_s\": " << sim_time << ", ";
                    if (isi_valid > 0) oss << "\"isi_cv_mean\": " << isi_cv_mean << ", ";
                    else oss << "\"isi_cv_mean\": null, ";
                    oss << "\"adapt_at_stim_end\": " << adapt_mean << ", ";

                    // Per-bin accuracy (only if computed)
                    if (!per_bin_acc.empty()) {
                        oss << "\"per_bin_accuracy\": [";
                        for (size_t i = 0; i < per_bin_acc.size(); i++) {
                            if (i > 0) oss << ", ";
                            oss << per_bin_acc[i];
                        }
                        oss << "], ";
                    }

                    // Distractor digit distribution (structured only, for confusability analysis)
                    if (noise_type == "structured" && !distractor_digits.empty()) {
                        // Count distractor digit frequencies
                        std::map<int, int> dist_counts;
                        for (int dd : distractor_digits) {
                            if (dd >= 0) dist_counts[dd]++;
                        }
                        oss << "\"distractor_digit_counts\": {";
                        bool first_dd = true;
                        for (auto& [dd, cnt] : dist_counts) {
                            if (!first_dd) oss << ", ";
                            oss << "\"" << dd << "\": " << cnt;
                            first_dd = false;
                        }
                        oss << "}, ";
                    }

                    // Per-repeat accuracies
                    oss << "\"cls_per_repeat_accuracy\": [";
                    for (size_t i = 0; i < cls_res.per_repeat_accuracy.size(); i++) {
                        if (i > 0) oss << ", ";
                        oss << cls_res.per_repeat_accuracy[i];
                    }
                    oss << "]";

                    oss << "}";
                    return oss.str();
                };

                // --- Branch A: Unmatched (natural rate) ---
                {
                    double natural_rate = noisy_measure_rate(base_cfg, cal_samples, dyn_ovr,
                                                              sim_cfg, n_workers);
                    printf("    [A] Natural rate: %.1f Hz\n", natural_rate);

                    auto json = evaluate_branch("A_unmatched",
                                                 sim_cfg.stimulus_current, natural_rate);
                    grid_results_json.push_back(json);
                    total_evaluations++;
                }

                // --- Branch B: Tonic-conductance-matched ---
                {
                    double prev_gt = 0.0, prev_rev = -80.0;
                    if (pt.tau_idx >= 0 && last_cal_by_tau.count(pt.tau_idx)) {
                        prev_gt = last_cal_by_tau[pt.tau_idx].first;
                        prev_rev = last_cal_by_tau[pt.tau_idx].second;
                    }

                    auto cal = noisy_calibrate_tonic(base_cfg, dyn_ovr, cal_samples, sim_cfg,
                                                      n_workers, target_rate_hz, prev_gt, prev_rev);

                    if (pt.tau_idx >= 0) last_cal_by_tau[pt.tau_idx] = {cal.g_tonic, cal.reversal};
                    const char* mode = (cal.reversal < -1.0) ? "inh" : "exc";
                    printf("    [B] Calibrated: g_tonic=%.4f (%s) -> %.1f Hz\n",
                           cal.g_tonic, mode, cal.rate_hz);

                    auto json = evaluate_branch("B_matched",
                                                 cal.g_tonic, cal.rate_hz, cal.reversal);
                    grid_results_json.push_back(json);
                    total_evaluations++;
                }

                double gp_elapsed = now_seconds() - gp_start;
                grid_point_times.push_back(gp_elapsed);
                printf("    %.1fmin  |  %d/%d evals  |  Elapsed: %.1fh\n",
                       gp_elapsed / 60.0, total_evaluations, total_evals_expected,
                       (now_seconds() - sweep_start) / 3600.0);

                noisy_save_checkpoint(output_dir, grid_results_json, bsa_baselines_json,
                                       taus, target_rate_hz, total_start, false);
            }
        }
    }

    double total_time = now_seconds() - total_start;

    printf("\n======================================================================\n");
    printf("  NOISY SWEEP COMPLETE\n");
    printf("  Total time: %.0fs (%.1f hours)\n", total_time, total_time / 3600.0);
    printf("  Evaluations: %d\n", total_evaluations);
    printf("======================================================================\n");

    noisy_save_checkpoint(output_dir, grid_results_json, bsa_baselines_json,
                           taus, target_rate_hz, total_start, true);

    printf("\nDone.\n");
    return 0;
}
