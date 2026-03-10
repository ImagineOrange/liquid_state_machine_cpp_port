#include "experiments.h"
#include <atomic>

// ============================================================
// GRID AXES — log-spaced, full coverage
// ============================================================
// adapt_inc: 0.0 baseline + 19 log-spaced values from 0.005 to 1.0 (ratio ~1.34x)
// adapt_tau: 15 log-spaced values from 30 to 5000 ms (ratio ~1.44x)
// Full grid: 20 x 15 = 300 points, no holes.
static const std::vector<double> UNIFIED_INC = {
    0.0,
    0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000,
};
static const std::vector<double> UNIFIED_TAU = {
    30.0, 43.2, 62.3, 89.8, 129.4,
    186.5, 268.7, 387.3, 558.1, 804.4,
    1159.2, 1670.6, 2407.5, 3469.5, 5000.0,
};

// ============================================================
// GRID POINT
// ============================================================
struct GridPoint {
    double adapt_inc, adapt_tau;
    std::string point_id;
    int inc_idx, tau_idx;
};

static std::vector<GridPoint> build_grid_points(const std::string& /*arms*/) {
    std::vector<GridPoint> points;
    points.reserve(UNIFIED_INC.size() * UNIFIED_TAU.size());
    for (int ii = 0; ii < (int)UNIFIED_INC.size(); ii++) {
        for (int ti = 0; ti < (int)UNIFIED_TAU.size(); ti++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "inc%.4f_tau%.1f",
                     UNIFIED_INC[ii], UNIFIED_TAU[ti]);
            points.push_back({UNIFIED_INC[ii], UNIFIED_TAU[ti], buf, ii, ti});
        }
    }
    return points;
}

// ============================================================
// ACTIVITY BINNING
// ============================================================
struct BinResult {
    Mat bins;
    int total_spikes;
};

static BinResult bin_activity_record(const std::vector<std::vector<int>>& activity_record,
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
struct WorkerResult {
    Mat res_bins;
    int total_res_spikes;
    int digit;
    int sample_idx;
    double isi_cv;
    double mean_adapt_stim_end;
};

static WorkerResult sim_worker(SphericalNetwork& net, const AudioSample& sample,
                                int sample_idx, const ZoneInfo& zone_info,
                                const SimConfig& sim_cfg, const StdMasks& masks) {
    double dt = sim_cfg.dt;
    double total_ms = sim_cfg.audio_duration_ms + POST_STIM_MS;
    int n_bins = (int)std::ceil(total_ms / BIN_MS);

    double stim_end_ms = *std::max_element(sample.spike_times_ms.begin(),
                                            sample.spike_times_ms.end());

    auto result = run_sample_with_std(net, sample, zone_info, sim_cfg,
                                       STD_U, STD_TAU_REC, masks, stim_end_ms);

    auto br = bin_activity_record(result.activity_record, zone_info.reservoir_zone_indices,
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
    int stim_step = std::min((int)(stim_end_ms / dt), (int)result.activity_record.size() - 1);
    if (stim_step >= 0 && stim_step < (int)net.adaptation.size()) {
        for (int nid : zone_info.reservoir_zone_indices)
            adapt_mean += net.adaptation[nid];
        adapt_mean /= zone_info.reservoir_zone_indices.size();
    }

    return {br.bins, br.total_spikes, sample.digit, sample_idx, isi_cv, adapt_mean};
}

// ============================================================
// RUN ALL SAMPLES (OpenMP parallel)
// ============================================================
struct AllSamplesResult {
    std::vector<Mat> res_bins_list;
    std::vector<int> digits;
    std::vector<double> total_spikes;
    int n_reservoir;
    std::vector<double> isi_cvs;
    std::vector<double> adapt_stim_ends;
};

// Apply tonic conductance to reservoir neurons only.
// g_tonic > 0, reversal determines direction:
//   E_i (-80 mV) = shunting inhibition (rate too high)
//   E_e (0 mV)   = tonic excitation (rate too low)
static void set_reservoir_tonic_conductance(SphericalNetwork& net,
                                             const ZoneInfo& zone_info,
                                             double g_tonic, double reversal) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zone_info.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g_tonic;
        net.tonic_reversal[nid] = reversal;
    }
}

static AllSamplesResult run_all_samples(const NetworkConfig& cfg,
                                         const std::vector<AudioSample>& samples,
                                         const DynamicalOverrides& dyn_ovr,
                                         const SimConfig& sim_cfg,
                                         int n_workers,
                                         bool verbose = true,
                                         double g_tonic = 0.0,
                                         double tonic_rev = -80.0) {
    int n_samples = (int)samples.size();
    AllSamplesResult out;
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
            build_full_network(net, zone_info, cfg, sim_cfg.dt, true,
                              &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);
        if (g_tonic != 0.0)
            set_reservoir_tonic_conductance(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp single
        {
            out.n_reservoir = (int)zone_info.reservoir_zone_indices.size();
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_samples; i++) {
            auto wr = sim_worker(net, samples[i], i, zone_info, sim_cfg, masks);
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
static double measure_rate(const NetworkConfig& cfg,
                           const std::vector<AudioSample>& samples_subset,
                           const DynamicalOverrides& dyn_ovr,
                           const SimConfig& sim_cfg, int n_workers,
                           double g_tonic = 0.0, double tonic_rev = -80.0) {
    auto res = run_all_samples(cfg, samples_subset, dyn_ovr, sim_cfg, n_workers, false, g_tonic, tonic_rev);
    double trial_dur_s = (sim_cfg.audio_duration_ms + POST_STIM_MS) / 1000.0;
    double sum = 0;
    for (int i = 0; i < (int)samples_subset.size(); i++) {
        sum += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
    }
    return sum / samples_subset.size();
}

// ============================================================
// CALIBRATION
// ============================================================
struct CalLogEntry { int iter; double g_tonic; double rate_hz; };

// ============================================================
// CALIBRATION — TONIC CONDUCTANCE (reservoir-only)
// ============================================================
// g_tonic >= 0. Reversal determines direction:
//   E_i (-80 mV): shunting inhibition — suppresses rate
//   E_e (0 mV):   tonic excitation — boosts rate
// Calibration first measures natural rate at g=0 to pick the right reversal,
// then binary-searches g upward from 0.
static constexpr double G_TONIC_HI = 5.0;      // initial search ceiling

struct CalResult {
    double g_tonic;
    double reversal;
    double rate_hz;
    std::vector<CalLogEntry> log;
};

static CalResult
calibrate_tonic_conductance(NetworkConfig cfg,
                             const DynamicalOverrides& dyn_ovr,
                             const std::vector<AudioSample>& cal_samples,
                             SimConfig sim_cfg, int n_workers,
                             double target_rate,
                             double initial_guess = 0.0,
                             double prev_reversal = -80.0) {
    std::vector<CalLogEntry> log;
    int iteration = 0;

    // Measure natural rate (g=0) to determine direction
    double natural_rate = measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers, 0.0);
    log.push_back({iteration, 0.0, natural_rate});
    printf("    cal[%d] g=0 (natural) -> %.1f Hz (target=%.1f)\n",
           iteration, natural_rate, target_rate);
    iteration++;

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate, log};

    // Pick reversal: need to suppress if too fast, excite if too slow
    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    const char* mode = (reversal < -1.0) ? "inhibitory" : "excitatory";
    printf("    cal: natural=%.1f Hz, target=%.1f Hz -> %s mode (E_rev=%.0f)\n",
           natural_rate, target_rate, mode, reversal);

    // For both modes: higher g = stronger effect = rate moves toward target
    double lo = 0.0, hi = G_TONIC_HI;

    // Try initial guess if nonzero and same direction
    if (initial_guess > 0.0 && prev_reversal == reversal) {
        double rate = measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers,
                                    initial_guess, reversal);
        log.push_back({iteration, initial_guess, rate});
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, initial_guess, rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, reversal, rate, log};

        // Did we overshoot or undershoot?
        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = initial_guess; else lo = initial_guess;
    }

    for (;; iteration++) {
        double mid = (lo + hi) / 2.0;
        double rate = measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers, mid, reversal);
        log.push_back({iteration, mid, rate});
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, mid, rate);

        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {mid, reversal, rate, log};

        // For both modes: did we overshoot the target?
        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = mid; else lo = mid;

        // If interval collapses, widen upward (more conductance)
        if (hi - lo < 1e-6) {
            double span = std::max(mid, 1.0);
            hi = mid + span;
            printf("    cal: interval collapsed, widening hi to %.4f\n", hi);
        }
    }
}

// ============================================================
// CLASSIFICATION
// ============================================================
struct ClassifyResult {
    double accuracy;
    double accuracy_std;
    std::vector<double> per_repeat_accuracy;
    std::vector<std::vector<int>> cm;
};

static ClassifyResult classify_flat_ridge(const std::vector<Mat>& bins_list,
                                           const std::vector<int>& y,
                                           int n_bins, int n_reservoir) {
    int n_samples = (int)bins_list.size();
    int n_features = n_bins * n_reservoir;

    Mat X_flat(n_samples, n_features, 0.0);
    for (int i = 0; i < n_samples; i++) {
        int actual_bins = bins_list[i].rows;
        int actual_res = bins_list[i].cols;
        for (int b = 0; b < std::min(n_bins, actual_bins); b++) {
            for (int r = 0; r < std::min(n_reservoir, actual_res); r++) {
                X_flat(i, b * n_reservoir + r) = bins_list[i](b, r);
            }
        }
    }

    std::vector<double> repeat_accs;
    std::vector<std::vector<int>> last_cm;

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

            // Build confusion matrix on last repeat, last fold
            if (rep == N_CV_REPEATS - 1 && f == N_CV_FOLDS - 1) {
                int n_classes = (int)DEFAULT_DIGITS.size();
                last_cm.assign(n_classes, std::vector<int>(n_classes, 0));
                std::map<int, int> digit_to_idx;
                for (int c = 0; c < n_classes; c++) digit_to_idx[DEFAULT_DIGITS[c]] = c;
                for (int i = 0; i < n_test; i++) {
                    int true_idx = digit_to_idx[y[split.test[i]]];
                    int pred_idx = digit_to_idx[best_preds[i]];
                    last_cm[true_idx][pred_idx]++;
                }
            }
        }

        repeat_accs.push_back(rep_correct / rep_total);
    }

    double mean = 0;
    for (double a : repeat_accs) mean += a;
    mean /= repeat_accs.size();
    double var = 0;
    for (double a : repeat_accs) var += (a - mean) * (a - mean);
    var /= repeat_accs.size();

    return {mean, std::sqrt(var), repeat_accs, last_cm};
}

static std::vector<double> classify_per_bin(const std::vector<Mat>& bins_list,
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
// PARTICIPATION RATIO
// ============================================================
static std::pair<double, std::vector<double>>
compute_participation_ratio(const std::vector<Mat>& bins_list, int n_bins, int n_reservoir) {
    int n_samples = (int)bins_list.size();
    int n_eff = std::min(n_samples, n_reservoir);
    std::vector<double> pr_per_bin(n_bins, 0.0);

    for (int b = 0; b < n_bins; b++) {
        Mat X(n_samples, n_reservoir, 0.0);
        for (int i = 0; i < n_samples; i++) {
            if (b < bins_list[i].rows) {
                for (int r = 0; r < std::min(n_reservoir, bins_list[i].cols); r++)
                    X(i, r) = bins_list[i](b, r);
            }
        }

        std::vector<double> col_mean(n_reservoir, 0.0);
        for (int j = 0; j < n_reservoir; j++) {
            for (int i = 0; i < n_samples; i++) col_mean[j] += X(i, j);
            col_mean[j] /= n_samples;
        }
        for (int i = 0; i < n_samples; i++)
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
            double lam = s * s / std::max(n_samples - 1, 1);
            sum_lam += lam;
            sum_lam2 += lam * lam;
        }
        if (sum_lam2 > 1e-20) {
            pr_per_bin[b] = (sum_lam * sum_lam) / (n_eff * sum_lam2);
        }
    }

    double mean_pr = 0;
    for (double v : pr_per_bin) mean_pr += v;
    mean_pr /= n_bins;
    return {mean_pr, pr_per_bin};
}

// ============================================================
// BSA FEATURE EXTRACTION
// ============================================================
static Mat extract_bsa_bins(const AudioSample& sample, double total_duration_ms) {
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
// JSON OUTPUT
// ============================================================
static void write_cm_json(FILE* f, const std::vector<std::vector<int>>& cm) {
    fprintf(f, "[");
    for (size_t i = 0; i < cm.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "[");
        for (size_t j = 0; j < cm[i].size(); j++) {
            if (j > 0) fprintf(f, ", ");
            fprintf(f, "%d", cm[i][j]);
        }
        fprintf(f, "]");
    }
    fprintf(f, "]");
}

static void save_checkpoint(const std::string& results_dir,
                            const std::vector<std::map<std::string, std::string>>& grid_results_json,
                            const std::string& bsa_json,
                            const std::string& baseline_json,
                            double target_rate_hz,
                            double total_start_time,
                            bool final_save) {
    std::string fname = final_save ? "classification_adaptation_sweep.json"
                                   : "classification_adaptation_sweep_checkpoint.json";
    std::string path = results_dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    double elapsed = now_seconds() - total_start_time;

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Classification Adaptation Parameter Sweep (C++ Port)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", elapsed);
    fprintf(f, "  \"task\": \"5-class digit classification\",\n");
    fprintf(f, "  \"digits\": [0, 1, 2, 3, 4],\n");
    fprintf(f, "  \"n_samples\": %d,\n", SAMPLES_PER_DIGIT * N_DIGITS);
    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.2f, \"tolerance_hz\": %.1f},\n",
            target_rate_hz, RATE_TOLERANCE_HZ);

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); json_write_double_array(f, UNIFIED_INC); fprintf(f, ",\n");
    fprintf(f, "    \"unified_tau\": "); json_write_double_array(f, UNIFIED_TAU); fprintf(f, ",\n");
    fprintf(f, "    \"n_inc\": %d, \"n_tau\": %d\n", (int)UNIFIED_INC.size(), (int)UNIFIED_TAU.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"bsa_baseline\": %s,\n", bsa_json.c_str());
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
// RASTER DUMP — full network spike raster for a single sample
// ============================================================
int run_raster_dump(const std::string& snapshot_path,
                    const std::string& trace_file,
                    const std::string& output_dir,
                    double stim_current_override,
                    double input_tau_e_override,
                    double input_adapt_inc_override,
                    double adapt_inc_override,
                    double adapt_tau_override,
                    double tonic_conductance_override) {
    // Load audio sample
    auto nf = load_npz(trace_file);
    AudioSample s;
    s.spike_times_ms = nf["spike_times_ms"].to_float64_vec();
    s.freq_bin_indices = nf["freq_bin_indices"].to_int32_vec();
    fs::path fp(trace_file);
    std::string fname = fp.stem().string();
    s.filename = fname;
    s.digit = 0;
    if (fname.size() > 12 && fname.substr(0, 12) == "spike_train_")
        s.digit = std::atoi(fname.substr(12, 1).c_str());

    // Shift BSA spike times forward by warmup period
    double warmup_ms = 50.0;
    for (auto& t : s.spike_times_ms) t += warmup_ms;

    double sample_max_ms = *std::max_element(
        s.spike_times_ms.begin(), s.spike_times_ms.end()) + 5.0;

    printf("[RASTER DUMP] file: %s (%d BSA spikes, digit %d, %.0fms warmup)\n",
           fname.c_str(), (int)s.spike_times_ms.size(), s.digit, warmup_ms);

    // Build network
    SimConfig sim;
    sim.dt = 0.1;
    sim.audio_duration_ms = sample_max_ms;
    sim.post_stimulus_ms = POST_STIM_MS;
    sim.stimulus_current = (stim_current_override > 0) ? stim_current_override : INPUT_STIM_CURRENT;

    DynamicalOverrides dyn_ovr;
    dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
    dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
    dyn_ovr.adapt_inc = (adapt_inc_override >= 0) ? adapt_inc_override : LHS021_ADAPT_INC;
    dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
    if (adapt_tau_override > 0)
        dyn_ovr.adapt_tau = adapt_tau_override;

    SphericalNetwork net;
    ZoneInfo zone_info;
    if (!snapshot_path.empty()) {
        load_network_snapshot(net, zone_info, snapshot_path, sim.dt, true);
        apply_dynamical_overrides(net, zone_info, sim.dt, dyn_ovr);
    } else {
        NetworkConfig base_cfg = make_base_config();
        base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
        build_full_network(net, zone_info, base_cfg, sim.dt, true, &dyn_ovr);
    }
    // Input neuron regime applied by apply_dynamical_overrides / build_full_network
    StdMasks masks = build_std_masks(net, zone_info);

    // CLI overrides (override the baked-in defaults if specified)
    if (input_tau_e_override > 0 && input_tau_e_override != INPUT_TAU_E) {
        for (int idx : zone_info.input_neuron_indices)
            net.tau_e[idx] = input_tau_e_override;
        net.precompute_decay_factors(sim.dt);
        printf("  Input tau_e overridden to %.2f ms\n", input_tau_e_override);
    }
    if (input_adapt_inc_override >= 0 && input_adapt_inc_override != INPUT_ADAPT_INC) {
        for (int idx : zone_info.input_neuron_indices)
            net.adaptation_increment[idx] = input_adapt_inc_override;
        printf("  Input adapt_inc overridden to %.4f\n", input_adapt_inc_override);
    }
    if (adapt_inc_override >= 0)
        printf("  Reservoir adapt_inc overridden to %.4f nS\n", adapt_inc_override);
    if (adapt_tau_override > 0)
        printf("  Reservoir adapt_tau overridden to %.1f ms\n", adapt_tau_override);

    // Apply tonic conductance if specified (rate-matching)
    double g_tonic_used = 0.0;
    if (tonic_conductance_override >= 0) {
        g_tonic_used = tonic_conductance_override;
        set_reservoir_tonic_conductance(net, zone_info, tonic_conductance_override, -80.0);
        printf("  Tonic conductance set to %.4f nS (E_rev=-80 mV)\n", tonic_conductance_override);
    }

    // Seed RNG (raster dump is single-threaded, not launched from OMP block)
    rng_seed(42);

    // Run simulation
    double total_ms = sim.audio_duration_ms + sim.post_stimulus_ms;
    printf("  Running %.1f ms (dt=%.1f, stim=%.4f)...\n",
           total_ms, sim.dt, sim.stimulus_current);

    auto result = run_sample_with_std(net, s, zone_info, sim,
                                       STD_U, STD_TAU_REC, masks);

    // Build zone lookup
    std::set<int> input_set(zone_info.input_neuron_indices.begin(),
                            zone_info.input_neuron_indices.end());

    // Per-reservoir-neuron: total drive and per-input-neuron drive breakdown
    // For each reservoir neuron, find which input neuron drives it most
    std::map<int, double> reservoir_total_drive;
    std::map<int, std::map<int, double>> reservoir_per_input;  // res_nid -> (input_nid -> weight_sum)
    for (int src : zone_info.input_neuron_indices) {
        int64_t start = net.csr_indptr[src];
        int64_t end = net.csr_indptr[src + 1];
        for (int64_t j = start; j < end; j++) {
            int tgt = net.csr_targets[j];
            if (!input_set.count(tgt)) {
                double w = std::abs(net.csr_weights[j]);
                reservoir_total_drive[tgt] += w;
                reservoir_per_input[tgt][src] += w;
            }
        }
    }

    // Write input_drive.csv with dominant input neuron
    {
        std::string drive_path = output_dir + "/input_drive.csv";
        FILE* fd = fopen(drive_path.c_str(), "w");
        fprintf(fd, "neuron_id,input_drive,dominant_input\n");
        for (int nid : zone_info.reservoir_zone_indices) {
            double total = reservoir_total_drive[nid];
            int dominant = -1;
            double max_w = 0.0;
            auto pit = reservoir_per_input.find(nid);
            if (pit != reservoir_per_input.end()) {
                for (auto& [inp, w] : pit->second) {
                    if (w > max_w) { max_w = w; dominant = inp; }
                }
            }
            fprintf(fd, "%d,%.6f,%d\n", nid, total, dominant);
        }
        fclose(fd);
        printf("  Wrote %s\n", drive_path.c_str());
    }

    // Write spikes.csv
    std::string spikes_path = output_dir + "/spikes.csv";
    FILE* f = fopen(spikes_path.c_str(), "w");
    fprintf(f, "time_ms,neuron_id,zone\n");
    int n_steps = (int)result.activity_record.size();
    int total_spikes = 0;
    for (int step = 0; step < n_steps; step++) {
        double t = step * sim.dt;
        for (int nid : result.activity_record[step]) {
            const char* zone = input_set.count(nid) ? "input" : "reservoir";
            fprintf(f, "%.4f,%d,%s\n", t, nid, zone);
            total_spikes++;
        }
    }
    fclose(f);
    printf("  Wrote %s (%d spikes)\n", spikes_path.c_str(), total_spikes);

    // Write bsa.csv
    std::string bsa_path = output_dir + "/bsa.csv";
    f = fopen(bsa_path.c_str(), "w");
    fprintf(f, "time_ms,freq_bin\n");
    for (size_t i = 0; i < s.spike_times_ms.size(); i++)
        fprintf(f, "%.4f,%d\n", s.spike_times_ms[i], s.freq_bin_indices[i]);
    fclose(f);
    printf("  Wrote %s (%d BSA spikes)\n", bsa_path.c_str(), (int)s.spike_times_ms.size());

    // Write neurons.csv — input neuron to freq bin mapping
    std::string neurons_path = output_dir + "/neurons.csv";
    f = fopen(neurons_path.c_str(), "w");
    fprintf(f, "neuron_id,freq_bins,weights\n");
    // Invert the mapping: freq_bin->[neurons] to neuron->[freq_bins]
    std::map<int, std::vector<std::pair<int, double>>> neuron_to_bins;
    for (const auto& [bin, neurons] : zone_info.input_neuron_mapping) {
        auto wit = zone_info.input_neuron_weights.find(bin);
        for (size_t j = 0; j < neurons.size(); j++) {
            double w = (wit != zone_info.input_neuron_weights.end() && j < wit->second.size())
                       ? wit->second[j] : 1.0;
            neuron_to_bins[neurons[j]].emplace_back(bin, w);
        }
    }
    for (int nid : zone_info.input_neuron_indices) {
        auto& bins = neuron_to_bins[nid];
        std::sort(bins.begin(), bins.end());
        fprintf(f, "%d,\"", nid);
        for (size_t j = 0; j < bins.size(); j++) {
            if (j > 0) fprintf(f, ";");
            fprintf(f, "%d", bins[j].first);
        }
        fprintf(f, "\",\"");
        for (size_t j = 0; j < bins.size(); j++) {
            if (j > 0) fprintf(f, ";");
            fprintf(f, "%.4f", bins[j].second);
        }
        fprintf(f, "\"\n");
    }
    fclose(f);
    printf("  Wrote %s (%d input neurons)\n", neurons_path.c_str(),
           (int)zone_info.input_neuron_indices.size());

    // Write meta.json
    std::string meta_path = output_dir + "/meta.json";
    f = fopen(meta_path.c_str(), "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"dt\": %.4f,\n", sim.dt);
    fprintf(f, "  \"audio_duration_ms\": %.2f,\n", sim.audio_duration_ms);
    fprintf(f, "  \"post_stimulus_ms\": %.2f,\n", sim.post_stimulus_ms);
    fprintf(f, "  \"total_ms\": %.2f,\n", total_ms);
    fprintf(f, "  \"warmup_ms\": %.2f,\n", warmup_ms);
    fprintf(f, "  \"stim_current\": %.6f,\n", sim.stimulus_current);
    fprintf(f, "  \"n_input\": %d,\n", (int)zone_info.input_neuron_indices.size());
    fprintf(f, "  \"n_reservoir\": %d,\n", (int)zone_info.reservoir_zone_indices.size());
    fprintf(f, "  \"n_total\": %d,\n", net.n_neurons);
    fprintf(f, "  \"adapt_inc\": %.6f,\n", dyn_ovr.adapt_inc);
    fprintf(f, "  \"adapt_tau\": %.1f,\n", dyn_ovr.adapt_tau);
    fprintf(f, "  \"tonic_conductance\": %.6f,\n", g_tonic_used);
    fprintf(f, "  \"digit\": %d,\n", s.digit);
    fprintf(f, "  \"filename\": \"%s\"\n", fname.c_str());
    fprintf(f, "}\n");
    fclose(f);
    printf("  Wrote %s\n", meta_path.c_str());

    return 0;
}

// ============================================================
// MAIN CLASSIFICATION SWEEP ENTRY POINT
// ============================================================
int run_classification_sweep(int argc, char** argv, const std::string& arms,
                              int n_workers, const std::string& output_dir,
                              const std::string& data_dir,
                              bool verify_only, const std::string& verify_output,
                              int trace_neuron, int trace_sample,
                              const std::string& trace_output,
                              const std::string& trace_file,
                              bool no_noise, bool no_input_nmda,
                              double stim_current_override,
                              double input_tau_e_override,
                              double input_adapt_inc_override,
                              double input_std_u_override) {
    auto grid_points = build_grid_points(arms);
    int n_grid = (int)grid_points.size();

    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  CLASSIFICATION ADAPTATION PARAMETER SWEEP (C++ PORT)\n");
    printf("  Grid: %d x %d = %d points x 2 branches\n",
           (int)UNIFIED_INC.size(), (int)UNIFIED_TAU.size(), n_grid);
    printf("  Task: 5-class digit classification (digits 0-4)\n");
    printf("  Samples: %d per digit = %d total\n", SAMPLES_PER_DIGIT, SAMPLES_PER_DIGIT * N_DIGITS);
    printf("  Workers: %d\n", n_workers);
    printf("  Readout: Flat Ridge (all %.0fms bins concatenated)\n", BIN_MS);
    printf("  CV: StratifiedKFold(%d) x %d repeats\n", N_CV_FOLDS, N_CV_REPEATS);
    if (!g_snapshot_path.empty()) {
        printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
        printf("  (Network loaded from Python export — RNG-independent)\n");
    }
    printf("======================================================================\n");

    // --trace-neuron with --trace-file: skip sample loader entirely
    if (trace_neuron >= 0 && !trace_file.empty()) {
        std::string t_output = trace_output.empty() ? "trace_cpp.csv" : trace_output;

        auto nf = load_npz(trace_file);
        AudioSample s;
        s.spike_times_ms = nf["spike_times_ms"].to_float64_vec();
        s.freq_bin_indices = nf["freq_bin_indices"].to_int32_vec();
        fs::path fp(trace_file);
        std::string fname = fp.stem().string();
        s.filename = fname;
        s.digit = 0;
        if (fname.size() > 12 && fname.substr(0, 12) == "spike_train_") {
            s.digit = std::atoi(fname.substr(12, 1).c_str());
        }

        printf("\n[TRACE] Neuron %d, file: %s (%d spikes)\n",
               trace_neuron, fname.c_str(), (int)s.spike_times_ms.size());

        double sample_max_ms = *std::max_element(
            s.spike_times_ms.begin(), s.spike_times_ms.end());
        printf("  Max spike time: %.2f ms\n", sample_max_ms);
        sample_max_ms += 5.0;

        SimConfig trace_sim;
        trace_sim.dt = 0.1;
        trace_sim.audio_duration_ms = sample_max_ms;
        trace_sim.post_stimulus_ms = POST_STIM_MS;
        trace_sim.stimulus_current = (stim_current_override > 0) ? stim_current_override : INPUT_STIM_CURRENT;

        DynamicalOverrides dyn_ovr;
        dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
        dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, trace_sim.dt, true);
            apply_dynamical_overrides(net, zone_info, trace_sim.dt, dyn_ovr);
        } else {
            NetworkConfig base_cfg = make_base_config();
            base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
            build_full_network(net, zone_info, base_cfg, trace_sim.dt, true, &dyn_ovr);
        }
        // Input neuron regime applied by apply_dynamical_overrides / build_full_network
        StdMasks masks = build_std_masks(net, zone_info);

        if (no_noise) {
            std::fill(net.v_noise_amp_arr.begin(), net.v_noise_amp_arr.end(), 0.0);
            std::fill(net.i_noise_amp_arr.begin(), net.i_noise_amp_arr.end(), 0.0);
            printf("  Noise DISABLED\n");
        }

        // CLI overrides (override the baked-in defaults if specified)
        if (input_tau_e_override > 0 && input_tau_e_override != INPUT_TAU_E) {
            for (int idx : zone_info.input_neuron_indices)
                net.tau_e[idx] = input_tau_e_override;
            net.precompute_decay_factors(trace_sim.dt);
            printf("  Input tau_e overridden to %.2f ms\n", input_tau_e_override);
        }
        if (input_adapt_inc_override >= 0 && input_adapt_inc_override != INPUT_ADAPT_INC) {
            for (int idx : zone_info.input_neuron_indices)
                net.adaptation_increment[idx] = input_adapt_inc_override;
            printf("  Input adapt_inc overridden to %.4f\n", input_adapt_inc_override);
        }
        if (input_std_u_override >= 0) {
            trace_sim.input_std_u = input_std_u_override;
            printf("  Input STD u overridden to %.4f\n", input_std_u_override);
        }

        printf("  INPUT_NEURON_INDICES:");
        for (int idx : zone_info.input_neuron_indices) printf(" %d", idx);
        printf("\n");

        net.trace_neuron_id = trace_neuron;
        net.trace.clear();

        double total_ms = trace_sim.audio_duration_ms + trace_sim.post_stimulus_ms;
        printf("  Running simulation (%.1f ms, dt=%.1f)...\n", total_ms, trace_sim.dt);

        auto result = run_sample_with_std(net, s, zone_info, trace_sim,
                                           STD_U, STD_TAU_REC, masks,
                                           trace_sim.audio_duration_ms);

        printf("  %d timesteps recorded\n", (int)net.trace.size());

        FILE* f = fopen(t_output.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", t_output.c_str()); return 1; }
        fprintf(f, "step,t_ms,v,g_e,g_i,g_i_slow,g_nmda,adaptation,"
                   "i_e,i_i,i_i_slow,i_nmda,i_adapt,"
                   "v_noise,ge_noise,gi_noise,spiked\n");
        for (size_t si = 0; si < net.trace.size(); si++) {
            auto& r = net.trace[si];
            fprintf(f, "%d,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,"
                       "%.8f,%.8f,%.8f,%.8f,%.8f,"
                       "%.8f,%.8f,%.8f,%d\n",
                    (int)si, si * trace_sim.dt,
                    r.v, r.g_e, r.g_i, r.g_i_slow, r.g_nmda, r.adaptation,
                    r.i_e, r.i_i, r.i_i_slow, r.i_nmda, r.i_adapt,
                    r.v_noise, r.ge_noise, r.gi_noise, r.spiked ? 1 : 0);
        }
        fclose(f);
        printf("  Wrote %s\n", t_output.c_str());
        return 0;
    }

    // 1. Load audio
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded\n", n_samples);

    std::vector<int> y(n_samples);
    for (int i = 0; i < n_samples; i++) y[i] = samples[i].digit;

    // --trace-neuron mode (without --trace-file): use loaded samples
    if (trace_neuron >= 0) {
        std::string t_output = trace_output.empty() ? "trace_cpp.csv" : trace_output;

        if (trace_sample >= n_samples) {
            fprintf(stderr, "trace_sample %d >= n_samples %d\n", trace_sample, n_samples);
            return 1;
        }

        printf("\n[TRACE] Neuron %d, sample %d (%s), output: %s\n",
               trace_neuron, trace_sample, samples[trace_sample].filename.c_str(),
               t_output.c_str());

        double sample_max_ms = *std::max_element(
            samples[trace_sample].spike_times_ms.begin(),
            samples[trace_sample].spike_times_ms.end());
        sample_max_ms += 5.0;

        SimConfig trace_sim;
        trace_sim.dt = 0.1;
        trace_sim.audio_duration_ms = sample_max_ms;
        trace_sim.post_stimulus_ms = POST_STIM_MS;
        trace_sim.stimulus_current = (stim_current_override > 0) ? stim_current_override : INPUT_STIM_CURRENT;

        DynamicalOverrides dyn_ovr;
        dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
        dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
        dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
        dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, trace_sim.dt, true);
            apply_dynamical_overrides(net, zone_info, trace_sim.dt, dyn_ovr);
        } else {
            NetworkConfig base_cfg = make_base_config();
            base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
            build_full_network(net, zone_info, base_cfg, trace_sim.dt, true, &dyn_ovr);
        }
        // Input neuron regime applied by apply_dynamical_overrides / build_full_network
        StdMasks masks = build_std_masks(net, zone_info);

        if (no_noise) {
            std::fill(net.v_noise_amp_arr.begin(), net.v_noise_amp_arr.end(), 0.0);
            std::fill(net.i_noise_amp_arr.begin(), net.i_noise_amp_arr.end(), 0.0);
            printf("  Noise DISABLED\n");
        }

        // CLI overrides (override the baked-in defaults if specified)
        if (input_tau_e_override > 0 && input_tau_e_override != INPUT_TAU_E) {
            for (int idx : zone_info.input_neuron_indices)
                net.tau_e[idx] = input_tau_e_override;
            net.precompute_decay_factors(trace_sim.dt);
            printf("  Input tau_e overridden to %.2f ms\n", input_tau_e_override);
        }
        if (input_adapt_inc_override >= 0 && input_adapt_inc_override != INPUT_ADAPT_INC) {
            for (int idx : zone_info.input_neuron_indices)
                net.adaptation_increment[idx] = input_adapt_inc_override;
            printf("  Input adapt_inc overridden to %.4f\n", input_adapt_inc_override);
        }
        if (input_std_u_override >= 0) {
            trace_sim.input_std_u = input_std_u_override;
            printf("  Input STD u overridden to %.4f\n", input_std_u_override);
        }

        net.trace_neuron_id = trace_neuron;
        net.trace.clear();

        printf("  Running simulation (%.1f ms, dt=%.1f)...\n",
               trace_sim.audio_duration_ms + trace_sim.post_stimulus_ms, trace_sim.dt);

        auto result = run_sample_with_std(net, samples[trace_sample], zone_info, trace_sim,
                                           STD_U, STD_TAU_REC, masks,
                                           trace_sim.audio_duration_ms);

        printf("  %d timesteps recorded\n", (int)net.trace.size());

        FILE* f = fopen(t_output.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", t_output.c_str()); return 1; }
        fprintf(f, "step,t_ms,v,g_e,g_i,g_i_slow,g_nmda,adaptation,"
                   "i_e,i_i,i_i_slow,i_nmda,i_adapt,"
                   "v_noise,ge_noise,gi_noise,spiked\n");
        for (size_t s = 0; s < net.trace.size(); s++) {
            auto& r = net.trace[s];
            fprintf(f, "%d,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,"
                       "%.8f,%.8f,%.8f,%.8f,%.8f,"
                       "%.8f,%.8f,%.8f,%d\n",
                    (int)s, s * trace_sim.dt,
                    r.v, r.g_e, r.g_i, r.g_i_slow, r.g_nmda, r.adaptation,
                    r.i_e, r.i_i, r.i_i_slow, r.i_nmda, r.i_adapt,
                    r.v_noise, r.ge_noise, r.gi_noise, r.spiked ? 1 : 0);
        }
        fclose(f);
        printf("  Wrote %s\n", t_output.c_str());
        return 0;
    }

    double max_audio_ms = 0;
    for (auto& s : samples) {
        double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (mx > max_audio_ms) max_audio_ms = mx;
    }
    max_audio_ms += 5.0;

    // 2. BSA baseline
    printf("[2] Computing BSA baseline...\n");
    double total_duration_ms = max_audio_ms + POST_STIM_MS;
    int n_bins_total = (int)std::ceil(total_duration_ms / BIN_MS);

    std::vector<Mat> bsa_bins_list(n_samples);
    int max_bsa_bins = 0;
    for (int i = 0; i < n_samples; i++) {
        bsa_bins_list[i] = extract_bsa_bins(samples[i], total_duration_ms);
        if (bsa_bins_list[i].rows > max_bsa_bins) max_bsa_bins = bsa_bins_list[i].rows;
    }

    auto bsa_result = classify_flat_ridge(bsa_bins_list, y, max_bsa_bins, 128);
    printf("  BSA: %.1f%% (std=%.3f)\n", bsa_result.accuracy * 100, bsa_result.accuracy_std);

    char bsa_json_buf[1024];
    snprintf(bsa_json_buf, sizeof(bsa_json_buf),
             "{\"accuracy\": %.10f, \"accuracy_std\": %.10f}",
             bsa_result.accuracy, bsa_result.accuracy_std);
    std::string bsa_json = bsa_json_buf;

    // 3. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_audio_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    // Stratified calibration subset: equal samples per digit to avoid
    // digit-dependent rate bias (samples are ordered by digit in the full set).
    std::vector<AudioSample> cal_samples;
    {
        int per_digit = std::max(1, CALIBRATION_N_SAMPLES / N_DIGITS);
        std::map<int, int> digit_count;
        for (auto& s : samples) {
            if ((int)cal_samples.size() >= CALIBRATION_N_SAMPLES) break;
            if (digit_count[s.digit] < per_digit) {
                cal_samples.push_back(s);
                digit_count[s.digit]++;
            }
        }
    }

    double trial_dur_s = (sim_cfg.audio_duration_ms + POST_STIM_MS) / 1000.0;

    // 4. LHS-021 baseline
    printf("\n[3] Running LHS-021 baseline...\n");
    DynamicalOverrides lhs021_ovr;
    lhs021_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
    lhs021_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
    lhs021_ovr.adapt_inc = LHS021_ADAPT_INC;
    lhs021_ovr.nmda_tau = FIXED_NMDA_TAU;

    printf("  Simulating %d samples (%d workers)...\n", n_samples, n_workers);
    double bl_t0 = now_seconds();
    auto bl_res = run_all_samples(base_cfg, samples, lhs021_ovr, sim_cfg, n_workers);
    double bl_sim_time = now_seconds() - bl_t0;

    // Branch B calibration target: fixed sensory-cortex evoked rate (literature-based)
    double target_rate_hz = RATE_TARGET_HZ;

    // Measure LHS-021 baseline rate for reporting (not used as calibration target)
    double bl_rate_hz = 0;
    for (int i = 0; i < n_samples; i++) {
        bl_rate_hz += bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s);
    }
    bl_rate_hz /= n_samples;

    int target_n_bins = 0;
    for (auto& b : bl_res.res_bins_list)
        if (b.rows > target_n_bins) target_n_bins = b.rows;
    target_n_bins = std::max(target_n_bins, n_bins_total);

    auto bl_cls = classify_flat_ridge(bl_res.res_bins_list, y, target_n_bins, bl_res.n_reservoir);
    auto bl_stats = compute_paired_stats(bl_cls.per_repeat_accuracy,
                                          bsa_result.per_repeat_accuracy,
                                          bl_cls.accuracy, bsa_result.accuracy);

    printf("  LHS-021: %.0fs\n", bl_sim_time);
    printf("  Rate: %.1f Hz (calibration target: %.1f Hz)\n", bl_rate_hz, target_rate_hz);
    printf("  Classification: %.1f%% (gap=%+.1fpp %s)\n",
           bl_cls.accuracy * 100, bl_stats.gap_pp, bl_stats.stars.c_str());

    if (!g_snapshot_path.empty()) {
        SphericalNetwork verify_net;
        ZoneInfo verify_zone;
        load_network_snapshot(verify_net, verify_zone, g_snapshot_path, sim_cfg.dt, true);
        apply_dynamical_overrides(verify_net, verify_zone, sim_cfg.dt, lhs021_ovr);
        print_network_fingerprint(verify_net, verify_zone);
    }

    // --verify-only: dump per-sample stats and exit
    if (verify_only) {
        std::string vout = verify_output.empty() ? "verify_cpp.json" : verify_output;
        FILE* f = fopen(vout.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", vout.c_str()); return 1; }

        fprintf(f, "{\n");
        fprintf(f, "  \"n_samples\": %d,\n", n_samples);
        fprintf(f, "  \"n_reservoir\": %d,\n", bl_res.n_reservoir);
        fprintf(f, "  \"mean_firing_rate_hz\": %.10f,\n", bl_rate_hz);
        fprintf(f, "  \"classification_accuracy\": %.10f,\n", bl_cls.accuracy);
        fprintf(f, "  \"classification_accuracy_std\": %.10f,\n", bl_cls.accuracy_std);
        fprintf(f, "  \"per_repeat_accuracy\": [");
        for (int r = 0; r < (int)bl_cls.per_repeat_accuracy.size(); r++) {
            if (r > 0) fprintf(f, ", ");
            fprintf(f, "%.10f", bl_cls.per_repeat_accuracy[r]);
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_spikes\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.1f", bl_res.total_spikes[i]);
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_rate_hz\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s));
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_isi_cv\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.isi_cvs[i]);
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_digit\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%d", bl_res.digits[i]);
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_adapt_stim_end\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.adapt_stim_ends[i]);
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"per_sample_filename\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "\"%s\"", samples[i].filename.c_str());
        }
        fprintf(f, "]\n");

        fprintf(f, "}\n");
        fclose(f);

        printf("\n  Verify-only: wrote %d-sample stats to %s\n", n_samples, vout.c_str());
        printf("  Total time: %.1fs\n", now_seconds() - total_start);
        return 0;
    }

    // Baseline JSON
    char bl_json_buf[2048];
    snprintf(bl_json_buf, sizeof(bl_json_buf),
             "{\"classification_accuracy\": %.10f, \"classification_accuracy_std\": %.10f, "
             "\"classification_gap_pp\": %.4f, \"classification_stars\": \"%s\", "
             "\"firing_rate_hz\": %.4f, \"sim_time_s\": %.1f, \"n_reservoir\": %d}",
             bl_cls.accuracy, bl_cls.accuracy_std,
             bl_stats.gap_pp, bl_stats.stars.c_str(),
             bl_rate_hz, bl_sim_time, bl_res.n_reservoir);
    std::string baseline_json = bl_json_buf;

    // ============================================================
    // EVALUATE ONE BRANCH — shared helper for Branch A and B
    // ============================================================
    auto evaluate_branch = [&](const GridPoint& pt,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& eval_sim,
                               const std::string& branch,
                               double cal_value,    // 0 (A) or g_tonic (B)
                               double cal_rate,
                               double tonic_rev = -80.0) -> std::string {
        double t0 = now_seconds();
        double gt = (branch == "B_matched") ? cal_value : 0.0;
        auto res = run_all_samples(base_cfg, samples, dyn_ovr, eval_sim, n_workers, true, gt, tonic_rev);
        double sim_time = now_seconds() - t0;

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

        int eval_n_bins = 0;
        for (auto& b : res.res_bins_list)
            if (b.rows > eval_n_bins) eval_n_bins = b.rows;

        auto cls_res = classify_flat_ridge(res.res_bins_list, y, eval_n_bins, res.n_reservoir);
        auto stats = compute_paired_stats(cls_res.per_repeat_accuracy,
                                           bsa_result.per_repeat_accuracy,
                                           cls_res.accuracy, bsa_result.accuracy);

        double isi_cv_mean = 0;
        int isi_valid = 0;
        for (int i = 0; i < n_samples; i++) {
            if (!std::isnan(res.isi_cvs[i])) { isi_cv_mean += res.isi_cvs[i]; isi_valid++; }
        }
        if (isi_valid > 0) isi_cv_mean /= isi_valid;

        double adapt_mean = 0;
        for (int i = 0; i < n_samples; i++) adapt_mean += res.adapt_stim_ends[i];
        adapt_mean /= n_samples;

        auto [pr_mean, pr_per_bin] = compute_participation_ratio(
            res.res_bins_list, eval_n_bins, res.n_reservoir);

        auto per_bin_acc = classify_per_bin(res.res_bins_list, y, eval_n_bins, res.n_reservoir);

        printf("    [%s] Rate: %.1f Hz | Acc: %.1f%% (gap=%+.1fpp %s) | ISI CV: %.3f | PR: %.4f\n",
               branch.c_str(), rate_mean, cls_res.accuracy * 100, stats.gap_pp, stats.stars.c_str(),
               isi_cv_mean, pr_mean);

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
        oss << "\"classification_accuracy\": " << cls_res.accuracy << ", ";
        oss.precision(6);
        oss << "\"classification_accuracy_std\": " << cls_res.accuracy_std << ", ";
        oss << "\"classification_gap_pp\": " << stats.gap_pp << ", ";
        oss << "\"classification_ci_lo_pp\": " << stats.ci_lo_pp << ", ";
        oss << "\"classification_ci_hi_pp\": " << stats.ci_hi_pp << ", ";
        oss << "\"classification_p_value\": " << stats.p_value << ", ";
        oss << "\"classification_cohens_d\": " << stats.cohens_d << ", ";
        oss << "\"classification_stars\": \"" << stats.stars << "\", ";
        oss << "\"firing_rate_hz\": " << rate_mean << ", ";
        oss << "\"firing_rate_std\": " << rate_std << ", ";
        oss << "\"n_reservoir\": " << res.n_reservoir << ", ";
        oss << "\"sim_time_s\": " << sim_time << ", ";
        if (isi_valid > 0) oss << "\"isi_cv_mean\": " << isi_cv_mean << ", ";
        else oss << "\"isi_cv_mean\": null, ";
        oss << "\"adapt_at_stim_end_mean\": " << adapt_mean << ", ";
        oss << "\"participation_ratio_mean\": " << pr_mean << ", ";
        oss << "\"per_bin_accuracy\": [";
        for (size_t i = 0; i < per_bin_acc.size(); i++) {
            if (i > 0) oss << ", ";
            oss << per_bin_acc[i];
        }
        oss << "], ";
        oss << "\"classification_per_repeat_accuracy\": [";
        for (size_t i = 0; i < cls_res.per_repeat_accuracy.size(); i++) {
            if (i > 0) oss << ", ";
            oss << cls_res.per_repeat_accuracy[i];
        }
        oss << "]";
        oss << "}";

        return oss.str();
    };

    // 5. Sweep — two branches per grid point
    printf("\n[4] Running %d grid points x 2 branches...\n", n_grid);
    printf("    Branch A: Unmatched (natural rate, fixed stim=%.4f)\n", sim_cfg.stimulus_current);
    printf("    Branch B: Background-current-matched (target=%.1f +/- %.0f Hz)\n",
           target_rate_hz, RATE_TOLERANCE_HZ);

    std::vector<std::map<std::string, std::string>> grid_results_json;
    std::vector<double> grid_point_times;
    double sweep_start = now_seconds();
    std::map<int, std::pair<double,double>> last_cal_by_tau;  // warm-start: {g_tonic, reversal}

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

        // --- Branch A: Unmatched (natural rate) ---
        // stimulus_current fixed at INPUT_STIM_CURRENT, no background current
        {
            SimConfig eval_a = sim_cfg;  // stim already set to INPUT_STIM_CURRENT
            // Measure natural rate for reporting (use calibration subset for speed)
            double natural_rate = measure_rate(base_cfg, cal_samples, dyn_ovr, eval_a, n_workers);
            printf("    [A] Natural rate: %.1f Hz (stim=%.4f, no bg)\n",
                   natural_rate, eval_a.stimulus_current);

            auto json = evaluate_branch(pt, dyn_ovr, eval_a, "A_unmatched",
                                         eval_a.stimulus_current, natural_rate);
            grid_results_json.push_back({{"json", json}});
        }

        // --- Branch B: Tonic-conductance-matched ---
        // stimulus_current fixed at INPUT_STIM_CURRENT, calibrate g_tonic
        {
            SimConfig eval_b = sim_cfg;  // stim fixed

            double prev_gt = 0.0, prev_rev = -80.0;
            if (pt.tau_idx >= 0 && last_cal_by_tau.count(pt.tau_idx)) {
                prev_gt = last_cal_by_tau[pt.tau_idx].first;
                prev_rev = last_cal_by_tau[pt.tau_idx].second;
            }

            auto cal = calibrate_tonic_conductance(base_cfg, dyn_ovr, cal_samples, eval_b,
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

        save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                        target_rate_hz, total_start, false);
    }

    double total_time = now_seconds() - total_start;

    printf("\n======================================================================\n");
    printf("  SWEEP COMPLETE\n");
    printf("  Total time: %.0fs (%.1f hours)\n", total_time, total_time / 3600.0);
    printf("  Grid points: %d results (%d grid x 2 branches)\n",
           (int)grid_results_json.size(), n_grid);
    printf("======================================================================\n");

    save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                    target_rate_hz, total_start, true);

    fs::path base_dir = fs::path(output_dir).parent_path().parent_path();
    std::string plot_script = (base_dir / "plot_results.py").string();
    if (fs::exists(plot_script)) {
        std::string json_path = output_dir + "/classification_adaptation_sweep.json";
        std::string cmd = "python3 " + plot_script + " " + json_path + " " + output_dir;
        printf("\n  Generating figures...\n");
        int ret = system(cmd.c_str());
        if (ret != 0) fprintf(stderr, "  Warning: plot script returned %d\n", ret);
    }

    printf("\nDone.\n");
    return 0;
}
