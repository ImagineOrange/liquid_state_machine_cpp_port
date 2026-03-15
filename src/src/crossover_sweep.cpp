#include "experiments.h"
#include <atomic>

// ============================================================
// CROSSOVER SWEEP
// For each (inc, tau) in the extended parameter grid, run
// 3 trials per digit (0-5), compute spike vs adaptation
// separability over time bins, and find the crossover time
// where adaptation separability first exceeds spike separability.
// ============================================================

// Parameter grid (same as extended_sweep)
static const std::vector<double> XO_INC = {
    0.0, 0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000
};

static const std::vector<double> XO_TAU = {
    5000.0, 7200.0, 10380.0, 14960.0, 21550.0, 31080.0, 44790.0
};

static constexpr int    XO_TRIALS_PER_DIGIT = 3;
static const std::vector<int> XO_DIGITS = {0, 1, 2, 3, 4, 5};
static constexpr double XO_BIN_MS = 20.0;

// ============================================================
// Rate calibration (reused from hidden_states pattern)
// ============================================================
struct XoCalResult { double g_tonic, reversal, rate_hz; };

static double xo_measure_rate(const std::vector<AudioSample>& cal_samples,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& sim_cfg, int n_workers,
                               double g_tonic, double tonic_rev) {
    int n = (int)cal_samples.size();
    std::vector<double> rates(n, 0.0);
    int n_reservoir = 0;

    #pragma omp parallel num_threads(n_workers)
    {
        SphericalNetwork net;
        ZoneInfo zone_info;
        load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
        apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        StdMasks masks = build_std_masks(net, zone_info);

        if (g_tonic != 0.0) {
            net.tonic_conductance.assign(net.n_neurons, 0.0);
            net.tonic_reversal.assign(net.n_neurons, 0.0);
            for (int nid : zone_info.reservoir_zone_indices) {
                net.tonic_conductance[nid] = g_tonic;
                net.tonic_reversal[nid] = tonic_rev;
            }
        }

        #pragma omp single
        n_reservoir = (int)zone_info.reservoir_zone_indices.size();

        rng_seed(604 + 77 + (uint64_t)omp_get_thread_num() * 1000 + (uint64_t)getpid());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; i++) {
            SimConfig sc = sim_cfg;
            double stim_end = *std::max_element(cal_samples[i].spike_times_ms.begin(),
                                                 cal_samples[i].spike_times_ms.end());
            sc.audio_duration_ms = stim_end + 5.0;

            auto result = run_sample_with_std(net, cal_samples[i], zone_info, sc,
                                               STD_U, STD_TAU_REC, masks, {});
            int spk = 0;
            std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                                   zone_info.reservoir_zone_indices.end());
            for (auto& step : result.activity_record)
                for (int nid : step)
                    if (res_set.count(nid)) spk++;

            double dur_s = (sc.audio_duration_ms + sc.post_stimulus_ms) / 1000.0;
            rates[i] = spk / (n_reservoir * dur_s);
        }
    }

    double sum = 0;
    for (double r : rates) sum += r;
    return sum / n;
}

static XoCalResult xo_calibrate(const std::vector<AudioSample>& cal_samples,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers) {
    double natural = xo_measure_rate(cal_samples, dyn_ovr, sim_cfg, n_workers, 0.0, 0.0);

    if (std::abs(natural - RATE_TARGET_HZ) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural};

    double reversal = (natural > RATE_TARGET_HZ) ? -80.0 : 0.0;
    double lo = 0.0, hi = 5.0;
    XoCalResult best = {0.0, reversal, natural};

    for (int iter = 1; iter <= 10; iter++) {
        double mid = (lo + hi) / 2.0;
        double rate = xo_measure_rate(cal_samples, dyn_ovr, sim_cfg, n_workers, mid, reversal);
        best = {mid, reversal, rate};
        if (std::abs(rate - RATE_TARGET_HZ) <= RATE_TOLERANCE_HZ) break;

        bool overshot = (reversal < -1.0) ? (rate < RATE_TARGET_HZ)
                                           : (rate > RATE_TARGET_HZ);
        if (overshot) hi = mid; else lo = mid;
    }
    return best;
}

// ============================================================
// Compute separability: mean pairwise Euclidean distance
// between digit centroids at each time bin
// ============================================================
struct SepResult {
    std::vector<double> spike_sep;  // per-bin
    std::vector<double> adapt_sep;  // per-bin
    double crossover_ms;            // -1 if no crossover
};

static SepResult compute_separability(
    const std::vector<std::vector<std::vector<double>>>& spike_data,  // [trial][bin][neuron]
    const std::vector<std::vector<std::vector<double>>>& adapt_data,
    const std::vector<int>& trial_digits,
    int n_bins, int n_res, double bin_ms)
{
    int n_trials = (int)trial_digits.size();
    int n_digits = 6;

    SepResult res;
    res.spike_sep.resize(n_bins, 0.0);
    res.adapt_sep.resize(n_bins, 0.0);
    res.crossover_ms = -1.0;

    for (int b = 0; b < n_bins; b++) {
        // Compute centroids per digit
        std::vector<std::vector<double>> spike_centroid(n_digits, std::vector<double>(n_res, 0.0));
        std::vector<std::vector<double>> adapt_centroid(n_digits, std::vector<double>(n_res, 0.0));
        std::vector<int> digit_count(n_digits, 0);

        for (int t = 0; t < n_trials; t++) {
            int d = trial_digits[t];
            digit_count[d]++;
            for (int n = 0; n < n_res; n++) {
                spike_centroid[d][n] += spike_data[t][b][n];
                adapt_centroid[d][n] += adapt_data[t][b][n];
            }
        }

        for (int d = 0; d < n_digits; d++) {
            if (digit_count[d] > 0) {
                for (int n = 0; n < n_res; n++) {
                    spike_centroid[d][n] /= digit_count[d];
                    adapt_centroid[d][n] /= digit_count[d];
                }
            }
        }

        // Mean pairwise Euclidean distance between centroids
        double spike_sum = 0, adapt_sum = 0;
        int n_pairs = 0;
        for (int i = 0; i < n_digits; i++) {
            if (digit_count[i] == 0) continue;
            for (int j = i + 1; j < n_digits; j++) {
                if (digit_count[j] == 0) continue;
                n_pairs++;
                double s_dist2 = 0, a_dist2 = 0;
                for (int n = 0; n < n_res; n++) {
                    double ds = spike_centroid[i][n] - spike_centroid[j][n];
                    double da = adapt_centroid[i][n] - adapt_centroid[j][n];
                    s_dist2 += ds * ds;
                    a_dist2 += da * da;
                }
                spike_sum += std::sqrt(s_dist2);
                adapt_sum += std::sqrt(a_dist2);
            }
        }

        if (n_pairs > 0) {
            res.spike_sep[b] = spike_sum / n_pairs;
            res.adapt_sep[b] = adapt_sum / n_pairs;
        }
    }

    // Find crossover: first bin where adapt > spike
    for (int b = 0; b < n_bins; b++) {
        if (res.adapt_sep[b] > res.spike_sep[b]) {
            res.crossover_ms = (b + 0.5) * bin_ms;
            break;
        }
    }

    return res;
}

// ============================================================
// MAIN ENTRY POINT
// ============================================================
int run_crossover_sweep(int /*argc*/, char** /*argv*/,
                         int n_workers, const std::string& output_dir,
                         const std::string& data_dir) {
    printf("\n=== CROSSOVER SWEEP ===\n");
    printf("  Grid: %d inc x %d tau = %d points\n",
           (int)XO_INC.size(), (int)XO_TAU.size(),
           (int)(XO_INC.size() * XO_TAU.size()));
    printf("  Digits: 0-5, %d trials each\n", XO_TRIALS_PER_DIGIT);

    // --- Load samples ---
    printf("\nLoading audio samples...\n");
    auto all_samples = load_audio_samples(data_dir, XO_DIGITS,
                                           SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);

    // Pick first N trials per digit
    std::map<int, std::vector<int>> by_digit;
    for (int i = 0; i < (int)all_samples.size(); i++)
        by_digit[all_samples[i].digit].push_back(i);

    struct Trial { int sample_idx; int digit; };
    std::vector<Trial> trials;
    for (int d : XO_DIGITS) {
        auto& pool = by_digit[d];
        int n = std::min(XO_TRIALS_PER_DIGIT, (int)pool.size());
        for (int t = 0; t < n; t++)
            trials.push_back({pool[t], d});
    }
    int n_trials = (int)trials.size();
    printf("  Total trials per grid point: %d\n", n_trials);

    // Calibration samples
    int cal_n = std::min(30, (int)all_samples.size());
    std::vector<AudioSample> cal_samples(all_samples.begin(), all_samples.begin() + cal_n);

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;

    // --- Output CSV ---
    std::string csv_path = output_dir + "/crossover_sweep.csv";
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (!csv) { fprintf(stderr, "Cannot open %s\n", csv_path.c_str()); return 1; }
    fprintf(csv, "inc,tau,g_tonic,rate_hz,crossover_ms,peak_spike_sep,peak_adapt_sep,final_spike_sep,final_adapt_sep\n");

    int total_points = (int)(XO_INC.size() * XO_TAU.size());
    int point_idx = 0;

    auto t_start = std::chrono::steady_clock::now();

    for (double tau : XO_TAU) {
        for (double inc : XO_INC) {
            point_idx++;
            printf("\n--- Point %d/%d: inc=%.4f, tau=%.1f ---\n",
                   point_idx, total_points, inc, tau);

            // Dynamical overrides
            DynamicalOverrides dyn_ovr;
            dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn_ovr.adapt_inc = inc;
            dyn_ovr.adapt_tau = tau;
            dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

            // Rate calibration
            auto cal = xo_calibrate(cal_samples, dyn_ovr, sim_cfg, n_workers);
            printf("  g_tonic=%.4f (%s) -> %.1f Hz\n",
                   cal.g_tonic, (cal.reversal < -1.0) ? "inh" : "exc", cal.rate_hz);

            // Run all trials and collect state data
            double bin_ms = XO_BIN_MS;

            // First pass: determine consistent bin count
            // Run trials and collect data
            struct TrialData {
                int digit;
                std::vector<std::vector<double>> spike_bins;  // [bin][neuron]
                std::vector<std::vector<double>> adapt_bins;
            };
            std::vector<TrialData> trial_results(n_trials);
            std::atomic<int> completed(0);
            int shared_n_res = 0;
            int shared_min_bins = INT_MAX;

            #pragma omp parallel num_threads(n_workers)
            {
                SphericalNetwork net;
                ZoneInfo zone_info;
                load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
                apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
                StdMasks masks = build_std_masks(net, zone_info);

                if (cal.g_tonic != 0.0) {
                    net.tonic_conductance.assign(net.n_neurons, 0.0);
                    net.tonic_reversal.assign(net.n_neurons, 0.0);
                    for (int nid : zone_info.reservoir_zone_indices) {
                        net.tonic_conductance[nid] = cal.g_tonic;
                        net.tonic_reversal[nid] = cal.reversal;
                    }
                }

                int n_res = (int)zone_info.reservoir_zone_indices.size();

                #pragma omp single
                shared_n_res = n_res;

                rng_seed(604 + 99 + (uint64_t)omp_get_thread_num() * 1000 + (uint64_t)getpid());

                #pragma omp for schedule(dynamic, 1)
                for (int ti = 0; ti < n_trials; ti++) {
                    const auto& sample = all_samples[trials[ti].sample_idx];

                    SimConfig sc = sim_cfg;
                    double stim_end = *std::max_element(sample.spike_times_ms.begin(),
                                                         sample.spike_times_ms.end());
                    sc.audio_duration_ms = stim_end + 5.0;

                    double total_ms = sc.audio_duration_ms + sc.post_stimulus_ms;
                    int n_bins = (int)std::ceil(total_ms / bin_ms);

                    // Snapshot times for adaptation
                    std::vector<double> snap_times;
                    for (int b = 0; b < n_bins; b++)
                        snap_times.push_back((b + 1) * bin_ms - 0.05);

                    auto result = run_sample_with_std(net, sample, zone_info, sc,
                                                       STD_U, STD_TAU_REC, masks, snap_times);

                    // Build reservoir index map
                    std::unordered_map<int, int> idx_to_pos;
                    for (int p = 0; p < n_res; p++)
                        idx_to_pos[zone_info.reservoir_zone_indices[p]] = p;

                    // Bin spikes
                    std::vector<std::vector<double>> spike_bins(n_bins, std::vector<double>(n_res, 0.0));
                    int n_steps = (int)result.activity_record.size();
                    for (int step = 0; step < n_steps; step++) {
                        double t = step * sc.dt;
                        int b = std::min((int)(t / bin_ms), n_bins - 1);
                        for (int nid : result.activity_record[step]) {
                            auto it = idx_to_pos.find(nid);
                            if (it != idx_to_pos.end())
                                spike_bins[b][it->second] += 1.0;
                        }
                    }

                    // Extract adaptation snapshots
                    std::vector<std::vector<double>> adapt_bins(n_bins, std::vector<double>(n_res, 0.0));
                    int n_snaps = std::min(n_bins, (int)result.adapt_snapshots.size());
                    for (int b = 0; b < n_snaps; b++) {
                        for (int p = 0; p < n_res; p++) {
                            int nid = zone_info.reservoir_zone_indices[p];
                            if (nid < (int)result.adapt_snapshots[b].size())
                                adapt_bins[b][p] = result.adapt_snapshots[b][nid];
                        }
                    }

                    trial_results[ti] = {trials[ti].digit,
                                          std::move(spike_bins),
                                          std::move(adapt_bins)};

                    #pragma omp critical
                    {
                        if (n_bins < shared_min_bins)
                            shared_min_bins = n_bins;
                    }

                    ++completed;
                }
            }

            // Normalize to min bins and compute separability
            int n_res = shared_n_res;
            int min_bins = shared_min_bins;

            std::vector<std::vector<std::vector<double>>> spike_data(n_trials);
            std::vector<std::vector<std::vector<double>>> adapt_data(n_trials);
            std::vector<int> trial_digits(n_trials);

            for (int ti = 0; ti < n_trials; ti++) {
                trial_digits[ti] = trial_results[ti].digit;
                spike_data[ti].resize(min_bins);
                adapt_data[ti].resize(min_bins);
                for (int b = 0; b < min_bins; b++) {
                    spike_data[ti][b] = trial_results[ti].spike_bins[b];
                    adapt_data[ti][b] = trial_results[ti].adapt_bins[b];
                }
            }

            auto sep = compute_separability(spike_data, adapt_data, trial_digits,
                                             min_bins, n_res, bin_ms);

            double peak_spike = *std::max_element(sep.spike_sep.begin(), sep.spike_sep.end());
            double peak_adapt = *std::max_element(sep.adapt_sep.begin(), sep.adapt_sep.end());
            double final_spike = sep.spike_sep.back();
            double final_adapt = sep.adapt_sep.back();

            printf("  crossover=%.1f ms, peak_spike=%.1f, peak_adapt=%.1f\n",
                   sep.crossover_ms, peak_spike, peak_adapt);

            fprintf(csv, "%.4f,%.1f,%.6f,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f\n",
                    inc, tau, cal.g_tonic, cal.rate_hz, sep.crossover_ms,
                    peak_spike, peak_adapt, final_spike, final_adapt);
            fflush(csv);

            auto t_now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
            double per_point = elapsed_s / point_idx;
            double remaining = per_point * (total_points - point_idx);
            printf("  ETA: %.0f min remaining\n", remaining / 60.0);
        }
    }

    fclose(csv);
    printf("\n=== CROSSOVER SWEEP COMPLETE ===\n");
    printf("  Wrote %s\n", csv_path.c_str());
    return 0;
}
