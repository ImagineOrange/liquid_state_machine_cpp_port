#include "experiments.h"
#include <atomic>

// ============================================================
// HIDDEN STATES EXPERIMENT
// Record spike and adaptation-conductance trajectories for
// single-digit presentations at the WM optimum, then dump
// per-bin state vectors for downstream PCA + separability analysis.
// ============================================================

static constexpr double HS_ADAPT_INC = 0.1710;
static constexpr double HS_ADAPT_TAU = 5000.0;
static constexpr int    HS_TRIALS_PER_DIGIT = 3;
static const std::vector<int> HS_DIGITS = {0, 1, 2, 3, 4, 5};

// ============================================================
// RATE CALIBRATION (single-digit, binary search for g_tonic)
// ============================================================
struct HsCalResult { double g_tonic, reversal, rate_hz; };

static double hs_measure_rate(const std::vector<AudioSample>& cal_samples,
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

static HsCalResult hs_calibrate(const std::vector<AudioSample>& cal_samples,
                                  const DynamicalOverrides& dyn_ovr,
                                  const SimConfig& sim_cfg, int n_workers) {
    double natural = hs_measure_rate(cal_samples, dyn_ovr, sim_cfg, n_workers, 0.0, 0.0);
    printf("  cal[0] natural -> %.1f Hz (target=%.1f)\n", natural, RATE_TARGET_HZ);

    if (std::abs(natural - RATE_TARGET_HZ) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural};

    double reversal = (natural > RATE_TARGET_HZ) ? -80.0 : 0.0;
    printf("  cal: %s mode\n", (reversal < -1.0) ? "inhibitory" : "excitatory");

    double lo = 0.0, hi = 5.0;
    HsCalResult best = {0.0, reversal, natural};

    for (int iter = 1; iter <= 12; iter++) {
        double mid = (lo + hi) / 2.0;
        double rate = hs_measure_rate(cal_samples, dyn_ovr, sim_cfg, n_workers, mid, reversal);
        printf("  cal[%d] g=%.4f -> %.1f Hz\n", iter, mid, rate);

        best = {mid, reversal, rate};
        if (std::abs(rate - RATE_TARGET_HZ) <= RATE_TOLERANCE_HZ) break;

        bool overshot = (reversal < -1.0) ? (rate < RATE_TARGET_HZ)
                                           : (rate > RATE_TARGET_HZ);
        if (overshot) hi = mid; else lo = mid;
    }
    return best;
}

// ============================================================
// MAIN ENTRY POINT
// ============================================================
int run_hidden_states(int /*argc*/, char** /*argv*/,
                       int n_workers, const std::string& output_dir,
                       const std::string& data_dir) {
    printf("\n=== HIDDEN STATES EXPERIMENT ===\n");
    printf("  WM optimum: inc=%.4f, tau=%.1f\n", HS_ADAPT_INC, HS_ADAPT_TAU);
    printf("  Digits: 0-5, %d trials each\n", HS_TRIALS_PER_DIGIT);

    // --- Load samples ---
    printf("\nLoading audio samples...\n");
    auto all_samples = load_audio_samples(data_dir, HS_DIGITS,
                                           SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);

    // Pick first N trials per digit
    std::map<int, std::vector<int>> by_digit;
    for (int i = 0; i < (int)all_samples.size(); i++)
        by_digit[all_samples[i].digit].push_back(i);

    struct Trial { int sample_idx; int digit; int trial_num; };
    std::vector<Trial> trials;
    for (int d : HS_DIGITS) {
        auto& pool = by_digit[d];
        int n = std::min(HS_TRIALS_PER_DIGIT, (int)pool.size());
        for (int t = 0; t < n; t++)
            trials.push_back({pool[t], d, t});
    }
    printf("  Total trials: %d\n", (int)trials.size());

    // --- Dynamical overrides ---
    DynamicalOverrides dyn_ovr;
    dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
    dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
    dyn_ovr.adapt_inc = HS_ADAPT_INC;
    dyn_ovr.adapt_tau = HS_ADAPT_TAU;
    dyn_ovr.nmda_tau = FIXED_NMDA_TAU;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;

    // --- Rate calibration ---
    printf("\nCalibrating tonic conductance...\n");
    // Use first 30 samples for calibration
    int cal_n = std::min(30, (int)all_samples.size());
    std::vector<AudioSample> cal_samples(all_samples.begin(), all_samples.begin() + cal_n);
    auto cal = hs_calibrate(cal_samples, dyn_ovr, sim_cfg, n_workers);
    printf("  Result: g_tonic=%.4f (%s) -> %.1f Hz\n\n",
           cal.g_tonic, (cal.reversal < -1.0) ? "inh" : "exc", cal.rate_hz);

    // --- Run all trials ---
    int n_trials = (int)trials.size();
    double bin_ms = BIN_MS;  // 20ms bins

    // Per-trial results: spike bins and adaptation snapshots
    struct TrajectoryResult {
        int digit, trial_num;
        int n_bins;
        Mat spike_bins;      // (n_bins x n_reservoir)
        Mat adapt_bins;      // (n_bins x n_reservoir)
    };
    std::vector<TrajectoryResult> results(n_trials);
    std::atomic<int> completed(0);

    printf("Running %d trials...\n", n_trials);

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

            // Build snapshot times: end of each 20ms bin
            std::vector<double> snap_times;
            for (int b = 0; b < n_bins; b++)
                snap_times.push_back((b + 1) * bin_ms - 0.05);  // just before bin end

            auto result = run_sample_with_std(net, sample, zone_info, sc,
                                               STD_U, STD_TAU_REC, masks, snap_times);

            // Build reservoir index map
            std::unordered_map<int, int> idx_to_pos;
            for (int p = 0; p < n_res; p++)
                idx_to_pos[zone_info.reservoir_zone_indices[p]] = p;

            // Bin spikes
            Mat spike_bins(n_bins, n_res, 0.0);
            int n_steps = (int)result.activity_record.size();
            for (int step = 0; step < n_steps; step++) {
                double t = step * sc.dt;
                int b = std::min((int)(t / bin_ms), n_bins - 1);
                for (int nid : result.activity_record[step]) {
                    auto it = idx_to_pos.find(nid);
                    if (it != idx_to_pos.end())
                        spike_bins(b, it->second) += 1.0;
                }
            }

            // Extract adaptation snapshots into matrix
            Mat adapt_bins(n_bins, n_res, 0.0);
            int n_snaps = std::min(n_bins, (int)result.adapt_snapshots.size());
            for (int b = 0; b < n_snaps; b++) {
                for (int p = 0; p < n_res; p++) {
                    int nid = zone_info.reservoir_zone_indices[p];
                    if (nid < (int)result.adapt_snapshots[b].size())
                        adapt_bins(b, p) = result.adapt_snapshots[b][nid];
                }
            }

            results[ti] = {trials[ti].digit, trials[ti].trial_num,
                           n_bins, std::move(spike_bins), std::move(adapt_bins)};

            int c = ++completed;
            #pragma omp critical
            printf("  %d/%d (digit=%d, trial=%d)\n", c, n_trials,
                   trials[ti].digit, trials[ti].trial_num);
        }
    }

    // --- Write output ---
    // Normalize bin counts to same number of bins (use minimum across trials)
    int min_bins = INT_MAX;
    int n_res = 0;
    for (auto& r : results) {
        min_bins = std::min(min_bins, r.n_bins);
        if (r.spike_bins.cols > 0) n_res = r.spike_bins.cols;
    }
    printf("\nUsing %d bins (%.0f ms each), %d reservoir neurons\n",
           min_bins, bin_ms, n_res);

    // Write spike trajectories
    {
        std::string fpath = output_dir + "/spike_trajectories.csv";
        FILE* f = fopen(fpath.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", fpath.c_str()); return 1; }

        // Header
        fprintf(f, "trial,digit,time_ms");
        for (int n = 0; n < n_res; n++) fprintf(f, ",n%d", n);
        fprintf(f, "\n");

        int trial_id = 0;
        for (auto& r : results) {
            for (int b = 0; b < min_bins; b++) {
                fprintf(f, "%d,%d,%.1f", trial_id, r.digit, (b + 0.5) * bin_ms);
                for (int n = 0; n < n_res; n++)
                    fprintf(f, ",%.0f", r.spike_bins(b, n));
                fprintf(f, "\n");
            }
            trial_id++;
        }
        fclose(f);
        printf("  Wrote %s\n", fpath.c_str());
    }

    // Write adaptation trajectories
    {
        std::string fpath = output_dir + "/adapt_trajectories.csv";
        FILE* f = fopen(fpath.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", fpath.c_str()); return 1; }

        fprintf(f, "trial,digit,time_ms");
        for (int n = 0; n < n_res; n++) fprintf(f, ",n%d", n);
        fprintf(f, "\n");

        int trial_id = 0;
        for (auto& r : results) {
            for (int b = 0; b < min_bins; b++) {
                fprintf(f, "%d,%d,%.1f", trial_id, r.digit, (b + 0.5) * bin_ms);
                for (int n = 0; n < n_res; n++)
                    fprintf(f, ",%.6f", r.adapt_bins(b, n));
                fprintf(f, "\n");
            }
            trial_id++;
        }
        fclose(f);
        printf("  Wrote %s\n", fpath.c_str());
    }

    // Write metadata
    {
        std::string fpath = output_dir + "/hidden_states_meta.json";
        FILE* f = fopen(fpath.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", fpath.c_str()); return 1; }

        fprintf(f, "{\n");
        fprintf(f, "  \"adapt_inc\": %.4f,\n", HS_ADAPT_INC);
        fprintf(f, "  \"adapt_tau\": %.1f,\n", HS_ADAPT_TAU);
        fprintf(f, "  \"g_tonic\": %.6f,\n", cal.g_tonic);
        fprintf(f, "  \"tonic_reversal\": %.1f,\n", cal.reversal);
        fprintf(f, "  \"calibrated_rate_hz\": %.1f,\n", cal.rate_hz);
        fprintf(f, "  \"bin_ms\": %.1f,\n", bin_ms);
        fprintf(f, "  \"n_bins\": %d,\n", min_bins);
        fprintf(f, "  \"n_reservoir\": %d,\n", n_res);
        fprintf(f, "  \"n_trials\": %d,\n", n_trials);
        fprintf(f, "  \"trials_per_digit\": %d,\n", HS_TRIALS_PER_DIGIT);
        fprintf(f, "  \"digits\": [0, 1, 2, 3, 4, 5]\n");
        fprintf(f, "}\n");
        fclose(f);
        printf("  Wrote %s\n", fpath.c_str());
    }

    printf("\n=== HIDDEN STATES COMPLETE ===\n");
    return 0;
}
