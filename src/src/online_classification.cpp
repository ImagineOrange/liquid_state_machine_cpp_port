#include "experiments.h"
#include <atomic>

// ============================================================
// GRID AXES — identical to classification.cpp / wm_sweep.cpp
// ============================================================
static const std::vector<double> SERIAL_UNIFIED_INC = {
    0.0,
    0.0050, 0.0067, 0.0090, 0.0121, 0.0162,
    0.0218, 0.0292, 0.0392, 0.0527, 0.0707,
    0.0949, 0.1274, 0.1710, 0.2295, 0.3081,
    0.4135, 0.5550, 0.7450, 1.0000,
};
static const std::vector<double> SERIAL_UNIFIED_TAU = {
    30.0, 43.2, 62.3, 89.8, 129.4,
    186.5, 268.7, 387.3, 558.1, 804.4,
    1159.2, 1670.6, 2407.5, 3469.5, 5000.0,
};

// ============================================================
// SERIAL CLASSIFICATION CONSTANTS
// ============================================================
static constexpr double SERIAL_GAP_MS     = 0.0;
static constexpr int    N_SERIAL_DIGITS   = 3;
static constexpr int    N_SEQUENCES       = 1500;
static constexpr int    SEQUENCE_SEED     = 42;
static constexpr int    SERIAL_CAL_N_SEQS = 200;
static constexpr double SERIAL_G_TONIC_HI = 5.0;

// ============================================================
// GRID POINT
// ============================================================
struct SerialGridPoint {
    double adapt_inc, adapt_tau;
    std::string point_id;
    int inc_idx, tau_idx;
};

static std::vector<SerialGridPoint> build_serial_grid_points() {
    std::vector<SerialGridPoint> points;
    points.reserve(SERIAL_UNIFIED_INC.size() * SERIAL_UNIFIED_TAU.size());
    for (int ii = 0; ii < (int)SERIAL_UNIFIED_INC.size(); ii++) {
        for (int ti = 0; ti < (int)SERIAL_UNIFIED_TAU.size(); ti++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "inc%.4f_tau%.1f",
                     SERIAL_UNIFIED_INC[ii], SERIAL_UNIFIED_TAU[ti]);
            points.push_back({SERIAL_UNIFIED_INC[ii], SERIAL_UNIFIED_TAU[ti], buf, ii, ti});
        }
    }
    return points;
}

// ============================================================
// SERIAL SEQUENCE
// ============================================================
struct SerialSequence {
    AudioSample compound;           // concatenated 3-digit spike train
    int digits[N_SERIAL_DIGITS];
    int sample_indices[N_SERIAL_DIGITS];
    double digit_onset_ms[N_SERIAL_DIGITS];
    double digit_end_ms[N_SERIAL_DIGITS];
    double total_stim_end_ms;
};

struct SerialBinInfo {
    int max_bins[N_SERIAL_DIGITS];  // max 20ms bins per position
    int total_max_bins;             // max across whole trial
};

// ============================================================
// SEQUENCE GENERATION
// ============================================================
static std::vector<SerialSequence> generate_serial_sequences(
    const std::vector<AudioSample>& samples,
    int n_sequences, int seed)
{
    // Organize samples by digit
    std::map<int, std::vector<int>> by_digit;
    for (int i = 0; i < (int)samples.size(); i++)
        by_digit[samples[i].digit].push_back(i);

    std::vector<int> digits;
    for (auto& [d, _] : by_digit) digits.push_back(d);
    std::sort(digits.begin(), digits.end());
    int n_digits = (int)digits.size();

    std::mt19937 rng(seed);

    std::vector<SerialSequence> seqs;
    seqs.reserve(n_sequences);

    for (int s = 0; s < n_sequences; s++) {
        SerialSequence seq;
        int prev_digit = -1;

        // Pick digits with no-consecutive-repeat constraint
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            int d;
            if (prev_digit < 0) {
                // Position 0: uniform over all digits
                d = digits[std::uniform_int_distribution<int>(0, n_digits - 1)(rng)];
            } else {
                // Draw from digits excluding prev_digit
                std::vector<int> valid;
                for (int di : digits) {
                    if (di != prev_digit) valid.push_back(di);
                }
                d = valid[std::uniform_int_distribution<int>(0, (int)valid.size() - 1)(rng)];
            }

            auto& pool = by_digit[d];
            int si = std::uniform_int_distribution<int>(0, (int)pool.size() - 1)(rng);
            seq.digits[p] = d;
            seq.sample_indices[p] = pool[si];
            prev_digit = d;
        }

        // Build compound spike train
        double current_offset = 0.0;
        AudioSample compound;
        compound.digit = seq.digits[0];
        compound.speaker = samples[seq.sample_indices[0]].speaker;

        std::string fname = "";
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            const auto& samp = samples[seq.sample_indices[p]];
            if (p > 0) fname += "+";
            fname += samp.filename;

            seq.digit_onset_ms[p] = current_offset;

            // Determine digit duration from max spike time
            double digit_dur = 0.0;
            if (!samp.spike_times_ms.empty()) {
                digit_dur = *std::max_element(samp.spike_times_ms.begin(),
                                               samp.spike_times_ms.end());
            }
            seq.digit_end_ms[p] = current_offset + digit_dur;

            // Append spikes shifted by current_offset
            for (size_t k = 0; k < samp.spike_times_ms.size(); k++) {
                compound.spike_times_ms.push_back(samp.spike_times_ms[k] + current_offset);
                compound.freq_bin_indices.push_back(samp.freq_bin_indices[k]);
            }

            current_offset += digit_dur;
            if (p < N_SERIAL_DIGITS - 1) {
                current_offset += SERIAL_GAP_MS;
            }
        }

        compound.filename = fname;
        seq.total_stim_end_ms = current_offset;
        seq.compound = std::move(compound);

        seqs.push_back(std::move(seq));
    }

    // Shuffle
    std::shuffle(seqs.begin(), seqs.end(), rng);
    return seqs;
}

// ============================================================
// BIN INFO
// ============================================================
static SerialBinInfo compute_serial_bin_info(
    const std::vector<SerialSequence>& sequences)
{
    SerialBinInfo info = {{0, 0, 0}, 0};
    for (auto& seq : sequences) {
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            double dur = seq.digit_end_ms[p] - seq.digit_onset_ms[p];
            int n = (int)std::ceil(dur / BIN_MS);
            if (n > info.max_bins[p]) info.max_bins[p] = n;
        }
        double total_dur = seq.total_stim_end_ms + POST_STIM_MS;
        int total = (int)std::ceil(total_dur / BIN_MS);
        if (total > info.total_max_bins) info.total_max_bins = total;
    }
    return info;
}

// ============================================================
// PER-POSITION ACTIVITY BINNING
// ============================================================
struct SerialBinResult {
    Mat position_bins[N_SERIAL_DIGITS];  // (max_bins[p], n_reservoir)
    Mat full_trial_bins;                  // (total_max_bins, n_reservoir)
    int total_spikes;
};

static SerialBinResult bin_serial_activity(
    const std::vector<std::vector<int>>& activity_record,
    const std::vector<int>& reservoir_indices,
    double dt, const SerialSequence& seq,
    const SerialBinInfo& bin_info)
{
    int n_res = (int)reservoir_indices.size();
    std::unordered_map<int, int> idx_to_pos;
    for (int p = 0; p < n_res; p++) idx_to_pos[reservoir_indices[p]] = p;

    SerialBinResult result;
    for (int p = 0; p < N_SERIAL_DIGITS; p++) {
        result.position_bins[p] = Mat(bin_info.max_bins[p], n_res, 0.0);
    }
    result.full_trial_bins = Mat(bin_info.total_max_bins, n_res, 0.0);
    result.total_spikes = 0;

    int n_steps = (int)activity_record.size();
    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;

        // Full trial bin
        int full_b = std::min((int)(t / BIN_MS), bin_info.total_max_bins - 1);

        // Determine which position this timestep belongs to
        int pos = -1;
        int pos_bin = -1;
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            if (t >= seq.digit_onset_ms[p] && t < seq.digit_end_ms[p]) {
                pos = p;
                pos_bin = (int)((t - seq.digit_onset_ms[p]) / BIN_MS);
                if (pos_bin >= bin_info.max_bins[p]) pos_bin = bin_info.max_bins[p] - 1;
                break;
            }
        }

        for (int nid : activity_record[step]) {
            auto it = idx_to_pos.find(nid);
            if (it != idx_to_pos.end()) {
                int col = it->second;
                result.full_trial_bins(full_b, col) += 1.0;
                if (pos >= 0 && pos_bin >= 0) {
                    result.position_bins[pos](pos_bin, col) += 1.0;
                }
                result.total_spikes++;
            }
        }
    }
    return result;
}

// ============================================================
// SIMULATION WORKER
// ============================================================
struct SerialWorkerResult {
    Mat position_bins[N_SERIAL_DIGITS];
    Mat full_trial_bins;
    int total_spikes;
    int digits[N_SERIAL_DIGITS];
    int seq_idx;
    double isi_cv;
    double adapt_at_onset[N_SERIAL_DIGITS];  // [0] always 0 (start of trial)
    double adapt_at_stim_end;
    double nmda_at_onset[N_SERIAL_DIGITS];   // mean g_nmda across reservoir
    double std_resource_at_onset[N_SERIAL_DIGITS];  // mean x_resource across reservoir
    double ge_at_onset[N_SERIAL_DIGITS];     // mean g_e (AMPA) across reservoir — negative control
};

static SerialWorkerResult serial_sim_worker(
    SphericalNetwork& net, const SerialSequence& seq, int seq_idx,
    const ZoneInfo& zone_info, const SimConfig& sim_cfg,
    const StdMasks& masks, const SerialBinInfo& bin_info)
{
    double dt = sim_cfg.dt;

    // Record adaptation at digit 2 onset, digit 3 onset, and stim end
    auto result = run_sample_with_std(net, seq.compound, zone_info, sim_cfg,
                                       STD_U, STD_TAU_REC, masks,
                                       {seq.digit_onset_ms[1], seq.digit_onset_ms[2],
                                        seq.total_stim_end_ms});

    // Bin activity per position
    auto br = bin_serial_activity(result.activity_record, zone_info.reservoir_zone_indices,
                                   dt, seq, bin_info);

    // ISI CV across full trial
    std::set<int> res_set(zone_info.reservoir_zone_indices.begin(),
                           zone_info.reservoir_zone_indices.end());
    std::unordered_map<int, double> last_spike;
    std::vector<double> all_isis;

    int n_steps = (int)result.activity_record.size();
    for (int step = 0; step < n_steps; step++) {
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

    // Helper: mean of a snapshot vector across reservoir neurons
    auto reservoir_mean = [&](const std::vector<double>& snap) -> double {
        if (snap.empty()) return 0.0;
        double sum = 0;
        for (int nid : zone_info.reservoir_zone_indices) sum += snap[nid];
        return sum / zone_info.reservoir_zone_indices.size();
    };

    // Extract conductance state at each digit onset from snapshots
    // Snapshots 0,1 = digit 2,3 onsets; snapshot 2 = stim end
    // Digit 1 onset = trial start (all zero / resource=1.0)
    double adapt_at_onset[N_SERIAL_DIGITS];
    double nmda_at_onset[N_SERIAL_DIGITS];
    double std_resource_at_onset[N_SERIAL_DIGITS];
    double ge_at_onset[N_SERIAL_DIGITS];

    // Position 0: trial start — network was reset
    adapt_at_onset[0] = 0.0;
    nmda_at_onset[0] = 0.0;
    std_resource_at_onset[0] = 1.0;
    ge_at_onset[0] = 0.0;

    for (int p = 1; p < N_SERIAL_DIGITS; p++) {
        int snap_idx = p - 1;  // snapshot 0 = digit 2 onset, 1 = digit 3 onset
        adapt_at_onset[p] = ((int)result.adapt_snapshots.size() > snap_idx)
                            ? reservoir_mean(result.adapt_snapshots[snap_idx]) : 0.0;
        nmda_at_onset[p] = ((int)result.nmda_snapshots.size() > snap_idx)
                           ? reservoir_mean(result.nmda_snapshots[snap_idx]) : 0.0;
        std_resource_at_onset[p] = ((int)result.std_resource_snapshots.size() > snap_idx)
                                   ? reservoir_mean(result.std_resource_snapshots[snap_idx]) : 1.0;
        ge_at_onset[p] = ((int)result.ge_snapshots.size() > snap_idx)
                         ? reservoir_mean(result.ge_snapshots[snap_idx]) : 0.0;
    }

    // Adaptation at stim end (3rd snapshot, index 2)
    double adapt_stim_end = ((int)result.adapt_snapshots.size() >= 3)
                            ? reservoir_mean(result.adapt_snapshots[2]) : 0.0;

    SerialWorkerResult wr;
    for (int p = 0; p < N_SERIAL_DIGITS; p++) {
        wr.position_bins[p] = std::move(br.position_bins[p]);
        wr.digits[p] = seq.digits[p];
        wr.adapt_at_onset[p] = adapt_at_onset[p];
        wr.nmda_at_onset[p] = nmda_at_onset[p];
        wr.std_resource_at_onset[p] = std_resource_at_onset[p];
        wr.ge_at_onset[p] = ge_at_onset[p];
    }
    wr.full_trial_bins = std::move(br.full_trial_bins);
    wr.total_spikes = br.total_spikes;
    wr.seq_idx = seq_idx;
    wr.isi_cv = isi_cv;
    wr.adapt_at_stim_end = adapt_stim_end;

    return wr;
}

// ============================================================
// RESULT AGGREGATION
// ============================================================
struct SerialAllResult {
    std::vector<std::array<Mat, N_SERIAL_DIGITS>> position_bins_list;
    std::vector<Mat> full_trial_bins_list;
    std::vector<std::array<int, N_SERIAL_DIGITS>> digit_list;
    std::vector<double> total_spikes;
    int n_reservoir;
    std::vector<double> isi_cvs;
    std::vector<std::array<double, N_SERIAL_DIGITS>> adapt_onsets;
    std::vector<double> adapt_stim_ends;
    std::vector<std::array<double, N_SERIAL_DIGITS>> nmda_onsets;
    std::vector<std::array<double, N_SERIAL_DIGITS>> std_resource_onsets;
    std::vector<std::array<double, N_SERIAL_DIGITS>> ge_onsets;
};

// ============================================================
// SET TONIC CONDUCTANCE
// ============================================================
static void set_serial_tonic_conductance(SphericalNetwork& net,
                                          const ZoneInfo& zone_info,
                                          double g_tonic, double reversal) {
    net.tonic_conductance.assign(net.n_neurons, 0.0);
    net.tonic_reversal.assign(net.n_neurons, 0.0);
    for (int nid : zone_info.reservoir_zone_indices) {
        net.tonic_conductance[nid] = g_tonic;
        net.tonic_reversal[nid] = reversal;
    }
}

// ============================================================
// PARALLEL RUNNER
// ============================================================
static SerialAllResult run_all_serial_sequences(
    const NetworkConfig& cfg,
    const std::vector<SerialSequence>& sequences,
    const DynamicalOverrides& dyn_ovr,
    const SimConfig& sim_cfg,
    const SerialBinInfo& bin_info,
    int n_workers,
    bool verbose = true,
    double g_tonic = 0.0,
    double tonic_rev = -80.0)
{
    int n_seqs = (int)sequences.size();
    SerialAllResult out;
    out.position_bins_list.resize(n_seqs);
    out.full_trial_bins_list.resize(n_seqs);
    out.digit_list.resize(n_seqs);
    out.total_spikes.resize(n_seqs);
    out.isi_cvs.resize(n_seqs);
    out.adapt_onsets.resize(n_seqs);
    out.adapt_stim_ends.resize(n_seqs);
    out.nmda_onsets.resize(n_seqs);
    out.std_resource_onsets.resize(n_seqs);
    out.ge_onsets.resize(n_seqs);
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
            set_serial_tonic_conductance(net, zone_info, g_tonic, tonic_rev);

        rng_seed(cfg.n_neurons + 42 + (uint64_t)omp_get_thread_num() * 1000 +
                 (uint64_t)getpid());

        #pragma omp single
        {
            out.n_reservoir = (int)zone_info.reservoir_zone_indices.size();
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_seqs; i++) {
            SimConfig seq_sim = sim_cfg;
            seq_sim.audio_duration_ms = sequences[i].total_stim_end_ms + 5.0;

            auto wr = serial_sim_worker(net, sequences[i], i, zone_info, seq_sim, masks, bin_info);

            for (int p = 0; p < N_SERIAL_DIGITS; p++) {
                out.position_bins_list[i][p] = std::move(wr.position_bins[p]);
                out.digit_list[i][p] = wr.digits[p];
                out.adapt_onsets[i][p] = wr.adapt_at_onset[p];
                out.nmda_onsets[i][p] = wr.nmda_at_onset[p];
                out.std_resource_onsets[i][p] = wr.std_resource_at_onset[p];
                out.ge_onsets[i][p] = wr.ge_at_onset[p];
            }
            out.full_trial_bins_list[i] = std::move(wr.full_trial_bins);
            out.total_spikes[i] = wr.total_spikes;
            out.isi_cvs[i] = wr.isi_cv;
            out.adapt_stim_ends[i] = wr.adapt_at_stim_end;

            int c = ++completed;
            if (verbose && (c % 100 == 0 || c == n_seqs)) {
                #pragma omp critical
                printf("        %5d/%d\n", c, n_seqs);
            }
        }
    }

    return out;
}

// ============================================================
// RATE MEASUREMENT
// ============================================================
static double serial_measure_rate(
    const NetworkConfig& cfg,
    const std::vector<SerialSequence>& seqs_subset,
    const DynamicalOverrides& dyn_ovr,
    const SimConfig& sim_cfg,
    const SerialBinInfo& bin_info,
    int n_workers,
    double g_tonic = 0.0, double tonic_rev = -80.0)
{
    auto res = run_all_serial_sequences(cfg, seqs_subset, dyn_ovr, sim_cfg, bin_info,
                                         n_workers, false, g_tonic, tonic_rev);
    int n = (int)seqs_subset.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double trial_dur_s = (seqs_subset[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        sum += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
    }
    return sum / n;
}

// ============================================================
// CALIBRATION
// ============================================================
struct SerialCalResult {
    double g_tonic;
    double reversal;
    double rate_hz;
};

static SerialCalResult serial_calibrate_tonic(
    const NetworkConfig& cfg,
    const DynamicalOverrides& dyn_ovr,
    const std::vector<SerialSequence>& cal_seqs,
    const SimConfig& sim_cfg,
    const SerialBinInfo& bin_info,
    int n_workers,
    double target_rate,
    double initial_guess = 0.0,
    double prev_reversal = -80.0)
{
    // Measure natural rate
    double natural_rate = serial_measure_rate(cfg, cal_seqs, dyn_ovr, sim_cfg, bin_info,
                                               n_workers, 0.0);
    printf("    cal[0] g=0 (natural) -> %.1f Hz (target=%.1f)\n", natural_rate, target_rate);

    if (std::abs(natural_rate - target_rate) <= RATE_TOLERANCE_HZ)
        return {0.0, 0.0, natural_rate};

    double reversal = (natural_rate > target_rate) ? -80.0 : 0.0;
    const char* mode = (reversal < -1.0) ? "inhibitory" : "excitatory";
    printf("    cal: natural=%.1f Hz, target=%.1f Hz -> %s mode\n",
           natural_rate, target_rate, mode);

    double lo = 0.0, hi = SERIAL_G_TONIC_HI;
    int iteration = 1;

    // Try warm-start
    if (initial_guess > 0.0 && prev_reversal == reversal) {
        double rate = serial_measure_rate(cfg, cal_seqs, dyn_ovr, sim_cfg, bin_info,
                                           n_workers, initial_guess, reversal);
        printf("    cal[%d] g=%.4f -> %.1f Hz\n", iteration, initial_guess, rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, reversal, rate};
        bool overshot = (reversal < -1.0) ? (rate < target_rate) : (rate > target_rate);
        if (overshot) hi = initial_guess; else lo = initial_guess;
    }

    // Binary search
    for (;; iteration++) {
        double mid = (lo + hi) / 2.0;
        double rate = serial_measure_rate(cfg, cal_seqs, dyn_ovr, sim_cfg, bin_info,
                                           n_workers, mid, reversal);
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
// PER-POSITION CLASSIFICATION
// ============================================================
struct SerialClassifyResult {
    double accuracy[N_SERIAL_DIGITS];
    double accuracy_std[N_SERIAL_DIGITS];
    std::vector<double> per_repeat_accuracy[N_SERIAL_DIGITS];
    double serial_accuracy;       // mean of position accuracies
    double serial_accuracy_std;
    std::vector<double> serial_per_repeat;  // mean across positions per repeat
};

static SerialClassifyResult classify_serial_positions(
    const std::vector<std::array<Mat, N_SERIAL_DIGITS>>& position_bins_list,
    const std::vector<std::array<int, N_SERIAL_DIGITS>>& digit_list,
    int n_reservoir, const SerialBinInfo& bin_info)
{
    int n_seqs = (int)position_bins_list.size();
    SerialClassifyResult out;

    // Classify each position independently
    for (int p = 0; p < N_SERIAL_DIGITS; p++) {
        int n_features = bin_info.max_bins[p] * n_reservoir;

        // Build feature matrix
        Mat X(n_seqs, n_features, 0.0);
        std::vector<int> y(n_seqs);
        for (int i = 0; i < n_seqs; i++) {
            y[i] = digit_list[i][p];
            int actual_bins = position_bins_list[i][p].rows;
            int actual_res = position_bins_list[i][p].cols;
            for (int b = 0; b < std::min(bin_info.max_bins[p], actual_bins); b++) {
                for (int r = 0; r < std::min(n_reservoir, actual_res); r++) {
                    X(i, b * n_reservoir + r) = position_bins_list[i][p](b, r);
                }
            }
        }

        // Ridge CV: 5 folds x 5 repeats
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
                        X_train(i, j) = X(split.train[i], j);
                    y_train[i] = y[split.train[i]];
                }

                Mat X_test((int)split.test.size(), n_features);
                std::vector<int> y_test(split.test.size());
                for (int i = 0; i < (int)split.test.size(); i++) {
                    for (int j = 0; j < n_features; j++)
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

        out.accuracy[p] = mean;
        out.accuracy_std[p] = std::sqrt(var);
        out.per_repeat_accuracy[p] = repeat_accs;
    }

    // Aggregate: mean across positions per repeat
    out.serial_per_repeat.resize(N_CV_REPEATS);
    for (int rep = 0; rep < N_CV_REPEATS; rep++) {
        double s = 0;
        for (int p = 0; p < N_SERIAL_DIGITS; p++)
            s += out.per_repeat_accuracy[p][rep];
        out.serial_per_repeat[rep] = s / N_SERIAL_DIGITS;
    }

    double mean = 0;
    for (double a : out.serial_per_repeat) mean += a;
    mean /= out.serial_per_repeat.size();
    double var = 0;
    for (double a : out.serial_per_repeat) var += (a - mean) * (a - mean);
    var /= out.serial_per_repeat.size();

    out.serial_accuracy = mean;
    out.serial_accuracy_std = std::sqrt(var);

    return out;
}

// ============================================================
// PER-BIN ACCURACY (full trial)
// ============================================================
static std::vector<double> serial_classify_per_bin(
    const std::vector<Mat>& full_trial_bins_list,
    const std::vector<std::array<int, N_SERIAL_DIGITS>>& digit_list,
    int n_bins, int n_reservoir)
{
    // Use position-0 digit as label for full-trial per-bin accuracy
    int n_samples = (int)full_trial_bins_list.size();
    std::vector<int> y(n_samples);
    for (int i = 0; i < n_samples; i++) y[i] = digit_list[i][0];

    // Per-bin: same pattern as classification.cpp's classify_per_bin
    // But we iterate over all bins in the full trial
    std::vector<double> bin_accs(n_bins, 0.0);

    for (int b = 0; b < n_bins; b++) {
        Mat X_bin(n_samples, n_reservoir, 0.0);
        for (int i = 0; i < n_samples; i++) {
            if (b < full_trial_bins_list[i].rows) {
                for (int r = 0; r < std::min(n_reservoir, full_trial_bins_list[i].cols); r++) {
                    X_bin(i, r) = full_trial_bins_list[i](b, r);
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
// PER-POSITION PER-BIN ACCURACY
// ============================================================
static std::vector<double> serial_classify_per_bin_position(
    const std::vector<std::array<Mat, N_SERIAL_DIGITS>>& position_bins_list,
    const std::vector<std::array<int, N_SERIAL_DIGITS>>& digit_list,
    int pos, int n_bins, int n_reservoir)
{
    int n_samples = (int)position_bins_list.size();
    std::vector<int> y(n_samples);
    for (int i = 0; i < n_samples; i++) y[i] = digit_list[i][pos];

    std::vector<double> bin_accs(n_bins, 0.0);

    for (int b = 0; b < n_bins; b++) {
        Mat X_bin(n_samples, n_reservoir, 0.0);
        for (int i = 0; i < n_samples; i++) {
            if (b < position_bins_list[i][pos].rows) {
                for (int r = 0; r < std::min(n_reservoir, position_bins_list[i][pos].cols); r++) {
                    X_bin(i, r) = position_bins_list[i][pos](b, r);
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
static double serial_participation_ratio(
    const std::vector<std::array<Mat, N_SERIAL_DIGITS>>& position_bins_list,
    const std::vector<std::array<int, N_SERIAL_DIGITS>>& digit_list,
    int n_reservoir, const SerialBinInfo& bin_info)
{
    // Compute PR over all position bins pooled
    int n_seqs = (int)position_bins_list.size();
    int n_eff = std::min(n_seqs, n_reservoir);

    double pr_sum = 0;
    int pr_count = 0;

    for (int p = 0; p < N_SERIAL_DIGITS; p++) {
        for (int b = 0; b < bin_info.max_bins[p]; b++) {
            Mat X(n_seqs, n_reservoir, 0.0);
            for (int i = 0; i < n_seqs; i++) {
                if (b < position_bins_list[i][p].rows) {
                    for (int r = 0; r < std::min(n_reservoir, position_bins_list[i][p].cols); r++)
                        X(i, r) = position_bins_list[i][p](b, r);
                }
            }

            // Center
            std::vector<double> col_mean(n_reservoir, 0.0);
            for (int j = 0; j < n_reservoir; j++) {
                for (int i = 0; i < n_seqs; i++) col_mean[j] += X(i, j);
                col_mean[j] /= n_seqs;
            }
            for (int i = 0; i < n_seqs; i++)
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
                double lam = s * s / std::max(n_seqs - 1, 1);
                sum_lam += lam;
                sum_lam2 += lam * lam;
            }
            if (sum_lam2 > 1e-20) {
                pr_sum += (sum_lam * sum_lam) / (n_eff * sum_lam2);
                pr_count++;
            }
        }
    }

    return pr_count > 0 ? pr_sum / pr_count : 0.0;
}

// ============================================================
// BSA SERIAL BASELINE
// ============================================================
static SerialClassifyResult serial_bsa_baseline(
    const std::vector<SerialSequence>& sequences,
    const SerialBinInfo& bin_info)
{
    int n_seqs = (int)sequences.size();
    int n_channels = 128;

    // Extract BSA features per position
    std::vector<std::array<Mat, N_SERIAL_DIGITS>> bsa_bins(n_seqs);
    std::vector<std::array<int, N_SERIAL_DIGITS>> digit_list(n_seqs);

    for (int i = 0; i < n_seqs; i++) {
        auto& seq = sequences[i];
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            digit_list[i][p] = seq.digits[p];
            bsa_bins[i][p] = Mat(bin_info.max_bins[p], n_channels, 0.0);
        }

        // Bin each BSA spike into the correct position
        for (size_t k = 0; k < seq.compound.spike_times_ms.size(); k++) {
            double t = seq.compound.spike_times_ms[k];
            int ch = seq.compound.freq_bin_indices[k];
            if (ch < 0 || ch >= n_channels) continue;

            for (int p = 0; p < N_SERIAL_DIGITS; p++) {
                if (t >= seq.digit_onset_ms[p] && t < seq.digit_end_ms[p]) {
                    int b = (int)((t - seq.digit_onset_ms[p]) / BIN_MS);
                    if (b >= 0 && b < bin_info.max_bins[p]) {
                        bsa_bins[i][p](b, ch) += 1.0;
                    }
                    break;
                }
            }
        }
    }

    // Run per-position classification on BSA features (128 channels instead of n_reservoir)
    SerialBinInfo bsa_bin_info;
    for (int p = 0; p < N_SERIAL_DIGITS; p++) bsa_bin_info.max_bins[p] = bin_info.max_bins[p];
    bsa_bin_info.total_max_bins = bin_info.total_max_bins;

    return classify_serial_positions(bsa_bins, digit_list, n_channels, bsa_bin_info);
}

// ============================================================
// JSON CHECKPOINT
// ============================================================
static void serial_save_checkpoint(
    const std::string& results_dir,
    const std::vector<std::map<std::string, std::string>>& grid_results_json,
    const std::string& bsa_json,
    const std::string& baseline_json,
    double target_rate_hz,
    int n_sequences,
    double total_start_time,
    bool final_save)
{
    std::string fname = final_save ? "serial_classification_adaptation_sweep.json"
                                   : "serial_classification_adaptation_sweep_checkpoint.json";
    std::string path = results_dir + "/" + fname;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }

    double elapsed = now_seconds() - total_start_time;

    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"Serial Classification Adaptation Sweep (C++ Port)\",\n");
    fprintf(f, "  \"total_time_s\": %.1f,\n", elapsed);
    fprintf(f, "  \"task\": \"3-position serial 5-class digit classification\",\n");
    fprintf(f, "  \"digits\": [0, 1, 2, 3, 4],\n");
    fprintf(f, "  \"n_sequences\": %d,\n", n_sequences);
    fprintf(f, "  \"n_serial_digits\": %d,\n", N_SERIAL_DIGITS);
    fprintf(f, "  \"serial_gap_ms\": %.1f,\n", SERIAL_GAP_MS);
    fprintf(f, "  \"no_consecutive_repeats\": true,\n");
    fprintf(f, "  \"sequence_seed\": %d,\n", SEQUENCE_SEED);
    fprintf(f, "  \"serial_chance_level\": 0.2,\n");
    fprintf(f, "  \"rate_matching\": {\"target_rate_hz\": %.2f, \"tolerance_hz\": %.1f},\n",
            target_rate_hz, RATE_TOLERANCE_HZ);

    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); json_write_double_array(f, SERIAL_UNIFIED_INC); fprintf(f, ",\n");
    fprintf(f, "    \"unified_tau\": "); json_write_double_array(f, SERIAL_UNIFIED_TAU); fprintf(f, ",\n");
    fprintf(f, "    \"n_inc\": %d, \"n_tau\": %d\n",
            (int)SERIAL_UNIFIED_INC.size(), (int)SERIAL_UNIFIED_TAU.size());
    fprintf(f, "  },\n");

    fprintf(f, "  \"bsa_serial_baseline\": %s,\n", bsa_json.c_str());
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
// MAIN ENTRY POINT
// ============================================================
int run_serial_sweep(int argc, char** argv,
                     int n_workers, const std::string& output_dir,
                     const std::string& data_dir)
{
    auto grid_points = build_serial_grid_points();
    int n_grid = (int)grid_points.size();

    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  SERIAL CLASSIFICATION ADAPTATION PARAMETER SWEEP (C++ PORT)\n");
    printf("  Grid: %d x %d = %d points x 2 branches\n",
           (int)SERIAL_UNIFIED_INC.size(), (int)SERIAL_UNIFIED_TAU.size(), n_grid);
    printf("  Task: 3-digit serial classification (%dms gap, no consecutive repeats)\n",
           (int)SERIAL_GAP_MS);
    printf("  Sequences: %d per grid point\n", N_SEQUENCES);
    printf("  Workers: %d\n", n_workers);
    printf("  CV: StratifiedKFold(%d) x %d repeats\n", N_CV_FOLDS, N_CV_REPEATS);
    if (!g_snapshot_path.empty()) {
        printf("  SNAPSHOT: %s\n", g_snapshot_path.c_str());
    }
    printf("======================================================================\n");

    // 1. Load audio samples
    printf("\n[1] Loading audio samples...\n");
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS, SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded (%d per digit)\n", n_samples, SAMPLES_PER_DIGIT);

    // 2. Generate serial sequences
    printf("\n[2] Generating %d serial sequences...\n", N_SEQUENCES);
    auto sequences = generate_serial_sequences(samples, N_SEQUENCES, SEQUENCE_SEED);
    printf("  %d sequences generated (3 digits each, %dms gap)\n",
           (int)sequences.size(), (int)SERIAL_GAP_MS);

    // Compute bin info
    auto bin_info = compute_serial_bin_info(sequences);
    printf("  Max bins per position: [%d, %d, %d], total: %d\n",
           bin_info.max_bins[0], bin_info.max_bins[1], bin_info.max_bins[2],
           bin_info.total_max_bins);

    // Find max trial duration
    double max_total_stim_ms = 0;
    for (auto& seq : sequences) {
        if (seq.total_stim_end_ms > max_total_stim_ms)
            max_total_stim_ms = seq.total_stim_end_ms;
    }
    max_total_stim_ms += 5.0;

    // 3. BSA serial baseline
    printf("\n[3] Computing BSA serial baseline...\n");
    auto bsa_result = serial_bsa_baseline(sequences, bin_info);
    printf("  BSA serial: %.1f%% (pos1=%.1f%%, pos2=%.1f%%, pos3=%.1f%%)\n",
           bsa_result.serial_accuracy * 100,
           bsa_result.accuracy[0] * 100,
           bsa_result.accuracy[1] * 100,
           bsa_result.accuracy[2] * 100);

    // BSA JSON
    std::ostringstream bsa_oss;
    bsa_oss << std::fixed;
    bsa_oss << "{";
    bsa_oss.precision(10);
    bsa_oss << "\"accuracy\": " << bsa_result.serial_accuracy << ", ";
    bsa_oss.precision(6);
    bsa_oss << "\"accuracy_std\": " << bsa_result.serial_accuracy_std << ", ";
    bsa_oss.precision(10);
    for (int p = 0; p < N_SERIAL_DIGITS; p++) {
        bsa_oss << "\"accuracy_pos" << (p+1) << "\": " << bsa_result.accuracy[p] << ", ";
    }
    bsa_oss << "\"per_repeat_accuracy\": [";
    for (int r = 0; r < N_CV_REPEATS; r++) {
        if (r > 0) bsa_oss << ", ";
        bsa_oss << bsa_result.serial_per_repeat[r];
    }
    bsa_oss << "]}";
    std::string bsa_json = bsa_oss.str();

    // 4. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_total_stim_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = INPUT_STIM_CURRENT;

    // Calibration subset (first SERIAL_CAL_N_SEQS sequences)
    std::vector<SerialSequence> cal_seqs(
        sequences.begin(),
        sequences.begin() + std::min(SERIAL_CAL_N_SEQS, (int)sequences.size()));

    double target_rate_hz = RATE_TARGET_HZ;

    // 5. LHS-021 baseline
    printf("\n[4] Running LHS-021 baseline...\n");
    DynamicalOverrides lhs021_ovr;
    lhs021_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
    lhs021_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
    lhs021_ovr.adapt_inc = LHS021_ADAPT_INC;
    lhs021_ovr.nmda_tau = FIXED_NMDA_TAU;

    printf("  Simulating %d sequences (%d workers)...\n", (int)sequences.size(), n_workers);
    double bl_t0 = now_seconds();
    auto bl_res = run_all_serial_sequences(base_cfg, sequences, lhs021_ovr, sim_cfg,
                                            bin_info, n_workers);
    double bl_sim_time = now_seconds() - bl_t0;

    // Rate
    double bl_rate_hz = 0;
    int n_seqs = (int)sequences.size();
    for (int i = 0; i < n_seqs; i++) {
        double trial_dur_s = (sequences[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
        bl_rate_hz += bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s);
    }
    bl_rate_hz /= n_seqs;

    // Classification
    auto bl_cls = classify_serial_positions(bl_res.position_bins_list, bl_res.digit_list,
                                             bl_res.n_reservoir, bin_info);
    printf("  LHS-021: %.0fs, Rate: %.1f Hz\n", bl_sim_time, bl_rate_hz);
    printf("  Serial: %.1f%% (pos1=%.1f%%, pos2=%.1f%%, pos3=%.1f%%)\n",
           bl_cls.serial_accuracy * 100,
           bl_cls.accuracy[0] * 100,
           bl_cls.accuracy[1] * 100,
           bl_cls.accuracy[2] * 100);

    // Baseline JSON
    char bl_json_buf[2048];
    snprintf(bl_json_buf, sizeof(bl_json_buf),
             "{\"serial_accuracy\": %.10f, \"serial_accuracy_std\": %.10f, "
             "\"accuracy_pos1\": %.10f, \"accuracy_pos2\": %.10f, \"accuracy_pos3\": %.10f, "
             "\"firing_rate_hz\": %.4f, \"sim_time_s\": %.1f, \"n_reservoir\": %d}",
             bl_cls.serial_accuracy, bl_cls.serial_accuracy_std,
             bl_cls.accuracy[0], bl_cls.accuracy[1], bl_cls.accuracy[2],
             bl_rate_hz, bl_sim_time, bl_res.n_reservoir);
    std::string baseline_json = bl_json_buf;

    // ============================================================
    // EVALUATE ONE BRANCH
    // ============================================================
    auto evaluate_branch = [&](const SerialGridPoint& pt,
                               const DynamicalOverrides& dyn_ovr,
                               const SimConfig& eval_sim,
                               const std::string& branch,
                               double cal_value,
                               double cal_rate,
                               double tonic_rev = -80.0) -> std::string {
        double t0 = now_seconds();
        double gt = (branch == "B_matched") ? cal_value : 0.0;
        auto res = run_all_serial_sequences(base_cfg, sequences, dyn_ovr, eval_sim,
                                             bin_info, n_workers, true, gt, tonic_rev);
        double sim_time = now_seconds() - t0;

        // Rate
        double rate_mean = 0, rate_std = 0;
        for (int i = 0; i < n_seqs; i++) {
            double trial_dur_s = (sequences[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
            rate_mean += res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
        }
        rate_mean /= n_seqs;
        for (int i = 0; i < n_seqs; i++) {
            double trial_dur_s = (sequences[i].total_stim_end_ms + 5.0 + POST_STIM_MS) / 1000.0;
            double r = res.total_spikes[i] / (res.n_reservoir * trial_dur_s);
            rate_std += (r - rate_mean) * (r - rate_mean);
        }
        rate_std = std::sqrt(rate_std / n_seqs);

        // Per-position classification
        auto cls_res = classify_serial_positions(res.position_bins_list, res.digit_list,
                                                  res.n_reservoir, bin_info);

        // Stats vs BSA
        auto stats = compute_paired_stats(cls_res.serial_per_repeat,
                                           bsa_result.serial_per_repeat,
                                           cls_res.serial_accuracy, bsa_result.serial_accuracy);

        // ISI CV
        double isi_cv_mean = 0;
        int isi_valid = 0;
        for (int i = 0; i < n_seqs; i++) {
            if (!std::isnan(res.isi_cvs[i])) { isi_cv_mean += res.isi_cvs[i]; isi_valid++; }
        }
        if (isi_valid > 0) isi_cv_mean /= isi_valid;

        // Conductance state at digit onsets and stim end
        double adapt_d2_mean = 0, adapt_d3_mean = 0, adapt_end_mean = 0;
        double nmda_onset_mean[N_SERIAL_DIGITS] = {};
        double std_resource_onset_mean[N_SERIAL_DIGITS] = {};
        double ge_onset_mean[N_SERIAL_DIGITS] = {};
        for (int i = 0; i < n_seqs; i++) {
            adapt_d2_mean += res.adapt_onsets[i][1];
            adapt_d3_mean += res.adapt_onsets[i][2];
            adapt_end_mean += res.adapt_stim_ends[i];
            for (int p = 0; p < N_SERIAL_DIGITS; p++) {
                nmda_onset_mean[p] += res.nmda_onsets[i][p];
                std_resource_onset_mean[p] += res.std_resource_onsets[i][p];
                ge_onset_mean[p] += res.ge_onsets[i][p];
            }
        }
        adapt_d2_mean /= n_seqs;
        adapt_d3_mean /= n_seqs;
        adapt_end_mean /= n_seqs;
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            nmda_onset_mean[p] /= n_seqs;
            std_resource_onset_mean[p] /= n_seqs;
            ge_onset_mean[p] /= n_seqs;
        }

        // Participation ratio
        double pr_mean = serial_participation_ratio(res.position_bins_list, res.digit_list,
                                                     res.n_reservoir, bin_info);

        // Interference: pos1 - pos3
        double interference_pp = (cls_res.accuracy[0] - cls_res.accuracy[2]) * 100.0;

        printf("    [%s] Rate: %.1f Hz | Serial: %.1f%% (gap=%+.1fpp %s) | "
               "Pos: [%.1f, %.1f, %.1f] | ISI CV: %.3f | PR: %.4f\n",
               branch.c_str(), rate_mean, cls_res.serial_accuracy * 100,
               stats.gap_pp, stats.stars.c_str(),
               cls_res.accuracy[0] * 100, cls_res.accuracy[1] * 100,
               cls_res.accuracy[2] * 100, isi_cv_mean, pr_mean);

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

        oss << "\"gap_ms\": " << SERIAL_GAP_MS << ", ";
        oss << "\"n_seq\": " << N_SERIAL_DIGITS << ", ";
        oss << "\"n_sequences\": " << n_seqs << ", ";

        oss << "\"serial_accuracy\": " << cls_res.serial_accuracy << ", ";
        oss.precision(6);
        oss << "\"serial_accuracy_std\": " << cls_res.serial_accuracy_std << ", ";
        oss << "\"serial_gap_pp\": " << stats.gap_pp << ", ";
        oss << "\"serial_ci_lo_pp\": " << stats.ci_lo_pp << ", ";
        oss << "\"serial_ci_hi_pp\": " << stats.ci_hi_pp << ", ";
        oss << "\"serial_p_value\": " << stats.p_value << ", ";
        oss << "\"serial_cohens_d\": " << stats.cohens_d << ", ";
        oss << "\"serial_stars\": \"" << stats.stars << "\", ";

        oss.precision(10);
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            oss << "\"serial_accuracy_pos" << (p+1) << "\": " << cls_res.accuracy[p] << ", ";
            oss.precision(6);
            oss << "\"serial_accuracy_std_pos" << (p+1) << "\": " << cls_res.accuracy_std[p] << ", ";
            oss.precision(10);
        }
        oss.precision(6);
        oss << "\"serial_interference_pp\": " << interference_pp << ", ";

        // Per-repeat accuracy
        oss << "\"serial_per_repeat_accuracy\": [";
        for (int r = 0; r < N_CV_REPEATS; r++) {
            if (r > 0) oss << ", ";
            oss.precision(10);
            oss << cls_res.serial_per_repeat[r];
        }
        oss << "], ";

        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            oss << "\"serial_per_repeat_pos" << (p+1) << "\": [";
            for (int r = 0; r < N_CV_REPEATS; r++) {
                if (r > 0) oss << ", ";
                oss << cls_res.per_repeat_accuracy[p][r];
            }
            oss << "], ";
        }

        // Dynamics — conductance state at each digit onset
        oss.precision(6);
        oss << "\"adapt_at_digit2_onset_mean\": " << adapt_d2_mean << ", ";
        oss << "\"adapt_at_digit3_onset_mean\": " << adapt_d3_mean << ", ";
        oss << "\"adapt_at_stim_end_mean\": " << adapt_end_mean << ", ";
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            oss << "\"nmda_conductance_at_digit" << (p+1) << "_onset\": " << nmda_onset_mean[p] << ", ";
            oss << "\"std_resource_at_digit" << (p+1) << "_onset\": " << std_resource_onset_mean[p] << ", ";
            oss << "\"excitatory_conductance_at_digit" << (p+1) << "_onset\": " << ge_onset_mean[p] << ", ";
        }

        oss << "\"firing_rate_hz\": " << rate_mean << ", ";
        oss << "\"firing_rate_std\": " << rate_std << ", ";
        oss << "\"n_reservoir\": " << res.n_reservoir << ", ";
        oss << "\"sim_time_s\": " << sim_time << ", ";
        if (isi_valid > 0) oss << "\"isi_cv_mean\": " << isi_cv_mean << ", ";
        else oss << "\"isi_cv_mean\": null, ";
        oss << "\"participation_ratio_mean\": " << pr_mean << ", ";

        // Per-bin accuracy (full trial, position-0 label)
        auto pba = serial_classify_per_bin(res.full_trial_bins_list, res.digit_list,
                                            bin_info.total_max_bins, res.n_reservoir);
        oss << "\"per_bin_accuracy\": [";
        for (int b = 0; b < (int)pba.size(); b++) {
            if (b > 0) oss << ", ";
            oss.precision(6);
            oss << pba[b];
        }
        oss << "], ";

        // Per-position per-bin accuracy
        for (int p = 0; p < N_SERIAL_DIGITS; p++) {
            auto pba_pos = serial_classify_per_bin_position(
                res.position_bins_list, res.digit_list,
                p, bin_info.max_bins[p], res.n_reservoir);
            oss << "\"per_bin_accuracy_pos" << (p+1) << "\": [";
            for (int b = 0; b < (int)pba_pos.size(); b++) {
                if (b > 0) oss << ", ";
                oss.precision(6);
                oss << pba_pos[b];
            }
            oss << "]";
            if (p < N_SERIAL_DIGITS - 1) oss << ", ";
        }

        oss << "}";
        return oss.str();
    };

    // ============================================================
    // SWEEP — two branches per grid point
    // ============================================================
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
            double natural_rate = serial_measure_rate(base_cfg, cal_seqs, dyn_ovr, eval_a,
                                                       bin_info, n_workers);
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

            auto cal = serial_calibrate_tonic(base_cfg, dyn_ovr, cal_seqs, eval_b,
                                               bin_info, n_workers, target_rate_hz,
                                               prev_gt, prev_rev);

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

        serial_save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                                target_rate_hz, n_seqs, total_start, false);
    }

    double total_time = now_seconds() - total_start;

    printf("\n======================================================================\n");
    printf("  SERIAL CLASSIFICATION SWEEP COMPLETE\n");
    printf("  Total time: %.0fs (%.1f hours)\n", total_time, total_time / 3600.0);
    printf("  Grid points: %d results (%d grid x 2 branches)\n",
           (int)grid_results_json.size(), n_grid);
    printf("======================================================================\n");

    serial_save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                            target_rate_hz, n_seqs, total_start, true);

    printf("\nDone.\n");
    return 0;
}
