#include "common.h"
#include "npz_reader.h"
#include "network.h"
#include "builder.h"
#include "ml.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace cls;

// ============================================================
// CONSTANTS
// ============================================================
static const std::vector<int> DEFAULT_DIGITS = {0, 1, 2, 3, 4};
static const int N_DIGITS = 5;
static const double BIN_MS = 20.0;
static const double POST_STIM_MS = 200.0;
static int SAMPLES_PER_DIGIT = 500;
static const int N_CV_REPEATS = 5;
static const int N_CV_FOLDS = 5;
static const std::vector<double> RIDGE_ALPHAS = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
static const int SEED = 42;
static const int SAMPLE_LOAD_SEED = 42;

// Network snapshot path (set via --snapshot CLI arg; empty = build from RNG)
static std::string g_snapshot_path;

// LHS-021 params
static const double LHS021_LAMBDA_CONNECT = 0.003288382505082908;
static const double LHS021_CORE_CORE_MULT = 0.8274594948671626;
[[maybe_unused]] static const double LHS021_SURFACE_SHELL_FRAC = 0.18171487839543513;
static const double LHS021_ADAPT_INC = 0.6264283936395093;
static const double LHS021_SHELL_CORE_MULT = 4.849737508561994;
static const double FIXED_NMDA_TAU = 50.0;

// Rate-matching
static const double RATE_TOLERANCE_HZ = 2.0;
static const int CALIBRATION_N_SAMPLES = 200;
static const int CALIBRATION_MAX_ITER = 100;
static const double CALIBRATION_STIM_LO = 0.01;
static const double CALIBRATION_STIM_HI = 5.0;

// Unified grid axes
static const std::vector<double> UNIFIED_INC = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.185714, 0.2, 0.321429, 0.457143, 0.592857,
    0.63, 0.728571, 0.864286, 1.0,
};
static const std::vector<double> UNIFIED_TAU = {
    30.0, 54.7, 99.6, 181.0, 330.6, 500.0, 602.4,
    1098.0, 2000.0, 3000.0, 3500.0, 4000.0, 5000.0,
};

// Forward declarations
static std::vector<AudioSample> load_audio_samples(
    const std::string& data_dir, const std::vector<int>& digits,
    int samples_per_digit, int random_seed);

// ============================================================
// INPUT NEURON GRID SEARCH
// ============================================================
static const std::vector<double> IG_STIM     = {0.04, 0.08, 0.12, 0.20, 0.40, 0.88};
static const std::vector<double> IG_TAU_E    = {0.5, 1.0, 2.0, 3.0, 5.0};
static const std::vector<double> IG_ADAPT    = {0.0, 0.3, 0.8, 1.5, 3.0};
static const std::vector<double> IG_STD_U    = {0.0, 0.15, 0.30, 0.50};
static const int IG_SAMPLES_PER_DIGIT = 6;
static const double IG_INPUT_STD_TAU = 500.0;

struct InputGridPoint {
    double stim_current, input_tau_e, input_adapt_inc, input_std_u;
    int idx;
};

struct InputGridMetrics {
    double mean_rate_hz;
    double mean_isi_cv;
    double mean_refrac_frac;
    double mean_mod_depth;
    double spike_bsa_r_10ms;
    double spike_bsa_r_20ms;
    double spike_bsa_r_50ms;
    int n_active_neurons;
};

static double pearson_r_vec(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)std::min(x.size(), y.size());
    if (n < 3) return 0.0;
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy;
    }
    double denom = std::sqrt(sxx * syy);
    return denom > 1e-12 ? sxy / denom : 0.0;
}

static std::vector<InputGridPoint> build_input_grid() {
    std::vector<InputGridPoint> pts;
    int idx = 0;
    for (double stim : IG_STIM)
        for (double tau : IG_TAU_E)
            for (double adapt : IG_ADAPT)
                for (double std_u : IG_STD_U)
                    pts.push_back({stim, tau, adapt, std_u, idx++});
    return pts;
}

// Compute per-input-neuron metrics from one simulation's activity_record
static InputGridMetrics compute_input_metrics(
    const RunResult& result,
    const ZoneInfo& zone_info,
    const AudioSample& sample,
    double dt, double stim_end_ms,
    const double bin_windows_ms[3])  // {10, 20, 50}
{
    InputGridMetrics m{};
    std::set<int> input_set(zone_info.input_neuron_indices.begin(),
                             zone_info.input_neuron_indices.end());
    int n_steps = (int)result.activity_record.size();

    // 1. Collect per-input-neuron spike times
    std::unordered_map<int, std::vector<double>> neuron_spikes;
    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        for (int nid : result.activity_record[step]) {
            if (input_set.count(nid)) {
                neuron_spikes[nid].push_back(t);
            }
        }
    }

    // 2. Build per-input-neuron BSA spike times
    std::unordered_map<int, std::vector<double>> neuron_bsa;
    for (size_t k = 0; k < sample.spike_times_ms.size(); k++) {
        int fb = sample.freq_bin_indices[k];
        auto mit = zone_info.input_neuron_mapping.find(fb);
        if (mit != zone_info.input_neuron_mapping.end()) {
            for (int nid : mit->second) {
                neuron_bsa[nid].push_back(sample.spike_times_ms[k]);
            }
        }
    }

    // 3. Per-neuron metrics
    double sum_rate = 0, sum_isi_cv = 0, sum_refrac = 0, sum_mod = 0;
    double sum_r[3] = {0, 0, 0};
    int n_with_isi = 0, n_active = 0, n_with_bsa = 0;

    for (int nid : zone_info.input_neuron_indices) {
        auto sit = neuron_spikes.find(nid);
        if (sit == neuron_spikes.end() || sit->second.empty()) continue;
        const auto& spk = sit->second;
        n_active++;

        // Firing rate (during stimulus period only)
        int n_stim_spikes = 0;
        for (double t : spk) if (t <= stim_end_ms) n_stim_spikes++;
        double rate_hz = n_stim_spikes / (stim_end_ms / 1000.0);
        sum_rate += rate_hz;

        // ISI stats
        std::vector<double> isis;
        for (size_t i = 1; i < spk.size(); i++) {
            if (spk[i] <= stim_end_ms) isis.push_back(spk[i] - spk[i-1]);
        }
        if (isis.size() >= 2) {
            double mean_isi = 0;
            for (double v : isis) mean_isi += v;
            mean_isi /= isis.size();
            double var = 0;
            for (double v : isis) var += (v - mean_isi) * (v - mean_isi);
            var /= isis.size();
            double cv = std::sqrt(var) / std::max(mean_isi, 1e-9);
            sum_isi_cv += cv;

            // Refractory fraction (ISI <= 5.0 ms)
            int n_ref = 0;
            for (double v : isis) if (v <= 5.0) n_ref++;
            sum_refrac += (double)n_ref / isis.size();

            n_with_isi++;
        }

        // Rate modulation depth (20ms windows during stim)
        {
            int n_bins_mod = std::max(1, (int)(stim_end_ms / 20.0));
            std::vector<double> rates_mod(n_bins_mod, 0.0);
            for (double t : spk) {
                if (t <= stim_end_ms) {
                    int b = std::min((int)(t / 20.0), n_bins_mod - 1);
                    rates_mod[b] += 1.0;
                }
            }
            double max_r = *std::max_element(rates_mod.begin(), rates_mod.end());
            double min_r = *std::min_element(rates_mod.begin(), rates_mod.end());
            if (max_r > 0) sum_mod += (max_r - min_r) / max_r;
        }

        // Spike-BSA correlation at multiple windows
        auto bit = neuron_bsa.find(nid);
        if (bit != neuron_bsa.end() && !bit->second.empty()) {
            n_with_bsa++;
            for (int wi = 0; wi < 3; wi++) {
                double win = bin_windows_ms[wi];
                int n_bins = std::max(1, (int)(stim_end_ms / win));
                std::vector<double> bsa_binned(n_bins, 0.0), spk_binned(n_bins, 0.0);
                for (double t : bit->second) {
                    if (t <= stim_end_ms) {
                        int b = std::min((int)(t / win), n_bins - 1);
                        bsa_binned[b] += 1.0;
                    }
                }
                for (double t : spk) {
                    if (t <= stim_end_ms) {
                        int b = std::min((int)(t / win), n_bins - 1);
                        spk_binned[b] += 1.0;
                    }
                }
                sum_r[wi] += pearson_r_vec(bsa_binned, spk_binned);
            }
        }
    }

    m.n_active_neurons = n_active;
    m.mean_rate_hz = n_active > 0 ? sum_rate / n_active : 0;
    m.mean_isi_cv = n_with_isi > 0 ? sum_isi_cv / n_with_isi : 0;
    m.mean_refrac_frac = n_with_isi > 0 ? sum_refrac / n_with_isi : 0;
    m.mean_mod_depth = n_active > 0 ? sum_mod / n_active : 0;
    m.spike_bsa_r_10ms = n_with_bsa > 0 ? sum_r[0] / n_with_bsa : 0;
    m.spike_bsa_r_20ms = n_with_bsa > 0 ? sum_r[1] / n_with_bsa : 0;
    m.spike_bsa_r_50ms = n_with_bsa > 0 ? sum_r[2] / n_with_bsa : 0;
    return m;
}

static int run_input_grid(int argc, char** argv, const std::string& snapshot_path,
                           const std::string& data_dir, int n_workers,
                           const std::string& output_csv) {
    printf("======================================================================\n");
    printf("  INPUT NEURON REGIME GRID SEARCH\n");
    printf("======================================================================\n");

    auto grid = build_input_grid();
    int n_grid = (int)grid.size();

    // Load audio
    printf("[1] Loading audio (%d per digit)...\n", IG_SAMPLES_PER_DIGIT);
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS,
                                       IG_SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded\n", n_samples);

    // Find max audio duration
    double max_audio_ms = 0;
    for (auto& s : samples) {
        double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (mx > max_audio_ms) max_audio_ms = mx;
    }
    max_audio_ms += 5.0;

    // Pre-compute per-sample stim end times
    std::vector<double> stim_ends(n_samples);
    for (int i = 0; i < n_samples; i++) {
        stim_ends[i] = *std::max_element(samples[i].spike_times_ms.begin(),
                                          samples[i].spike_times_ms.end());
    }

    double bin_windows[3] = {10.0, 20.0, 50.0};

    printf("[2] Running grid search: %d grid points x %d samples = %d sims\n",
           n_grid, n_samples, n_grid * n_samples);
    printf("    Axes: stim_current(%d) x tau_e(%d) x adapt_inc(%d) x std_u(%d)\n",
           (int)IG_STIM.size(), (int)IG_TAU_E.size(),
           (int)IG_ADAPT.size(), (int)IG_STD_U.size());
    printf("    Workers: %d\n", n_workers);

    // Open output CSV
    FILE* csv = fopen(output_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "Cannot open %s\n", output_csv.c_str()); return 1; }
    fprintf(csv, "stim_current,input_tau_e,input_adapt_inc,input_std_u,"
                 "mean_rate_hz,mean_isi_cv,mean_refrac_frac,mean_mod_depth,"
                 "spike_bsa_r_10ms,spike_bsa_r_20ms,spike_bsa_r_50ms,"
                 "n_active_neurons,composite_score\n");

    double t_start = now_seconds();

    for (int gp = 0; gp < n_grid; gp++) {
        auto& pt = grid[gp];

        // Accumulate metrics across samples
        double sum_rate = 0, sum_cv = 0, sum_ref = 0, sum_mod = 0;
        double sum_r10 = 0, sum_r20 = 0, sum_r50 = 0;
        double sum_active = 0;

        #pragma omp parallel num_threads(n_workers) \
            reduction(+:sum_rate,sum_cv,sum_ref,sum_mod,sum_r10,sum_r20,sum_r50,sum_active)
        {
            // Each thread builds its own network with this grid point's params
            SphericalNetwork net;
            ZoneInfo zone_info;
            if (!snapshot_path.empty()) {
                load_network_snapshot(net, zone_info, snapshot_path, 0.1, true);
                DynamicalOverrides dyn_ovr;
                dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
                dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
                dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
                dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
                apply_dynamical_overrides(net, zone_info, 0.1, dyn_ovr);
            } else {
                NetworkConfig base_cfg = make_base_config();
                base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
                DynamicalOverrides dyn_ovr;
                dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
                dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
                dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
                dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
                build_full_network(net, zone_info, base_cfg, 0.1, true,
                                  &dyn_ovr, "default", true);
            }

            // Apply input neuron overrides
            for (int nid : zone_info.input_neuron_indices) {
                net.tau_e[nid] = pt.input_tau_e;
                net.adaptation_increment[nid] = pt.input_adapt_inc;
            }
            net.precompute_decay_factors(0.1);
            StdMasks masks = build_std_masks(net, zone_info);

            rng_seed(42 + (uint64_t)omp_get_thread_num() * 1000 + (uint64_t)getpid());

            #pragma omp for schedule(dynamic, 1)
            for (int si = 0; si < n_samples; si++) {
                SimConfig sim;
                sim.dt = 0.1;
                sim.audio_duration_ms = max_audio_ms;
                sim.post_stimulus_ms = POST_STIM_MS;
                sim.stimulus_current = pt.stim_current;
                sim.input_std_u = pt.input_std_u;
                sim.input_std_tau_rec = IG_INPUT_STD_TAU;

                auto result = run_sample_with_std(net, samples[si], zone_info, sim,
                                                   STD_U, STD_TAU_REC, masks,
                                                   stim_ends[si]);

                auto m = compute_input_metrics(result, zone_info, samples[si],
                                                sim.dt, stim_ends[si], bin_windows);

                sum_rate += m.mean_rate_hz;
                sum_cv += m.mean_isi_cv;
                sum_ref += m.mean_refrac_frac;
                sum_mod += m.mean_mod_depth;
                sum_r10 += m.spike_bsa_r_10ms;
                sum_r20 += m.spike_bsa_r_20ms;
                sum_r50 += m.spike_bsa_r_50ms;
                sum_active += m.n_active_neurons;
            }
        }

        // Average across samples
        double avg_rate = sum_rate / n_samples;
        double avg_cv = sum_cv / n_samples;
        double avg_ref = sum_ref / n_samples;
        double avg_mod = sum_mod / n_samples;
        double avg_r10 = sum_r10 / n_samples;
        double avg_r20 = sum_r20 / n_samples;
        double avg_r50 = sum_r50 / n_samples;
        double avg_active = sum_active / n_samples;

        // Composite score: reward correlation + variability + modulation,
        // penalize refractory locking
        double best_r = std::max({avg_r10, avg_r20, avg_r50});
        double score = 1.5 * best_r + 0.3 * avg_cv + 0.3 * avg_mod
                       - 1.5 * avg_ref;
        // Penalty for too-low activity (< 10 Hz)
        if (avg_rate < 10.0) score -= 0.5 * (10.0 - avg_rate) / 10.0;

        fprintf(csv, "%.4f,%.2f,%.4f,%.4f,"
                     "%.2f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.1f,%.4f\n",
                pt.stim_current, pt.input_tau_e, pt.input_adapt_inc, pt.input_std_u,
                avg_rate, avg_cv, avg_ref, avg_mod,
                avg_r10, avg_r20, avg_r50,
                avg_active, score);
        fflush(csv);

        int gc = gp + 1;
        double elapsed = now_seconds() - t_start;
        double eta = (elapsed / gc) * (n_grid - gc);
        printf("  [%4d/%d] stim=%.2f tau_e=%.1f adapt=%.2f std_u=%.2f | "
               "rate=%.0fHz r50=%.3f refr=%.2f score=%.3f | "
               "%.0fs elapsed, ETA %.0fs\n",
               gc, n_grid,
               pt.stim_current, pt.input_tau_e, pt.input_adapt_inc, pt.input_std_u,
               avg_rate, avg_r50, avg_ref, score,
               elapsed, eta);
    }

    fclose(csv);
    double total = now_seconds() - t_start;
    printf("\n======================================================================\n");
    printf("  DONE: %d grid points in %.1fs (%.1f min)\n", n_grid, total, total / 60.0);
    printf("  Results: %s\n", output_csv.c_str());
    printf("======================================================================\n");
    return 0;
}

// ============================================================
// GRID POINT
// ============================================================
struct GridPoint {
    double adapt_inc, adapt_tau;
    std::string source, point_id;
    int inc_idx, tau_idx;
};

static std::pair<int,int> snap_to_unified(double inc, double tau) {
    int i_idx = 0, t_idx = 0;
    double best_i = 1e9, best_t = 1e9;
    for (int i = 0; i < (int)UNIFIED_INC.size(); i++) {
        double d = std::abs(UNIFIED_INC[i] - inc);
        if (d < best_i) { best_i = d; i_idx = i; }
    }
    for (int i = 0; i < (int)UNIFIED_TAU.size(); i++) {
        double d = std::abs(UNIFIED_TAU[i] - tau);
        if (d < best_t) { best_t = d; t_idx = i; }
    }
    if (best_i > 0.005 || best_t > 5.0) return {-1, -1};
    return {i_idx, t_idx};
}

static std::vector<GridPoint> build_grid_points(const std::string& arms) {
    std::set<std::string> requested;
    std::istringstream ss(arms);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" "));
        tok.erase(tok.find_last_not_of(" ") + 1);
        requested.insert(tok);
    }
    bool all = requested.count("all") > 0;

    std::vector<GridPoint> points;
    std::set<std::pair<int,int>> seen; // (round(inc*10000), round(tau))

    auto add = [&](double inc, double tau, const std::string& source) {
        auto key = std::make_pair((int)std::round(inc * 10000), (int)std::round(tau));
        if (seen.count(key)) return;
        seen.insert(key);
        auto [ii, ti] = snap_to_unified(inc, tau);
        char buf[256];
        snprintf(buf, sizeof(buf), "%s_inc%.6f_tau%.1f", source.c_str(), inc, tau);
        points.push_back({inc, tau, source, buf, ii, ti});
    };

    // Original 8x8 grid
    if (all || requested.count("original")) {
        for (int i = 0; i < 8; i++) {
            double inc = 0.05 + (1.0 - 0.05) * i / 7.0;
            for (int j = 0; j < 8; j++) {
                double tau = 30.0 * std::pow(2000.0 / 30.0, j / 7.0);
                add(inc, tau, "original");
            }
        }
    }

    // Arm A: sAHP Zoom
    if (all || requested.count("A")) {
        double a_inc[] = {0.005, 0.01, 0.02, 0.03, 0.04, 0.05};
        double a_tau[] = {500.0, 1098.0, 2000.0, 3000.0, 4000.0, 5000.0};
        for (double inc : a_inc) for (double tau : a_tau) add(inc, tau, "A");
    }

    // Arm E: Grid Infill
    if (all || requested.count("E")) {
        // inc=0 baseline
        for (double tau : UNIFIED_TAU) add(0.0, tau, "E");
        // Low inc x short tau
        double e_combos[][2] = {
            {0.005,30},{0.005,54.7},{0.005,99.6},{0.005,181},{0.005,330.6},
            {0.01,30},{0.01,54.7},{0.01,99.6},{0.01,181},
            {0.02,30},{0.02,54.7},{0.03,30},{0.03,54.7},{0.04,30},
            // Mid-high inc x high tau
            {0.185714,3500},{0.185714,4000},{0.185714,5000},
            {0.2,3500},{0.2,4000},{0.2,5000},
            {0.321429,3000},{0.321429,3500},{0.321429,4000},{0.321429,5000},
            {0.457143,3000},{0.457143,3500},{0.457143,4000},{0.457143,5000},
            {0.592857,3000},{0.592857,3500},{0.592857,4000},{0.592857,5000},
            {0.63,3000},{0.63,3500},{0.63,4000},{0.63,5000},
            {0.728571,3000},{0.728571,3500},{0.728571,4000},{0.728571,5000},
            {0.864286,3000},{0.864286,3500},{0.864286,4000},{0.864286,5000},
            {1.0,3000},{1.0,3500},{1.0,4000},{1.0,5000},
        };
        for (auto& c : e_combos) add(c[0], c[1], "E");
    }

    return points;
}

// ============================================================
// LOAD AUDIO SAMPLES
// ============================================================
static std::vector<AudioSample> load_audio_samples(
    const std::string& data_dir, const std::vector<int>& digits,
    int samples_per_digit, int random_seed)
{
    rng_seed(random_seed + 100);
    std::string spike_dir = data_dir + "/spike_trains_bsa";
    std::vector<AudioSample> samples;

    for (int digit : digits) {
        std::vector<std::string> files;
        std::string prefix = "spike_train_" + std::to_string(digit) + "_";
        for (auto& entry : fs::directory_iterator(spike_dir)) {
            if (entry.path().extension() == ".npz" &&
                entry.path().filename().string().substr(0, prefix.size()) == prefix) {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());

        std::vector<std::string> selected;
        if ((int)files.size() <= samples_per_digit) {
            selected = files;
        } else {
            auto indices = rng_choice((int)files.size(), samples_per_digit, false);
            std::sort(indices.begin(), indices.end());
            for (int i : indices) selected.push_back(files[i]);
        }

        printf("  Digit %d: %d files found, loading %d...\n",
               digit, (int)files.size(), (int)selected.size());
        int loaded = 0;
        for (const auto& fpath : selected) {
            try {
                NpzFile npz = load_npz(fpath);
                AudioSample s;
                s.spike_times_ms = npz["spike_times_ms"].to_float64_vec();
                s.freq_bin_indices = npz["freq_bin_indices"].to_int32_vec();
                s.digit = (int)npz["digit"].as_scalar_int();
                s.speaker = npz["speaker"].as_string();
                // filename = stem
                fs::path p(fpath);
                s.filename = p.stem().string();
                samples.push_back(std::move(s));
                loaded++;
                if (loaded % 50 == 0)
                    printf("    %d/%d\n", loaded, (int)selected.size());
            } catch (const std::exception& e) {
                fprintf(stderr, "WARNING: Skipping %s: %s\n", fpath.c_str(), e.what());
            }
        }
    }

    printf("\n  Loaded %d audio samples:\n", (int)samples.size());
    for (int d : digits) {
        int count = 0;
        for (auto& s : samples) if (s.digit == d) count++;
        printf("    Digit %d: %d samples\n", d, count);
    }
    return samples;
}

// ============================================================
// ACTIVITY BINNING
// ============================================================
struct BinResult {
    Mat bins; // (n_bins, n_reservoir)
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
    std::unordered_map<int, double> neuron_last;
    std::vector<double> all_isis;

    for (int step = 0; step < stim_end_step; step++) {
        for (int nid : result.activity_record[step]) {
            if (res_set.count(nid)) {
                double t = step * dt;
                auto it = neuron_last.find(nid);
                if (it != neuron_last.end()) {
                    all_isis.push_back(t - it->second);
                }
                neuron_last[nid] = t;
            }
        }
    }

    double isi_cv = NAN;
    if (all_isis.size() >= 2) {
        double mean = 0;
        for (double v : all_isis) mean += v;
        mean /= all_isis.size();
        double var = 0;
        for (double v : all_isis) var += (v - mean) * (v - mean);
        var /= all_isis.size();
        isi_cv = std::sqrt(var) / std::max(mean, 1e-9);
    }

    // Adaptation at stimulus end
    double adapt_mean = 0;
    if (!result.adapt_snapshot.empty()) {
        for (int ri : zone_info.reservoir_zone_indices) {
            adapt_mean += result.adapt_snapshot[ri];
        }
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

static AllSamplesResult run_all_samples(const NetworkConfig& cfg,
                                         const std::vector<AudioSample>& samples,
                                         const DynamicalOverrides& dyn_ovr,
                                         const SimConfig& sim_cfg,
                                         int n_workers,
                                         bool verbose = true) {
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
        // Each thread builds its own network
        SphericalNetwork net;
        ZoneInfo zone_info;
        if (!g_snapshot_path.empty()) {
            load_network_snapshot(net, zone_info, g_snapshot_path, sim_cfg.dt, true);
            // Apply per-grid-point adaptation overrides on top of snapshot
            apply_dynamical_overrides(net, zone_info, sim_cfg.dt, dyn_ovr);
        } else {
            build_full_network(net, zone_info, cfg, sim_cfg.dt, true,
                              &dyn_ovr, "default", true);
        }
        StdMasks masks = build_std_masks(net, zone_info);

        // Reseed per-thread
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
                           const SimConfig& sim_cfg, int n_workers) {
    auto res = run_all_samples(cfg, samples_subset, dyn_ovr, sim_cfg, n_workers, false);
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
struct CalLogEntry { int iter; double stim_current; double rate_hz; };

static std::tuple<double, double, std::vector<CalLogEntry>>
calibrate_stimulus_current(NetworkConfig cfg,
                           const DynamicalOverrides& dyn_ovr,
                           const std::vector<AudioSample>& cal_samples,
                           SimConfig sim_cfg, int n_workers,
                           double target_rate,
                           double lo = CALIBRATION_STIM_LO,
                           double hi = CALIBRATION_STIM_HI,
                           double initial_guess = -1.0) {
    std::vector<CalLogEntry> log;
    int iteration = 0;

    if (initial_guess > 0) {
        sim_cfg.stimulus_current = initial_guess;
        double rate = measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers);
        log.push_back({iteration, initial_guess, rate});
        printf("    cal[%d] stim=%.4f -> %.1f Hz (target=%.1f)\n",
               iteration, initial_guess, rate, target_rate);
        iteration++;
        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {initial_guess, rate, log};
        double margin = 3.0;
        lo = std::max(lo, initial_guess / margin);
        hi = std::min(hi, initial_guess * margin);
        if (rate > target_rate) hi = initial_guess; else lo = initial_guess;
    }

    for (; iteration < CALIBRATION_MAX_ITER; iteration++) {
        double mid = (lo + hi) / 2.0;
        sim_cfg.stimulus_current = mid;
        double rate = measure_rate(cfg, cal_samples, dyn_ovr, sim_cfg, n_workers);
        log.push_back({iteration, mid, rate});
        printf("    cal[%d] stim=%.4f -> %.1f Hz (target=%.1f)\n",
               iteration, mid, rate, target_rate);

        if (std::abs(rate - target_rate) <= RATE_TOLERANCE_HZ)
            return {mid, rate, log};

        if (rate > target_rate) hi = mid; else lo = mid;
        if (hi - lo < 1e-5) break;
    }

    // Return best
    auto best = std::min_element(log.begin(), log.end(),
        [&](auto& a, auto& b) { return std::abs(a.rate_hz - target_rate) < std::abs(b.rate_hz - target_rate); });
    return {best->stim_current, best->rate_hz, log};
}

// ============================================================
// CLASSIFICATION
// ============================================================
struct ClassifyResult {
    double accuracy;
    double accuracy_std;
    std::vector<double> per_repeat_accuracy;
    std::vector<std::vector<int>> cm; // last confusion matrix
};

static ClassifyResult classify_flat_ridge(const std::vector<Mat>& bins_list,
                                           const std::vector<int>& y,
                                           int n_bins, int n_reservoir) {
    int n_samples = (int)bins_list.size();
    int n_features = n_bins * n_reservoir;

    // Flatten to (n_samples, n_features)
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
            int correct = 0;
            std::vector<int> y_test_for_cm(n_test);
            for (int i = 0; i < n_test; i++) {
                y_test_for_cm[i] = y[split.test[i]];
                if (best_preds[i] == y_test_for_cm[i]) correct++;
            }
            rep_correct += correct;
            rep_total += n_test;
            last_cm = cls::confusion_matrix(y_test_for_cm, best_preds, DEFAULT_DIGITS);
        }

        repeat_accs.push_back(rep_correct / rep_total);
    }

    double mean_acc = 0, std_acc = 0;
    for (double a : repeat_accs) mean_acc += a;
    mean_acc /= repeat_accs.size();
    for (double a : repeat_accs) std_acc += (a - mean_acc) * (a - mean_acc);
    std_acc = std::sqrt(std_acc / repeat_accs.size());

    return {mean_acc, std_acc, repeat_accs, last_cm};
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

        // Center
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

    // Grid axes
    fprintf(f, "  \"grid\": {\n");
    fprintf(f, "    \"unified_inc\": "); json_write_double_array(f, UNIFIED_INC); fprintf(f, ",\n");
    fprintf(f, "    \"unified_tau\": "); json_write_double_array(f, UNIFIED_TAU); fprintf(f, ",\n");
    fprintf(f, "    \"n_inc\": %d, \"n_tau\": %d\n", (int)UNIFIED_INC.size(), (int)UNIFIED_TAU.size());
    fprintf(f, "  },\n");

    // BSA baseline
    fprintf(f, "  \"bsa_baseline\": %s,\n", bsa_json.c_str());

    // LHS-021 baseline
    fprintf(f, "  \"lhs021_baseline\": %s,\n", baseline_json.c_str());

    // Grid results
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
// MAIN
// ============================================================
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0); // line-buffered stdout
    // Parse args
    std::string arms = "all";
    int n_workers = 8;
    std::string output_dir = "";

    std::string snapshot_path = "";
    bool no_snapshot = false;
    bool verify_only = false;
    std::string verify_output = "";
    std::string data_dir_override = "";
    int trace_neuron = -1;
    int trace_sample = 0;
    std::string trace_output = "";
    std::string trace_file = "";  // direct path to a single .npz audio file
    bool no_noise = false;
    bool no_input_nmda = false;
    double stim_current_override = -1.0;  // -1 = use default 0.88
    bool input_grid = false;
    std::string input_grid_output = "input_grid_results.csv";

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--arms" && i + 1 < argc) arms = argv[++i];
        else if (std::string(argv[i]) == "--n-workers" && i + 1 < argc) n_workers = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (std::string(argv[i]) == "--snapshot" && i + 1 < argc) snapshot_path = argv[++i];
        else if (std::string(argv[i]) == "--no-snapshot") no_snapshot = true;
        else if (std::string(argv[i]) == "--verify-only") verify_only = true;
        else if (std::string(argv[i]) == "--verify-output" && i + 1 < argc) verify_output = argv[++i];
        else if (std::string(argv[i]) == "--samples-per-digit" && i + 1 < argc) SAMPLES_PER_DIGIT = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--data-dir" && i + 1 < argc) data_dir_override = argv[++i];
        else if (std::string(argv[i]) == "--trace-neuron" && i + 1 < argc) trace_neuron = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--trace-sample" && i + 1 < argc) trace_sample = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--trace-output" && i + 1 < argc) trace_output = argv[++i];
        else if (std::string(argv[i]) == "--trace-file" && i + 1 < argc) trace_file = argv[++i];
        else if (std::string(argv[i]) == "--no-noise") no_noise = true;
        else if (std::string(argv[i]) == "--no-input-nmda") no_input_nmda = true;
        else if (std::string(argv[i]) == "--stim-current" && i + 1 < argc) stim_current_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--input-grid") input_grid = true;
        else if (std::string(argv[i]) == "--input-grid-output" && i + 1 < argc) input_grid_output = argv[++i];
    }

    // Default: use network_snapshot.npz next to the binary if it exists
    if (snapshot_path.empty() && !no_snapshot) {
        fs::path default_snap = fs::path(argv[0]).parent_path() / "network_snapshot.npz";
        if (fs::exists(default_snap)) {
            snapshot_path = default_snap.string();
            printf("  Using default snapshot: %s\n", snapshot_path.c_str());
        }
    }
    g_snapshot_path = snapshot_path;

    #ifdef _OPENMP
    omp_set_num_threads(n_workers);
    #endif

    // --input-grid mode: early exit
    if (input_grid) {
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        fs::path base_dir = exe_dir.parent_path();
        if (base_dir.empty()) base_dir = ".";
        std::string ig_data_dir = data_dir_override.empty()
            ? (base_dir / "data").string() : data_dir_override;
        return run_input_grid(argc, argv, g_snapshot_path, ig_data_dir,
                               n_workers, input_grid_output);
    }

    // Determine paths
    fs::path exe_dir = fs::path(argv[0]).parent_path();
    fs::path base_dir = exe_dir.parent_path();
    if (base_dir.empty()) base_dir = ".";
    std::string data_dir = data_dir_override.empty()
        ? (base_dir / "data").string() : data_dir_override;
    if (output_dir.empty()) {
        output_dir = (base_dir / "results" / "classification_adaptation_sweep").string();
    }
    fs::create_directories(output_dir);

    auto grid_points = build_grid_points(arms);
    int n_grid = (int)grid_points.size();

    double total_start = now_seconds();

    printf("======================================================================\n");
    printf("  CLASSIFICATION ADAPTATION PARAMETER SWEEP (C++ PORT)\n");
    printf("  Grid points: %d\n", n_grid);
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
        if (trace_output.empty()) trace_output = "trace_cpp.csv";

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
        trace_sim.stimulus_current = (stim_current_override > 0) ? stim_current_override : 0.88;

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
        StdMasks masks = build_std_masks(net, zone_info);

        if (no_noise) {
            std::fill(net.v_noise_amp_arr.begin(), net.v_noise_amp_arr.end(), 0.0);
            std::fill(net.i_noise_amp_arr.begin(), net.i_noise_amp_arr.end(), 0.0);
            printf("  Noise DISABLED\n");
        }
        if (no_input_nmda) {
            net.skip_stim_nmda = true;
            printf("  Input NMDA DISABLED (skip_stim_nmda=true)\n");
        }

        // Print input neuron indices for sweep scripts
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

        FILE* f = fopen(trace_output.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", trace_output.c_str()); return 1; }
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
        printf("  Wrote %s\n", trace_output.c_str());
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
        if (trace_output.empty()) trace_output = "trace_cpp.csv";

        if (trace_sample >= n_samples) {
            fprintf(stderr, "trace_sample %d >= n_samples %d\n", trace_sample, n_samples);
            return 1;
        }

        printf("\n[TRACE] Neuron %d, sample %d (%s), output: %s\n",
               trace_neuron, trace_sample, samples[trace_sample].filename.c_str(),
               trace_output.c_str());

        double sample_max_ms = *std::max_element(
            samples[trace_sample].spike_times_ms.begin(),
            samples[trace_sample].spike_times_ms.end());
        sample_max_ms += 5.0;

        SimConfig trace_sim;
        trace_sim.dt = 0.1;
        trace_sim.audio_duration_ms = sample_max_ms;
        trace_sim.post_stimulus_ms = POST_STIM_MS;
        trace_sim.stimulus_current = (stim_current_override > 0) ? stim_current_override : 0.88;

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
        StdMasks masks = build_std_masks(net, zone_info);

        if (no_noise) {
            std::fill(net.v_noise_amp_arr.begin(), net.v_noise_amp_arr.end(), 0.0);
            std::fill(net.i_noise_amp_arr.begin(), net.i_noise_amp_arr.end(), 0.0);
            printf("  Noise DISABLED\n");
        }
        if (no_input_nmda) {
            net.skip_stim_nmda = true;
            printf("  Input NMDA DISABLED (skip_stim_nmda=true)\n");
        }

        net.trace_neuron_id = trace_neuron;
        net.trace.clear();

        printf("  Running simulation (%.1f ms, dt=%.1f)...\n",
               trace_sim.audio_duration_ms + trace_sim.post_stimulus_ms, trace_sim.dt);

        auto result = run_sample_with_std(net, samples[trace_sample], zone_info, trace_sim,
                                           STD_U, STD_TAU_REC, masks,
                                           trace_sim.audio_duration_ms);

        printf("  %d timesteps recorded\n", (int)net.trace.size());

        // Write CSV
        FILE* f = fopen(trace_output.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot open %s\n", trace_output.c_str()); return 1; }
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
        printf("  Wrote %s\n", trace_output.c_str());
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

    // BSA JSON
    char bsa_json_buf[1024];
    snprintf(bsa_json_buf, sizeof(bsa_json_buf),
             "{\"accuracy\": %.10f, \"accuracy_std\": %.10f}",
             bsa_result.accuracy, bsa_result.accuracy_std);
    std::string bsa_json = bsa_json_buf;

    // 3. Base config
    NetworkConfig base_cfg = make_base_config();
    base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
    // surface_shell_fraction is used in builder via constant

    SimConfig sim_cfg;
    sim_cfg.dt = 0.1;
    sim_cfg.audio_duration_ms = max_audio_ms;
    sim_cfg.post_stimulus_ms = POST_STIM_MS;
    sim_cfg.stimulus_current = 0.88;

    std::vector<AudioSample> cal_samples(samples.begin(),
                                          samples.begin() + std::min(CALIBRATION_N_SAMPLES, n_samples));

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

    double target_rate_hz = 0;
    for (int i = 0; i < n_samples; i++) {
        target_rate_hz += bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s);
    }
    target_rate_hz /= n_samples;

    int target_n_bins = 0;
    for (auto& b : bl_res.res_bins_list)
        if (b.rows > target_n_bins) target_n_bins = b.rows;
    target_n_bins = std::max(target_n_bins, n_bins_total);

    auto bl_cls = classify_flat_ridge(bl_res.res_bins_list, y, target_n_bins, bl_res.n_reservoir);
    auto bl_stats = compute_paired_stats(bl_cls.per_repeat_accuracy,
                                          bsa_result.per_repeat_accuracy,
                                          bl_cls.accuracy, bsa_result.accuracy);

    printf("  LHS-021: %.0fs\n", bl_sim_time);
    printf("  Rate: %.1f Hz -> calibration target\n", target_rate_hz);
    printf("  Classification: %.1f%% (gap=%+.1fpp %s)\n",
           bl_cls.accuracy * 100, bl_stats.gap_pp, bl_stats.stars.c_str());

    // Print fingerprint for verification against Python export
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
        fprintf(f, "  \"mean_firing_rate_hz\": %.10f,\n", target_rate_hz);
        fprintf(f, "  \"classification_accuracy\": %.10f,\n", bl_cls.accuracy);
        fprintf(f, "  \"classification_accuracy_std\": %.10f,\n", bl_cls.accuracy_std);
        fprintf(f, "  \"per_repeat_accuracy\": [");
        for (int r = 0; r < (int)bl_cls.per_repeat_accuracy.size(); r++) {
            if (r > 0) fprintf(f, ", ");
            fprintf(f, "%.10f", bl_cls.per_repeat_accuracy[r]);
        }
        fprintf(f, "],\n");

        // Per-sample spike counts
        fprintf(f, "  \"per_sample_spikes\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.1f", bl_res.total_spikes[i]);
        }
        fprintf(f, "],\n");

        // Per-sample firing rates
        fprintf(f, "  \"per_sample_rate_hz\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.total_spikes[i] / (bl_res.n_reservoir * trial_dur_s));
        }
        fprintf(f, "],\n");

        // Per-sample ISI CVs
        fprintf(f, "  \"per_sample_isi_cv\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.isi_cvs[i]);
        }
        fprintf(f, "],\n");

        // Per-sample digits
        fprintf(f, "  \"per_sample_digit\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%d", bl_res.digits[i]);
        }
        fprintf(f, "],\n");

        // Per-sample adaptation at stim end
        fprintf(f, "  \"per_sample_adapt_stim_end\": [");
        for (int i = 0; i < n_samples; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.6f", bl_res.adapt_stim_ends[i]);
        }
        fprintf(f, "],\n");

        // Per-sample filenames
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
             target_rate_hz, bl_sim_time, bl_res.n_reservoir);
    std::string baseline_json = bl_json_buf;

    // 5. Sweep
    printf("\n[4] Running %d grid points (rate-matched to %.1f +/- %.0f Hz)...\n",
           n_grid, target_rate_hz, RATE_TOLERANCE_HZ);

    std::vector<std::map<std::string, std::string>> grid_results_json;
    std::vector<double> grid_point_times;
    double sweep_start = now_seconds();
    std::map<int, double> last_stim_by_tau;

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

        // Calibration
        double prev_stim = -1;
        if (pt.tau_idx >= 0 && last_stim_by_tau.count(pt.tau_idx))
            prev_stim = last_stim_by_tau[pt.tau_idx];

        SimConfig cal_sim = sim_cfg;
        double lo = CALIBRATION_STIM_LO, hi = CALIBRATION_STIM_HI;
        if (prev_stim > 0) {
            lo = std::max(lo, prev_stim / 3.0);
            hi = std::min(hi, prev_stim * 3.0);
        }

        auto [matched_stim, cal_rate, cal_log] =
            calibrate_stimulus_current(base_cfg, dyn_ovr, cal_samples, cal_sim,
                                       n_workers, target_rate_hz, lo, hi,
                                       prev_stim > 0 ? prev_stim : -1);

        // Retry if missed
        if (std::abs(cal_rate - target_rate_hz) > RATE_TOLERANCE_HZ) {
            std::tie(matched_stim, cal_rate, cal_log) =
                calibrate_stimulus_current(base_cfg, dyn_ovr, cal_samples, cal_sim,
                                           n_workers, target_rate_hz);
        }

        if (pt.tau_idx >= 0) last_stim_by_tau[pt.tau_idx] = matched_stim;
        printf("    Calibrated: stim=%.4f -> %.1f Hz\n", matched_stim, cal_rate);

        // Full evaluation
        SimConfig eval_sim = sim_cfg;
        eval_sim.stimulus_current = matched_stim;

        double t0 = now_seconds();
        auto res = run_all_samples(base_cfg, samples, dyn_ovr, eval_sim, n_workers);
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

        // Results printed after classification below

        // Classification
        int eval_n_bins = 0;
        for (auto& b : res.res_bins_list)
            if (b.rows > eval_n_bins) eval_n_bins = b.rows;

        auto cls_res = classify_flat_ridge(res.res_bins_list, y, eval_n_bins, res.n_reservoir);
        auto stats = compute_paired_stats(cls_res.per_repeat_accuracy,
                                           bsa_result.per_repeat_accuracy,
                                           cls_res.accuracy, bsa_result.accuracy);

        // (printed below after all metrics computed)

        // ISI stats
        double isi_cv_mean = 0;
        int isi_valid = 0;
        for (int i = 0; i < n_samples; i++) {
            if (!std::isnan(res.isi_cvs[i])) { isi_cv_mean += res.isi_cvs[i]; isi_valid++; }
        }
        if (isi_valid > 0) isi_cv_mean /= isi_valid;

        double adapt_mean = 0;
        for (int i = 0; i < n_samples; i++) adapt_mean += res.adapt_stim_ends[i];
        adapt_mean /= n_samples;

        // Participation ratio
        auto [pr_mean, pr_per_bin] = compute_participation_ratio(
            res.res_bins_list, eval_n_bins, res.n_reservoir);

        // Per-bin accuracy
        auto per_bin_acc = classify_per_bin(res.res_bins_list, y, eval_n_bins, res.n_reservoir);

        printf("    Rate: %.1f Hz | Acc: %.1f%% (gap=%+.1fpp %s) | ISI CV: %.3f | PR: %.4f\n",
               rate_mean, cls_res.accuracy * 100, stats.gap_pp, stats.stars.c_str(),
               isi_cv_mean, pr_mean);

        // Build JSON for this grid point
        std::ostringstream oss;
        oss << "{";
        oss << "\"point_id\": \"" << pt.point_id << "\", ";
        oss << "\"source\": \"" << pt.source << "\", ";
        oss << "\"inc_idx\": " << pt.inc_idx << ", ";
        oss << "\"tau_idx\": " << pt.tau_idx << ", ";
        oss << std::fixed;
        oss.precision(10);
        oss << "\"adapt_inc\": " << inc_val << ", ";
        oss << "\"adapt_tau\": " << tau_val << ", ";
        oss << "\"matched_stimulus_current\": " << matched_stim << ", ";
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
        // per_bin_accuracy
        oss << "\"per_bin_accuracy\": [";
        for (size_t i = 0; i < per_bin_acc.size(); i++) {
            if (i > 0) oss << ", ";
            oss << per_bin_acc[i];
        }
        oss << "], ";
        // per_repeat_accuracy
        oss << "\"classification_per_repeat_accuracy\": [";
        for (size_t i = 0; i < cls_res.per_repeat_accuracy.size(); i++) {
            if (i > 0) oss << ", ";
            oss << cls_res.per_repeat_accuracy[i];
        }
        oss << "]";
        oss << "}";

        grid_results_json.push_back({{"json", oss.str()}});

        double gp_elapsed = now_seconds() - gp_start;
        grid_point_times.push_back(gp_elapsed);
        printf("    %.1fmin  |  %d/%d  |  Elapsed: %.1fh\n",
               gp_elapsed / 60.0, pt_num + 1, n_grid,
               (now_seconds() - sweep_start) / 3600.0);

        // Checkpoint
        save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                        target_rate_hz, total_start, false);
    }

    double total_time = now_seconds() - total_start;

    // Summary
    printf("\n======================================================================\n");
    printf("  SWEEP COMPLETE\n");
    printf("  Total time: %.0fs (%.1f hours)\n", total_time, total_time / 3600.0);
    printf("  Grid points: %d/%d\n", (int)grid_results_json.size(), n_grid);
    printf("======================================================================\n");

    // Final save
    save_checkpoint(output_dir, grid_results_json, bsa_json, baseline_json,
                    target_rate_hz, total_start, true);

    // Generate figures via Python script
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
