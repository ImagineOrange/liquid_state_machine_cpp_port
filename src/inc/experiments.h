#pragma once

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
#include <future>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace cls;

// ============================================================
// SHARED CONSTANTS
// ============================================================
inline const std::vector<int> DEFAULT_DIGITS = {0, 1, 2, 3, 4};
inline const int N_DIGITS = 5;
inline const double BIN_MS = 20.0;
inline const double POST_STIM_MS = 200.0;
inline int SAMPLES_PER_DIGIT = 500;
inline const int N_CV_REPEATS = 5;
inline const int N_CV_FOLDS = 5;
inline const std::vector<double> RIDGE_ALPHAS = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
inline const int SEED = 42;
inline const int SAMPLE_LOAD_SEED = 42;

// Network snapshot path (set via --snapshot CLI arg; empty = build from RNG)
inline std::string g_snapshot_path;

// LHS-021 params
inline const double LHS021_LAMBDA_CONNECT = 0.003288382505082908;
inline const double LHS021_CORE_CORE_MULT = 0.8274594948671626;
[[maybe_unused]] inline const double LHS021_SURFACE_SHELL_FRAC = 0.18171487839543513;
inline const double LHS021_ADAPT_INC = 0.6264283936395093;
inline const double LHS021_SHELL_CORE_MULT = 4.849737508561994;
inline const double FIXED_NMDA_TAU = 50.0;

// Rate-matching
// Target: sensory cortex evoked firing rate during auditory processing.
// Auditory cortex neurons fire at ~10-40 Hz during stimulus presentation
// (Hromádka et al. 2008, DeWeese et al. 2003). 20 Hz is mid-range.
inline const double RATE_TARGET_HZ = 20.0;
inline const double RATE_TOLERANCE_HZ = 2.0;
inline const int CALIBRATION_N_SAMPLES = 200;

// ============================================================
// SHARED HELPERS
// ============================================================
inline double pearson_r_vec(const std::vector<double>& x, const std::vector<double>& y) {
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

// Binned mutual information estimate (discrete binning of rate values)
inline double binned_mi(const std::vector<double>& x, const std::vector<double>& y, int n_quant = 4) {
    int n = (int)std::min(x.size(), y.size());
    if (n < 4) return 0.0;

    auto quantize = [&](const std::vector<double>& v) -> std::vector<int> {
        int m = (int)v.size();
        std::vector<int> idx(m), q(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return v[a] < v[b]; });
        if (v[idx[0]] == v[idx[m - 1]]) {
            std::fill(q.begin(), q.end(), 0);
        } else {
            for (int i = 0; i < m; i++) q[idx[i]] = std::min(i * n_quant / m, n_quant - 1);
        }
        return q;
    };

    auto qx = quantize(x);
    auto qy = quantize(y);

    std::vector<double> px(n_quant, 0), py(n_quant, 0);
    std::vector<std::vector<double>> pxy(n_quant, std::vector<double>(n_quant, 0));
    for (int i = 0; i < n; i++) {
        px[qx[i]] += 1.0;
        py[qy[i]] += 1.0;
        pxy[qx[i]][qy[i]] += 1.0;
    }
    for (int i = 0; i < n_quant; i++) { px[i] /= n; py[i] /= n; }
    for (int i = 0; i < n_quant; i++)
        for (int j = 0; j < n_quant; j++) pxy[i][j] /= n;

    double mi = 0;
    for (int i = 0; i < n_quant; i++)
        for (int j = 0; j < n_quant; j++)
            if (pxy[i][j] > 1e-12 && px[i] > 1e-12 && py[j] > 1e-12)
                mi += pxy[i][j] * std::log2(pxy[i][j] / (px[i] * py[j]));
    return mi;
}

// Build per-input-neuron spike time maps from activity_record and BSA sample.
inline void build_spike_maps(
    const RunResult& result, const ZoneInfo& zone_info,
    const AudioSample& sample, double dt,
    const std::set<int>& input_set,
    std::unordered_map<int, std::vector<double>>& neuron_spikes,
    std::unordered_map<int, std::vector<double>>& neuron_bsa)
{
    int n_steps = (int)result.activity_record.size();
    for (int step = 0; step < n_steps; step++) {
        double t = step * dt;
        for (int nid : result.activity_record[step]) {
            if (input_set.count(nid))
                neuron_spikes[nid].push_back(t);
        }
    }
    for (size_t k = 0; k < sample.spike_times_ms.size(); k++) {
        int fb = sample.freq_bin_indices[k];
        auto mit = zone_info.input_neuron_mapping.find(fb);
        if (mit != zone_info.input_neuron_mapping.end()) {
            for (int nid : mit->second)
                neuron_bsa[nid].push_back(sample.spike_times_ms[k]);
        }
    }
}

struct InputGridMetrics {
    double mean_rate_hz;
    double mean_isi_cv;
    double mean_refrac_frac;
    double mean_mod_depth;
    double spike_bsa_r_5ms;
    double spike_bsa_r_10ms;
    double spike_bsa_r_20ms;
    double spike_bsa_r_50ms;
    double spike_bsa_r_100ms;
    double mean_burst_frac;
    double mean_fano_factor;
    double mean_temporal_sparseness;
    double mean_dynamic_range;
    double mean_mi_bits;
    int n_active_neurons;
};

inline InputGridMetrics compute_input_metrics(
    const ZoneInfo& zone_info,
    const std::unordered_map<int, std::vector<double>>& neuron_spikes,
    const std::unordered_map<int, std::vector<double>>& neuron_bsa,
    double stim_end_ms,
    const double bin_windows_ms[5])
{
    InputGridMetrics m{};

    double sum_rate = 0, sum_isi_cv = 0, sum_refrac = 0, sum_mod = 0;
    double sum_r[5] = {0, 0, 0, 0, 0};
    double sum_burst = 0, sum_fano = 0, sum_sparseness = 0;
    double sum_drange = 0;
    int n_with_isi = 0, n_active = 0, n_with_bsa = 0;
    int n_with_fano = 0, n_with_drange = 0;

    for (int nid : zone_info.input_neuron_indices) {
        auto sit = neuron_spikes.find(nid);
        if (sit == neuron_spikes.end() || sit->second.empty()) continue;
        const auto& spk = sit->second;
        n_active++;

        int n_stim_spikes = 0;
        for (double t : spk) if (t <= stim_end_ms) n_stim_spikes++;
        double rate_hz = n_stim_spikes / (stim_end_ms / 1000.0);
        sum_rate += rate_hz;

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

            int n_ref = 0;
            for (double v : isis) if (v <= 5.0) n_ref++;
            sum_refrac += (double)n_ref / isis.size();
            sum_burst += (double)n_ref / isis.size();
            n_with_isi++;
        }

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

            double min_nz = 1e30;
            for (double v : rates_mod) if (v > 0 && v < min_nz) min_nz = v;
            if (max_r > 0 && min_nz < 1e29) {
                sum_drange += std::log10(max_r / min_nz);
                n_with_drange++;
            }

            if (n_bins_mod > 1) {
                double s1 = 0, s2 = 0;
                for (double v : rates_mod) { s1 += v; s2 += v * v; }
                s1 /= n_bins_mod;
                s2 /= n_bins_mod;
                if (s2 > 1e-12) {
                    double sp = (1.0 - (s1 * s1) / s2) / (1.0 - 1.0 / n_bins_mod);
                    sum_sparseness += std::max(0.0, std::min(1.0, sp));
                }
            }
        }

        {
            int n_bins_ff = std::max(1, (int)(stim_end_ms / 50.0));
            std::vector<double> counts(n_bins_ff, 0.0);
            for (double t : spk) {
                if (t <= stim_end_ms) {
                    int b = std::min((int)(t / 50.0), n_bins_ff - 1);
                    counts[b] += 1.0;
                }
            }
            if (n_bins_ff >= 3) {
                double mn = 0, mn2 = 0;
                for (double c : counts) { mn += c; mn2 += c * c; }
                mn /= n_bins_ff;
                mn2 /= n_bins_ff;
                double var_ff = mn2 - mn * mn;
                if (mn > 1e-9) {
                    sum_fano += var_ff / mn;
                    n_with_fano++;
                }
            }
        }

        auto bit = neuron_bsa.find(nid);
        if (bit != neuron_bsa.end() && !bit->second.empty()) {
            n_with_bsa++;
            for (int wi = 0; wi < 5; wi++) {
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
    m.spike_bsa_r_5ms  = n_with_bsa > 0 ? sum_r[0] / n_with_bsa : 0;
    m.spike_bsa_r_10ms = n_with_bsa > 0 ? sum_r[1] / n_with_bsa : 0;
    m.spike_bsa_r_20ms = n_with_bsa > 0 ? sum_r[2] / n_with_bsa : 0;
    m.spike_bsa_r_50ms = n_with_bsa > 0 ? sum_r[3] / n_with_bsa : 0;
    m.spike_bsa_r_100ms = n_with_bsa > 0 ? sum_r[4] / n_with_bsa : 0;
    m.mean_burst_frac = n_with_isi > 0 ? sum_burst / n_with_isi : 0;
    m.mean_fano_factor = n_with_fano > 0 ? sum_fano / n_with_fano : 0;
    m.mean_temporal_sparseness = n_active > 0 ? sum_sparseness / n_active : 0;
    m.mean_dynamic_range = n_with_drange > 0 ? sum_drange / n_with_drange : 0;
    m.mean_mi_bits = 0;
    return m;
}

// Load a single npz file into an AudioSample (thread-safe, no shared state)
inline AudioSample load_single_sample(const std::string& fpath) {
    NpzFile npz = load_npz(fpath);
    AudioSample s;
    s.spike_times_ms = npz["spike_times_ms"].to_float64_vec();
    s.freq_bin_indices = npz["freq_bin_indices"].to_int32_vec();
    s.digit = (int)npz["digit"].as_scalar_int();
    s.speaker = npz["speaker"].as_string();
    s.filename = fs::path(fpath).stem().string();
    return s;
}

// Load audio samples from data directory (parallel I/O + decompression)
inline std::vector<AudioSample> load_audio_samples(
    const std::string& data_dir, const std::vector<int>& digits,
    int samples_per_digit, int random_seed)
{
    rng_seed(random_seed + 100);
    std::string spike_dir = data_dir + "/spike_trains_bsa";

    // Phase 1: collect all file paths deterministically (same order as before)
    struct LoadEntry { std::string path; int digit; };
    std::vector<LoadEntry> all_files;

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

        printf("  Digit %d: %d files found, selected %d\n",
               digit, (int)files.size(), (int)selected.size());
        for (auto& f : selected)
            all_files.push_back({f, digit});
    }

    // Phase 2: parallel load all files
    int n_files = (int)all_files.size();
    int n_threads = std::min((int)std::thread::hardware_concurrency(), n_files);
    if (n_threads < 1) n_threads = 1;
    printf("  Loading %d files with %d threads...\n", n_files, n_threads);

    std::vector<AudioSample> results(n_files);
    std::vector<bool> success(n_files, false);
    std::atomic<int> next_idx{0};
    std::atomic<int> done_count{0};

    auto worker = [&]() {
        while (true) {
            int idx = next_idx.fetch_add(1);
            if (idx >= n_files) break;
            try {
                results[idx] = load_single_sample(all_files[idx].path);
                success[idx] = true;
            } catch (const std::exception& e) {
                fprintf(stderr, "WARNING: Skipping %s: %s\n",
                        all_files[idx].path.c_str(), e.what());
            }
            int d = done_count.fetch_add(1) + 1;
            if (d % 200 == 0 || d == n_files)
                printf("    %d/%d loaded\n", d, n_files);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    // Phase 3: collect successful loads in original order (preserves deterministic ordering)
    std::vector<AudioSample> samples;
    samples.reserve(n_files);
    for (int i = 0; i < n_files; i++) {
        if (success[i])
            samples.push_back(std::move(results[i]));
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
// EXPERIMENT ENTRY POINTS
// ============================================================
int run_input_grid(int argc, char** argv, const std::string& snapshot_path,
                   const std::string& data_dir, int n_workers,
                   const std::string& output_csv);

int run_mi_refine(int argc, char** argv, const std::string& snapshot_path,
                  const std::string& data_dir, int n_workers,
                  const std::string& input_csv, int top_n,
                  int samples_per_digit, const std::string& output_csv);

int run_raster_dump(const std::string& snapshot_path,
                    const std::string& trace_file,
                    const std::string& output_dir,
                    double stim_current_override,
                    double input_tau_e_override,
                    double input_adapt_inc_override,
                    double adapt_inc_override = -1.0,
                    double adapt_tau_override = -1.0,
                    double tonic_conductance_override = -1.0);

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
                              double input_std_u_override);

int run_wm_sweep(int argc, char** argv,
                  int n_workers, const std::string& output_dir,
                  const std::string& data_dir);

int run_serial_sweep(int argc, char** argv,
                     int n_workers, const std::string& output_dir,
                     const std::string& data_dir);

int run_noisy_sweep(int argc, char** argv,
                     int n_workers, const std::string& output_dir,
                     const std::string& data_dir,
                     const std::vector<double>& tau_values,
                     bool do_per_bin = false);

int run_mechanistic_interp(int argc, char** argv,
                            int n_workers, const std::string& output_dir,
                            const std::string& data_dir);

int run_mech_raster(int n_workers, const std::string& output_dir,
                     const std::string& data_dir);

int run_extended_tau(int argc, char** argv,
                      int n_workers, const std::string& output_dir,
                      const std::string& data_dir);

int run_gap_sweep(int argc, char** argv,
                   int n_workers, const std::string& output_dir,
                   const std::string& data_dir);

int run_gap_sweep_ext(int argc, char** argv,
                       int n_workers, const std::string& output_dir,
                       const std::string& data_dir);

int run_hidden_states(int argc, char** argv,
                       int n_workers, const std::string& output_dir,
                       const std::string& data_dir);

int run_crossover_sweep(int argc, char** argv,
                         int n_workers, const std::string& output_dir,
                         const std::string& data_dir);

int run_ablation(int argc, char** argv,
                  int n_workers, const std::string& output_dir,
                  const std::string& data_dir);

int run_multi_seed(int argc, char** argv,
                    int n_workers, const std::string& output_dir,
                    const std::string& data_dir);
