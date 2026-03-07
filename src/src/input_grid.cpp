#include "experiments.h"

// ============================================================
// INPUT NEURON GRID SEARCH
// ============================================================
static const std::vector<double> IG_STIM = {
    0.0100, 0.0139, 0.0193, 0.0268, 0.0373, 0.0518, 0.0720, 0.1000,
    0.1389, 0.1931, 0.2683, 0.3728, 0.5179, 0.7197, 1.0000,
    1.3895, 1.9307, 2.6827, 3.7276, 5.0000
};
static const std::vector<double> IG_TAU_E = {
    0.050, 0.092, 0.169, 0.311, 0.571, 1.050, 1.931, 3.550, 6.527, 12.000
};
static const std::vector<double> IG_ADAPT = {
    0.0, 0.005, 0.0158, 0.05, 0.1581, 0.50, 1.5811, 5.0
};
struct StdPair { double u; double tau_rec; };
static const std::vector<StdPair> IG_STD_PAIRS = {
    {0.0,   0.0},
    {0.10, 30.0},
    {0.15, 80.0},
    {0.25, 50.0},
    {0.35, 120.0},
};
static const int IG_SAMPLES_PER_DIGIT = 6;

struct InputGridPoint {
    double stim_current, input_tau_e, input_adapt_inc;
    double input_std_u, input_std_tau_rec;
    int idx;
};

static std::vector<InputGridPoint> build_input_grid() {
    std::vector<InputGridPoint> pts;
    int idx = 0;
    for (double stim : IG_STIM)
        for (double tau : IG_TAU_E)
            for (double adapt : IG_ADAPT)
                for (auto& sp : IG_STD_PAIRS)
                    pts.push_back({stim, tau, adapt, sp.u, sp.tau_rec, idx++});
    return pts;
}

int run_input_grid(int argc, char** argv, const std::string& snapshot_path,
                   const std::string& data_dir, int n_workers,
                   const std::string& output_csv) {
    printf("======================================================================\n");
    printf("  INPUT NEURON REGIME GRID SEARCH\n");
    printf("======================================================================\n");

    auto grid = build_input_grid();
    int n_grid = (int)grid.size();

    printf("[1] Loading audio (%d per digit)...\n", IG_SAMPLES_PER_DIGIT);
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS,
                                       IG_SAMPLES_PER_DIGIT, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded\n", n_samples);

    double max_audio_ms = 0;
    for (auto& s : samples) {
        double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (mx > max_audio_ms) max_audio_ms = mx;
    }
    max_audio_ms += 5.0;

    std::vector<double> stim_ends(n_samples);
    for (int i = 0; i < n_samples; i++) {
        stim_ends[i] = *std::max_element(samples[i].spike_times_ms.begin(),
                                          samples[i].spike_times_ms.end());
    }

    double bin_windows[5] = {5.0, 10.0, 20.0, 50.0, 100.0};

    printf("[2] Running grid search: %d grid points x %d samples = %d sims\n",
           n_grid, n_samples, n_grid * n_samples);
    printf("    Axes: stim_current(%d) x tau_e(%d) x adapt_inc(%d) x std_pairs(%d)\n",
           (int)IG_STIM.size(), (int)IG_TAU_E.size(),
           (int)IG_ADAPT.size(), (int)IG_STD_PAIRS.size());
    printf("    NMDA on input: DISABLED\n");
    printf("    Workers: %d\n", n_workers);

    FILE* csv = fopen(output_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "Cannot open %s\n", output_csv.c_str()); return 1; }
    fprintf(csv, "stim_current,input_tau_e,input_adapt_inc,"
                 "input_std_u,input_std_tau_rec,"
                 "mean_rate_hz,mean_isi_cv,mean_refrac_frac,mean_mod_depth,"
                 "spike_bsa_r_5ms,spike_bsa_r_10ms,spike_bsa_r_20ms,"
                 "spike_bsa_r_50ms,spike_bsa_r_100ms,"
                 "mean_burst_frac,mean_fano_factor,mean_temporal_sparseness,"
                 "mean_dynamic_range,mean_mi_bits,"
                 "n_active_neurons,composite_score\n");

    SphericalNetwork ref_net;
    ZoneInfo ref_zone;
    if (!snapshot_path.empty()) {
        load_network_snapshot(ref_net, ref_zone, snapshot_path, 0.1, true);
    } else {
        NetworkConfig ref_cfg = make_base_config();
        ref_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
        build_full_network(ref_net, ref_zone, ref_cfg, 0.1, true,
                          nullptr, "default", true);
    }
    int n_input = (int)ref_zone.input_neuron_indices.size();
    std::unordered_map<int, int> nid_to_local;
    for (int i = 0; i < n_input; i++)
        nid_to_local[ref_zone.input_neuron_indices[i]] = i;
    printf("    Input neurons: %d (for pooled MI)\n", n_input);

    struct ThreadState {
        SphericalNetwork net;
        ZoneInfo zone_info;
        StdMasks masks;
        std::set<int> input_set;
    };
    std::vector<ThreadState> tstate(n_workers);

    #pragma omp parallel num_threads(n_workers)
    {
        int tid = omp_get_thread_num();
        auto& ts = tstate[tid];
        if (!snapshot_path.empty()) {
            load_network_snapshot(ts.net, ts.zone_info, snapshot_path, 0.1, true);
            DynamicalOverrides dyn_ovr;
            dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
            dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
            apply_dynamical_overrides(ts.net, ts.zone_info, 0.1, dyn_ovr);
        } else {
            NetworkConfig base_cfg = make_base_config();
            base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
            DynamicalOverrides dyn_ovr;
            dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
            dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
            build_full_network(ts.net, ts.zone_info, base_cfg, 0.1, true,
                              &dyn_ovr, "default", true);
        }
        ts.net.skip_stim_nmda = true;
        ts.masks = build_std_masks(ts.net, ts.zone_info);
        ts.input_set.insert(ts.zone_info.input_neuron_indices.begin(),
                            ts.zone_info.input_neuron_indices.end());
    }
    printf("[3] Built %d thread-local networks\n", n_workers);

    double t_start = now_seconds();

    for (int gp = 0; gp < n_grid; gp++) {
        auto& pt = grid[gp];

        for (int tid = 0; tid < n_workers; tid++) {
            auto& ts = tstate[tid];
            for (int nid : ts.zone_info.input_neuron_indices) {
                ts.net.tau_e[nid] = pt.input_tau_e;
                ts.net.adaptation_increment[nid] = pt.input_adapt_inc;
            }
            ts.net.precompute_decay_factors(0.1);
        }

        double sum_rate = 0, sum_cv = 0, sum_ref = 0, sum_mod = 0;
        double sum_r5 = 0, sum_r10 = 0, sum_r20 = 0, sum_r50 = 0, sum_r100 = 0;
        double sum_burst = 0, sum_fano = 0, sum_sparse = 0, sum_drange = 0;
        double sum_active = 0;

        std::vector<std::vector<double>> all_bsa_bins(n_samples * n_input);
        std::vector<std::vector<double>> all_spk_bins(n_samples * n_input);

        #pragma omp parallel num_threads(n_workers) \
            reduction(+:sum_rate,sum_cv,sum_ref,sum_mod,sum_r5,sum_r10,sum_r20,sum_r50,sum_r100,sum_burst,sum_fano,sum_sparse,sum_drange,sum_active)
        {
            int tid = omp_get_thread_num();
            auto& net = tstate[tid].net;
            auto& zone_info = tstate[tid].zone_info;
            auto& masks = tstate[tid].masks;
            auto& input_set = tstate[tid].input_set;

            rng_seed(42 + (uint64_t)tid * 1000 + (uint64_t)getpid());

            #pragma omp for schedule(dynamic, 1)
            for (int si = 0; si < n_samples; si++) {
                SimConfig sim;
                sim.dt = 0.1;
                sim.audio_duration_ms = max_audio_ms;
                sim.post_stimulus_ms = POST_STIM_MS;
                sim.stimulus_current = pt.stim_current;
                sim.input_std_u = pt.input_std_u;
                sim.input_std_tau_rec = pt.input_std_tau_rec;

                auto result = run_sample_with_std(net, samples[si], zone_info, sim,
                                                   STD_U, STD_TAU_REC, masks,
                                                   stim_ends[si]);

                std::unordered_map<int, std::vector<double>> neuron_spikes, neuron_bsa;
                build_spike_maps(result, zone_info, samples[si], sim.dt,
                                 input_set, neuron_spikes, neuron_bsa);

                auto m = compute_input_metrics(zone_info, neuron_spikes, neuron_bsa,
                                                stim_ends[si], bin_windows);

                sum_rate += m.mean_rate_hz;
                sum_cv += m.mean_isi_cv;
                sum_ref += m.mean_refrac_frac;
                sum_mod += m.mean_mod_depth;
                sum_r5 += m.spike_bsa_r_5ms;
                sum_r10 += m.spike_bsa_r_10ms;
                sum_r20 += m.spike_bsa_r_20ms;
                sum_r50 += m.spike_bsa_r_50ms;
                sum_r100 += m.spike_bsa_r_100ms;
                sum_burst += m.mean_burst_frac;
                sum_fano += m.mean_fano_factor;
                sum_sparse += m.mean_temporal_sparseness;
                sum_drange += m.mean_dynamic_range;
                sum_active += m.n_active_neurons;

                double stim_end = stim_ends[si];
                int n_mi_bins = std::max(1, (int)(stim_end / 20.0));

                for (int nid : zone_info.input_neuron_indices) {
                    int li = nid_to_local.at(nid);
                    int idx = si * n_input + li;

                    std::vector<double> bsa_b(n_mi_bins, 0.0);
                    auto bit = neuron_bsa.find(nid);
                    if (bit != neuron_bsa.end()) {
                        for (double t : bit->second) {
                            if (t <= stim_end) {
                                int b = std::min((int)(t / 20.0), n_mi_bins - 1);
                                bsa_b[b] += 1.0;
                            }
                        }
                    }
                    all_bsa_bins[idx] = std::move(bsa_b);

                    std::vector<double> spk_b(n_mi_bins, 0.0);
                    auto sit = neuron_spikes.find(nid);
                    if (sit != neuron_spikes.end()) {
                        for (double t : sit->second) {
                            if (t <= stim_end) {
                                int b = std::min((int)(t / 20.0), n_mi_bins - 1);
                                spk_b[b] += 1.0;
                            }
                        }
                    }
                    all_spk_bins[idx] = std::move(spk_b);
                }
            }
        }

        double total_mi = 0;
        int n_mi_neurons = 0;
        for (int li = 0; li < n_input; li++) {
            std::vector<double> pooled_bsa, pooled_spk;
            for (int si = 0; si < n_samples; si++) {
                int idx = si * n_input + li;
                auto& bb = all_bsa_bins[idx];
                auto& sb = all_spk_bins[idx];
                pooled_bsa.insert(pooled_bsa.end(), bb.begin(), bb.end());
                pooled_spk.insert(pooled_spk.end(), sb.begin(), sb.end());
            }
            if ((int)pooled_bsa.size() >= 32) {
                total_mi += binned_mi(pooled_bsa, pooled_spk, 8);
                n_mi_neurons++;
            }
        }
        double avg_mi = n_mi_neurons > 0 ? total_mi / n_mi_neurons : 0;

        double avg_rate = sum_rate / n_samples;
        double avg_cv = sum_cv / n_samples;
        double avg_ref = sum_ref / n_samples;
        double avg_mod = sum_mod / n_samples;
        double avg_r5 = sum_r5 / n_samples;
        double avg_r10 = sum_r10 / n_samples;
        double avg_r20 = sum_r20 / n_samples;
        double avg_r50 = sum_r50 / n_samples;
        double avg_r100 = sum_r100 / n_samples;
        double avg_burst = sum_burst / n_samples;
        double avg_fano = sum_fano / n_samples;
        double avg_sparse = sum_sparse / n_samples;
        double avg_drange = sum_drange / n_samples;
        double avg_active = sum_active / n_samples;

        bool bio_valid = (avg_rate >= 5.0 && avg_rate <= 150.0)
                      && (avg_cv >= 0.3 && avg_cv <= 2.0)
                      && (avg_ref < 0.10)
                      && (avg_burst < 0.15);

        double score = 0;
        if (bio_valid) {
            score = avg_mi;
            score += 0.15 * avg_r20;
            score += 0.05 * avg_mod;
        }

        fprintf(csv, "%.4f,%.2f,%.4f,"
                     "%.4f,%.1f,"
                     "%.2f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,%.4f,%.4f,"
                     "%.1f,%.4f\n",
                pt.stim_current, pt.input_tau_e, pt.input_adapt_inc,
                pt.input_std_u, pt.input_std_tau_rec,
                avg_rate, avg_cv, avg_ref, avg_mod,
                avg_r5, avg_r10, avg_r20, avg_r50, avg_r100,
                avg_burst, avg_fano, avg_sparse, avg_drange, avg_mi,
                avg_active, score);
        fflush(csv);

        int gc = gp + 1;
        double elapsed = now_seconds() - t_start;
        double eta = (elapsed / gc) * (n_grid - gc);
        printf("  [%4d/%d] stim=%.3f tau=%.1f adpt=%.2f std=%.2f/%.0f | "
               "rate=%.0fHz r20=%.3f MI=%.3f fano=%.2f refr=%.2f score=%.3f | "
               "%.0fs ETA %.0fs\n",
               gc, n_grid,
               pt.stim_current, pt.input_tau_e, pt.input_adapt_inc,
               pt.input_std_u, pt.input_std_tau_rec,
               avg_rate, avg_r20, avg_mi, avg_fano, avg_ref, score,
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
// MI REFINEMENT
// ============================================================
struct MiRefineConfig {
    double stim_current, input_tau_e, input_adapt_inc;
    double input_std_u, input_std_tau_rec;
    double original_mi;
    double original_r20;
};

static std::vector<MiRefineConfig> load_top_configs(const std::string& csv_path, int top_n) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        fprintf(stderr, "Cannot open %s\n", csv_path.c_str());
        return {};
    }
    std::string header;
    std::getline(in, header);

    struct Row {
        MiRefineConfig cfg;
        double composite;
    };
    std::vector<Row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(ss, tok, ',')) vals.push_back(std::atof(tok.c_str()));
        if (vals.size() < 21) continue;
        Row r;
        r.cfg.stim_current = vals[0];
        r.cfg.input_tau_e = vals[1];
        r.cfg.input_adapt_inc = vals[2];
        r.cfg.input_std_u = vals[3];
        r.cfg.input_std_tau_rec = vals[4];
        r.cfg.original_mi = vals[18];
        r.cfg.original_r20 = vals[12];
        r.composite = vals[18];
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.composite > b.composite;
    });
    std::vector<MiRefineConfig> out;
    for (int i = 0; i < std::min(top_n, (int)rows.size()); i++)
        out.push_back(rows[i].cfg);
    return out;
}

int run_mi_refine(int argc, char** argv, const std::string& snapshot_path,
                  const std::string& data_dir, int n_workers,
                  const std::string& input_csv, int top_n,
                  int samples_per_digit, const std::string& output_csv) {
    printf("======================================================================\n");
    printf("  MI REFINEMENT — TOP %d CONFIGURATIONS\n", top_n);
    printf("======================================================================\n");

    auto configs = load_top_configs(input_csv, top_n);
    if (configs.empty()) {
        fprintf(stderr, "No configs loaded from %s\n", input_csv.c_str());
        return 1;
    }
    int n_configs = (int)configs.size();
    printf("  Loaded %d configs from %s\n", n_configs, input_csv.c_str());

    printf("[1] Loading audio (%d per digit)...\n", samples_per_digit);
    auto samples = load_audio_samples(data_dir, DEFAULT_DIGITS,
                                       samples_per_digit, SAMPLE_LOAD_SEED);
    int n_samples = (int)samples.size();
    printf("  %d samples loaded\n", n_samples);

    double max_audio_ms = 0;
    for (auto& s : samples) {
        double mx = *std::max_element(s.spike_times_ms.begin(), s.spike_times_ms.end());
        if (mx > max_audio_ms) max_audio_ms = mx;
    }
    max_audio_ms += 5.0;

    std::vector<double> stim_ends(n_samples);
    for (int i = 0; i < n_samples; i++) {
        stim_ends[i] = *std::max_element(samples[i].spike_times_ms.begin(),
                                          samples[i].spike_times_ms.end());
    }

    const std::vector<double> MI_BIN_WIDTHS = {5.0, 10.0, 20.0, 50.0};
    const std::vector<int> MI_QUANTILES = {4, 8, 16, 32};

    printf("[2] Running MI refinement: %d configs x %d samples = %d sims\n",
           n_configs, n_samples, n_configs * n_samples);
    printf("    MI resolutions: bin_widths={5,10,20,50}ms x quantiles={4,8,16,32}\n");
    printf("    Workers: %d\n", n_workers);

    SphericalNetwork ref_net;
    ZoneInfo ref_zone;
    if (!snapshot_path.empty()) {
        load_network_snapshot(ref_net, ref_zone, snapshot_path, 0.1, true);
    } else {
        NetworkConfig ref_cfg = make_base_config();
        ref_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
        build_full_network(ref_net, ref_zone, ref_cfg, 0.1, true,
                          nullptr, "default", true);
    }
    int n_input = (int)ref_zone.input_neuron_indices.size();
    std::unordered_map<int, int> nid_to_local;
    for (int i = 0; i < n_input; i++)
        nid_to_local[ref_zone.input_neuron_indices[i]] = i;
    printf("    Input neurons: %d\n", n_input);

    struct ThreadState {
        SphericalNetwork net;
        ZoneInfo zone_info;
        StdMasks masks;
        std::set<int> input_set;
    };
    std::vector<ThreadState> tstate(n_workers);

    #pragma omp parallel num_threads(n_workers)
    {
        int tid = omp_get_thread_num();
        auto& ts = tstate[tid];
        if (!snapshot_path.empty()) {
            load_network_snapshot(ts.net, ts.zone_info, snapshot_path, 0.1, true);
            DynamicalOverrides dyn_ovr;
            dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
            dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
            apply_dynamical_overrides(ts.net, ts.zone_info, 0.1, dyn_ovr);
        } else {
            NetworkConfig base_cfg = make_base_config();
            base_cfg.lambda_connect = LHS021_LAMBDA_CONNECT;
            DynamicalOverrides dyn_ovr;
            dyn_ovr.shell_core_mult = LHS021_SHELL_CORE_MULT;
            dyn_ovr.core_core_mult = LHS021_CORE_CORE_MULT;
            dyn_ovr.adapt_inc = LHS021_ADAPT_INC;
            dyn_ovr.nmda_tau = FIXED_NMDA_TAU;
            build_full_network(ts.net, ts.zone_info, base_cfg, 0.1, true,
                              &dyn_ovr, "default", true);
        }
        ts.net.skip_stim_nmda = true;
        ts.masks = build_std_masks(ts.net, ts.zone_info);
        ts.input_set.insert(ts.zone_info.input_neuron_indices.begin(),
                            ts.zone_info.input_neuron_indices.end());
    }
    printf("[3] Built %d thread-local networks\n", n_workers);

    FILE* csv = fopen(output_csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "Cannot open %s\n", output_csv.c_str()); return 1; }
    fprintf(csv, "rank,stim_current,input_tau_e,input_adapt_inc,"
                 "input_std_u,input_std_tau_rec,"
                 "original_mi,original_r20,mean_rate_hz,spike_bsa_r_20ms");
    for (double bw : MI_BIN_WIDTHS)
        for (int nq : MI_QUANTILES)
            fprintf(csv, ",mi_%.0fms_q%d", bw, nq);
    fprintf(csv, "\n");

    double t_start = now_seconds();

    for (int ci = 0; ci < n_configs; ci++) {
        auto& cfg = configs[ci];

        for (int tid = 0; tid < n_workers; tid++) {
            auto& ts = tstate[tid];
            for (int nid : ts.zone_info.input_neuron_indices) {
                ts.net.tau_e[nid] = cfg.input_tau_e;
                ts.net.adaptation_increment[nid] = cfg.input_adapt_inc;
            }
            ts.net.precompute_decay_factors(0.1);
        }

        double sum_rate = 0, sum_r20 = 0;

        int n_bw = (int)MI_BIN_WIDTHS.size();
        std::vector<std::vector<std::vector<double>>> all_bsa_bins(n_bw,
            std::vector<std::vector<double>>(n_samples * n_input));
        std::vector<std::vector<std::vector<double>>> all_spk_bins(n_bw,
            std::vector<std::vector<double>>(n_samples * n_input));

        double bin_windows[5] = {5.0, 10.0, 20.0, 50.0, 100.0};

        #pragma omp parallel num_threads(n_workers) \
            reduction(+:sum_rate,sum_r20)
        {
            int tid = omp_get_thread_num();
            auto& net = tstate[tid].net;
            auto& zone_info = tstate[tid].zone_info;
            auto& masks = tstate[tid].masks;
            auto& input_set = tstate[tid].input_set;

            rng_seed(42 + (uint64_t)tid * 1000 + (uint64_t)getpid());

            #pragma omp for schedule(dynamic, 1)
            for (int si = 0; si < n_samples; si++) {
                SimConfig sim;
                sim.dt = 0.1;
                sim.audio_duration_ms = max_audio_ms;
                sim.post_stimulus_ms = POST_STIM_MS;
                sim.stimulus_current = cfg.stim_current;
                sim.input_std_u = cfg.input_std_u;
                sim.input_std_tau_rec = cfg.input_std_tau_rec;

                auto result = run_sample_with_std(net, samples[si], zone_info, sim,
                                                   STD_U, STD_TAU_REC, masks,
                                                   stim_ends[si]);

                std::unordered_map<int, std::vector<double>> neuron_spikes, neuron_bsa;
                build_spike_maps(result, zone_info, samples[si], sim.dt,
                                 input_set, neuron_spikes, neuron_bsa);

                auto m = compute_input_metrics(zone_info, neuron_spikes, neuron_bsa,
                                                stim_ends[si], bin_windows);
                sum_rate += m.mean_rate_hz;
                sum_r20 += m.spike_bsa_r_20ms;

                double stim_end = stim_ends[si];

                for (int bwi = 0; bwi < n_bw; bwi++) {
                    double bw = MI_BIN_WIDTHS[bwi];
                    int n_bins = std::max(1, (int)(stim_end / bw));

                    for (int nid : zone_info.input_neuron_indices) {
                        int li = nid_to_local.at(nid);
                        int idx = si * n_input + li;

                        std::vector<double> bsa_b(n_bins, 0.0);
                        auto bit = neuron_bsa.find(nid);
                        if (bit != neuron_bsa.end()) {
                            for (double t : bit->second) {
                                if (t <= stim_end) {
                                    int b = std::min((int)(t / bw), n_bins - 1);
                                    bsa_b[b] += 1.0;
                                }
                            }
                        }
                        all_bsa_bins[bwi][idx] = std::move(bsa_b);

                        std::vector<double> spk_b(n_bins, 0.0);
                        auto sit = neuron_spikes.find(nid);
                        if (sit != neuron_spikes.end()) {
                            for (double t : sit->second) {
                                if (t <= stim_end) {
                                    int b = std::min((int)(t / bw), n_bins - 1);
                                    spk_b[b] += 1.0;
                                }
                            }
                        }
                        all_spk_bins[bwi][idx] = std::move(spk_b);
                    }
                }
            }
        }

        std::vector<std::vector<double>> mi_results(n_bw,
            std::vector<double>(MI_QUANTILES.size(), 0.0));

        for (int bwi = 0; bwi < n_bw; bwi++) {
            for (int qi = 0; qi < (int)MI_QUANTILES.size(); qi++) {
                int nq = MI_QUANTILES[qi];
                double total_mi = 0;
                int n_mi_neurons = 0;
                for (int li = 0; li < n_input; li++) {
                    std::vector<double> pooled_bsa, pooled_spk;
                    for (int si = 0; si < n_samples; si++) {
                        int idx = si * n_input + li;
                        auto& bb = all_bsa_bins[bwi][idx];
                        auto& sb = all_spk_bins[bwi][idx];
                        pooled_bsa.insert(pooled_bsa.end(), bb.begin(), bb.end());
                        pooled_spk.insert(pooled_spk.end(), sb.begin(), sb.end());
                    }
                    int min_pts = std::max(4 * nq, 32);
                    if ((int)pooled_bsa.size() >= min_pts) {
                        total_mi += binned_mi(pooled_bsa, pooled_spk, nq);
                        n_mi_neurons++;
                    }
                }
                mi_results[bwi][qi] = n_mi_neurons > 0 ? total_mi / n_mi_neurons : 0;
            }
        }

        double avg_rate = sum_rate / n_samples;
        double avg_r20 = sum_r20 / n_samples;

        fprintf(csv, "%d,%.4f,%.2f,%.4f,%.4f,%.1f,%.4f,%.4f,%.2f,%.4f",
                ci + 1, cfg.stim_current, cfg.input_tau_e, cfg.input_adapt_inc,
                cfg.input_std_u, cfg.input_std_tau_rec,
                cfg.original_mi, cfg.original_r20, avg_rate, avg_r20);
        for (int bwi = 0; bwi < n_bw; bwi++)
            for (int qi = 0; qi < (int)MI_QUANTILES.size(); qi++)
                fprintf(csv, ",%.4f", mi_results[bwi][qi]);
        fprintf(csv, "\n");
        fflush(csv);

        int gc = ci + 1;
        double elapsed = now_seconds() - t_start;
        double eta = (elapsed / gc) * (n_configs - gc);
        printf("  [%2d/%d] stim=%.4f tau=%.2f adpt=%.4f | rate=%.0fHz r20=%.3f | "
               "MI@20ms: q4=%.3f q8=%.3f q16=%.3f q32=%.3f (orig=%.3f) | "
               "%.0fs ETA %.0fs\n",
               gc, n_configs,
               cfg.stim_current, cfg.input_tau_e, cfg.input_adapt_inc,
               avg_rate, avg_r20,
               mi_results[2][0], mi_results[2][1], mi_results[2][2], mi_results[2][3],
               cfg.original_mi,
               elapsed, eta);
    }

    fclose(csv);
    double total = now_seconds() - t_start;
    printf("\n======================================================================\n");
    printf("  MI REFINEMENT DONE: %d configs in %.1fs (%.1f min)\n",
           n_configs, total, total / 60.0);
    printf("  Results: %s\n", output_csv.c_str());
    printf("======================================================================\n");
    return 0;
}
