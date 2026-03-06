#include "builder.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <set>

namespace cls {

NetworkConfig make_base_config() {
    NetworkConfig cfg;
    cfg.n_neurons = 1000;
    cfg.inhibitory_fraction = 0.185;
    cfg.connection_probabilities = {{"ee", 0.10}, {"ei", 0.15}, {"ie", 0.25}, {"ii", 0.15}};
    cfg.weight_scale = 0.55;
    cfg.weight_min = 0.0;
    cfg.transmission_delay = 1.0;
    cfg.distance_lambda = 0.18;
    cfg.lambda_decay_ie = 0.15;
    cfg.layout = "sphere";
    cfg.v_noise_amp = 0.1;
    cfg.i_noise_amp = 0.001;
    cfg.e_reversal = 0.0;
    cfg.i_reversal = -80.0;
    cfg.jitter_v_rest = 3.0;
    cfg.jitter_v_threshold = 2.5;
    cfg.jitter_tau_m = 0.3;
    cfg.jitter_tau_ref = 0.1;
    cfg.jitter_tau_e = 0.3;
    cfg.jitter_tau_i = 0.3;
    cfg.jitter_adaptation_increment = 0.4;
    cfg.jitter_tau_adaptation = 0.35;
    cfg.nmda_ratio = 0.5;
    cfg.tau_nmda = 100.0;
    cfg.mg_concentration = 1.0;
    cfg.jitter_tau_nmda = 0.0;
    cfg.radius_multiplier = 1.5;
    cfg.lambda_connect = 0.0;
    return cfg;
}

void create_ring_zone_network(SphericalNetwork& net, ZoneInfo& zone_info,
                              const NetworkConfig& cfg, bool quiet,
                              const std::string& connectivity_regime) {
    rng_seed(42); // config random_seed

    net.init(cfg);

    int n = cfg.n_neurons;
    int n_mel = 128; // n_mel_bins
    double radius = net.sphere_radius;

    // Compute radial distance for each neuron
    std::vector<double> radii(n);
    for (int i = 0; i < n; i++) {
        auto& p = net.positions[i];
        radii[i] = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    }

    // Surface shell
    double shell_frac = SURFACE_SHELL_FRACTION;
    double r_threshold = (1.0 - shell_frac) * radius;
    std::vector<bool> surface_mask(n), interior_mask(n);
    for (int i = 0; i < n; i++) {
        surface_mask[i] = radii[i] >= r_threshold;
        interior_mask[i] = !surface_mask[i];
    }

    // Equatorial band
    double band_half = RING_LATITUDE_BAND * radius;
    std::vector<bool> equatorial_mask(n);
    for (int i = 0; i < n; i++) {
        equatorial_mask[i] = std::abs(net.positions[i][1]) <= band_half;
    }

    // Ring candidates: surface AND equatorial
    std::vector<int> ring_excitatory;
    for (int i = 0; i < n; i++) {
        if (surface_mask[i] && equatorial_mask[i] && !net.is_inhibitory[i]) {
            ring_excitatory.push_back(i);
        }
    }

    // Compute azimuthal angle phi = atan2(z, x)
    std::vector<double> phis(ring_excitatory.size());
    for (size_t k = 0; k < ring_excitatory.size(); k++) {
        int i = ring_excitatory[k];
        phis[k] = std::atan2(net.positions[i][2], net.positions[i][0]);
    }
    std::vector<double> phis_deg(phis.size());
    for (size_t k = 0; k < phis.size(); k++) phis_deg[k] = phis[k] * 180.0 / M_PI;

    // Arc: exclude gap
    double gap_half = (360.0 - ARC_DEGREES) / 2.0;
    double gap_center = ARC_GAP_CENTER_DEG;
    if (gap_center > 180) gap_center -= 360;

    std::vector<int> arc_neurons;
    std::vector<double> arc_phis;
    for (size_t k = 0; k < ring_excitatory.size(); k++) {
        double ang_dist = std::abs(std::fmod(phis_deg[k] - gap_center + 180.0 + 360.0, 360.0) - 180.0);
        if (ang_dist > gap_half) {
            arc_neurons.push_back(ring_excitatory[k]);
            arc_phis.push_back(phis[k]);
        }
    }

    // Sort by azimuthal angle
    auto sort_idx = argsort(arc_phis);
    std::vector<int> sorted_arc(arc_neurons.size());
    std::vector<double> sorted_arc_phis(arc_phis.size());
    for (size_t k = 0; k < sort_idx.size(); k++) {
        sorted_arc[k] = arc_neurons[sort_idx[k]];
        sorted_arc_phis[k] = arc_phis[sort_idx[k]];
    }

    // Surface indices
    std::vector<int> surface_indices;
    for (int i = 0; i < n; i++) if (surface_mask[i]) surface_indices.push_back(i);

    if (!quiet) {
        int n_surf = 0, n_int = 0;
        for (int i = 0; i < n; i++) { if (surface_mask[i]) n_surf++; else n_int++; }
        printf("  Surface shell (r > %.2f): %d neurons\n", r_threshold, n_surf);
        printf("  Arc (%d deg): %d neurons\n", ARC_DEGREES, (int)arc_neurons.size());
        printf("  Interior: %d neurons\n", n_int);
    }

    // Overlapping spectral mapping
    int n_arc = (int)sorted_arc.size();
    std::map<int, std::vector<int>> input_mapping;
    std::set<int> all_input_set;

    if (n_arc > 0) {
        double phi_min = sorted_arc_phis[0];
        double phi_max = sorted_arc_phis[n_arc - 1];
        std::vector<double> channel_centers(n_mel);
        for (int m = 0; m < n_mel; m++) {
            channel_centers[m] = phi_min + (phi_max - phi_min) * m / (n_mel - 1);
        }

        int k = std::min(OVERLAP_K, n_arc);
        for (int mel_bin = 0; mel_bin < n_mel; mel_bin++) {
            std::vector<std::pair<double, int>> dists(n_arc);
            for (int a = 0; a < n_arc; a++) {
                dists[a] = {std::abs(sorted_arc_phis[a] - channel_centers[mel_bin]), a};
            }
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
            std::vector<int> mapped;
            for (int kk = 0; kk < k; kk++) {
                int nid = sorted_arc[dists[kk].second];
                mapped.push_back(nid);
                all_input_set.insert(nid);
            }
            input_mapping[mel_bin] = mapped;
        }
    }

    std::vector<int> input_neuron_indices(all_input_set.begin(), all_input_set.end());
    std::sort(input_neuron_indices.begin(), input_neuron_indices.end());

    // Interior = reservoir
    std::vector<int> reservoir_indices;
    for (int i = 0; i < n; i++) if (interior_mask[i]) reservoir_indices.push_back(i);

    zone_info.input_zone_indices = surface_indices;
    zone_info.reservoir_zone_indices = reservoir_indices;
    zone_info.input_neuron_mapping = input_mapping;
    zone_info.input_neuron_indices = input_neuron_indices;
    zone_info.y_threshold = r_threshold;
    zone_info.sphere_radius = radius;
    zone_info.connectivity_regime = connectivity_regime;

    // Connectivity pruning
    std::set<int> input_set(surface_indices.begin(), surface_indices.end());
    std::set<int> reservoir_set(reservoir_indices.begin(), reservoir_indices.end());
    std::set<int> arc_set(input_neuron_indices.begin(), input_neuron_indices.end());

    if (connectivity_regime == "default") {
        // Remove ALL reservoir->input zone feedback
        for (int ri : reservoir_indices) {
            for (int si : surface_indices) {
                net.weights[ri * n + si] = 0.0;
                net.delays[ri * n + si] = 0.0;
            }
        }
        // Remove ALL intra-shell
        for (int si : surface_indices) {
            for (int sj : surface_indices) {
                net.weights[si * n + sj] = 0.0;
                net.delays[si * n + sj] = 0.0;
            }
        }
    }
    // Other regimes omitted for brevity but follow same pattern

    net.build_csr();

    if (!quiet) {
        int n_ff = 0;
        for (int si : surface_indices)
            for (int ri : reservoir_indices)
                if (net.weights[si * n + ri] != 0.0) n_ff++;
        printf("  Connectivity regime: %s\n", connectivity_regime.c_str());
        printf("  Shell->core feedforward: %d\n", n_ff);
    }
}

void apply_config_b_overrides(SphericalNetwork& net, ZoneInfo& zone_info, double dt) {
    int n = net.n_neurons;
    // Scale input->reservoir weights
    for (int i : zone_info.input_zone_indices) {
        for (int j : zone_info.reservoir_zone_indices) {
            double& w = net.weights[i * n + j];
            if (w != 0.0) w *= WEIGHT_MULT;
        }
    }

    // Ratio-preserving adaptation rescale
    double orig_mean = 0;
    for (int i = 0; i < n; i++) orig_mean += net.adaptation_increment[i];
    orig_mean /= n;
    if (orig_mean > 1e-9) {
        double scale = ADAPT_INC_DEFAULT / orig_mean;
        for (int i = 0; i < n; i++) net.adaptation_increment[i] *= scale;
    } else {
        for (int i = 0; i < n; i++) net.adaptation_increment[i] = ADAPT_INC_DEFAULT;
    }
    clip_vec(net.adaptation_increment, 0.0, 1.0);

    // NMDA tau rescale
    double nmda_mean = 0;
    for (int i = 0; i < n; i++) nmda_mean += net.tau_nmda[i];
    nmda_mean /= n;
    if (nmda_mean > 1e-9) {
        double scale = NMDA_TAU_DEFAULT / nmda_mean;
        for (int i = 0; i < n; i++) net.tau_nmda[i] *= scale;
    } else {
        for (int i = 0; i < n; i++) net.tau_nmda[i] = NMDA_TAU_DEFAULT;
    }
    clip_vec(net.tau_nmda, 20.0, 300.0);

    net.build_csr();
    net.precompute_decay_factors(dt);
}

void apply_dynamical_overrides(SphericalNetwork& net, ZoneInfo& zone_info,
                               double dt, const DynamicalOverrides& ovr) {
    int n = net.n_neurons;

    // 1. Shell->core weight scaling
    if (ovr.shell_core_mult != 1.0) {
        for (int i : zone_info.input_zone_indices) {
            for (int j : zone_info.reservoir_zone_indices) {
                double& w = net.weights[i * n + j];
                if (w != 0.0) w *= ovr.shell_core_mult;
            }
        }
    }

    // 2. Core->core (excitatory only)
    if (ovr.core_core_mult != 1.0) {
        for (int i : zone_info.reservoir_zone_indices) {
            if (net.is_inhibitory[i]) continue;
            for (int j : zone_info.reservoir_zone_indices) {
                double& w = net.weights[i * n + j];
                if (w != 0.0) w *= ovr.core_core_mult;
            }
        }
    }

    // 3. Feedback scaling
    if (ovr.feedback_scale != 1.0) {
        for (int i : zone_info.reservoir_zone_indices) {
            for (int j : zone_info.input_zone_indices) {
                double& w = net.weights[i * n + j];
                if (w != 0.0) w *= ovr.feedback_scale;
            }
        }
    }

    // 4. Adaptation increment rescale
    double orig_mean = 0;
    for (int i = 0; i < n; i++) orig_mean += net.adaptation_increment[i];
    orig_mean /= n;
    if (orig_mean > 1e-9) {
        double scale = ovr.adapt_inc / orig_mean;
        for (int i = 0; i < n; i++) net.adaptation_increment[i] *= scale;
    } else {
        for (int i = 0; i < n; i++) net.adaptation_increment[i] = ovr.adapt_inc;
    }
    clip_vec(net.adaptation_increment, 0.0, 5.0);

    // 5. Adaptation tau rescale (optional)
    if (ovr.adapt_tau > 0) {
        double orig_tau = 0;
        for (int i = 0; i < n; i++) orig_tau += net.tau_adaptation[i];
        orig_tau /= n;
        if (orig_tau > 1e-9) {
            double scale = ovr.adapt_tau / orig_tau;
            for (int i = 0; i < n; i++) net.tau_adaptation[i] *= scale;
        } else {
            for (int i = 0; i < n; i++) net.tau_adaptation[i] = ovr.adapt_tau;
        }
        clip_vec(net.tau_adaptation, 5.0, 10000.0);
    }

    // 6. NMDA tau rescale
    double nmda_mean = 0;
    for (int i = 0; i < n; i++) nmda_mean += net.tau_nmda[i];
    nmda_mean /= n;
    if (nmda_mean > 1e-9) {
        double scale = ovr.nmda_tau / nmda_mean;
        for (int i = 0; i < n; i++) net.tau_nmda[i] *= scale;
    } else {
        for (int i = 0; i < n; i++) net.tau_nmda[i] = ovr.nmda_tau;
    }
    clip_vec(net.tau_nmda, 20.0, 300.0);

    net.build_csr();
    net.precompute_decay_factors(dt);
}

void compact_network(SphericalNetwork& net, ZoneInfo& zone_info,
                     double dt, bool quiet) {
    std::set<int> arc_set(zone_info.input_neuron_indices.begin(),
                          zone_info.input_neuron_indices.end());
    std::set<int> input_set(zone_info.input_zone_indices.begin(),
                            zone_info.input_zone_indices.end());
    std::set<int> nonarc_set;
    for (int x : input_set) {
        if (arc_set.find(x) == arc_set.end()) nonarc_set.insert(x);
    }

    if (nonarc_set.empty()) return;

    int n_old = net.n_neurons;
    std::vector<int> keep;
    for (int i = 0; i < n_old; i++) {
        if (nonarc_set.find(i) == nonarc_set.end()) keep.push_back(i);
    }
    int n_new = (int)keep.size();

    std::map<int, int> old_to_new;
    for (int i = 0; i < n_new; i++) old_to_new[keep[i]] = i;

    if (!quiet) {
        printf("  Compacting: %d -> %d neurons (%d non-arc removed)\n",
               n_old, n_new, (int)nonarc_set.size());
    }

    // Slice per-neuron arrays
    auto slice_vec_d = [&](std::vector<double>& vec) {
        if ((int)vec.size() != n_old) return;
        std::vector<double> nv(n_new);
        for (int i = 0; i < n_new; i++) nv[i] = vec[keep[i]];
        vec = std::move(nv);
    };
    auto slice_vec_b = [&](std::vector<bool>& vec) {
        if ((int)vec.size() != n_old) return;
        std::vector<bool> nv(n_new);
        for (int i = 0; i < n_new; i++) nv[i] = vec[keep[i]];
        vec = std::move(nv);
    };

    slice_vec_b(net.is_inhibitory);
    slice_vec_b(net.is_slow_inhibitory);
    slice_vec_d(net.v_rest); slice_vec_d(net.v_threshold); slice_vec_d(net.v_reset);
    slice_vec_d(net.tau_m); slice_vec_d(net.tau_ref);
    slice_vec_d(net.tau_e); slice_vec_d(net.tau_i); slice_vec_d(net.tau_i_slow);
    slice_vec_d(net.tau_nmda);
    slice_vec_d(net.adaptation_increment); slice_vec_d(net.tau_adaptation);
    slice_vec_d(net.e_reversal_arr); slice_vec_d(net.i_reversal_arr);
    slice_vec_d(net.k_reversal_arr);
    slice_vec_d(net.v_noise_amp_arr); slice_vec_d(net.i_noise_amp_arr);
    slice_vec_d(net.v); slice_vec_d(net.g_e); slice_vec_d(net.g_i);
    slice_vec_d(net.g_i_slow); slice_vec_d(net.g_nmda);
    slice_vec_d(net.adaptation); slice_vec_d(net.t_since_spike);

    // Decay factors
    slice_vec_d(net.exp_decay_e); slice_vec_d(net.exp_decay_i);
    slice_vec_d(net.exp_decay_i_slow);
    slice_vec_d(net.exp_decay_adapt); slice_vec_d(net.exp_decay_nmda);

    // Weight/delay matrices
    std::vector<double> new_weights(n_new * n_new, 0.0);
    std::vector<double> new_delays(n_new * n_new, 0.0);
    for (int ni = 0; ni < n_new; ni++) {
        for (int nj = 0; nj < n_new; nj++) {
            new_weights[ni * n_new + nj] = net.weights[keep[ni] * n_old + keep[nj]];
            new_delays[ni * n_new + nj] = net.delays[keep[ni] * n_old + keep[nj]];
        }
    }
    net.weights = std::move(new_weights);
    net.delays = std::move(new_delays);

    // Positions
    std::vector<std::array<double, 3>> new_pos(n_new);
    for (int i = 0; i < n_new; i++) new_pos[i] = net.positions[keep[i]];
    net.positions = std::move(new_pos);

    net.n_neurons = n_new;
    net.ring_initialized = false;
    net.build_csr();
    net.precompute_decay_factors(dt);

    // Remap zone_info
    auto remap = [&](const std::vector<int>& old_ids) {
        std::vector<int> nv;
        for (int x : old_ids) {
            auto it = old_to_new.find(x);
            if (it != old_to_new.end()) nv.push_back(it->second);
        }
        return nv;
    };

    zone_info.input_zone_indices = remap(zone_info.input_neuron_indices);
    zone_info.reservoir_zone_indices = remap(zone_info.reservoir_zone_indices);
    zone_info.input_neuron_indices = remap(zone_info.input_neuron_indices);

    std::map<int, std::vector<int>> new_mapping;
    for (auto& [mel, old_ids] : zone_info.input_neuron_mapping) {
        std::vector<int> new_ids;
        for (int nid : old_ids) {
            auto it = old_to_new.find(nid);
            if (it != old_to_new.end()) new_ids.push_back(it->second);
        }
        new_mapping[mel] = new_ids;
    }
    zone_info.input_neuron_mapping = std::move(new_mapping);

    if (!quiet) {
        int n_ff = 0, n_nz = 0;
        for (int i = 0; i < n_new * n_new; i++) if (net.weights[i] != 0.0) n_nz++;
        for (int si : zone_info.input_zone_indices)
            for (int ri : zone_info.reservoir_zone_indices)
                if (net.weights[si * n_new + ri] != 0.0) n_ff++;
        printf("  Post-compact: %d neurons, %d nonzero weights, %d feedforward\n",
               n_new, n_nz, n_ff);
    }
}

StdMasks build_std_masks(const SphericalNetwork& net, const ZoneInfo& zone_info) {
    StdMasks masks;
    std::set<int> reservoir_set(zone_info.reservoir_zone_indices.begin(),
                                zone_info.reservoir_zone_indices.end());

    for (int nid : zone_info.reservoir_zone_indices) {
        if (net.is_inhibitory[nid]) continue;
        masks.std_eligible.insert(nid);

        int64_t start = net.csr_indptr[nid];
        int64_t end = net.csr_indptr[nid + 1];
        int n_out = (int)(end - start);
        if (n_out > 0) {
            std::vector<bool> mask(n_out, false);
            bool any = false;
            for (int k = 0; k < n_out; k++) {
                if (reservoir_set.count(net.csr_targets[start + k])) {
                    mask[k] = true;
                    any = true;
                }
            }
            if (any) masks.recurrent_masks[nid] = mask;
        }
    }
    return masks;
}

RunResult run_sample_with_std(SphericalNetwork& net, const AudioSample& sample,
                              const ZoneInfo& zone_info, const SimConfig& sim_cfg,
                              double std_u, double std_tau_rec,
                              const StdMasks& masks,
                              double record_adapt_at_ms) {
    net.reset_all();
    bool use_std = std_u > 0.0 && !masks.std_eligible.empty();

    double dt = sim_cfg.dt;
    double total_ms = sim_cfg.audio_duration_ms + sim_cfg.post_stimulus_ms;
    int n_steps = (int)(total_ms / dt);
    double stim_current = sim_cfg.stimulus_current;

    // Pre-bin spike indices by step
    int n_spikes = (int)sample.spike_times_ms.size();
    std::unordered_map<int, std::vector<int>> step_to_spikes;
    for (int i = 0; i < n_spikes; i++) {
        int step = std::max(0, std::min((int)(sample.spike_times_ms[i] / dt), n_steps - 1));
        step_to_spikes[step].push_back(i);
    }

    // STD state
    int n_neurons = net.n_neurons;
    std::vector<double> x_resource;
    std::vector<double> last_update_time;
    std::map<int, std::vector<double>> original_weights;
    std::set<int> dirty_neurons;

    if (use_std) {
        x_resource.assign(n_neurons, 1.0);
        last_update_time.assign(n_neurons, 0.0);
        for (auto& [nid, mask] : masks.recurrent_masks) {
            int64_t start = net.csr_indptr[nid];
            int64_t end = net.csr_indptr[nid + 1];
            original_weights[nid].assign(net.csr_weights.begin() + start,
                                         net.csr_weights.begin() + end);
        }
    }

    bool record_adapt = record_adapt_at_ms >= 0;
    int adapt_step = record_adapt ? std::max(0, (int)(record_adapt_at_ms / dt)) : -1;
    bool adapt_recorded = false;

    RunResult result;
    result.activity_record.resize(n_steps);

    for (int step = 0; step < n_steps; step++) {
        double t_ms = step * dt;

        // Inject BSA input
        auto it = step_to_spikes.find(step);
        if (it != step_to_spikes.end()) {
            std::vector<int> neuron_indices;
            for (int si : it->second) {
                int fb = sample.freq_bin_indices[si];
                auto mit = zone_info.input_neuron_mapping.find(fb);
                if (mit != zone_info.input_neuron_mapping.end()) {
                    for (int nid : mit->second) neuron_indices.push_back(nid);
                }
            }
            if (!neuron_indices.empty()) {
                std::vector<double> currents(neuron_indices.size(), stim_current);
                net.stimulate_neurons(neuron_indices, currents);
            }
        }

        auto active = net.update_network(dt);
        result.activity_record[step] = active;

        // Adaptation snapshot
        if (record_adapt && !adapt_recorded && step >= adapt_step) {
            result.adapt_snapshot = net.adaptation;
            adapt_recorded = true;
        }

        // STD
        if (use_std && !active.empty()) {
            for (int nid : active) {
                if (masks.std_eligible.find(nid) == masks.std_eligible.end()) continue;
                auto mit = masks.recurrent_masks.find(nid);
                if (mit == masks.recurrent_masks.end()) continue;
                const auto& mask = mit->second;

                double delta_t = t_ms - last_update_time[nid];
                if (delta_t > 0 && x_resource[nid] < 1.0) {
                    x_resource[nid] = 1.0 - (1.0 - x_resource[nid]) * std::exp(-delta_t / std_tau_rec);
                    x_resource[nid] = std::min(x_resource[nid], 1.0);
                }

                x_resource[nid] *= (1.0 - std_u);
                last_update_time[nid] = t_ms;

                const auto& orig = original_weights[nid];
                int64_t start = net.csr_indptr[nid];
                int64_t end = net.csr_indptr[nid + 1];
                int n_out = (int)(end - start);
                for (int k = 0; k < n_out; k++) {
                    if (mask[k]) {
                        net.csr_weights[start + k] = orig[k] * x_resource[nid];
                    } else {
                        net.csr_weights[start + k] = orig[k];
                    }
                }
                dirty_neurons.insert(nid);
            }
        }
    }

    // Restore weights
    if (use_std) {
        for (int nid : dirty_neurons) {
            int64_t start = net.csr_indptr[nid];
            int64_t end = net.csr_indptr[nid + 1];
            const auto& orig = original_weights[nid];
            for (int64_t k = start; k < end; k++) {
                net.csr_weights[k] = orig[k - start];
            }
        }
    }

    return result;
}

void build_full_network(SphericalNetwork& net, ZoneInfo& zone_info,
                        const NetworkConfig& cfg, double dt,
                        bool quiet,
                        const DynamicalOverrides* dyn_ovr,
                        const std::string& connectivity_regime,
                        bool remove_nonarc) {
    create_ring_zone_network(net, zone_info, cfg, quiet, connectivity_regime);

    if (dyn_ovr != nullptr) {
        apply_dynamical_overrides(net, zone_info, dt, *dyn_ovr);
    } else {
        apply_config_b_overrides(net, zone_info, dt);
    }

    if (remove_nonarc) {
        compact_network(net, zone_info, dt, quiet);
    }
}

} // namespace cls
