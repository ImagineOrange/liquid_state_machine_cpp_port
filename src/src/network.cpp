#include "network.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>

namespace cls {

double SphericalNetwork::jitter_gaussian(double base, double sigma) {
    if (sigma > 0) return base + rng_normal(0.0, sigma);
    return base;
}

double SphericalNetwork::jitter_lognormal(double base, double cv) {
    if (cv > 0) {
        double sigma_log = std::sqrt(std::log(1 + cv * cv));
        double mu_log = std::log(base) - sigma_log * sigma_log / 2.0;
        return rng_lognormal(mu_log, sigma_log);
    }
    return base;
}

std::vector<double> SphericalNetwork::jitter_gaussian_vec(double base, double sigma, int n) {
    std::vector<double> out(n);
    if (sigma > 0) {
        for (int i = 0; i < n; i++) out[i] = base + rng_normal(0.0, sigma);
    } else {
        std::fill(out.begin(), out.end(), base);
    }
    return out;
}

std::vector<double> SphericalNetwork::jitter_lognormal_vec(double base, double cv, int n) {
    std::vector<double> out(n);
    if (cv > 0) {
        double sigma_log = std::sqrt(std::log(1 + cv * cv));
        double mu_log = std::log(base) - sigma_log * sigma_log / 2.0;
        for (int i = 0; i < n; i++) out[i] = rng_lognormal(mu_log, sigma_log);
    } else {
        std::fill(out.begin(), out.end(), base);
    }
    return out;
}

void SphericalNetwork::generate_3d_positions(const std::string& layout) {
    positions.resize(n_neurons);
    if (layout == "sphere-surface") {
        double golden = (1.0 + std::sqrt(5.0)) / 2.0;
        for (int i = 0; i < n_neurons; i++) {
            double theta = 2.0 * M_PI * i / golden;
            double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n_neurons);
            positions[i] = {
                sphere_radius * std::sin(phi) * std::cos(theta),
                sphere_radius * std::sin(phi) * std::sin(theta),
                sphere_radius * std::cos(phi)
            };
        }
    } else {
        // Uniform inside sphere
        int count = 0;
        while (count < n_neurons) {
            double x = rng_uniform(-sphere_radius, sphere_radius);
            double y = rng_uniform(-sphere_radius, sphere_radius);
            double z = rng_uniform(-sphere_radius, sphere_radius);
            if (x*x + y*y + z*z <= sphere_radius * sphere_radius) {
                positions[count] = {x, y, z};
                count++;
            }
        }
    }
}

void SphericalNetwork::create_connections(double ws, double wmin,
                                           double trans_delay, double dist_lambda,
                                           double lam_ie, double lam_connect) {
    double sphere_diameter = 2.0 * sphere_radius;
    attempted_connections = 0;
    nonzero_connections = 0;
    connection_counts_by_type = {{"ee", 0}, {"ei", 0}, {"ie", 0}, {"ii", 0}};
    std::map<std::string, int> attempts = {{"ee", 0}, {"ei", 0}, {"ie", 0}, {"ii", 0}};

    weights.assign(n_neurons * n_neurons, 0.0);
    delays.assign(n_neurons * n_neurons, 0.0);

    for (int i = 0; i < n_neurons; i++) {
        for (int j = 0; j < n_neurons; j++) {
            if (i == j) continue;

            bool si = is_inhibitory[i], tj = is_inhibitory[j];
            std::string ct;
            if (si && tj) ct = "ii";
            else if (si) ct = "ie";
            else if (tj) ct = "ei";
            else ct = "ee";

            double cp = connection_probabilities[ct];

            double dist = 0.0;
            if (!positions.empty()) {
                double dx = positions[i][0] - positions[j][0];
                double dy = positions[i][1] - positions[j][1];
                double dz = positions[i][2] - positions[j][2];
                dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (lam_connect > 0) cp *= std::exp(-lam_connect * dist);
            }

            if (rng_uniform() < cp) {
                attempted_connections++;
                attempts[ct]++;

                double distance_factor = 1.0;
                if (!positions.empty()) {
                    bool is_ie = si && !tj;
                    double eff_lambda = is_ie ? lam_ie : dist_lambda;
                    distance_factor = std::exp(-eff_lambda * dist);
                }

                double base_weight;
                if (si) {
                    base_weight = -rng_uniform(wmin, ws * 4.0);
                } else {
                    base_weight = rng_uniform(wmin, ws);
                }

                double w = base_weight * distance_factor;
                double d;
                if (!positions.empty()) {
                    d = trans_delay * (0.5 + 0.5 * dist / sphere_diameter);
                } else {
                    d = trans_delay;
                }

                if (w > 0.01 || w < -0.01) {
                    weights[i * n_neurons + j] = w;
                    delays[i * n_neurons + j] = d;
                    nonzero_connections++;
                    connection_counts_by_type[ct]++;
                }
            }
        }
    }
}

void SphericalNetwork::init(const NetworkConfig& cfg) {
    n_neurons = cfg.n_neurons;
    weight_scale = cfg.weight_scale;
    distance_lambda = cfg.distance_lambda;
    lambda_decay_ie = cfg.lambda_decay_ie;
    nmda_ratio = cfg.nmda_ratio;
    mg_concentration = cfg.mg_concentration;
    _mg_factor = cfg.mg_concentration / 3.57;
    e_reversal = cfg.e_reversal;
    i_reversal = cfg.i_reversal;
    v_noise_amp_scalar = cfg.v_noise_amp;
    i_noise_amp_scalar = cfg.i_noise_amp;
    lambda_connect = cfg.lambda_connect;

    sphere_radius = std::pow(3.0 * n_neurons / (4.0 * M_PI), 1.0/3.0) * cfg.radius_multiplier;

    // Connection probabilities
    connection_probabilities = {{"ee", 0.1}, {"ei", 0.1}, {"ie", 0.1}, {"ii", 0.1}};
    for (auto& [k, v] : cfg.connection_probabilities) {
        connection_probabilities[k] = v;
    }

    // Determine inhibitory neurons
    is_inhibitory.resize(n_neurons);
    for (int i = 0; i < n_neurons; i++) {
        is_inhibitory[i] = rng_uniform() < cfg.inhibitory_fraction;
    }
    is_slow_inhibitory.assign(n_neurons, false);

    // Base parameters
    double base_v_rest = -65.0, base_v_threshold = -55.0, base_v_reset = -75.0;
    double base_tau_m = 10.0, base_tau_ref_exc = 4.0, base_tau_ref_inh = 2.5;
    double base_tau_e = 3.0, base_tau_i = 7.0;
    double base_adapt_inc = 0.2, base_tau_adapt = 100.0;
    double k_rev = -90.0;

    v_rest = jitter_gaussian_vec(base_v_rest, cfg.jitter_v_rest, n_neurons);
    v_threshold = jitter_gaussian_vec(base_v_threshold, cfg.jitter_v_threshold, n_neurons);
    v_reset.assign(n_neurons, base_v_reset);
    tau_m = jitter_lognormal_vec(base_tau_m, cfg.jitter_tau_m, n_neurons);
    tau_e = jitter_lognormal_vec(base_tau_e, cfg.jitter_tau_e, n_neurons);
    tau_i = jitter_lognormal_vec(base_tau_i, cfg.jitter_tau_i, n_neurons);
    adaptation_increment = jitter_lognormal_vec(base_adapt_inc, cfg.jitter_adaptation_increment, n_neurons);
    tau_adaptation = jitter_lognormal_vec(base_tau_adapt, cfg.jitter_tau_adaptation, n_neurons);
    tau_nmda = jitter_lognormal_vec(cfg.tau_nmda, cfg.jitter_tau_nmda, n_neurons);

    // Refractory period depends on neuron type
    tau_ref.resize(n_neurons);
    if (cfg.jitter_tau_ref > 0) {
        double sigma_log = std::sqrt(std::log(1 + cfg.jitter_tau_ref * cfg.jitter_tau_ref));
        for (int i = 0; i < n_neurons; i++) {
            double base = is_inhibitory[i] ? base_tau_ref_inh : base_tau_ref_exc;
            double mu_log = std::log(base) - sigma_log * sigma_log / 2.0;
            tau_ref[i] = rng_lognormal(mu_log, sigma_log);
        }
    } else {
        for (int i = 0; i < n_neurons; i++)
            tau_ref[i] = is_inhibitory[i] ? base_tau_ref_inh : base_tau_ref_exc;
    }

    // Biologically plausible clipping
    clip_vec(v_rest, -80.0, -55.0);
    for (int i = 0; i < n_neurons; i++)
        v_threshold[i] = clip(v_threshold[i], v_rest[i] + 5.0, -40.0);
    clip_vec(tau_m, 3.0, 30.0);
    clip_vec(tau_ref, 1.0, 10.0);
    clip_vec(tau_e, 0.5, 10.0);
    clip_vec(tau_i, 2.0, 20.0);
    clip_vec(adaptation_increment, 0.0, 1.0);
    clip_vec(tau_adaptation, 20.0, 300.0);
    clip_vec(tau_nmda, 20.0, 300.0);

    // Reversal potentials
    e_reversal_arr.assign(n_neurons, cfg.e_reversal);
    i_reversal_arr.assign(n_neurons, cfg.i_reversal);
    k_reversal_arr.assign(n_neurons, k_rev);

    // Noise
    v_noise_amp_arr.assign(n_neurons, cfg.v_noise_amp);
    i_noise_amp_arr.assign(n_neurons, cfg.i_noise_amp);

    // Slow inhibitory
    tau_i_slow.assign(n_neurons, 50.0);

    // Dynamic state
    v = v_rest;
    g_e.assign(n_neurons, 0.0);
    g_i.assign(n_neurons, 0.0);
    g_i_slow.assign(n_neurons, 0.0);
    g_nmda.assign(n_neurons, 0.0);
    adaptation.assign(n_neurons, 0.0);
    t_since_spike.resize(n_neurons);
    for (int i = 0; i < n_neurons; i++) t_since_spike[i] = tau_ref[i] + 1e-5;

    // 3D positions
    generate_3d_positions(cfg.layout);

    // Create connections
    create_connections(cfg.weight_scale, cfg.weight_min, cfg.transmission_delay,
                       cfg.distance_lambda, cfg.lambda_decay_ie, cfg.lambda_connect);

    // Print stats
    int n_exc = 0;
    for (int i = 0; i < n_neurons; i++) if (!is_inhibitory[i]) n_exc++;
    int n_inh = n_neurons - n_exc;
    printf("\n===== 3D Spherical Network (C++) =====\n");
    printf("Neurons: %d (E: %d, I: %d)\n", n_neurons, n_exc, n_inh);
    printf("Sphere radius: %.2f\n", sphere_radius);
    printf("Non-zero connections: %d\n", nonzero_connections);
    printf("======================================\n");

    build_csr();
}

void SphericalNetwork::build_csr() {
    // Count nonzero per source
    std::vector<int64_t> counts(n_neurons, 0);
    int total = 0;
    for (int i = 0; i < n_neurons; i++) {
        for (int j = 0; j < n_neurons; j++) {
            if (weights[i * n_neurons + j] != 0.0) {
                counts[i]++;
                total++;
            }
        }
    }

    csr_indptr.resize(n_neurons + 1);
    csr_indptr[0] = 0;
    for (int i = 0; i < n_neurons; i++)
        csr_indptr[i + 1] = csr_indptr[i] + counts[i];

    csr_targets.resize(total);
    csr_weights.resize(total);
    csr_delays.resize(total);
    csr_slow_inh.resize(total, false);
    csr_lengths.resize(n_neurons);

    std::vector<int64_t> pos(n_neurons, 0);
    for (int i = 0; i < n_neurons; i++) {
        csr_lengths[i] = counts[i];
        int64_t base = csr_indptr[i];
        int64_t idx = 0;
        for (int j = 0; j < n_neurons; j++) {
            double w = weights[i * n_neurons + j];
            if (w != 0.0) {
                csr_targets[base + idx] = j;
                csr_weights[base + idx] = w;
                csr_delays[base + idx] = delays[i * n_neurons + j];
                if (is_slow_inhibitory[i] && w < 0)
                    csr_slow_inh[base + idx] = true;
                idx++;
            }
        }
    }

    (void)total;
}

void SphericalNetwork::precompute_decay_factors(double dt) {
    exp_decay_e.resize(n_neurons);
    exp_decay_i.resize(n_neurons);
    exp_decay_i_slow.resize(n_neurons);
    exp_decay_adapt.resize(n_neurons);
    exp_decay_nmda.resize(n_neurons);

    for (int i = 0; i < n_neurons; i++) {
        exp_decay_e[i] = std::exp(-dt / std::max(tau_e[i], 1e-9));
        exp_decay_i[i] = std::exp(-dt / std::max(tau_i[i], 1e-9));
        exp_decay_i_slow[i] = std::exp(-dt / std::max(tau_i_slow[i], 1e-9));
        exp_decay_adapt[i] = std::exp(-dt / std::max(tau_adaptation[i], 1e-9));
        exp_decay_nmda[i] = std::exp(-dt / std::max(tau_nmda[i], 1e-9));
    }
    cached_dt = dt;
}

void SphericalNetwork::init_ring_buffer(double dt) {
    double max_delay = 0.0;
    csr_delay_steps.resize(csr_delays.size());
    for (size_t i = 0; i < csr_delays.size(); i++) {
        if (csr_delays[i] > max_delay) max_delay = csr_delays[i];
        csr_delay_steps[i] = (int64_t)std::round(csr_delays[i] / dt);
    }

    ring_size = (max_delay > 0) ? (int)std::ceil(max_delay / dt) + 2 : 2;
    ring_targets.resize(ring_size);
    ring_weights.resize(ring_size);
    ring_slow_inh.resize(ring_size);
    ring_initialized = true;
}

void SphericalNetwork::clear_ring_buffer() {
    for (int i = 0; i < ring_size; i++) {
        ring_targets[i].clear();
        ring_weights[i].clear();
        ring_slow_inh[i].clear();
    }
}

std::vector<int> SphericalNetwork::update_network(double dt) {
    if (cached_dt != dt) precompute_decay_factors(dt);
    if (!ring_initialized) init_ring_buffer(dt);

    int current_step = step_counter;
    int slot = current_step % ring_size;

    // 1. Deliver due spikes from ring buffer
    if (!ring_targets[slot].empty()) {
        int n_deliver = (int)ring_targets[slot].size();
        for (int k = 0; k < n_deliver; k++) {
            int tgt = ring_targets[slot][k];
            double w = ring_weights[slot][k];
            bool slow = ring_slow_inh[slot][k];
            if (w > 0) {
                g_e[tgt] += w;
                g_nmda[tgt] += w * nmda_ratio;
            } else {
                if (slow) {
                    g_i_slow[tgt] += -w;
                } else {
                    g_i[tgt] += -w;
                }
            }
        }
        ring_targets[slot].clear();
        ring_weights[slot].clear();
        ring_slow_inh[slot].clear();
    }

    // Ensure persistent refractory buffer is sized
    if ((int)_refractory.size() != n_neurons)
        _refractory.resize(n_neurons);

    // Pointers into contiguous arrays for cache-friendly access
    double* __restrict__ pv  = v.data();
    double* __restrict__ pge = g_e.data();
    double* __restrict__ pgi = g_i.data();
    double* __restrict__ pgis = g_i_slow.data();
    double* __restrict__ pgnmda = g_nmda.data();
    double* __restrict__ padapt = adaptation.data();
    const double* __restrict__ pvrest = v_rest.data();
    const double* __restrict__ pvthr = v_threshold.data();
    const double* __restrict__ pvrst = v_reset.data();
    const double* __restrict__ ptaum = tau_m.data();
    const double* __restrict__ ptref = tau_ref.data();
    const double* __restrict__ perev = e_reversal_arr.data();
    const double* __restrict__ pirev = i_reversal_arr.data();
    const double* __restrict__ pkrev = k_reversal_arr.data();
    const double* __restrict__ pvnoise = v_noise_amp_arr.data();
    const double* __restrict__ pinoise = i_noise_amp_arr.data();
    const double* __restrict__ padaptinc = adaptation_increment.data();
    const double* __restrict__ pdece = exp_decay_e.data();
    const double* __restrict__ pdeci = exp_decay_i.data();
    const double* __restrict__ pdecis = exp_decay_i_slow.data();
    const double* __restrict__ pdecnmda = exp_decay_nmda.data();
    const double* __restrict__ pdecadapt = exp_decay_adapt.data();
    double* __restrict__ ptss = t_since_spike.data();

    // 2-6: Fused loop — refractory, membrane dynamics, conductance decay,
    //       synaptic noise, spike detection — single pass over neurons.
    double trace_v_noise = 0.0, trace_ge_noise = 0.0, trace_gi_noise = 0.0;
    std::vector<int> active;
    active.reserve(32);

    for (int i = 0; i < n_neurons; i++) {
        // Refractory check
        bool ref = ptss[i] < ptref[i];
        ptss[i] += dt;

        // Membrane dynamics
        double vi = pv[i];
        double i_e = pge[i] * (perev[i] - vi);
        double i_i = pgi[i] * (pirev[i] - vi);
        double i_is = pgis[i] * (pirev[i] - vi);
        double mg_block = 1.0 / (1.0 + _mg_factor * std::exp(-0.062 * vi));
        double i_nmda = pgnmda[i] * mg_block * (perev[i] - vi);
        double i_adapt = padapt[i] * (pkrev[i] - vi);

        double dv = dt * ((-(vi - pvrest[i]) / ptaum[i]) + i_e + i_nmda + i_i + i_is + i_adapt);
        vi += dv;
        double vn = rng_normal() * pvnoise[i];
        vi += vn;

        // Clamp if refractory
        if (ref) vi = pvrst[i];
        pv[i] = vi;

        // Decay conductances
        pge[i]    *= pdece[i];
        pgi[i]    *= pdeci[i];
        pgis[i]   *= pdecis[i];
        pgnmda[i] *= pdecnmda[i];
        padapt[i] *= pdecadapt[i];

        // Synaptic noise
        double ne = rng_normal() * pinoise[i];
        double ni = rng_normal() * pinoise[i];
        if (ne > 0) pge[i] += ne;
        if (ni > 0) pgi[i] += ni;

        // Spike detection
        if (vi >= pvthr[i] && !ref) {
            active.push_back(i);
        }

        // Trace recording (rare — only when trace_neuron_id is set)
        if (i == trace_neuron_id) {
            trace_v_noise = vn;
            trace_ge_noise = ne > 0 ? ne : 0.0;
            trace_gi_noise = ni > 0 ? ni : 0.0;
        }
    }

    // 7. Post-spike updates
    for (int idx : active) {
        pv[idx] = pvrst[idx];
        ptss[idx] = 0.0;
        padapt[idx] += padaptinc[idx];
    }

    // 8. Record trace for target neuron
    if (trace_neuron_id >= 0 && trace_neuron_id < n_neurons) {
        int ti = trace_neuron_id;
        double vi = pv[ti];
        double ie = pge[ti] * (perev[ti] - vi);
        double ii = pgi[ti] * (pirev[ti] - vi);
        double iis = pgis[ti] * (pirev[ti] - vi);
        double mgb = 1.0 / (1.0 + _mg_factor * std::exp(-0.062 * vi));
        double inmda = pgnmda[ti] * mgb * (perev[ti] - vi);
        double iadapt = padapt[ti] * (pkrev[ti] - vi);
        bool spiked = false;
        for (int idx : active) { if (idx == ti) { spiked = true; break; } }
        trace.push_back({pv[ti], pge[ti], pgi[ti], pgis[ti], pgnmda[ti], padapt[ti],
                         ie, ii, iis, inmda, iadapt,
                         trace_v_noise, trace_ge_noise, trace_gi_noise, spiked});
    }

    // 9. Step counter
    step_counter++;

    // 10. Queue new spikes into ring buffer
    if (!active.empty()) {
        for (int nid : active) {
            int64_t start = csr_indptr[nid];
            int64_t end = csr_indptr[nid + 1];
            for (int64_t c = start; c < end; c++) {
                int delivery_step = current_step + (int)csr_delay_steps[c];
                int rs = delivery_step % ring_size;
                ring_targets[rs].push_back(csr_targets[c]);
                ring_weights[rs].push_back(csr_weights[c]);
                ring_slow_inh[rs].push_back(csr_slow_inh[c]);
            }
        }
    }

    return active;
}

void SphericalNetwork::reset_all() {
    v = v_rest;
    std::fill(g_e.begin(), g_e.end(), 0.0);
    std::fill(g_i.begin(), g_i.end(), 0.0);
    std::fill(g_i_slow.begin(), g_i_slow.end(), 0.0);
    std::fill(g_nmda.begin(), g_nmda.end(), 0.0);
    std::fill(adaptation.begin(), adaptation.end(), 0.0);
    trace.clear();
    for (int i = 0; i < n_neurons; i++) t_since_spike[i] = tau_ref[i] + 1e-5;

    step_counter = 0;
    network_activity.clear();
    current_avalanche_size = 0;
    current_avalanche_start = -1;

    if (ring_initialized) clear_ring_buffer();
}

void SphericalNetwork::stimulate_neuron(int idx, double current) {
    if (idx < 0 || idx >= n_neurons) return;
    if (current > 0) {
        g_e[idx] += current;
        if (!skip_stim_nmda)
            g_nmda[idx] += current * nmda_ratio;
    } else {
        g_i[idx] += -current;
    }
}

void SphericalNetwork::stimulate_neurons(const std::vector<int>& indices,
                                          const std::vector<double>& currents) {
    for (size_t k = 0; k < indices.size(); k++) {
        int idx = indices[k];
        double c = currents[k];
        if (c > 0) {
            g_e[idx] += c;
            if (!skip_stim_nmda)
                g_nmda[idx] += c * nmda_ratio;
        } else if (c < 0) {
            g_i[idx] += -c;
        }
    }
}

} // namespace cls
