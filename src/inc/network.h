#pragma once
#include "common.h"
#include <vector>
#include <map>
#include <tuple>

namespace cls {

struct NetworkConfig {
    int n_neurons = 1000;
    double inhibitory_fraction = 0.2;
    std::map<std::string, double> connection_probabilities;
    double weight_scale = 0.55;
    double weight_min = 0.0;
    double transmission_delay = 1.0;
    double distance_lambda = 0.18;
    double lambda_decay_ie = 0.15;
    std::string layout = "sphere";
    double v_noise_amp = 0.1;
    double i_noise_amp = 0.001;
    double e_reversal = 0.0;
    double i_reversal = -80.0;
    double jitter_v_rest = 3.0;
    double jitter_v_threshold = 2.5;
    double jitter_tau_m = 0.3;
    double jitter_tau_ref = 0.1;
    double jitter_tau_e = 0.3;
    double jitter_tau_i = 0.3;
    double jitter_adaptation_increment = 0.4;
    double jitter_tau_adaptation = 0.35;
    double nmda_ratio = 0.5;
    double tau_nmda = 100.0;
    double mg_concentration = 1.0;
    double jitter_tau_nmda = 0.0;
    double radius_multiplier = 1.5;
    double lambda_connect = 0.0;
};

class SphericalNetwork {
public:
    int n_neurons;
    double sphere_radius;
    double weight_scale;
    double distance_lambda;
    double lambda_decay_ie;
    double nmda_ratio;
    double mg_concentration;
    double _mg_factor;
    double e_reversal, i_reversal;
    double v_noise_amp_scalar, i_noise_amp_scalar;
    double lambda_connect;

    // Per-neuron boolean
    std::vector<bool> is_inhibitory;
    std::vector<bool> is_slow_inhibitory;

    // Per-neuron parameters
    std::vector<double> v_rest, v_threshold, v_reset;
    std::vector<double> tau_m, tau_ref, tau_e, tau_i, tau_i_slow, tau_nmda;
    std::vector<double> adaptation_increment, tau_adaptation;
    std::vector<double> e_reversal_arr, i_reversal_arr, k_reversal_arr;
    std::vector<double> v_noise_amp_arr, i_noise_amp_arr;

    // Dynamic state
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;

    // Tonic background current (nA) — applied per-neuron each timestep.
    // Used for rate-matching without modulating input encoding.
    // Positive = depolarizing (increases rate), negative = hyperpolarizing.
    std::vector<double> background_current;

    // Decay factors
    std::vector<double> exp_decay_e, exp_decay_i, exp_decay_i_slow;
    std::vector<double> exp_decay_adapt, exp_decay_nmda;
    double cached_dt = -1.0;

    // Weight/delay matrices (n x n, row-major)
    std::vector<double> weights;
    std::vector<double> delays;

    // 3D positions
    std::vector<std::array<double, 3>> positions;

    // CSR outgoing connections
    std::vector<int64_t> csr_indptr;
    std::vector<int32_t> csr_targets;
    std::vector<double> csr_weights;
    std::vector<double> csr_delays;
    std::vector<int64_t> csr_delay_steps;
    std::vector<bool> csr_slow_inh;
    std::vector<int64_t> csr_lengths;

    // Ring buffer
    bool ring_initialized = false;
    int ring_size = 0;
    std::vector<std::vector<int32_t>> ring_targets;
    std::vector<std::vector<double>> ring_weights;
    std::vector<std::vector<bool>> ring_slow_inh;

    // Skip NMDA on stimulate_neuron/stimulate_neurons calls (experiment flag)
    bool skip_stim_nmda = false;

    // Single-neuron trace recording (optional)
    int trace_neuron_id = -1;  // -1 = disabled
    struct TraceRow {
        double v, g_e, g_i, g_i_slow, g_nmda, adaptation;
        double i_e, i_i, i_i_slow, i_nmda, i_adapt;
        double v_noise, ge_noise, gi_noise;
        bool spiked;
    };
    std::vector<TraceRow> trace;

    // Activity tracking
    int step_counter = 0;
    std::vector<int> network_activity;  // kept for callers that read it; not grown in hot path
    int current_avalanche_size = 0;
    int current_avalanche_start = -1;

    // Persistent scratch buffers (avoid per-step allocation)
    std::vector<bool> _refractory;

    // Connection stats
    int attempted_connections = 0;
    int nonzero_connections = 0;
    std::map<std::string, int> connection_counts_by_type;
    std::map<std::string, double> connection_probabilities;

    // Construction
    SphericalNetwork() = default;
    void init(const NetworkConfig& cfg);
    void build_csr();
    void precompute_decay_factors(double dt);

    // Simulation
    std::vector<int> update_network(double dt);
    void reset_all();
    void stimulate_neuron(int idx, double current);
    void stimulate_neurons(const std::vector<int>& indices, const std::vector<double>& currents);

private:
    void generate_3d_positions(const std::string& layout);
    void create_connections(double weight_scale, double weight_min,
                           double transmission_delay, double distance_lambda,
                           double lambda_decay_ie, double lambda_connect);
    void init_ring_buffer(double dt);
    void clear_ring_buffer();

    double jitter_gaussian(double base, double sigma);
    double jitter_lognormal(double base, double cv);
    std::vector<double> jitter_gaussian_vec(double base, double sigma, int n);
    std::vector<double> jitter_lognormal_vec(double base, double cv, int n);
};

} // namespace cls
