#pragma once
#include "network.h"
#include "npz_reader.h"
#include "common.h"
#include <map>
#include <set>

namespace cls {

// Base config
NetworkConfig make_base_config();

// Ring zone constants
constexpr double SURFACE_SHELL_FRACTION = 0.15;
constexpr double RING_LATITUDE_BAND = 0.40;
constexpr int ARC_DEGREES = 300;
constexpr int ARC_GAP_CENTER_DEG = 180;
constexpr int OVERLAP_K = 4;

// Gaussian tuning curve: sigma in units of channel spacing.
// sigma=1.5 means weight drops to ~0.80 at 1 channel, ~0.41 at 2, ~0.14 at 3.
constexpr double TUNING_SIGMA_CHANNELS = 1.5;

// Optimized input neuron regime (from 8,000-point grid search)
// These are applied automatically after network construction/loading.
constexpr double INPUT_STIM_CURRENT = 0.0518;  // BSA→input injection current (nA)
constexpr double INPUT_TAU_E = 1.05;            // excitatory synaptic time constant (ms)
constexpr double INPUT_ADAPT_INC = 0.0;         // spike-frequency adaptation increment
constexpr double INPUT_STD_U = 0.0;             // short-term depression on input (disabled)

// Config B constants
constexpr double WEIGHT_MULT = 5.0;
constexpr double ADAPT_INC_DEFAULT = 0.5;
constexpr double NMDA_TAU_DEFAULT = 50.0;
constexpr double STD_U = 0.1;
constexpr double STD_TAU_REC = 500.0;

struct ZoneInfo {
    std::vector<int> input_zone_indices;
    std::vector<int> reservoir_zone_indices;
    std::map<int, std::vector<int>> input_neuron_mapping;
    std::map<int, std::vector<double>> input_neuron_weights;  // Gaussian tuning weights, parallel to mapping
    std::vector<int> input_neuron_indices;
    double y_threshold;
    double sphere_radius;
    std::string connectivity_regime;
};

struct DynamicalOverrides {
    double shell_core_mult = 5.0;
    double core_core_mult = 1.0;
    double feedback_scale = 1.0;
    double adapt_inc = 0.5;
    double adapt_tau = -1.0;  // -1 = don't override
    double nmda_tau = 50.0;
};

// Network construction
void create_ring_zone_network(SphericalNetwork& net, ZoneInfo& zone_info,
                              const NetworkConfig& cfg, bool quiet = false,
                              const std::string& connectivity_regime = "default",
                              int override_seed = -1);

// Apply optimized input neuron regime (tau_e, adapt_inc, skip_stim_nmda).
// Called automatically at the end of apply_config_b_overrides and
// apply_dynamical_overrides, so input neurons always get these params.
void apply_input_neuron_regime(SphericalNetwork& net, const ZoneInfo& zone_info,
                                double dt);

void apply_config_b_overrides(SphericalNetwork& net, ZoneInfo& zone_info,
                              double dt);

void apply_dynamical_overrides(SphericalNetwork& net, ZoneInfo& zone_info,
                               double dt, const DynamicalOverrides& ovr);

void compact_network(SphericalNetwork& net, ZoneInfo& zone_info,
                     double dt, bool quiet = false);

struct StdMasks {
    std::set<int> std_eligible;
    std::map<int, std::vector<bool>> recurrent_masks;
};

StdMasks build_std_masks(const SphericalNetwork& net, const ZoneInfo& zone_info);

struct SimConfig {
    double dt = 0.1;
    double audio_duration_ms = 800.0;
    double post_stimulus_ms = 200.0;
    double stimulus_current = INPUT_STIM_CURRENT;
    double input_std_u = INPUT_STD_U;
    double input_std_tau_rec = 500.0; // recovery time constant for input STD
};

struct RunResult {
    std::vector<std::vector<int>> activity_record;
    std::vector<std::vector<double>> adapt_snapshots;
    std::vector<std::vector<double>> nmda_snapshots;         // g_nmda per neuron
    std::vector<std::vector<double>> std_resource_snapshots;  // x_resource per neuron (recurrent STD)
    std::vector<std::vector<double>> ge_snapshots;            // g_e (AMPA) per neuron
};

RunResult run_sample_with_std(SphericalNetwork& net, const AudioSample& sample,
                              const ZoneInfo& zone_info, const SimConfig& sim_cfg,
                              double std_u, double std_tau_rec,
                              const StdMasks& masks,
                              const std::vector<double>& record_adapt_at_ms = {});

void build_full_network(SphericalNetwork& net, ZoneInfo& zone_info,
                        const NetworkConfig& cfg, double dt,
                        bool quiet = false,
                        const DynamicalOverrides* dyn_ovr = nullptr,
                        const std::string& connectivity_regime = "default",
                        bool remove_nonarc = false,
                        int override_seed = -1);

// Load pre-built network from Python-exported .npz snapshot.
// This bypasses all RNG-dependent construction, giving bit-identical
// network topology regardless of the C++ random library implementation.
void load_network_snapshot(SphericalNetwork& net, ZoneInfo& zone_info,
                           const std::string& npz_path, double dt,
                           bool quiet = false);

// Print network fingerprint for cross-checking with Python export.
void print_network_fingerprint(const SphericalNetwork& net,
                                const ZoneInfo& zone_info);

} // namespace cls
