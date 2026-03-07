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
constexpr int OVERLAP_K = 5;

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
                              const std::string& connectivity_regime = "default");

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
    double stimulus_current = 0.88;
    double input_std_u = 0.0;        // STD on BSA->input injection (0=off)
    double input_std_tau_rec = 500.0; // recovery time constant for input STD
};

struct RunResult {
    std::vector<std::vector<int>> activity_record;
    std::vector<double> adapt_snapshot;
};

RunResult run_sample_with_std(SphericalNetwork& net, const AudioSample& sample,
                              const ZoneInfo& zone_info, const SimConfig& sim_cfg,
                              double std_u, double std_tau_rec,
                              const StdMasks& masks,
                              double record_adapt_at_ms = -1.0);

void build_full_network(SphericalNetwork& net, ZoneInfo& zone_info,
                        const NetworkConfig& cfg, double dt,
                        bool quiet = false,
                        const DynamicalOverrides* dyn_ovr = nullptr,
                        const std::string& connectivity_regime = "default",
                        bool remove_nonarc = false);

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
