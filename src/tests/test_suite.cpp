#include "catch2/catch_all.hpp"
#include "common.h"
#include "network.h"
#include "builder.h"
#include "ml.h"
#include "npz_reader.h"

#include <cmath>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>

// Test rng_seed and rng_uniform
TEST_CASE("Random number generation", "[rng]") {
    // Test seeding produces reproducible sequences
    cls::rng_seed(42);
    double r1 = cls::rng_uniform();
    cls::rng_seed(42);
    double r2 = cls::rng_uniform();
    REQUIRE(r1 == r2);

    // Test uniform distribution bounds
    cls::rng_seed(123);
    for (int i = 0; i < 100; i++) {
        double u = cls::rng_uniform();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
    }
}

// Test StandardScaler
TEST_CASE("StandardScaler", "[ml]") {
    cls::Mat X(3, 2);
    // Row 0: [1, 2]
    X(0, 0) = 1.0; X(0, 1) = 2.0;
    // Row 1: [3, 4]
    X(1, 0) = 3.0; X(1, 1) = 4.0;
    // Row 2: [5, 6]
    X(2, 0) = 5.0; X(2, 1) = 6.0;

    cls::StandardScaler scaler;
    auto X_scaled = scaler.fit_transform(X);

    // Check means are approximately zero
    for (int j = 0; j < 2; j++) {
        double mean = 0.0;
        for (int i = 0; i < 3; i++) mean += X_scaled(i, j);
        mean /= 3.0;
        REQUIRE(std::abs(mean) < 1e-10);
    }

    // Check standard deviations are approximately one
    for (int j = 0; j < 2; j++) {
        double mean = 0.0;
        for (int i = 0; i < 3; i++) mean += X_scaled(i, j);
        mean /= 3.0;

        double var = 0.0;
        for (int i = 0; i < 3; i++) {
            double d = X_scaled(i, j) - mean;
            var += d * d;
        }
        var /= 3.0;
        REQUIRE(std::abs(std::sqrt(var) - 1.0) < 1e-10);
    }
}

// Test accuracy_score
TEST_CASE("Accuracy score", "[ml]") {
    std::vector<int> y_true = {0, 1, 2, 2, 1};
    std::vector<int> y_pred = {0, 1, 2, 1, 1};

    double acc = cls::accuracy_score(y_true, y_pred);
    REQUIRE(acc == 0.8); // 4/5 correct
}

// Test confusion matrix
TEST_CASE("Confusion matrix", "[ml]") {
    std::vector<int> y_true = {0, 1, 2, 2, 1};
    std::vector<int> y_pred = {0, 1, 2, 1, 1};
    std::vector<int> labels = {0, 1, 2};

    auto cm = cls::confusion_matrix(y_true, y_pred, labels);

    // Expected:
    // [[1, 0, 0],
    //  [0, 2, 0],
    //  [0, 1, 1]]
    REQUIRE(cm[0][0] == 1);
    REQUIRE(cm[0][1] == 0);
    REQUIRE(cm[0][2] == 0);
    REQUIRE(cm[1][0] == 0);
    REQUIRE(cm[1][1] == 2);
    REQUIRE(cm[1][2] == 0);
    REQUIRE(cm[2][0] == 0);
    REQUIRE(cm[2][1] == 1);
    REQUIRE(cm[2][2] == 1);
}

// Test SphericalNetwork initialization
TEST_CASE("Network initialization", "[network]") {
    cls::NetworkConfig cfg;
    cfg.n_neurons = 100;
    cfg.inhibitory_fraction = 0.2;
    cfg.connection_probabilities = {{"ee", 0.1}, {"ei", 0.1}, {"ie", 0.1}, {"ii", 0.1}};
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

    cls::SphericalNetwork net;
    net.init(cfg);

    REQUIRE(net.n_neurons == 100);
    REQUIRE(net.nonzero_connections > 0);
    REQUIRE(net.sphere_radius > 0.0);
}

// Test network update step
TEST_CASE("Network update", "[network]") {
    cls::NetworkConfig cfg;
    cfg.n_neurons = 50;
    cfg.inhibitory_fraction = 0.2;
    cfg.connection_probabilities = {{"ee", 0.1}, {"ei", 0.1}, {"ie", 0.1}, {"ii", 0.1}};
    cfg.weight_scale = 0.55;
    cfg.layout = "sphere";

    cls::SphericalNetwork net;
    net.init(cfg);
    net.reset_all();

    double dt = 0.1;
    for (int step = 0; step < 100; step++) {
        auto active = net.update_network(dt);
        // Check that no neuron spikes immediately after reset
        REQUIRE((active.empty() || step > 0));
    }
}

// Test stimulus injection
TEST_CASE("Stimulus injection", "[network]") {
    cls::NetworkConfig cfg;
    cfg.n_neurons = 50;
    cfg.inhibitory_fraction = 0.0;
    cfg.connection_probabilities = {{"ee", 0.0}, {"ei", 0.0}, {"ie", 0.0}, {"ii", 0.0}};
    cfg.weight_scale = 0.0;
    cfg.layout = "sphere";

    cls::SphericalNetwork net;
    net.init(cfg);
    net.reset_all();

    // Stimulate a neuron with strong current
    net.stimulate_neuron(0, 10.0);

    // Check that the neuron's excitation conductance increased
    REQUIRE(net.g_e[0] > 0.0);

    // Simulate for a short time
    double dt = 0.1;
    for (int step = 0; step < 10; step++) {
        net.update_network(dt);
    }

    // Neuron should have fired if current is sufficient
    REQUIRE(net.network_activity[0] > 0);
}

