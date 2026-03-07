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

TEST_CASE("Matrix construction and indexing", "[common]") {
    cls::Mat m(3, 4, 1.0);
    REQUIRE(m.rows == 3);
    REQUIRE(m.cols == 4);
    REQUIRE(m.data.size() == 12);
    
    // Check initialization value
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            REQUIRE(m(i, j) == 1.0);
        }
    }
    
    // Test assignment
    m(1, 2) = 5.5;
    REQUIRE(m(1, 2) == 5.5);
    
    // Test const access
    const cls::Mat cm = m;
    REQUIRE(cm(1, 2) == 5.5);
}

// Test Mat resize and fill
TEST_CASE("Matrix resize and fill", "[common]") {
    cls::Mat m(2, 3, 0.0);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    
    m.resize(3, 2, -1.0);
    REQUIRE(m.rows == 3);
    REQUIRE(m.cols == 2);
    REQUIRE(m.data.size() == 6);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            REQUIRE(m(i, j) == -1.0);
        }
    }
    
    m.fill(7.0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            REQUIRE(m(i, j) == 7.0);
        }
    }
}

// Test argsort functionality
TEST_CASE("Argsort functionality", "[common]") {
    std::vector<double> v = {3.0, 1.0, 4.0, 1.5, 2.0};
    auto idx = cls::argsort(v);
    
    // Expected indices: [1, 3, 4, 0, 2] (values: 1.0, 1.5, 2.0, 3.0, 4.0)
    REQUIRE(idx.size() == 5);
    REQUIRE(idx[0] == 1);  // smallest value at index 1
    REQUIRE(idx[1] == 3);  // next smallest at index 3
    REQUIRE(idx[2] == 4);
    REQUIRE(idx[3] == 0);
    REQUIRE(idx[4] == 2);  // largest value at index 2
    
    // Verify indices produce sorted values
    std::vector<double> sorted;
    for (int i : idx) sorted.push_back(v[i]);
    REQUIRE(std::is_sorted(sorted.begin(), sorted.end()));
}

// Test clip function
TEST_CASE("Clip function", "[common]") {
    REQUIRE(cls::clip(0.5, 0.0, 1.0) == 0.5);
    REQUIRE(cls::clip(-1.0, 0.0, 1.0) == 0.0);
    REQUIRE(cls::clip(2.0, 0.0, 1.0) == 1.0);
    REQUIRE(cls::clip(0.0, 0.0, 1.0) == 0.0);
    REQUIRE(cls::clip(1.0, 0.0, 1.0) == 1.0);
}

// Test clip_vec function
TEST_CASE("Clip vector function", "[common]") {
    std::vector<double> v = {-1.0, 0.5, 2.0, 0.0, 1.0};
    cls::clip_vec(v, 0.0, 1.0);
    
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 0.0);
    REQUIRE(v[1] == 0.5);
    REQUIRE(v[2] == 1.0);
    REQUIRE(v[3] == 0.0);
    REQUIRE(v[4] == 1.0);
}

// Test SVD decomposition
TEST_CASE("SVD decomposition", "[common]") {
    // Simple 2x2 matrix: [[1, 2], [3, 4]]
    cls::Mat A(2, 2);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;
    
    std::vector<double> S;
    cls::Mat U, Vt;
    cls::svd_econ(A, S, U, Vt);
    
    REQUIRE(S.size() == 2);
    REQUIRE(S[0] > 0); // Largest singular value
    REQUIRE(S[1] > 0); // Smaller singular value
    
    // Verify dimensions
    REQUIRE((U.rows == 2 && U.cols == 2));
    REQUIRE((Vt.rows == 2 && Vt.cols == 2));
    
    // Check that U and Vt are orthogonal (U^T * U ≈ I)
    cls::Mat UTU(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double sum = 0.0;
            for (int k = 0; k < 2; k++) {
                sum += U(k, i) * U(k, j);
            }
            UTU(i, j) = sum;
        }
    }
    
    // Diagonal should be ~1, off-diagonal ~0
    REQUIRE(std::abs(UTU(0, 0) - 1.0) < 0.001);
    REQUIRE(std::abs(UTU(1, 1) - 1.0) < 0.001);
    REQUIRE(std::abs(UTU(0, 1)) < 0.001);
    REQUIRE(std::abs(UTU(1, 0)) < 0.001);
    
    // Reconstruct original matrix: A ≈ U * diag(S) * Vt
    cls::Mat A_recon(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double sum = 0.0;
            for (int k = 0; k < 2; k++) {
                sum += U(i, k) * S[k] * Vt(k, j);
            }
            A_recon(i, j) = sum;
        }
    }
    
    // Check reconstruction error
    double max_error = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            max_error = std::max(max_error, std::abs(A(i, j) - A_recon(i, j)));
        }
    }
    REQUIRE(max_error < 0.001);
}

// Test rng_normal and rng_lognormal
TEST_CASE("Normal and lognormal distributions", "[rng]") {
    cls::rng_seed(42);
    std::vector<double> samples;
    for (int i = 0; i < 1000; i++) {
        samples.push_back(cls::rng_normal(0.0, 1.0));
    }
    
    // Check mean and variance approximately match
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double var = 0.0;
    for (double x : samples) var += (x - mean) * (x - mean);
    var /= samples.size();
    
    REQUIRE(std::abs(mean) < 0.1); // Mean should be close to 0
    REQUIRE(std::abs(var - 1.0) < 0.2); // Variance should be close to 1
    
    // Test lognormal
    cls::rng_seed(42);
    samples.clear();
    for (int i = 0; i < 1000; i++) {
        samples.push_back(cls::rng_lognormal(0.0, 1.0));
    }
    
    // For lognormal with mu=0, sigma=1, mean = exp(0.5) ≈ 1.65
    double expected_mean = std::exp(0.5);
    mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    REQUIRE(std::abs(mean - expected_mean) < 0.2);
}

// Test rng_choice
TEST_CASE("RNG choice", "[rng]") {
    // Test with replacement
    cls::rng_seed(42);
    auto choice_replace = cls::rng_choice(10, 5, true);
    REQUIRE(choice_replace.size() == 5);
    for (int c : choice_replace) {
        REQUIRE((c >= 0 && c < 10));
    }
    
    // Test without replacement
    cls::rng_seed(42);
    auto choice_no_replace = cls::rng_choice(10, 5, false);
    REQUIRE(choice_no_replace.size() == 5);
    
    // Check uniqueness
    std::set<int> s(choice_no_replace.begin(), choice_no_replace.end());
    REQUIRE(s.size() == 5);
    
    // Check all values in range
    for (int c : choice_no_replace) {
        REQUIRE((c >= 0 && c < 10));
    }
}

// Test rng_uniform with custom bounds
TEST_CASE("RNG uniform with bounds", "[rng]") {
    cls::rng_seed(42);
    for (int i = 0; i < 100; i++) {
        double u = cls::rng_uniform(-5.0, 3.0);
        REQUIRE(u >= -5.0);
        REQUIRE(u < 3.0);
    }
}

// Test rng_normal_vec
TEST_CASE("RNG normal vector", "[rng]") {
    cls::rng_seed(42);
    auto v = cls::rng_normal_vec(5, 2.0, 0.5);
    
    REQUIRE(v.size() == 5);
    for (double x : v) {
        REQUIRE(!std::isnan(x));
        REQUIRE(!std::isinf(x));
    }
}

// Test argsort with integer vector
TEST_CASE("Argsort with integers", "[common]") {
    std::vector<int> v = {5, 2, 8, 1, 3};
    auto idx = cls::argsort(v);
    
    // Expected indices: [3, 1, 4, 0, 2] (values: 1, 2, 3, 5, 8)
    REQUIRE(idx[0] == 3);
    REQUIRE(idx[1] == 1);
    REQUIRE(idx[2] == 4);
    REQUIRE(idx[3] == 0);
    REQUIRE(idx[4] == 2);
}

// Test argsort with pointer interface
TEST_CASE("Argsort with pointer", "[common]") {
    double arr[] = {3.5, 1.2, 4.8, 2.1};
    auto idx = cls::argsort(arr, 4);
    
    // Expected indices: [1, 3, 0, 2] (values: 1.2, 2.1, 3.5, 4.8)
    REQUIRE(idx[0] == 1);
    REQUIRE(idx[1] == 3);
    REQUIRE(idx[2] == 0);
    REQUIRE(idx[3] == 2);
}
