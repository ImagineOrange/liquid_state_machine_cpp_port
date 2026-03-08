/*
 * SoA vs AoS benchmark for the neuron update hot loop.
 *
 * Replicates the exact computation from update_network() lines 409-457:
 *   - Refractory check
 *   - Membrane dynamics (leak, 4 synaptic currents, adaptation, tonic, noise)
 *   - Conductance decay (5 channels)
 *   - Synaptic noise injection
 *   - Spike detection
 *
 * Both layouts use identical data and produce identical results.
 * Measures wall-clock time over many iterations at realistic neuron counts.
 *
 * Build:
 *   c++ -O3 -std=c++17 -o bench_soa_vs_aos experiments/bench_soa_vs_aos.cpp -lm
 *
 * Run:
 *   ./bench_soa_vs_aos [n_neurons] [n_steps] [n_trials]
 *   Default: 604 neurons, 10000 steps, 5 trials
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// RNG (thread-local for reproducibility)
// ---------------------------------------------------------------------------
static thread_local std::mt19937_64 g_rng(42);
static thread_local std::normal_distribution<double> g_norm(0.0, 1.0);

static inline double rng_normal() { return g_norm(g_rng); }
static inline double rng_uniform(double lo = 0.0, double hi = 1.0) {
    return std::uniform_real_distribution<double>(lo, hi)(g_rng);
}

// ---------------------------------------------------------------------------
// SoA layout — current production code structure
// ---------------------------------------------------------------------------
struct NeuronsSoA {
    int n;

    // Dynamic state (read+write every step)
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;

    // Parameters (read-only in hot loop)
    std::vector<double> v_rest, v_threshold, v_reset;
    std::vector<double> tau_m, tau_ref;
    std::vector<double> e_reversal, i_reversal, k_reversal;
    std::vector<double> v_noise_amp, i_noise_amp;
    std::vector<double> adaptation_increment;

    // Precomputed decay factors (read-only)
    std::vector<double> exp_decay_e, exp_decay_i, exp_decay_i_slow;
    std::vector<double> exp_decay_nmda, exp_decay_adapt;

    // Tonic conductance
    std::vector<double> tonic_conductance, tonic_reversal;

    double mg_factor;
    double nmda_ratio;

    void init(int n_neurons);
    int update(double dt);
};

void NeuronsSoA::init(int n_neurons) {
    n = n_neurons;
    mg_factor = 1.0 / 3.57;
    nmda_ratio = 0.5;

    auto fill_jitter = [&](int sz, double base, double sigma) {
        std::vector<double> out(sz);
        for (int i = 0; i < sz; i++) out[i] = base + rng_normal() * sigma;
        return out;
    };

    v_rest = fill_jitter(n, -65.0, 3.0);
    v_threshold = fill_jitter(n, -55.0, 2.5);
    v_reset.assign(n, -75.0);
    tau_m = fill_jitter(n, 10.0, 1.0);
    tau_ref.assign(n, 4.0);
    e_reversal.assign(n, 0.0);
    i_reversal.assign(n, -80.0);
    k_reversal.assign(n, -90.0);
    v_noise_amp.assign(n, 0.1);
    i_noise_amp.assign(n, 0.001);
    adaptation_increment = fill_jitter(n, 0.2, 0.05);

    double dt = 0.1;
    exp_decay_e.resize(n);
    exp_decay_i.resize(n);
    exp_decay_i_slow.resize(n);
    exp_decay_nmda.resize(n);
    exp_decay_adapt.resize(n);
    for (int i = 0; i < n; i++) {
        exp_decay_e[i]      = std::exp(-dt / 3.0);
        exp_decay_i[i]      = std::exp(-dt / 7.0);
        exp_decay_i_slow[i] = std::exp(-dt / 50.0);
        exp_decay_nmda[i]   = std::exp(-dt / 100.0);
        exp_decay_adapt[i]  = std::exp(-dt / 100.0);
    }

    tonic_conductance.assign(n, 0.001);
    tonic_reversal.assign(n, -80.0);

    // Dynamic state
    v = v_rest;
    g_e.assign(n, 0.01);
    g_i.assign(n, 0.005);
    g_i_slow.assign(n, 0.002);
    g_nmda.assign(n, 0.005);
    adaptation.assign(n, 0.01);
    t_since_spike.assign(n, 10.0);

    for (int i = 0; i < n; i++) {
        tau_m[i] = std::max(tau_m[i], 3.0);
        adaptation_increment[i] = std::max(adaptation_increment[i], 0.0);
    }
}

int NeuronsSoA::update(double dt) {
    const double* __restrict__ pvrest    = v_rest.data();
    const double* __restrict__ pvthr    = v_threshold.data();
    const double* __restrict__ pvrst    = v_reset.data();
    const double* __restrict__ ptaum    = tau_m.data();
    const double* __restrict__ ptref    = tau_ref.data();
    const double* __restrict__ perev    = e_reversal.data();
    const double* __restrict__ pirev    = i_reversal.data();
    const double* __restrict__ pkrev    = k_reversal.data();
    const double* __restrict__ pvnoise  = v_noise_amp.data();
    const double* __restrict__ pinoise  = i_noise_amp.data();
    const double* __restrict__ padaptinc = adaptation_increment.data();
    const double* __restrict__ pdece    = exp_decay_e.data();
    const double* __restrict__ pdeci    = exp_decay_i.data();
    const double* __restrict__ pdecis   = exp_decay_i_slow.data();
    const double* __restrict__ pdecnmda = exp_decay_nmda.data();
    const double* __restrict__ pdecadapt = exp_decay_adapt.data();
    const double* __restrict__ ptonic   = tonic_conductance.data();
    const double* __restrict__ ptonic_rev = tonic_reversal.data();

    double* __restrict__ pv     = v.data();
    double* __restrict__ pge    = g_e.data();
    double* __restrict__ pgi    = g_i.data();
    double* __restrict__ pgis   = g_i_slow.data();
    double* __restrict__ pgnmda = g_nmda.data();
    double* __restrict__ padapt = adaptation.data();
    double* __restrict__ ptss   = t_since_spike.data();

    int n_spikes = 0;

    for (int i = 0; i < n; i++) {
        bool ref = ptss[i] < ptref[i];
        ptss[i] += dt;

        double vi = pv[i];
        double i_e    = pge[i] * (perev[i] - vi);
        double i_i    = pgi[i] * (pirev[i] - vi);
        double i_is   = pgis[i] * (pirev[i] - vi);
        double mg_block = 1.0 / (1.0 + mg_factor * std::exp(-0.062 * vi));
        double i_nmda = pgnmda[i] * mg_block * (perev[i] - vi);
        double i_adapt = padapt[i] * (pkrev[i] - vi);
        double i_tonic = ptonic[i] * (ptonic_rev[i] - vi);

        double dv = dt * ((-(vi - pvrest[i]) / ptaum[i])
                          + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
        vi += dv;
        vi += rng_normal() * pvnoise[i];

        if (ref) vi = pvrst[i];
        pv[i] = vi;

        pge[i]    *= pdece[i];
        pgi[i]    *= pdeci[i];
        pgis[i]   *= pdecis[i];
        pgnmda[i] *= pdecnmda[i];
        padapt[i] *= pdecadapt[i];

        double ne = rng_normal() * pinoise[i];
        double ni = rng_normal() * pinoise[i];
        if (ne > 0) pge[i] += ne;
        if (ni > 0) pgi[i] += ni;

        if (vi >= pvthr[i] && !ref) {
            pv[i] = pvrst[i];
            ptss[i] = 0.0;
            padapt[i] += padaptinc[i];
            n_spikes++;
        }
    }
    return n_spikes;
}


// ---------------------------------------------------------------------------
// AoS layout — all hot-path data packed per neuron
// ---------------------------------------------------------------------------

// Pack dynamic state + read-only params into a single struct.
// Target: everything the hot loop touches fits in 2-3 cache lines per neuron.
struct alignas(64) NeuronAoS {
    // Dynamic state (read+write) — 56 bytes
    double v;
    double g_e;
    double g_i;
    double g_i_slow;
    double g_nmda;
    double adaptation;
    double t_since_spike;

    // Parameters (read-only in hot loop) — 88 bytes
    double v_rest;
    double v_threshold;
    double v_reset;
    double tau_m;
    double tau_ref;
    double e_reversal;
    double i_reversal;
    double k_reversal;
    double v_noise_amp;
    double i_noise_amp;
    double adaptation_increment;

    // Decay factors (read-only) — 40 bytes
    double exp_decay_e;
    double exp_decay_i;
    double exp_decay_i_slow;
    double exp_decay_nmda;
    double exp_decay_adapt;

    // Tonic — 16 bytes
    double tonic_conductance;
    double tonic_reversal;
    // Total: 200 bytes = 3.125 cache lines (64-byte aligned)
};

struct NeuronsAoS {
    int n;
    std::vector<NeuronAoS> neurons;
    double mg_factor;
    double nmda_ratio;

    void init(int n_neurons);
    void init_from_soa(const NeuronsSoA& soa);
    int update(double dt);
};

void NeuronsAoS::init(int n_neurons) {
    // Initialize via SoA then convert (ensures identical data)
}

void NeuronsAoS::init_from_soa(const NeuronsSoA& soa) {
    n = soa.n;
    mg_factor = soa.mg_factor;
    nmda_ratio = soa.nmda_ratio;
    neurons.resize(n);

    for (int i = 0; i < n; i++) {
        auto& ne = neurons[i];
        ne.v              = soa.v[i];
        ne.g_e            = soa.g_e[i];
        ne.g_i            = soa.g_i[i];
        ne.g_i_slow       = soa.g_i_slow[i];
        ne.g_nmda         = soa.g_nmda[i];
        ne.adaptation     = soa.adaptation[i];
        ne.t_since_spike  = soa.t_since_spike[i];

        ne.v_rest         = soa.v_rest[i];
        ne.v_threshold    = soa.v_threshold[i];
        ne.v_reset        = soa.v_reset[i];
        ne.tau_m          = soa.tau_m[i];
        ne.tau_ref        = soa.tau_ref[i];
        ne.e_reversal     = soa.e_reversal[i];
        ne.i_reversal     = soa.i_reversal[i];
        ne.k_reversal     = soa.k_reversal[i];
        ne.v_noise_amp    = soa.v_noise_amp[i];
        ne.i_noise_amp    = soa.i_noise_amp[i];
        ne.adaptation_increment = soa.adaptation_increment[i];

        ne.exp_decay_e      = soa.exp_decay_e[i];
        ne.exp_decay_i      = soa.exp_decay_i[i];
        ne.exp_decay_i_slow = soa.exp_decay_i_slow[i];
        ne.exp_decay_nmda   = soa.exp_decay_nmda[i];
        ne.exp_decay_adapt  = soa.exp_decay_adapt[i];

        ne.tonic_conductance = soa.tonic_conductance[i];
        ne.tonic_reversal    = soa.tonic_reversal[i];
    }
}

int NeuronsAoS::update(double dt) {
    NeuronAoS* __restrict__ data = neurons.data();
    int n_spikes = 0;

    for (int i = 0; i < n; i++) {
        NeuronAoS& ne = data[i];

        bool ref = ne.t_since_spike < ne.tau_ref;
        ne.t_since_spike += dt;

        double vi = ne.v;
        double i_e    = ne.g_e * (ne.e_reversal - vi);
        double i_i    = ne.g_i * (ne.i_reversal - vi);
        double i_is   = ne.g_i_slow * (ne.i_reversal - vi);
        double mg_block = 1.0 / (1.0 + mg_factor * std::exp(-0.062 * vi));
        double i_nmda = ne.g_nmda * mg_block * (ne.e_reversal - vi);
        double i_adapt = ne.adaptation * (ne.k_reversal - vi);
        double i_tonic = ne.tonic_conductance * (ne.tonic_reversal - vi);

        double dv = dt * ((-(vi - ne.v_rest) / ne.tau_m)
                          + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
        vi += dv;
        vi += rng_normal() * ne.v_noise_amp;

        if (ref) vi = ne.v_reset;
        ne.v = vi;

        ne.g_e      *= ne.exp_decay_e;
        ne.g_i      *= ne.exp_decay_i;
        ne.g_i_slow *= ne.exp_decay_i_slow;
        ne.g_nmda   *= ne.exp_decay_nmda;
        ne.adaptation *= ne.exp_decay_adapt;

        double noise_e = rng_normal() * ne.i_noise_amp;
        double noise_i = rng_normal() * ne.i_noise_amp;
        if (noise_e > 0) ne.g_e += noise_e;
        if (noise_i > 0) ne.g_i += noise_i;

        if (vi >= ne.v_threshold && !ref) {
            ne.v = ne.v_reset;
            ne.t_since_spike = 0.0;
            ne.adaptation += ne.adaptation_increment;
            n_spikes++;
        }
    }
    return n_spikes;
}


// ---------------------------------------------------------------------------
// Hybrid SoA: split into hot (written every step) and cold (read-only params)
// ---------------------------------------------------------------------------
struct NeuronsSoAHot {
    // Dynamic state — 7 arrays, each contiguous
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;
};

struct alignas(64) NeuronCold {
    double v_rest, v_threshold, v_reset;
    double tau_m, tau_ref;
    double e_reversal, i_reversal, k_reversal;
    double v_noise_amp, i_noise_amp;
    double adaptation_increment;
    double exp_decay_e, exp_decay_i, exp_decay_i_slow, exp_decay_nmda, exp_decay_adapt;
    double tonic_conductance, tonic_reversal;
    // 144 bytes = 2.25 cache lines
};

struct NeuronsHybrid {
    int n;
    NeuronsSoAHot hot;
    std::vector<NeuronCold> cold;
    double mg_factor;
    double nmda_ratio;

    void init_from_soa(const NeuronsSoA& soa);
    int update(double dt);
};

void NeuronsHybrid::init_from_soa(const NeuronsSoA& soa) {
    n = soa.n;
    mg_factor = soa.mg_factor;
    nmda_ratio = soa.nmda_ratio;

    hot.v = soa.v;
    hot.g_e = soa.g_e;
    hot.g_i = soa.g_i;
    hot.g_i_slow = soa.g_i_slow;
    hot.g_nmda = soa.g_nmda;
    hot.adaptation = soa.adaptation;
    hot.t_since_spike = soa.t_since_spike;

    cold.resize(n);
    for (int i = 0; i < n; i++) {
        auto& c = cold[i];
        c.v_rest         = soa.v_rest[i];
        c.v_threshold    = soa.v_threshold[i];
        c.v_reset        = soa.v_reset[i];
        c.tau_m          = soa.tau_m[i];
        c.tau_ref        = soa.tau_ref[i];
        c.e_reversal     = soa.e_reversal[i];
        c.i_reversal     = soa.i_reversal[i];
        c.k_reversal     = soa.k_reversal[i];
        c.v_noise_amp    = soa.v_noise_amp[i];
        c.i_noise_amp    = soa.i_noise_amp[i];
        c.adaptation_increment = soa.adaptation_increment[i];
        c.exp_decay_e      = soa.exp_decay_e[i];
        c.exp_decay_i      = soa.exp_decay_i[i];
        c.exp_decay_i_slow = soa.exp_decay_i_slow[i];
        c.exp_decay_nmda   = soa.exp_decay_nmda[i];
        c.exp_decay_adapt  = soa.exp_decay_adapt[i];
        c.tonic_conductance = soa.tonic_conductance[i];
        c.tonic_reversal    = soa.tonic_reversal[i];
    }
}

int NeuronsHybrid::update(double dt) {
    double* __restrict__ pv     = hot.v.data();
    double* __restrict__ pge    = hot.g_e.data();
    double* __restrict__ pgi    = hot.g_i.data();
    double* __restrict__ pgis   = hot.g_i_slow.data();
    double* __restrict__ pgnmda = hot.g_nmda.data();
    double* __restrict__ padapt = hot.adaptation.data();
    double* __restrict__ ptss   = hot.t_since_spike.data();
    const NeuronCold* __restrict__ pc = cold.data();

    int n_spikes = 0;

    for (int i = 0; i < n; i++) {
        const NeuronCold& c = pc[i];

        bool ref = ptss[i] < c.tau_ref;
        ptss[i] += dt;

        double vi = pv[i];
        double i_e    = pge[i] * (c.e_reversal - vi);
        double i_i    = pgi[i] * (c.i_reversal - vi);
        double i_is   = pgis[i] * (c.i_reversal - vi);
        double mg_block = 1.0 / (1.0 + mg_factor * std::exp(-0.062 * vi));
        double i_nmda = pgnmda[i] * mg_block * (c.e_reversal - vi);
        double i_adapt = padapt[i] * (c.k_reversal - vi);
        double i_tonic = c.tonic_conductance * (c.tonic_reversal - vi);

        double dv = dt * ((-(vi - c.v_rest) / c.tau_m)
                          + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
        vi += dv;
        vi += rng_normal() * c.v_noise_amp;

        if (ref) vi = c.v_reset;
        pv[i] = vi;

        pge[i]    *= c.exp_decay_e;
        pgi[i]    *= c.exp_decay_i;
        pgis[i]   *= c.exp_decay_i_slow;
        pgnmda[i] *= c.exp_decay_nmda;
        padapt[i] *= c.exp_decay_adapt;

        double ne = rng_normal() * c.i_noise_amp;
        double ni = rng_normal() * c.i_noise_amp;
        if (ne > 0) pge[i] += ne;
        if (ni > 0) pgi[i] += ni;

        if (vi >= c.v_threshold && !ref) {
            pv[i] = c.v_reset;
            ptss[i] = 0.0;
            padapt[i] += c.adaptation_increment;
            n_spikes++;
        }
    }
    return n_spikes;
}


// ---------------------------------------------------------------------------
// SoA with pre-buffered noise: generate all 3*n normals before the loop
// ---------------------------------------------------------------------------
struct NeuronsSoABuffered {
    int n;
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;
    std::vector<double> v_rest, v_threshold, v_reset;
    std::vector<double> tau_m, tau_ref;
    std::vector<double> e_reversal, i_reversal, k_reversal;
    std::vector<double> v_noise_amp, i_noise_amp;
    std::vector<double> adaptation_increment;
    std::vector<double> exp_decay_e, exp_decay_i, exp_decay_i_slow;
    std::vector<double> exp_decay_nmda, exp_decay_adapt;
    std::vector<double> tonic_conductance, tonic_reversal;
    double mg_factor, nmda_ratio;

    // Pre-allocated noise buffer: 3*n doubles (~14 KB for 604 neurons)
    std::vector<double> noise_buf;

    void init_from_soa(const NeuronsSoA& soa) {
        n = soa.n;
        mg_factor = soa.mg_factor;
        nmda_ratio = soa.nmda_ratio;
        v = soa.v; g_e = soa.g_e; g_i = soa.g_i; g_i_slow = soa.g_i_slow;
        g_nmda = soa.g_nmda; adaptation = soa.adaptation; t_since_spike = soa.t_since_spike;
        v_rest = soa.v_rest; v_threshold = soa.v_threshold; v_reset = soa.v_reset;
        tau_m = soa.tau_m; tau_ref = soa.tau_ref;
        e_reversal = soa.e_reversal; i_reversal = soa.i_reversal; k_reversal = soa.k_reversal;
        v_noise_amp = soa.v_noise_amp; i_noise_amp = soa.i_noise_amp;
        adaptation_increment = soa.adaptation_increment;
        exp_decay_e = soa.exp_decay_e; exp_decay_i = soa.exp_decay_i;
        exp_decay_i_slow = soa.exp_decay_i_slow;
        exp_decay_nmda = soa.exp_decay_nmda; exp_decay_adapt = soa.exp_decay_adapt;
        tonic_conductance = soa.tonic_conductance; tonic_reversal = soa.tonic_reversal;
        noise_buf.resize(3 * n);
    }

    int update(double dt) {
        // Phase 1: Fill noise buffer (tight loop, no dependencies)
        double* __restrict__ nb = noise_buf.data();
        for (int i = 0; i < 3 * n; i++) {
            nb[i] = rng_normal();
        }

        // Phase 2: Neuron update with pre-buffered noise (pure arithmetic)
        const double* __restrict__ pvrest    = v_rest.data();
        const double* __restrict__ pvthr    = v_threshold.data();
        const double* __restrict__ pvrst    = v_reset.data();
        const double* __restrict__ ptaum    = tau_m.data();
        const double* __restrict__ ptref    = tau_ref.data();
        const double* __restrict__ perev    = e_reversal.data();
        const double* __restrict__ pirev    = i_reversal.data();
        const double* __restrict__ pkrev    = k_reversal.data();
        const double* __restrict__ pvnoise  = v_noise_amp.data();
        const double* __restrict__ pinoise  = i_noise_amp.data();
        const double* __restrict__ padaptinc = adaptation_increment.data();
        const double* __restrict__ pdece    = exp_decay_e.data();
        const double* __restrict__ pdeci    = exp_decay_i.data();
        const double* __restrict__ pdecis   = exp_decay_i_slow.data();
        const double* __restrict__ pdecnmda = exp_decay_nmda.data();
        const double* __restrict__ pdecadapt = exp_decay_adapt.data();
        const double* __restrict__ ptonic   = tonic_conductance.data();
        const double* __restrict__ ptonic_rev = tonic_reversal.data();

        double* __restrict__ pv     = v.data();
        double* __restrict__ pge    = g_e.data();
        double* __restrict__ pgi    = g_i.data();
        double* __restrict__ pgis   = g_i_slow.data();
        double* __restrict__ pgnmda = g_nmda.data();
        double* __restrict__ padapt = adaptation.data();
        double* __restrict__ ptss   = t_since_spike.data();

        int n_spikes = 0;

        for (int i = 0; i < n; i++) {
            bool ref = ptss[i] < ptref[i];
            ptss[i] += dt;

            double vi = pv[i];
            double i_e    = pge[i] * (perev[i] - vi);
            double i_i    = pgi[i] * (pirev[i] - vi);
            double i_is   = pgis[i] * (pirev[i] - vi);
            double mg_block = 1.0 / (1.0 + mg_factor * std::exp(-0.062 * vi));
            double i_nmda = pgnmda[i] * mg_block * (perev[i] - vi);
            double i_adapt = padapt[i] * (pkrev[i] - vi);
            double i_tonic = ptonic[i] * (ptonic_rev[i] - vi);

            double dv = dt * ((-(vi - pvrest[i]) / ptaum[i])
                              + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
            vi += dv;
            vi += nb[i * 3 + 0] * pvnoise[i];

            if (ref) vi = pvrst[i];
            pv[i] = vi;

            pge[i]    *= pdece[i];
            pgi[i]    *= pdeci[i];
            pgis[i]   *= pdecis[i];
            pgnmda[i] *= pdecnmda[i];
            padapt[i] *= pdecadapt[i];

            double ne = nb[i * 3 + 1] * pinoise[i];
            double ni = nb[i * 3 + 2] * pinoise[i];
            if (ne > 0) pge[i] += ne;
            if (ni > 0) pgi[i] += ni;

            if (vi >= pvthr[i] && !ref) {
                pv[i] = pvrst[i];
                ptss[i] = 0.0;
                padapt[i] += padaptinc[i];
                n_spikes++;
            }
        }
        return n_spikes;
    }
};


// ---------------------------------------------------------------------------
// SoA with multi-step noise pre-generation: generate 3*n*K normals for K
// steps at once. The neuron loop is pure arithmetic for K steps, then a
// bulk RNG refill. Amortizes RNG overhead further vs single-step buffering.
// ---------------------------------------------------------------------------
struct NeuronsSoAMultiStep {
    int n;
    int K;  // batch size (steps per refill)
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;
    std::vector<double> v_rest, v_threshold, v_reset;
    std::vector<double> tau_m, tau_ref;
    std::vector<double> e_reversal, i_reversal, k_reversal;
    std::vector<double> v_noise_amp, i_noise_amp;
    std::vector<double> adaptation_increment;
    std::vector<double> exp_decay_e, exp_decay_i, exp_decay_i_slow;
    std::vector<double> exp_decay_nmda, exp_decay_adapt;
    std::vector<double> tonic_conductance, tonic_reversal;
    double mg_factor, nmda_ratio;

    // Pre-allocated noise buffer: 3*n*K doubles
    std::vector<double> noise_buf;
    int noise_cursor;  // which step within current batch (0..K-1)

    void init_from_soa(const NeuronsSoA& soa, int batch_k) {
        n = soa.n;
        K = batch_k;
        mg_factor = soa.mg_factor;
        nmda_ratio = soa.nmda_ratio;
        v = soa.v; g_e = soa.g_e; g_i = soa.g_i; g_i_slow = soa.g_i_slow;
        g_nmda = soa.g_nmda; adaptation = soa.adaptation; t_since_spike = soa.t_since_spike;
        v_rest = soa.v_rest; v_threshold = soa.v_threshold; v_reset = soa.v_reset;
        tau_m = soa.tau_m; tau_ref = soa.tau_ref;
        e_reversal = soa.e_reversal; i_reversal = soa.i_reversal; k_reversal = soa.k_reversal;
        v_noise_amp = soa.v_noise_amp; i_noise_amp = soa.i_noise_amp;
        adaptation_increment = soa.adaptation_increment;
        exp_decay_e = soa.exp_decay_e; exp_decay_i = soa.exp_decay_i;
        exp_decay_i_slow = soa.exp_decay_i_slow;
        exp_decay_nmda = soa.exp_decay_nmda; exp_decay_adapt = soa.exp_decay_adapt;
        tonic_conductance = soa.tonic_conductance; tonic_reversal = soa.tonic_reversal;
        noise_buf.resize(3 * n * K);
        noise_cursor = K;  // force refill on first call
    }

    void refill_noise() {
        double* __restrict__ nb = noise_buf.data();
        int total = 3 * n * K;
        for (int i = 0; i < total; i++) {
            nb[i] = rng_normal();
        }
        noise_cursor = 0;
    }

    int update(double dt) {
        // Refill when batch exhausted
        if (noise_cursor >= K) {
            refill_noise();
        }

        // Offset into noise buffer for this step
        const double* __restrict__ nb = noise_buf.data() + noise_cursor * 3 * n;
        noise_cursor++;

        const double* __restrict__ pvrest    = v_rest.data();
        const double* __restrict__ pvthr    = v_threshold.data();
        const double* __restrict__ pvrst    = v_reset.data();
        const double* __restrict__ ptaum    = tau_m.data();
        const double* __restrict__ ptref    = tau_ref.data();
        const double* __restrict__ perev    = e_reversal.data();
        const double* __restrict__ pirev    = i_reversal.data();
        const double* __restrict__ pkrev    = k_reversal.data();
        const double* __restrict__ pvnoise  = v_noise_amp.data();
        const double* __restrict__ pinoise  = i_noise_amp.data();
        const double* __restrict__ padaptinc = adaptation_increment.data();
        const double* __restrict__ pdece    = exp_decay_e.data();
        const double* __restrict__ pdeci    = exp_decay_i.data();
        const double* __restrict__ pdecis   = exp_decay_i_slow.data();
        const double* __restrict__ pdecnmda = exp_decay_nmda.data();
        const double* __restrict__ pdecadapt = exp_decay_adapt.data();
        const double* __restrict__ ptonic   = tonic_conductance.data();
        const double* __restrict__ ptonic_rev = tonic_reversal.data();

        double* __restrict__ pv     = v.data();
        double* __restrict__ pge    = g_e.data();
        double* __restrict__ pgi    = g_i.data();
        double* __restrict__ pgis   = g_i_slow.data();
        double* __restrict__ pgnmda = g_nmda.data();
        double* __restrict__ padapt = adaptation.data();
        double* __restrict__ ptss   = t_since_spike.data();

        int n_spikes = 0;

        for (int i = 0; i < n; i++) {
            bool ref = ptss[i] < ptref[i];
            ptss[i] += dt;

            double vi = pv[i];
            double i_e    = pge[i] * (perev[i] - vi);
            double i_i    = pgi[i] * (pirev[i] - vi);
            double i_is   = pgis[i] * (pirev[i] - vi);
            double mg_block = 1.0 / (1.0 + mg_factor * std::exp(-0.062 * vi));
            double i_nmda = pgnmda[i] * mg_block * (perev[i] - vi);
            double i_adapt = padapt[i] * (pkrev[i] - vi);
            double i_tonic = ptonic[i] * (ptonic_rev[i] - vi);

            double dv = dt * ((-(vi - pvrest[i]) / ptaum[i])
                              + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
            vi += dv;
            vi += nb[i * 3 + 0] * pvnoise[i];

            if (ref) vi = pvrst[i];
            pv[i] = vi;

            pge[i]    *= pdece[i];
            pgi[i]    *= pdeci[i];
            pgis[i]   *= pdecis[i];
            pgnmda[i] *= pdecnmda[i];
            padapt[i] *= pdecadapt[i];

            double ne = nb[i * 3 + 1] * pinoise[i];
            double ni = nb[i * 3 + 2] * pinoise[i];
            if (ne > 0) pge[i] += ne;
            if (ni > 0) pgi[i] += ni;

            if (vi >= pvthr[i] && !ref) {
                pv[i] = pvrst[i];
                ptss[i] = 0.0;
                padapt[i] += padaptinc[i];
                n_spikes++;
            }
        }
        return n_spikes;
    }
};


// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    double mean_ms;
    double min_ms;
    double max_ms;
    double std_ms;
    int total_spikes;
};

template<typename UpdateFn>
BenchResult benchmark(const char* name, int n_steps, int n_trials, UpdateFn&& fn) {
    std::vector<double> times(n_trials);
    int total_spikes = 0;

    for (int t = 0; t < n_trials; t++) {
        g_rng.seed(42 + t);  // deterministic per trial
        int spikes = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < n_steps; s++) {
            spikes += fn();
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        times[t] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_spikes += spikes;
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n_trials;
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double std_dev = std::sqrt(sq_sum / n_trials);

    return {
        mean,
        *std::min_element(times.begin(), times.end()),
        *std::max_element(times.begin(), times.end()),
        std_dev,
        total_spikes
    };
}

// ---------------------------------------------------------------------------
// Behavioral verification: run all 3 layouts with identical RNG state,
// compare every dynamic state variable bitwise after N steps.
// ---------------------------------------------------------------------------

// Extract dynamic state from AoS into flat vectors for comparison
struct DynState {
    std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;
    int total_spikes;
};

DynState extract_soa(const NeuronsSoA& s) {
    return {s.v, s.g_e, s.g_i, s.g_i_slow, s.g_nmda, s.adaptation, s.t_since_spike, 0};
}

DynState extract_aos(const NeuronsAoS& a) {
    DynState d;
    int n = a.n;
    d.v.resize(n); d.g_e.resize(n); d.g_i.resize(n); d.g_i_slow.resize(n);
    d.g_nmda.resize(n); d.adaptation.resize(n); d.t_since_spike.resize(n);
    for (int i = 0; i < n; i++) {
        d.v[i]             = a.neurons[i].v;
        d.g_e[i]           = a.neurons[i].g_e;
        d.g_i[i]           = a.neurons[i].g_i;
        d.g_i_slow[i]      = a.neurons[i].g_i_slow;
        d.g_nmda[i]        = a.neurons[i].g_nmda;
        d.adaptation[i]    = a.neurons[i].adaptation;
        d.t_since_spike[i] = a.neurons[i].t_since_spike;
    }
    return d;
}

DynState extract_hybrid(const NeuronsHybrid& h) {
    return {h.hot.v, h.hot.g_e, h.hot.g_i, h.hot.g_i_slow,
            h.hot.g_nmda, h.hot.adaptation, h.hot.t_since_spike, 0};
}

// Compare two state snapshots bitwise. Returns number of mismatches.
int compare_states(const char* name_a, const char* name_b,
                   const DynState& a, const DynState& b, int n, bool verbose) {
    int mismatches = 0;
    auto check = [&](const char* field, const std::vector<double>& va, const std::vector<double>& vb) {
        for (int i = 0; i < n; i++) {
            // Bitwise comparison via memcmp — no epsilon tolerance
            if (std::memcmp(&va[i], &vb[i], sizeof(double)) != 0) {
                mismatches++;
                if (verbose && mismatches <= 10) {
                    printf("  MISMATCH %s vs %s: %s[%d] = %.17g vs %.17g (diff=%.2e)\n",
                           name_a, name_b, field, i, va[i], vb[i], va[i] - vb[i]);
                }
            }
        }
    };
    check("v", a.v, b.v);
    check("g_e", a.g_e, b.g_e);
    check("g_i", a.g_i, b.g_i);
    check("g_i_slow", a.g_i_slow, b.g_i_slow);
    check("g_nmda", a.g_nmda, b.g_nmda);
    check("adaptation", a.adaptation, b.adaptation);
    check("t_since_spike", a.t_since_spike, b.t_since_spike);
    return mismatches;
}

bool run_verification(int n_neurons, int verify_steps, double dt) {
    printf("========================================\n");
    printf("  PHASE 1: BEHAVIORAL VERIFICATION\n");
    printf("========================================\n");
    printf("Running %d neurons x %d steps with identical RNG seed...\n\n", n_neurons, verify_steps);

    // Initialize canonical SoA with fixed seed
    g_rng.seed(12345);
    NeuronsSoA soa;
    soa.init(n_neurons);

    // Clone into AoS, Hybrid, and Buffered (copies state exactly)
    NeuronsAoS aos;
    aos.init_from_soa(soa);
    NeuronsHybrid hybrid;
    hybrid.init_from_soa(soa);
    NeuronsSoABuffered buffered;
    buffered.init_from_soa(soa);
    NeuronsSoAMultiStep multistep;
    multistep.init_from_soa(soa, 10);  // K=10 for verification

    // Verify initial state is identical
    {
        auto ds = extract_soa(soa);
        auto da = extract_aos(aos);
        auto dh = extract_hybrid(hybrid);
        DynState db = {buffered.v, buffered.g_e, buffered.g_i, buffered.g_i_slow,
                       buffered.g_nmda, buffered.adaptation, buffered.t_since_spike, 0};
        DynState dm = {multistep.v, multistep.g_e, multistep.g_i, multistep.g_i_slow,
                       multistep.g_nmda, multistep.adaptation, multistep.t_since_spike, 0};
        int m1 = compare_states("SoA", "AoS", ds, da, n_neurons, true);
        int m2 = compare_states("SoA", "Hybrid", ds, dh, n_neurons, true);
        int m3 = compare_states("SoA", "Buffered", ds, db, n_neurons, true);
        int m4 = compare_states("SoA", "MultiStep", ds, dm, n_neurons, true);
        if (m1 > 0 || m2 > 0 || m3 > 0 || m4 > 0) {
            printf("FAIL: Initial state not identical! AoS:%d Hybrid:%d Buf:%d Multi:%d mismatches\n", m1, m2, m3, m4);
            return false;
        }
        printf("  Initial state: IDENTICAL (all 7 state vars x %d neurons)\n", n_neurons);
    }

    // Run all five with identical RNG seeds, collecting per-step spike counts
    std::vector<int> spikes_soa(verify_steps), spikes_aos(verify_steps);
    std::vector<int> spikes_hyb(verify_steps), spikes_buf(verify_steps);
    std::vector<int> spikes_multi(verify_steps);

    g_rng.seed(99999);
    for (int s = 0; s < verify_steps; s++)
        spikes_soa[s] = soa.update(dt);

    g_rng.seed(99999);
    for (int s = 0; s < verify_steps; s++)
        spikes_aos[s] = aos.update(dt);

    g_rng.seed(99999);
    for (int s = 0; s < verify_steps; s++)
        spikes_hyb[s] = hybrid.update(dt);

    g_rng.seed(99999);
    for (int s = 0; s < verify_steps; s++)
        spikes_buf[s] = buffered.update(dt);

    g_rng.seed(99999);
    multistep.noise_cursor = multistep.K;  // force refill
    for (int s = 0; s < verify_steps; s++)
        spikes_multi[s] = multistep.update(dt);

    // Compare per-step spike counts
    bool spike_match = true;
    int first_mismatch_step = -1;
    const char* first_mismatch_who = "";
    for (int s = 0; s < verify_steps; s++) {
        if (spikes_soa[s] != spikes_aos[s]) {
            spike_match = false; if (first_mismatch_step < 0) { first_mismatch_step = s; first_mismatch_who = "AoS"; }
        }
        if (spikes_soa[s] != spikes_hyb[s]) {
            spike_match = false; if (first_mismatch_step < 0) { first_mismatch_step = s; first_mismatch_who = "Hybrid"; }
        }
        if (spikes_soa[s] != spikes_buf[s]) {
            spike_match = false; if (first_mismatch_step < 0) { first_mismatch_step = s; first_mismatch_who = "Buffered"; }
        }
        if (spikes_soa[s] != spikes_multi[s]) {
            spike_match = false; if (first_mismatch_step < 0) { first_mismatch_step = s; first_mismatch_who = "MultiStep"; }
        }
    }

    int total_soa = std::accumulate(spikes_soa.begin(), spikes_soa.end(), 0);
    int total_aos = std::accumulate(spikes_aos.begin(), spikes_aos.end(), 0);
    int total_hyb = std::accumulate(spikes_hyb.begin(), spikes_hyb.end(), 0);
    int total_buf = std::accumulate(spikes_buf.begin(), spikes_buf.end(), 0);
    int total_multi = std::accumulate(spikes_multi.begin(), spikes_multi.end(), 0);

    printf("  Per-step spike counts: %s\n", spike_match ? "IDENTICAL" : "MISMATCH");
    if (!spike_match)
        printf("    First mismatch at step %d (%s): SoA=%d AoS=%d Hyb=%d Buf=%d Multi=%d\n",
               first_mismatch_step, first_mismatch_who,
               spikes_soa[first_mismatch_step], spikes_aos[first_mismatch_step],
               spikes_hyb[first_mismatch_step], spikes_buf[first_mismatch_step],
               spikes_multi[first_mismatch_step]);
    printf("  Total spikes: SoA=%d  AoS=%d  Hybrid=%d  Buffered=%d  MultiStep=%d\n",
           total_soa, total_aos, total_hyb, total_buf, total_multi);

    // Compare final state bitwise
    auto ds = extract_soa(soa);
    auto da = extract_aos(aos);
    auto dh = extract_hybrid(hybrid);
    DynState db = {buffered.v, buffered.g_e, buffered.g_i, buffered.g_i_slow,
                   buffered.g_nmda, buffered.adaptation, buffered.t_since_spike, 0};
    DynState dm = {multistep.v, multistep.g_e, multistep.g_i, multistep.g_i_slow,
                   multistep.g_nmda, multistep.adaptation, multistep.t_since_spike, 0};

    int m_aos = compare_states("SoA", "AoS", ds, da, n_neurons, true);
    int m_hyb = compare_states("SoA", "Hybrid", ds, dh, n_neurons, true);
    int m_buf = compare_states("SoA", "Buffered", ds, db, n_neurons, true);
    int m_multi = compare_states("SoA", "MultiStep", ds, dm, n_neurons, true);

    printf("\n  Final state comparison after %d steps:\n", verify_steps);
    printf("    SoA vs AoS:       %d mismatches across %d values — %s\n",
           m_aos, n_neurons * 7, m_aos == 0 ? "BITWISE IDENTICAL" : "MISMATCH");
    printf("    SoA vs Hybrid:    %d mismatches across %d values — %s\n",
           m_hyb, n_neurons * 7, m_hyb == 0 ? "BITWISE IDENTICAL" : "MISMATCH");
    printf("    SoA vs Buffered:  %d mismatches across %d values — %s\n",
           m_buf, n_neurons * 7, m_buf == 0 ? "BITWISE IDENTICAL" : "MISMATCH");
    printf("    SoA vs MultiStep: %d mismatches across %d values — %s\n",
           m_multi, n_neurons * 7, m_multi == 0 ? "BITWISE IDENTICAL" : "MISMATCH");

    // Summary stats on final state
    double v_mean = 0, v_min = 1e9, v_max = -1e9;
    for (int i = 0; i < n_neurons; i++) {
        v_mean += ds.v[i];
        v_min = std::min(v_min, ds.v[i]);
        v_max = std::max(v_max, ds.v[i]);
    }
    v_mean /= n_neurons;
    printf("\n  Final V stats: mean=%.2f  min=%.2f  max=%.2f\n", v_mean, v_min, v_max);
    printf("  Firing rate: %.1f Hz (%.0f spikes / %d neurons / %.0f ms)\n",
           total_soa * 1000.0 / (n_neurons * verify_steps * dt),
           (double)total_soa, n_neurons, verify_steps * dt);

    bool pass = (m_aos == 0 && m_hyb == 0 && m_buf == 0 && m_multi == 0 && spike_match);
    printf("\n  VERIFICATION: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int n_neurons = argc > 1 ? std::atoi(argv[1]) : 604;
    int n_steps   = argc > 2 ? std::atoi(argv[2]) : 10000;
    int n_trials  = argc > 3 ? std::atoi(argv[3]) : 5;
    double dt = 0.1;

    printf("=== SoA vs AoS Neuron Update Benchmark ===\n");
    printf("Neurons: %d  Steps: %d  Trials: %d  dt=%.1f ms\n", n_neurons, n_steps, n_trials, dt);
    printf("NOTE: This is a standalone benchmark — no core code is modified.\n\n");

    printf("NeuronAoS struct size: %zu bytes (%.1f cache lines)\n",
           sizeof(NeuronAoS), sizeof(NeuronAoS) / 64.0);
    printf("NeuronCold struct size: %zu bytes (%.1f cache lines)\n",
           sizeof(NeuronCold), sizeof(NeuronCold) / 64.0);

    size_t soa_hot_bytes = 25UL * n_neurons * sizeof(double);
    printf("SoA total array footprint: %zu bytes (%.1f KB)\n", soa_hot_bytes, soa_hot_bytes / 1024.0);
    printf("AoS total struct footprint: %zu bytes (%.1f KB)\n",
           n_neurons * sizeof(NeuronAoS), n_neurons * sizeof(NeuronAoS) / 1024.0);
    printf("Hybrid cold array: %zu bytes (%.1f KB)\n",
           n_neurons * sizeof(NeuronCold), n_neurons * sizeof(NeuronCold) / 1024.0);
    printf("Hybrid hot arrays (7x): %zu bytes (%.1f KB)\n",
           7UL * n_neurons * sizeof(double), 7UL * n_neurons * sizeof(double) / 1024.0);
    printf("Noise buffer (K=1):  %zu bytes (%.1f KB)  [L1]\n",
           3UL * n_neurons * sizeof(double), 3.0 * n_neurons * sizeof(double) / 1024.0);
    printf("Noise buffer (K=10): %zu bytes (%.1f KB)  [L2]\n",
           30UL * n_neurons * sizeof(double), 30.0 * n_neurons * sizeof(double) / 1024.0);
    printf("Noise buffer (K=100): %zu bytes (%.1f KB) [L2/L3]\n\n",
           300UL * n_neurons * sizeof(double), 300.0 * n_neurons * sizeof(double) / 1024.0);

    // ====== PHASE 1: Verification ======
    int verify_steps = 5000;  // enough steps for divergence to manifest
    bool verified = run_verification(n_neurons, verify_steps, dt);
    if (!verified) {
        printf("ABORTING: Behavioral verification failed. No timing results.\n");
        return 1;
    }

    // ====== PHASE 2: Performance benchmark ======
    printf("========================================\n");
    printf("  PHASE 2: PERFORMANCE BENCHMARK\n");
    printf("========================================\n\n");

    // Fresh init for each layout — same starting state
    g_rng.seed(77777);
    NeuronsSoA soa;
    soa.init(n_neurons);
    NeuronsAoS aos;
    aos.init_from_soa(soa);
    NeuronsHybrid hybrid;
    hybrid.init_from_soa(soa);
    NeuronsSoABuffered buffered;
    buffered.init_from_soa(soa);

    // --- Benchmark SoA ---
    printf("Running SoA (%d steps x %d trials)...\n", n_steps, n_trials);
    auto soa_result = benchmark("SoA", n_steps, n_trials, [&]() {
        return soa.update(dt);
    });

    // Re-init all from same seed for fair comparison
    g_rng.seed(77777);
    soa.init(n_neurons);
    aos.init_from_soa(soa);
    hybrid.init_from_soa(soa);
    buffered.init_from_soa(soa);

    // --- Benchmark AoS ---
    printf("Running AoS (%d steps x %d trials)...\n", n_steps, n_trials);
    auto aos_result = benchmark("AoS", n_steps, n_trials, [&]() {
        return aos.update(dt);
    });

    // Re-init
    g_rng.seed(77777);
    soa.init(n_neurons);
    hybrid.init_from_soa(soa);
    buffered.init_from_soa(soa);

    // --- Benchmark Hybrid ---
    printf("Running Hybrid (%d steps x %d trials)...\n", n_steps, n_trials);
    auto hybrid_result = benchmark("Hybrid", n_steps, n_trials, [&]() {
        return hybrid.update(dt);
    });

    // Re-init
    g_rng.seed(77777);
    soa.init(n_neurons);
    buffered.init_from_soa(soa);

    // --- Benchmark Buffered ---
    printf("Running Buffered (%d steps x %d trials)...\n", n_steps, n_trials);
    auto buf_result = benchmark("Buffered", n_steps, n_trials, [&]() {
        return buffered.update(dt);
    });

    // --- Benchmark MultiStep K-sweep ---
    int k_values[] = {1, 5, 10, 25, 50, 100, 250, 500};
    int n_k = sizeof(k_values) / sizeof(k_values[0]);
    std::vector<BenchResult> multi_results(n_k);

    for (int ki = 0; ki < n_k; ki++) {
        int K = k_values[ki];
        g_rng.seed(77777);
        soa.init(n_neurons);
        NeuronsSoAMultiStep ms;
        ms.init_from_soa(soa, K);

        printf("Running MultiStep K=%d (%d steps x %d trials)...\n", K, n_steps, n_trials);
        multi_results[ki] = benchmark("MultiStep", n_steps, n_trials, [&]() {
            return ms.update(dt);
        });
    }

    // --- Results ---
    printf("\n");
    printf("%-14s  %10s  %10s  %10s  %10s  %12s\n",
           "Layout", "Mean(ms)", "Min(ms)", "Max(ms)", "Std(ms)", "Spikes");
    printf("%-14s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "SoA", soa_result.mean_ms, soa_result.min_ms, soa_result.max_ms,
           soa_result.std_ms, soa_result.total_spikes);
    printf("%-14s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "AoS", aos_result.mean_ms, aos_result.min_ms, aos_result.max_ms,
           aos_result.std_ms, aos_result.total_spikes);
    printf("%-14s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "Hybrid", hybrid_result.mean_ms, hybrid_result.min_ms, hybrid_result.max_ms,
           hybrid_result.std_ms, hybrid_result.total_spikes);
    printf("%-14s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "Buffered(K=1)", buf_result.mean_ms, buf_result.min_ms, buf_result.max_ms,
           buf_result.std_ms, buf_result.total_spikes);

    for (int ki = 0; ki < n_k; ki++) {
        char label[32];
        snprintf(label, sizeof(label), "Multi(K=%d)", k_values[ki]);
        printf("%-14s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
               label, multi_results[ki].mean_ms, multi_results[ki].min_ms,
               multi_results[ki].max_ms, multi_results[ki].std_ms,
               multi_results[ki].total_spikes);
    }

    printf("\n--- Speedup (vs SoA, using min times) ---\n");
    printf("AoS:          %.3fx %s\n", soa_result.min_ms / aos_result.min_ms,
           aos_result.min_ms < soa_result.min_ms ? "(faster)" : "(slower)");
    printf("Hybrid:       %.3fx %s\n", soa_result.min_ms / hybrid_result.min_ms,
           hybrid_result.min_ms < soa_result.min_ms ? "(faster)" : "(slower)");
    printf("Buffered(K=1):%.3fx %s\n", soa_result.min_ms / buf_result.min_ms,
           buf_result.min_ms < soa_result.min_ms ? "(faster)" : "(slower)");

    double best_multi_ms = 1e9;
    int best_k = 1;
    for (int ki = 0; ki < n_k; ki++) {
        int K = k_values[ki];
        double speedup = soa_result.min_ms / multi_results[ki].min_ms;
        size_t buf_bytes = 3UL * n_neurons * K * sizeof(double);
        const char* cache_tier = buf_bytes <= 48*1024 ? "L1" :
                                 buf_bytes <= 512*1024 ? "L2" :
                                 buf_bytes <= 8*1024*1024 ? "L3" : "RAM";
        printf("Multi(K=%-3d): %.3fx %s  [buf=%.1fKB %s]\n",
               K, speedup,
               multi_results[ki].min_ms < soa_result.min_ms ? "(faster)" : "(slower)",
               buf_bytes / 1024.0, cache_tier);
        if (multi_results[ki].min_ms < best_multi_ms) {
            best_multi_ms = multi_results[ki].min_ms;
            best_k = K;
        }
    }

    // Per-step throughput
    double soa_us = soa_result.min_ms * 1000.0 / n_steps;
    double buf_us = buf_result.min_ms * 1000.0 / n_steps;
    double best_us = best_multi_ms * 1000.0 / n_steps;
    printf("\n--- Per-step time (best variants) ---\n");
    printf("SoA (baseline):   %.2f us/step (%.2f ns/neuron)\n", soa_us, soa_us * 1000.0 / n_neurons);
    printf("Buffered (K=1):   %.2f us/step (%.2f ns/neuron)\n", buf_us, buf_us * 1000.0 / n_neurons);
    printf("Best Multi (K=%d): %.2f us/step (%.2f ns/neuron)\n", best_k, best_us, best_us * 1000.0 / n_neurons);

    // Simulation context: 604 neurons * ~5000 steps/sample * 1500 samples
    double sim_soa_s = soa_us * 5000.0 * 1500.0 / 1e6;
    double sim_buf_s = buf_us * 5000.0 * 1500.0 / 1e6;
    double sim_best_s = best_us * 5000.0 * 1500.0 / 1e6;
    printf("\n--- Projected sweep impact (1500 samples x 5000 steps) ---\n");
    printf("SoA:             %.1f s per grid point\n", sim_soa_s);
    printf("Buffered (K=1):  %.1f s per grid point\n", sim_buf_s);
    printf("Best Multi K=%d: %.1f s per grid point\n", best_k, sim_best_s);
    printf("Savings vs SoA:  %.1f s per grid point (%.1f%%)\n",
           sim_soa_s - sim_best_s, (sim_soa_s - sim_best_s) / sim_soa_s * 100);
    printf("300-point sweep savings: %.0f s (%.1f min)\n",
           (sim_soa_s - sim_best_s) * 300, (sim_soa_s - sim_best_s) * 300 / 60.0);

    return 0;
}
