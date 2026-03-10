/*
 * Caller-loop efficiency benchmark.
 *
 * Replicates the full run_sample_with_std() loop structure:
 *   - Neuron update (with pre-buffered noise, already verified)
 *   - Activity recording (vector<vector<int>> vs CSR-flat)
 *   - STD lookups (std::set/map vs flat vectors)
 *   - Input injection allocation (per-step alloc vs scratch reuse)
 *   - Spike return (new vector per step vs persistent buffer)
 *
 * Tests "Baseline" (current code patterns) vs "Optimized" (allocation-free).
 * Verifies bitwise-identical neuron state + identical activity records.
 *
 * Build:
 *   c++ -O3 -std=c++17 -o bench_caller_loop \
 *       experiments/sim_efficiency_benchmarking/bench_caller_loop.cpp -lm
 *
 * Run:
 *   ./bench_caller_loop [n_neurons] [n_steps] [n_trials]
 *   Default: 604 neurons, 10000 steps, 5 trials
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <array>

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------
static thread_local std::mt19937_64 g_rng(42);
static thread_local std::normal_distribution<double> g_norm(0.0, 1.0);
static inline double rng_normal() { return g_norm(g_rng); }

// ---------------------------------------------------------------------------
// Shared neuron state (SoA with pre-buffered noise — already verified)
// ---------------------------------------------------------------------------
struct NeuronState {
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
    std::vector<double> noise_buf;

    void init(int n_neurons) {
        n = n_neurons;
        mg_factor = 1.0 / 3.57;
        nmda_ratio = 0.5;

        auto fill = [&](int sz, double base, double sigma) {
            std::vector<double> out(sz);
            for (int i = 0; i < sz; i++) out[i] = base + rng_normal() * sigma;
            return out;
        };

        v_rest = fill(n, -65.0, 3.0);
        v_threshold = fill(n, -55.0, 2.5);
        v_reset.assign(n, -75.0);
        tau_m = fill(n, 10.0, 1.0);
        tau_ref.assign(n, 4.0);
        e_reversal.assign(n, 0.0);
        i_reversal.assign(n, -80.0);
        k_reversal.assign(n, -90.0);
        v_noise_amp.assign(n, 0.1);
        i_noise_amp.assign(n, 0.001);
        adaptation_increment = fill(n, 0.2, 0.05);

        double dt = 0.1;
        exp_decay_e.resize(n); exp_decay_i.resize(n); exp_decay_i_slow.resize(n);
        exp_decay_nmda.resize(n); exp_decay_adapt.resize(n);
        for (int i = 0; i < n; i++) {
            exp_decay_e[i]      = std::exp(-dt / 3.0);
            exp_decay_i[i]      = std::exp(-dt / 7.0);
            exp_decay_i_slow[i] = std::exp(-dt / 50.0);
            exp_decay_nmda[i]   = std::exp(-dt / 100.0);
            exp_decay_adapt[i]  = std::exp(-dt / 100.0);
        }
        tonic_conductance.assign(n, 0.001);
        tonic_reversal.assign(n, -80.0);

        v = v_rest;
        g_e.assign(n, 0.01); g_i.assign(n, 0.005);
        g_i_slow.assign(n, 0.002); g_nmda.assign(n, 0.005);
        adaptation.assign(n, 0.01);
        t_since_spike.assign(n, 10.0);

        for (int i = 0; i < n; i++) {
            tau_m[i] = std::max(tau_m[i], 3.0);
            adaptation_increment[i] = std::max(adaptation_increment[i], 0.0);
        }

        noise_buf.resize(3 * n);
    }

    // Save/restore for fair repeated benchmarking
    struct Snapshot {
        std::vector<double> v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike;
    };
    Snapshot save() const {
        return {v, g_e, g_i, g_i_slow, g_nmda, adaptation, t_since_spike};
    }
    void restore(const Snapshot& s) {
        v = s.v; g_e = s.g_e; g_i = s.g_i; g_i_slow = s.g_i_slow;
        g_nmda = s.g_nmda; adaptation = s.adaptation; t_since_spike = s.t_since_spike;
    }
};

// Core neuron update — identical to verified NeuronsSoABuffered
// Returns spike count (spikes written into caller-provided buffer)
static int neuron_update_into(NeuronState& ns, double dt,
                               int* __restrict__ spike_out) {
    double* __restrict__ nb = ns.noise_buf.data();
    for (int i = 0; i < 3 * ns.n; i++) nb[i] = rng_normal();

    const double* __restrict__ pvrest = ns.v_rest.data();
    const double* __restrict__ pvthr = ns.v_threshold.data();
    const double* __restrict__ pvrst = ns.v_reset.data();
    const double* __restrict__ ptaum = ns.tau_m.data();
    const double* __restrict__ ptref = ns.tau_ref.data();
    const double* __restrict__ perev = ns.e_reversal.data();
    const double* __restrict__ pirev = ns.i_reversal.data();
    const double* __restrict__ pkrev = ns.k_reversal.data();
    const double* __restrict__ pvnoise = ns.v_noise_amp.data();
    const double* __restrict__ pinoise = ns.i_noise_amp.data();
    const double* __restrict__ padaptinc = ns.adaptation_increment.data();
    const double* __restrict__ pdece = ns.exp_decay_e.data();
    const double* __restrict__ pdeci = ns.exp_decay_i.data();
    const double* __restrict__ pdecis = ns.exp_decay_i_slow.data();
    const double* __restrict__ pdecnmda = ns.exp_decay_nmda.data();
    const double* __restrict__ pdecadapt = ns.exp_decay_adapt.data();
    const double* __restrict__ ptonic = ns.tonic_conductance.data();
    const double* __restrict__ ptonic_rev = ns.tonic_reversal.data();
    double* __restrict__ pv = ns.v.data();
    double* __restrict__ pge = ns.g_e.data();
    double* __restrict__ pgi = ns.g_i.data();
    double* __restrict__ pgis = ns.g_i_slow.data();
    double* __restrict__ pgnmda = ns.g_nmda.data();
    double* __restrict__ padapt = ns.adaptation.data();
    double* __restrict__ ptss = ns.t_since_spike.data();

    int n_spikes = 0;
    for (int i = 0; i < ns.n; i++) {
        bool ref = ptss[i] < ptref[i];
        ptss[i] += dt;
        double vi = pv[i];
        double i_e = pge[i] * (perev[i] - vi);
        double i_i = pgi[i] * (pirev[i] - vi);
        double i_is = pgis[i] * (pirev[i] - vi);
        double mg_block = 1.0 / (1.0 + ns.mg_factor * std::exp(-0.062 * vi));
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
            spike_out[n_spikes++] = i;
        }
    }
    return n_spikes;
}

// ---------------------------------------------------------------------------
// Fake connectivity + STD masks (realistic structure)
// ---------------------------------------------------------------------------
struct FakeConnectivity {
    // CSR outgoing connections (sparse, ~200 targets per neuron)
    std::vector<int64_t> csr_indptr;
    std::vector<int32_t> csr_targets;
    std::vector<double>  csr_weights;

    // STD masks — current code uses set<int> + map<int, vector<bool>>
    std::set<int> std_eligible_set;
    std::map<int, std::vector<bool>> recurrent_masks_map;

    // STD masks — optimized: flat vectors indexed by neuron id
    std::vector<bool> std_eligible_flat;     // n_neurons, true if eligible
    std::vector<bool> has_recurrent_mask;    // n_neurons
    // Flat-packed mask data: for neuron i, mask starts at mask_offsets[i]
    // with mask_lengths[i] entries
    std::vector<int> mask_offsets;
    std::vector<int> mask_lengths;
    std::vector<bool> mask_data_flat;

    // Input spikes: which steps have input, and which neurons to stimulate
    std::vector<int> input_steps;  // sorted steps that have input
    std::vector<std::vector<int>> input_neurons_per_step;
    std::vector<std::vector<double>> input_weights_per_step;

    int n_neurons;
    int n_reservoir;

    void build(int n, int n_steps) {
        n_neurons = n;
        n_reservoir = (int)(n * 0.8);  // 80% reservoir
        int n_input = n - n_reservoir;

        // Build sparse connectivity: ~200 outgoing per neuron
        csr_indptr.resize(n + 1, 0);
        std::mt19937 conn_rng(123);
        std::uniform_int_distribution<int> tgt_dist(0, n - 1);
        std::normal_distribution<double> w_dist(0.05, 0.02);

        for (int i = 0; i < n; i++) {
            int n_out = 150 + (conn_rng() % 100);  // 150-249 targets
            csr_indptr[i + 1] = csr_indptr[i] + n_out;
        }
        int total_edges = (int)csr_indptr[n];
        csr_targets.resize(total_edges);
        csr_weights.resize(total_edges);
        for (int i = 0; i < total_edges; i++) {
            csr_targets[i] = tgt_dist(conn_rng);
            csr_weights[i] = std::abs(w_dist(conn_rng));
        }

        // Build STD masks — excitatory reservoir neurons are eligible
        // (~60% of reservoir = ~290 neurons for n=604)
        std_eligible_flat.assign(n, false);
        has_recurrent_mask.assign(n, false);
        mask_offsets.assign(n, 0);
        mask_lengths.assign(n, 0);

        for (int i = 0; i < n_reservoir; i++) {
            if (conn_rng() % 5 != 0) {  // ~80% eligible
                std_eligible_set.insert(i);
                std_eligible_flat[i] = true;

                int64_t start = csr_indptr[i];
                int64_t end = csr_indptr[i + 1];
                int n_out = (int)(end - start);
                std::vector<bool> mask(n_out);
                bool any = false;
                for (int k = 0; k < n_out; k++) {
                    mask[k] = csr_targets[start + k] < n_reservoir;
                    if (mask[k]) any = true;
                }
                if (any) {
                    recurrent_masks_map[i] = mask;
                    has_recurrent_mask[i] = true;
                    mask_offsets[i] = (int)mask_data_flat.size();
                    mask_lengths[i] = n_out;
                    for (int k = 0; k < n_out; k++)
                        mask_data_flat.push_back(mask[k]);
                }
            }
        }

        // Build input spikes: ~30% of steps have input (realistic for BSA)
        std::uniform_int_distribution<int> input_nid(n_reservoir, n - 1);
        for (int s = 0; s < n_steps; s++) {
            if (conn_rng() % 3 == 0) {  // ~33% of steps
                input_steps.push_back(s);
                int n_inp = 3 + (conn_rng() % 8);  // 3-10 input neurons
                std::vector<int> nids(n_inp);
                std::vector<double> wts(n_inp);
                for (int k = 0; k < n_inp; k++) {
                    nids[k] = input_nid(conn_rng);
                    wts[k] = 0.02 + (conn_rng() % 100) * 0.001;
                }
                input_neurons_per_step.push_back(nids);
                input_weights_per_step.push_back(wts);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// BASELINE: mimics current run_sample_with_std patterns
// ---------------------------------------------------------------------------
struct BaselineResult {
    // vector<vector<int>> activity_record — current format
    std::vector<std::vector<int>> activity_record;
    int total_spikes;
};

BaselineResult run_baseline(NeuronState& ns, FakeConnectivity& conn,
                             int n_steps, double dt) {
    BaselineResult result;
    result.activity_record.resize(n_steps);
    result.total_spikes = 0;

    // STD state
    std::vector<double> x_resource(ns.n, 1.0);
    std::vector<double> last_update_time(ns.n, 0.0);

    // Original weights backup (map, like current code)
    std::map<int, std::vector<double>> original_weights;
    for (auto& [nid, mask] : conn.recurrent_masks_map) {
        int64_t start = conn.csr_indptr[nid];
        int64_t end = conn.csr_indptr[nid + 1];
        original_weights[nid].assign(conn.csr_weights.begin() + start,
                                      conn.csr_weights.begin() + end);
    }
    std::set<int> dirty_neurons;

    int input_idx = 0;

    for (int step = 0; step < n_steps; step++) {
        double t_ms = step * dt;

        // Input injection: allocate vectors each step (current pattern)
        if (input_idx < (int)conn.input_steps.size() &&
            conn.input_steps[input_idx] == step) {
            // Current code: allocate neuron_indices, tuning_weights, currents per step
            std::vector<int> neuron_indices = conn.input_neurons_per_step[input_idx];
            std::vector<double> currents(neuron_indices.size());
            const auto& wts = conn.input_weights_per_step[input_idx];
            for (size_t k = 0; k < neuron_indices.size(); k++) {
                currents[k] = wts[k];
            }
            // Stimulate
            for (size_t k = 0; k < neuron_indices.size(); k++) {
                ns.g_e[neuron_indices[k]] += currents[k];
                ns.g_nmda[neuron_indices[k]] += currents[k] * ns.nmda_ratio;
            }
            input_idx++;
        }

        // Neuron update — returns new vector<int> each step (current pattern)
        std::vector<int> active;
        active.reserve(32);
        // Use same kernel but through vector allocation
        {
            double* __restrict__ nb = ns.noise_buf.data();
            for (int i = 0; i < 3 * ns.n; i++) nb[i] = rng_normal();

            for (int i = 0; i < ns.n; i++) {
                bool ref = ns.t_since_spike[i] < ns.tau_ref[i];
                ns.t_since_spike[i] += dt;
                double vi = ns.v[i];
                double i_e = ns.g_e[i] * (ns.e_reversal[i] - vi);
                double i_i = ns.g_i[i] * (ns.i_reversal[i] - vi);
                double i_is = ns.g_i_slow[i] * (ns.i_reversal[i] - vi);
                double mg_block = 1.0 / (1.0 + ns.mg_factor * std::exp(-0.062 * vi));
                double i_nmda = ns.g_nmda[i] * mg_block * (ns.e_reversal[i] - vi);
                double i_adapt = ns.adaptation[i] * (ns.k_reversal[i] - vi);
                double i_tonic = ns.tonic_conductance[i] * (ns.tonic_reversal[i] - vi);
                double dv = dt * ((-(vi - ns.v_rest[i]) / ns.tau_m[i])
                                  + i_e + i_nmda + i_i + i_is + i_adapt + i_tonic);
                vi += dv;
                vi += nb[i * 3 + 0] * ns.v_noise_amp[i];
                if (ref) vi = ns.v_reset[i];
                ns.v[i] = vi;

                ns.g_e[i]        *= ns.exp_decay_e[i];
                ns.g_i[i]        *= ns.exp_decay_i[i];
                ns.g_i_slow[i]   *= ns.exp_decay_i_slow[i];
                ns.g_nmda[i]     *= ns.exp_decay_nmda[i];
                ns.adaptation[i] *= ns.exp_decay_adapt[i];

                double ne = nb[i * 3 + 1] * ns.i_noise_amp[i];
                double ni = nb[i * 3 + 2] * ns.i_noise_amp[i];
                if (ne > 0) ns.g_e[i] += ne;
                if (ni > 0) ns.g_i[i] += ni;

                if (vi >= ns.v_threshold[i] && !ref) {
                    ns.v[i] = ns.v_reset[i];
                    ns.t_since_spike[i] = 0.0;
                    ns.adaptation[i] += ns.adaptation_increment[i];
                    active.push_back(i);
                }
            }
        }

        // Store activity — copy into vector<vector<int>> (current pattern)
        result.activity_record[step] = active;
        result.total_spikes += (int)active.size();

        // STD processing — set::find + map::find per spiking neuron
        if (!active.empty()) {
            for (int nid : active) {
                if (conn.std_eligible_set.find(nid) == conn.std_eligible_set.end())
                    continue;
                auto mit = conn.recurrent_masks_map.find(nid);
                if (mit == conn.recurrent_masks_map.end()) continue;
                const auto& mask = mit->second;

                double delta_t = t_ms - last_update_time[nid];
                if (delta_t > 0 && x_resource[nid] < 1.0) {
                    x_resource[nid] = 1.0 - (1.0 - x_resource[nid]) * std::exp(-delta_t / 500.0);
                    x_resource[nid] = std::min(x_resource[nid], 1.0);
                }
                x_resource[nid] *= 0.9;
                last_update_time[nid] = t_ms;

                const auto& orig = original_weights[nid];
                int64_t start = conn.csr_indptr[nid];
                int n_out = (int)(conn.csr_indptr[nid + 1] - start);
                for (int k = 0; k < n_out; k++) {
                    conn.csr_weights[start + k] = mask[k]
                        ? orig[k] * x_resource[nid] : orig[k];
                }
                dirty_neurons.insert(nid);
            }
        }
    }

    // Restore weights
    for (int nid : dirty_neurons) {
        int64_t start = conn.csr_indptr[nid];
        const auto& orig = original_weights[nid];
        for (size_t k = 0; k < orig.size(); k++)
            conn.csr_weights[start + (int64_t)k] = orig[k];
    }

    return result;
}


// ---------------------------------------------------------------------------
// OPTIMIZED: allocation-free caller loop
//   #1: persistent spike buffer (no per-step vector<int> allocation)
//   #2: CSR-flat activity record (one big vector + offsets)
//   #3: flat STD lookups (vector<bool> indexed by neuron id)
//   #4: pre-allocated input scratch buffers
// ---------------------------------------------------------------------------
struct OptimizedResult {
    // CSR-flat activity record: spike_ids[offsets[s]..offsets[s+1]]
    std::vector<int> spike_ids;
    std::vector<int> offsets;  // n_steps + 1
    int total_spikes;
};

OptimizedResult run_optimized(NeuronState& ns, FakeConnectivity& conn,
                               int n_steps, double dt) {
    OptimizedResult result;
    result.offsets.resize(n_steps + 1);
    result.spike_ids.reserve(n_steps * 2);  // rough estimate
    result.total_spikes = 0;

    // #1: persistent spike buffer
    std::vector<int> spike_buf(ns.n);

    // #3: flat STD state
    std::vector<double> x_resource(ns.n, 1.0);
    std::vector<double> last_update_time(ns.n, 0.0);

    // Original weights backup — flat vector per eligible neuron
    // (still need per-neuron backup, but use flat lookup)
    std::vector<std::vector<double>> original_weights(ns.n);
    std::vector<bool> dirty(ns.n, false);
    for (int i = 0; i < ns.n; i++) {
        if (conn.has_recurrent_mask[i]) {
            int64_t start = conn.csr_indptr[i];
            int64_t end = conn.csr_indptr[i + 1];
            original_weights[i].assign(conn.csr_weights.begin() + start,
                                        conn.csr_weights.begin() + end);
        }
    }

    // #4: pre-allocated input scratch
    std::vector<int> inp_neurons;
    std::vector<double> inp_currents;
    inp_neurons.reserve(32);
    inp_currents.reserve(32);

    int input_idx = 0;

    for (int step = 0; step < n_steps; step++) {
        double t_ms = step * dt;

        // Input injection: reuse scratch buffers
        if (input_idx < (int)conn.input_steps.size() &&
            conn.input_steps[input_idx] == step) {
            inp_neurons.clear();
            inp_currents.clear();
            const auto& nids = conn.input_neurons_per_step[input_idx];
            const auto& wts = conn.input_weights_per_step[input_idx];
            for (size_t k = 0; k < nids.size(); k++) {
                inp_neurons.push_back(nids[k]);
                inp_currents.push_back(wts[k]);
            }
            for (size_t k = 0; k < inp_neurons.size(); k++) {
                ns.g_e[inp_neurons[k]] += inp_currents[k];
                ns.g_nmda[inp_neurons[k]] += inp_currents[k] * ns.nmda_ratio;
            }
            input_idx++;
        }

        // #1: neuron update writes into persistent buffer
        int n_active = neuron_update_into(ns, dt, spike_buf.data());

        // #2: append to CSR-flat activity record
        result.offsets[step] = result.total_spikes;
        for (int k = 0; k < n_active; k++) {
            result.spike_ids.push_back(spike_buf[k]);
        }
        result.total_spikes += n_active;

        // #3: STD with flat lookups — no set::find, no map::find
        if (n_active > 0) {
            for (int k = 0; k < n_active; k++) {
                int nid = spike_buf[k];
                if (!conn.std_eligible_flat[nid]) continue;
                if (!conn.has_recurrent_mask[nid]) continue;

                double delta_t = t_ms - last_update_time[nid];
                if (delta_t > 0 && x_resource[nid] < 1.0) {
                    x_resource[nid] = 1.0 - (1.0 - x_resource[nid]) * std::exp(-delta_t / 500.0);
                    x_resource[nid] = std::min(x_resource[nid], 1.0);
                }
                x_resource[nid] *= 0.9;
                last_update_time[nid] = t_ms;

                int off = conn.mask_offsets[nid];
                int len = conn.mask_lengths[nid];
                const auto& orig = original_weights[nid];
                int64_t start = conn.csr_indptr[nid];
                for (int m = 0; m < len; m++) {
                    conn.csr_weights[start + m] = conn.mask_data_flat[off + m]
                        ? orig[m] * x_resource[nid] : orig[m];
                }
                dirty[nid] = true;
            }
        }
    }
    result.offsets[n_steps] = result.total_spikes;

    // Restore weights
    for (int i = 0; i < ns.n; i++) {
        if (!dirty[i]) continue;
        int64_t start = conn.csr_indptr[i];
        const auto& orig = original_weights[i];
        for (size_t k = 0; k < orig.size(); k++)
            conn.csr_weights[start + (int64_t)k] = orig[k];
    }

    return result;
}


// ---------------------------------------------------------------------------
// Verification: compare baseline and optimized results
// ---------------------------------------------------------------------------
bool verify(const NeuronState& ns_base, const NeuronState& ns_opt,
            const BaselineResult& base, const OptimizedResult& opt,
            int n_steps) {
    printf("========================================\n");
    printf("  PHASE 1: BEHAVIORAL VERIFICATION\n");
    printf("========================================\n\n");

    bool pass = true;

    // Compare total spikes
    printf("  Total spikes: baseline=%d  optimized=%d  %s\n",
           base.total_spikes, opt.total_spikes,
           base.total_spikes == opt.total_spikes ? "MATCH" : "MISMATCH");
    if (base.total_spikes != opt.total_spikes) pass = false;

    // Compare per-step activity records
    int step_mismatches = 0;
    int first_mismatch = -1;
    for (int s = 0; s < n_steps; s++) {
        int base_n = (int)base.activity_record[s].size();
        int opt_start = opt.offsets[s];
        int opt_n = opt.offsets[s + 1] - opt_start;
        if (base_n != opt_n) {
            step_mismatches++;
            if (first_mismatch < 0) first_mismatch = s;
            continue;
        }
        for (int k = 0; k < base_n; k++) {
            if (base.activity_record[s][k] != opt.spike_ids[opt_start + k]) {
                step_mismatches++;
                if (first_mismatch < 0) first_mismatch = s;
                break;
            }
        }
    }
    printf("  Per-step activity: %d/%d steps match — %s\n",
           n_steps - step_mismatches, n_steps,
           step_mismatches == 0 ? "IDENTICAL" : "MISMATCH");
    if (step_mismatches > 0) {
        printf("    First mismatch at step %d\n", first_mismatch);
        pass = false;
    }

    // Compare final neuron state bitwise
    int state_mismatches = 0;
    auto check = [&](const char* name, const std::vector<double>& a, const std::vector<double>& b) {
        for (int i = 0; i < ns_base.n; i++) {
            if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
                state_mismatches++;
                if (state_mismatches <= 5)
                    printf("    MISMATCH %s[%d]: %.17g vs %.17g\n", name, i, a[i], b[i]);
            }
        }
    };
    check("v", ns_base.v, ns_opt.v);
    check("g_e", ns_base.g_e, ns_opt.g_e);
    check("g_i", ns_base.g_i, ns_opt.g_i);
    check("g_i_slow", ns_base.g_i_slow, ns_opt.g_i_slow);
    check("g_nmda", ns_base.g_nmda, ns_opt.g_nmda);
    check("adaptation", ns_base.adaptation, ns_opt.adaptation);
    check("t_since_spike", ns_base.t_since_spike, ns_opt.t_since_spike);

    printf("  Final state: %d mismatches across %d values — %s\n",
           state_mismatches, ns_base.n * 7,
           state_mismatches == 0 ? "BITWISE IDENTICAL" : "MISMATCH");
    if (state_mismatches > 0) pass = false;

    printf("\n  VERIFICATION: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}


// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------
struct BenchResult {
    double mean_ms, min_ms, max_ms, std_ms;
    int total_spikes;
};

template<typename Fn>
BenchResult bench(const char* name, int n_trials, Fn&& fn) {
    std::vector<double> times(n_trials);
    int total_spikes = 0;
    for (int t = 0; t < n_trials; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int spikes = fn(t);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[t] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_spikes += spikes;
    }
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n_trials;
    double sq = 0;
    for (double t : times) sq += (t - mean) * (t - mean);
    return {mean,
            *std::min_element(times.begin(), times.end()),
            *std::max_element(times.begin(), times.end()),
            std::sqrt(sq / n_trials), total_spikes};
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int n_neurons = argc > 1 ? std::atoi(argv[1]) : 604;
    int n_steps   = argc > 2 ? std::atoi(argv[2]) : 10000;
    int n_trials  = argc > 3 ? std::atoi(argv[3]) : 5;
    double dt = 0.1;

    printf("=== Caller-Loop Efficiency Benchmark ===\n");
    printf("Neurons: %d  Steps: %d  Trials: %d  dt=%.1f ms\n\n", n_neurons, n_steps, n_trials, dt);
    printf("Optimizations tested:\n");
    printf("  #1: Persistent spike buffer (vs new vector<int> per step)\n");
    printf("  #2: CSR-flat activity record (vs vector<vector<int>>)\n");
    printf("  #3: Flat STD lookups (vs std::set::find + std::map::find)\n");
    printf("  #4: Pre-allocated input scratch (vs per-step allocation)\n\n");

    // Build shared connectivity and input schedule
    FakeConnectivity conn;
    conn.build(n_neurons, n_steps);
    printf("Connectivity: %d edges, %d STD-eligible neurons, %d recurrent masks\n",
           (int)conn.csr_targets.size(), (int)conn.std_eligible_set.size(),
           (int)conn.recurrent_masks_map.size());
    printf("Input steps: %d/%d (%.0f%%)\n\n",
           (int)conn.input_steps.size(), n_steps,
           100.0 * conn.input_steps.size() / n_steps);

    // Memory comparison
    size_t baseline_activity_bytes = (size_t)n_steps * (sizeof(std::vector<int>) + 2 * sizeof(int));
    size_t opt_activity_bytes = (size_t)n_steps * sizeof(int) + sizeof(int) * 2;
    printf("Activity record memory estimate:\n");
    printf("  Baseline (vector<vector<int>>): ~%zu KB overhead + spike data\n",
           baseline_activity_bytes / 1024);
    printf("  Optimized (CSR-flat):           ~%zu KB overhead + spike data\n\n",
           opt_activity_bytes / 1024);

    // ====== PHASE 1: Verification ======
    g_rng.seed(12345);
    NeuronState ns_base;
    ns_base.init(n_neurons);
    auto snapshot = ns_base.save();

    // Make a copy for optimized run
    NeuronState ns_opt;
    ns_opt = ns_base;  // deep copy

    // Save connectivity weights (both variants modify them during STD)
    std::vector<double> weights_backup = conn.csr_weights;

    // Run baseline
    g_rng.seed(99999);
    auto base_result = run_baseline(ns_base, conn, n_steps, dt);

    // Restore connectivity + neuron state for optimized run
    conn.csr_weights = weights_backup;
    ns_opt.restore(snapshot);

    // Run optimized with same RNG seed
    g_rng.seed(99999);
    auto opt_result = run_optimized(ns_opt, conn, n_steps, dt);

    bool verified = verify(ns_base, ns_opt, base_result, opt_result, n_steps);
    if (!verified) {
        printf("ABORTING: Behavioral verification failed.\n");
        return 1;
    }

    // ====== PHASE 2: Performance ======
    printf("========================================\n");
    printf("  PHASE 2: PERFORMANCE BENCHMARK\n");
    printf("========================================\n\n");

    // Benchmark baseline
    printf("Running Baseline (%d steps x %d trials)...\n", n_steps, n_trials);
    auto br = bench("Baseline", n_trials, [&](int trial) {
        g_rng.seed(12345);
        NeuronState ns;
        ns.init(n_neurons);
        conn.csr_weights = weights_backup;
        g_rng.seed(42 + trial);
        auto r = run_baseline(ns, conn, n_steps, dt);
        return r.total_spikes;
    });

    // Benchmark optimized
    printf("Running Optimized (%d steps x %d trials)...\n", n_steps, n_trials);
    auto optbr = bench("Optimized", n_trials, [&](int trial) {
        g_rng.seed(12345);
        NeuronState ns;
        ns.init(n_neurons);
        conn.csr_weights = weights_backup;
        g_rng.seed(42 + trial);
        auto r = run_optimized(ns, conn, n_steps, dt);
        return r.total_spikes;
    });

    // Results
    printf("\n%-12s  %10s  %10s  %10s  %10s  %12s\n",
           "Variant", "Mean(ms)", "Min(ms)", "Max(ms)", "Std(ms)", "Spikes");
    printf("%-12s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "Baseline", br.mean_ms, br.min_ms, br.max_ms, br.std_ms, br.total_spikes);
    printf("%-12s  %10.2f  %10.2f  %10.2f  %10.2f  %12d\n",
           "Optimized", optbr.mean_ms, optbr.min_ms, optbr.max_ms, optbr.std_ms, optbr.total_spikes);

    double speedup = br.min_ms / optbr.min_ms;
    double savings_pct = (1.0 - optbr.min_ms / br.min_ms) * 100.0;
    printf("\n--- Speedup (min times) ---\n");
    printf("Optimized: %.3fx (%+.1f%%)\n", speedup, savings_pct);

    double base_us = br.min_ms * 1000.0 / n_steps;
    double opt_us = optbr.min_ms * 1000.0 / n_steps;
    printf("\n--- Per-step time ---\n");
    printf("Baseline:  %.2f us/step (%.2f ns/neuron)\n", base_us, base_us * 1000.0 / n_neurons);
    printf("Optimized: %.2f us/step (%.2f ns/neuron)\n", opt_us, opt_us * 1000.0 / n_neurons);

    // Breakdown: where the savings come from
    printf("\n--- Optimization breakdown ---\n");
    printf("  #1 Persistent spike buf: eliminates %d malloc/free per trial\n", n_steps);
    printf("  #2 CSR activity record:  eliminates %d small vector allocs per trial\n", n_steps);
    printf("  #3 Flat STD lookups:     O(1) vs O(log n) for %d eligible neurons\n",
           (int)conn.std_eligible_set.size());
    printf("  #4 Scratch reuse:        eliminates %d allocs for input injection\n",
           (int)conn.input_steps.size());

    // Projected sweep savings
    double sim_base_s = base_us * 5000.0 * 1500.0 / 1e6;
    double sim_opt_s = opt_us * 5000.0 * 1500.0 / 1e6;
    printf("\n--- Projected sweep impact (1500 samples x 5000 steps) ---\n");
    printf("Baseline:  %.1f s per grid point\n", sim_base_s);
    printf("Optimized: %.1f s per grid point\n", sim_opt_s);
    printf("Savings:   %.1f s per point (%.1f%%)\n",
           sim_base_s - sim_opt_s, savings_pct);
    printf("300-point sweep: %.0f s (%.1f min) saved\n",
           (sim_base_s - sim_opt_s) * 300, (sim_base_s - sim_opt_s) * 300 / 60.0);

    return 0;
}
