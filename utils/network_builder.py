"""
Self-contained network construction for liquid_state_machine_expanded.

Replicates the ring-zone tonotopic network construction from
liquid_state_machine_initial/lsm_xor_experiment_novel, including:
  - BASE_CONFIG: canonical parameter dict
  - Ring zone creation: surface shell, equatorial arc, K-overlap spectral mapping
  - Config B overrides: weight scaling, adaptation, NMDA rescaling
  - STD mask construction

No imports from liquid_state_machine_initial/.
"""

import numpy as np
import io
import contextlib
import sys
from pathlib import Path

# Import network class from local LIF_objects
EXPANDED_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPANDED_DIR))
from LIF_objects.SphericalNeuronalNetworkVectorized import SphericalNeuronalNetworkVectorized


# ============================================================
# BASE CONFIG — exact replica from lsm_BSA_simulation.py
# ============================================================
BASE_CONFIG = {
    # Network architecture
    'n_neurons': 1000,
    'inhibitory_fraction': 0.185,
    'connection_probabilities': {'ee': 0.10, 'ei': 0.15, 'ie': 0.25, 'ii': 0.15},
    'weight_scale': 0.55,
    'transmission_delay': 1.0,
    'distance_lambda': 0.18,
    'lambda_decay_ie': 0.15,
    'layout': 'sphere',

    # Noise
    'v_noise_amp': 0.1,
    'i_noise_amp': 0.001,
    'e_reversal': 0.0,
    'i_reversal': -80.0,

    # Jitter (heterogeneity — matching critical regime)
    'jitter_v_rest': 3.0,
    'jitter_v_threshold': 2.5,
    'jitter_tau_m': 0.3,
    'jitter_tau_ref': 0.1,
    'jitter_tau_e': 0.3,
    'jitter_tau_i': 0.3,
    'jitter_adaptation_increment': 0.4,
    'jitter_tau_adaptation': 0.35,

    # Simulation
    'dt': 0.1,
    'post_stimulus_ms': 500.0,
    'random_seed': 42,

    # Audio / stimulus
    'n_mel_bins': 128,
    'stimulus_current': 0.88,
    'audio_duration_ms': 800.0,
    'samples_per_digit': 5,
    'input_fan_out': 3,

    # Spatial zones
    'input_zone_fraction': 1 / 3,

    # Connectivity decay
    'lambda_connect': 0.0,
}

# Ring zone constants
SURFACE_SHELL_FRACTION = 0.15
RING_LATITUDE_BAND = 0.40
ARC_DEGREES = 300
ARC_GAP_CENTER_DEG = 180
OVERLAP_K = 5

# Config B constants
WEIGHT_MULT = 5.0
ADAPT_INC = 0.5
NMDA_TAU = 50.0
STD_U = 0.1
STD_TAU_REC = 500.0


# ============================================================
# LOW-LEVEL NETWORK CONSTRUCTION
# ============================================================

def _create_base_network(config):
    """Instantiate SphericalNeuronalNetworkVectorized from config dict.

    Replicates create_lsm_network() parameter passing from lsm_BSA_simulation.py.
    """
    kwargs = dict(
        n_neurons=config['n_neurons'],
        connection_p=0.1,
        connection_probabilities=config['connection_probabilities'],
        weight_scale=config['weight_scale'],
        spatial=True,
        transmission_delay=config['transmission_delay'],
        inhibitory_fraction=config['inhibitory_fraction'],
        layout=config['layout'],
        v_noise_amp=config['v_noise_amp'],
        i_noise_amp=config['i_noise_amp'],
        e_reversal=config['e_reversal'],
        i_reversal=config['i_reversal'],
        distance_lambda=config['distance_lambda'],
        lambda_decay_ie=config['lambda_decay_ie'],
        jitter_v_rest=config['jitter_v_rest'],
        jitter_v_threshold=config['jitter_v_threshold'],
        jitter_tau_m=config['jitter_tau_m'],
        jitter_tau_ref=config['jitter_tau_ref'],
        jitter_tau_e=config['jitter_tau_e'],
        jitter_tau_i=config['jitter_tau_i'],
        jitter_adaptation_increment=config['jitter_adaptation_increment'],
        jitter_tau_adaptation=config['jitter_tau_adaptation'],
    )
    if 'radius_multiplier' in config:
        kwargs['radius_multiplier'] = config['radius_multiplier']
    if 'lambda_connect' in config:
        kwargs['lambda_connect'] = config['lambda_connect']

    network = SphericalNeuronalNetworkVectorized(**kwargs)
    return network


# ============================================================
# RING ZONE CREATION
# Replicated from ring_input_diagnostic.py::create_ring_zone_network()
# ============================================================

def create_ring_zone_network(config, quiet=False, connectivity_regime='default'):
    """Build network, then reassign zones: tonotopic ring on surface, interior = reservoir.

    Parameters
    ----------
    connectivity_regime : str
        Controls surface-shell connectivity pruning:
        - 'default': Remove ALL reservoir→input + ALL intra-shell (current behavior)
        - 'nonarc_recurrent': Remove reservoir→arc only; keep reservoir↔non-arc
          bidirectional + non-arc↔non-arc local recurrence (arc isolated)
        - 'nonarc_reservoir_only': Remove reservoir→arc only; keep reservoir↔non-arc
          bidirectional; remove ALL intra-shell (non-arc loops through reservoir only)

    Returns:
        network: SphericalNeuronalNetworkVectorized instance
        zone_info: dict with zone indices and input neuron mapping
    """
    np.random.seed(config['random_seed'])

    with contextlib.redirect_stdout(io.StringIO()):
        network = _create_base_network(config)

    n = config['n_neurons']
    n_mel = config['n_mel_bins']
    radius = network.sphere_radius

    # Compute radial distance from origin for each neuron
    positions = np.array([network.neuron_3d_positions[i] for i in range(n)])
    radii = np.linalg.norm(positions, axis=1)

    # Surface shell: neurons with r > (1 - shell_frac) * radius
    shell_frac = config.get('surface_shell_fraction', SURFACE_SHELL_FRACTION)
    r_threshold = (1.0 - shell_frac) * radius
    surface_mask = radii >= r_threshold
    interior_mask = ~surface_mask

    # Equatorial band: constrain y-coordinate to narrow belt around equator
    y_coords = positions[:, 1]
    band_half = RING_LATITUDE_BAND * radius
    equatorial_mask = np.abs(y_coords) <= band_half

    # Ring candidates: on the surface AND in the equatorial band
    ring_candidate_mask = surface_mask & equatorial_mask
    ring_candidate_indices = np.where(ring_candidate_mask)[0]
    ring_excitatory = ring_candidate_indices[
        ~network.is_inhibitory[ring_candidate_indices]]

    # Compute azimuthal angle phi = atan2(z, x) for ring excitatory neurons
    phis = np.arctan2(positions[ring_excitatory, 2],
                      positions[ring_excitatory, 0])
    phis_deg = np.degrees(phis)

    # Define the arc: exclude a gap centered at ARC_GAP_CENTER_DEG
    gap_half = (360 - ARC_DEGREES) / 2
    gap_center = ARC_GAP_CENTER_DEG
    if gap_center > 180:
        gap_center -= 360

    angular_dist = np.abs(((phis_deg - gap_center + 180) % 360) - 180)
    on_arc = angular_dist > gap_half

    arc_neurons = ring_excitatory[on_arc]
    arc_phis = phis[on_arc]

    # Sort by azimuthal angle for tonotopic ordering
    sort_order = np.argsort(arc_phis)
    sorted_arc = arc_neurons[sort_order]
    sorted_arc_phis = arc_phis[sort_order]

    surface_indices = np.where(surface_mask)[0]

    if not quiet:
        print(f"  Surface shell (r > {r_threshold:.2f}): {surface_mask.sum()} neurons")
        print(f"  Equatorial band: {ring_candidate_mask.sum()} surface, "
              f"{len(ring_excitatory)} excitatory")
        print(f"    On arc ({ARC_DEGREES}°): {len(arc_neurons)}")
        print(f"  Interior: {interior_mask.sum()} neurons")

    # Overlapping spectral mapping: K nearest ring neurons per channel
    n_arc = len(sorted_arc)
    if n_arc == 0:
        # Degenerate case: no arc neurons (very small network or extreme params)
        input_neuron_mapping = {i: [] for i in range(n_mel)}
        input_neuron_indices = np.array([], dtype=np.int64)
    else:
        arc_phi_min = sorted_arc_phis[0]
        arc_phi_max = sorted_arc_phis[-1]
        channel_centers = np.linspace(arc_phi_min, arc_phi_max, n_mel)

        k = min(OVERLAP_K, n_arc)
        input_neuron_mapping = {}
        all_input_set = set()
        for mel_bin in range(n_mel):
            angular_dists = np.abs(sorted_arc_phis - channel_centers[mel_bin])
            nearest_k = np.argsort(angular_dists)[:k]
            mapped_neurons = sorted_arc[nearest_k].tolist()
            input_neuron_mapping[mel_bin] = [int(nn) for nn in mapped_neurons]
            all_input_set.update(mapped_neurons)

        input_neuron_indices = np.array(sorted(all_input_set), dtype=np.int64)

    # Input zone = all surface neurons; reservoir zone = all interior neurons
    input_zone_indices = surface_indices
    reservoir_zone_indices = np.where(interior_mask)[0]

    zone_info = {
        'input_zone_indices': input_zone_indices,
        'reservoir_zone_indices': reservoir_zone_indices,
        'input_neuron_mapping': input_neuron_mapping,
        'input_neuron_indices': input_neuron_indices,
        'y_threshold': float(r_threshold),
        'sphere_radius': float(radius),
    }

    # --- Connectivity pruning (regime-dependent) ---
    input_set = set(int(x) for x in input_zone_indices)
    reservoir_set = set(int(x) for x in reservoir_zone_indices)
    arc_set = set(int(x) for x in input_neuron_indices)
    nonarc_set = input_set - arc_set

    n_res_feedback_removed = 0
    n_res_nonarc_kept = 0
    n_intrashell_removed = 0
    n_intrashell_kept = 0

    if connectivity_regime == 'default':
        # Remove ALL reservoir→input zone feedback
        n_res_feedback_removed = int(np.count_nonzero(
            network.weights[np.ix_(reservoir_zone_indices, input_zone_indices)]))
        network.weights[np.ix_(reservoir_zone_indices, input_zone_indices)] = 0.0
        network.delays[np.ix_(reservoir_zone_indices, input_zone_indices)] = 0.0
        edges = [(u, v) for u, v in network.graph.edges()
                 if u in reservoir_set and v in input_set]
        network.graph.remove_edges_from(edges)

        # Remove ALL intra-shell
        n_intrashell_removed = int(np.count_nonzero(
            network.weights[np.ix_(input_zone_indices, input_zone_indices)]))
        network.weights[np.ix_(input_zone_indices, input_zone_indices)] = 0.0
        network.delays[np.ix_(input_zone_indices, input_zone_indices)] = 0.0
        edges = [(u, v) for u, v in network.graph.edges()
                 if u in input_set and v in input_set]
        network.graph.remove_edges_from(edges)

    elif connectivity_regime in ('nonarc_recurrent', 'nonarc_reservoir_only'):
        # Remove reservoir→ARC only (keep reservoir→non-arc bidirectional)
        if len(input_neuron_indices) > 0:
            n_res_feedback_removed = int(np.count_nonzero(
                network.weights[np.ix_(reservoir_zone_indices, input_neuron_indices)]))
            network.weights[np.ix_(reservoir_zone_indices, input_neuron_indices)] = 0.0
            network.delays[np.ix_(reservoir_zone_indices, input_neuron_indices)] = 0.0
        edges = [(u, v) for u, v in network.graph.edges()
                 if u in reservoir_set and v in arc_set]
        network.graph.remove_edges_from(edges)

        nonarc_arr = np.array(sorted(nonarc_set), dtype=np.int64)
        if len(nonarc_arr) > 0:
            n_res_nonarc_kept = int(np.count_nonzero(
                network.weights[np.ix_(reservoir_zone_indices, nonarc_arr)]))

        if connectivity_regime == 'nonarc_recurrent':
            # Isolate arc from all intra-shell; keep non-arc↔non-arc
            if len(input_neuron_indices) > 0:
                n_arc_out = int(np.count_nonzero(
                    network.weights[np.ix_(input_neuron_indices, input_zone_indices)]))
                network.weights[np.ix_(input_neuron_indices, input_zone_indices)] = 0.0
                network.delays[np.ix_(input_neuron_indices, input_zone_indices)] = 0.0
                # Second pass: zero input_zone→arc (arc→arc already zeroed above)
                n_to_arc = int(np.count_nonzero(
                    network.weights[np.ix_(input_zone_indices, input_neuron_indices)]))
                network.weights[np.ix_(input_zone_indices, input_neuron_indices)] = 0.0
                network.delays[np.ix_(input_zone_indices, input_neuron_indices)] = 0.0
                n_intrashell_removed = n_arc_out + n_to_arc
            if len(nonarc_arr) > 0:
                n_intrashell_kept = int(np.count_nonzero(
                    network.weights[np.ix_(nonarc_arr, nonarc_arr)]))
            edges = [(u, v) for u, v in network.graph.edges()
                     if u in input_set and v in input_set
                     and (u in arc_set or v in arc_set)]
            network.graph.remove_edges_from(edges)

        else:  # nonarc_reservoir_only
            # Remove ALL intra-shell (non-arc connects only to reservoir)
            n_intrashell_removed = int(np.count_nonzero(
                network.weights[np.ix_(input_zone_indices, input_zone_indices)]))
            network.weights[np.ix_(input_zone_indices, input_zone_indices)] = 0.0
            network.delays[np.ix_(input_zone_indices, input_zone_indices)] = 0.0
            edges = [(u, v) for u, v in network.graph.edges()
                     if u in input_set and v in input_set]
            network.graph.remove_edges_from(edges)

    else:
        raise ValueError(f"Unknown connectivity_regime: {connectivity_regime!r}")

    with contextlib.redirect_stdout(io.StringIO()):
        network._build_csr()

    zone_info['connectivity_regime'] = connectivity_regime

    if not quiet:
        n_feedforward = int(np.count_nonzero(
            network.weights[np.ix_(input_zone_indices, reservoir_zone_indices)]))
        print(f"  Connectivity regime: {connectivity_regime}")
        print(f"  Reservoir→arc feedback removed: {n_res_feedback_removed}")
        if n_res_nonarc_kept > 0:
            print(f"  Reservoir→non-arc kept: {n_res_nonarc_kept}")
        print(f"  Intra-shell removed: {n_intrashell_removed}")
        if n_intrashell_kept > 0:
            print(f"  Intra-shell kept (non-arc↔non-arc): {n_intrashell_kept}")
        print(f"  Shell→core feedforward: {n_feedforward}")

    return network, zone_info


# ============================================================
# CONFIG B OVERRIDES
# Replicated from input_penetration_analysis.py::build_v4_network()
# ============================================================

def apply_config_b_overrides(network, zone_info, config):
    """Apply Config B parameter overrides to an existing network.

    - Scale input→reservoir weights × WEIGHT_MULT
    - Ratio-preserving rescale adaptation_increment to mean ADAPT_INC
    - Ratio-preserving rescale NMDA tau to mean NMDA_TAU
    - Recompute outgoing connections + decay factors
    """
    # Scale input→reservoir weights
    if WEIGHT_MULT != 1.0:
        input_idx = zone_info['input_zone_indices']
        res_idx = zone_info['reservoir_zone_indices']
        submatrix = network.weights[np.ix_(input_idx, res_idx)]
        network.weights[np.ix_(input_idx, res_idx)] = submatrix * WEIGHT_MULT

        # Update graph edge weights too
        for i_pos, i in enumerate(input_idx):
            for j_pos, j in enumerate(res_idx):
                w = submatrix[i_pos, j_pos]
                if w != 0.0:
                    new_w = w * WEIGHT_MULT
                    if network.graph.has_edge(int(i), int(j)):
                        network.graph[int(i)][int(j)]['weight'] = new_w

    # Ratio-preserving adaptation rescale
    orig_adapt_mean = network.adaptation_increment.mean()
    if orig_adapt_mean > 1e-9:
        network.adaptation_increment *= (ADAPT_INC / orig_adapt_mean)
    else:
        network.adaptation_increment[:] = ADAPT_INC
    network.adaptation_increment = np.clip(network.adaptation_increment, 0.0, 1.0)

    # Ratio-preserving NMDA tau rescale
    orig_nmda_mean = network.tau_nmda.mean()
    if orig_nmda_mean > 1e-9:
        network.tau_nmda *= (NMDA_TAU / orig_nmda_mean)
    else:
        network.tau_nmda[:] = NMDA_TAU
    network.tau_nmda = np.clip(network.tau_nmda, 20.0, 300.0)

    # Recompute caches
    with contextlib.redirect_stdout(io.StringIO()):
        network._build_csr()
    network._precompute_decay_factors(config['dt'])


# ============================================================
# DYNAMICAL OVERRIDES (configurable version for Phase 3 sweep)
# ============================================================

def apply_dynamical_overrides(network, zone_info, config, *,
                               shell_core_mult=5.0,
                               core_core_mult=1.0,
                               feedback_scale=1.0,
                               adapt_inc=0.5,
                               adapt_tau=None,
                               nmda_tau=50.0):
    """Apply dynamical parameter overrides with explicit values.

    Configurable replacement for apply_config_b_overrides() used in
    Phase 3 dynamical parameter sweeps.

    Parameters
    ----------
    shell_core_mult : float
        Multiplier for shell→core (input→reservoir) weights.
    core_core_mult : float
        Multiplier for excitatory core→core (reservoir→reservoir) weights.
        Controls recurrence amplification for signal propagation through depth.
    feedback_scale : float
        Multiplier for reservoir→input (core→shell) feedback weights.
        Only has effect when connectivity_regime != 'default' (otherwise
        those weights are already zeroed). Scales the res→non-arc weights
        that survive pruning.
    adapt_inc : float
        Target mean for adaptation_increment (ratio-preserving rescale).
    adapt_tau : float, optional
        Target mean for tau_adaptation (ratio-preserving rescale).
        If None, tau_adaptation is left at its original jittered values.
    nmda_tau : float
        Target mean for tau_nmda (ratio-preserving rescale).
    """
    input_idx = zone_info['input_zone_indices']
    res_idx = zone_info['reservoir_zone_indices']
    is_exc = ~network.is_inhibitory

    # 1. Shell→core weight scaling
    if shell_core_mult != 1.0:
        submatrix = network.weights[np.ix_(input_idx, res_idx)]
        network.weights[np.ix_(input_idx, res_idx)] = submatrix * shell_core_mult
        for i_pos, i in enumerate(input_idx):
            for j_pos, j in enumerate(res_idx):
                w = submatrix[i_pos, j_pos]
                if w != 0.0:
                    if network.graph.has_edge(int(i), int(j)):
                        network.graph[int(i)][int(j)]['weight'] = w * shell_core_mult

    # 2. Core→core weight scaling (excitatory only)
    if core_core_mult != 1.0:
        cc_sub = network.weights[np.ix_(res_idx, res_idx)]
        # Build mask: source must be excitatory
        exc_mask_rows = is_exc[res_idx]  # bool array, True for excitatory core neurons
        scale_matrix = np.ones_like(cc_sub)
        scale_matrix[exc_mask_rows, :] = core_core_mult
        # Only scale nonzero weights
        nonzero = cc_sub != 0.0
        cc_sub[nonzero] *= scale_matrix[nonzero]
        network.weights[np.ix_(res_idx, res_idx)] = cc_sub
        # Update graph edges for excitatory core→core
        for i_pos, i in enumerate(res_idx):
            if not is_exc[int(i)]:
                continue
            for j_pos, j in enumerate(res_idx):
                orig_w = cc_sub[i_pos, j_pos] / core_core_mult if nonzero[i_pos, j_pos] else 0.0
                if orig_w != 0.0:
                    if network.graph.has_edge(int(i), int(j)):
                        network.graph[int(i)][int(j)]['weight'] = cc_sub[i_pos, j_pos]

    # 3. Core→shell feedback scaling (reservoir→input zone)
    if feedback_scale != 1.0:
        fb_sub = network.weights[np.ix_(res_idx, input_idx)]
        nonzero = fb_sub != 0.0
        if nonzero.any():
            fb_sub[nonzero] *= feedback_scale
            network.weights[np.ix_(res_idx, input_idx)] = fb_sub
            for i_pos, i in enumerate(res_idx):
                for j_pos, j in enumerate(input_idx):
                    if nonzero[i_pos, j_pos]:
                        if network.graph.has_edge(int(i), int(j)):
                            network.graph[int(i)][int(j)]['weight'] = fb_sub[i_pos, j_pos]

    # 4. Ratio-preserving adaptation increment rescale
    orig_adapt_mean = network.adaptation_increment.mean()
    if orig_adapt_mean > 1e-9:
        network.adaptation_increment *= (adapt_inc / orig_adapt_mean)
    else:
        network.adaptation_increment[:] = adapt_inc
    network.adaptation_increment = np.clip(network.adaptation_increment, 0.0, 5.0)

    # 5. Ratio-preserving adaptation tau rescale (optional)
    if adapt_tau is not None:
        orig_tau_mean = network.tau_adaptation.mean()
        if orig_tau_mean > 1e-9:
            network.tau_adaptation *= (adapt_tau / orig_tau_mean)
        else:
            network.tau_adaptation[:] = adapt_tau
        network.tau_adaptation = np.clip(network.tau_adaptation, 5.0, 10000.0)

    # 6. Ratio-preserving NMDA tau rescale
    orig_nmda_mean = network.tau_nmda.mean()
    if orig_nmda_mean > 1e-9:
        network.tau_nmda *= (nmda_tau / orig_nmda_mean)
    else:
        network.tau_nmda[:] = nmda_tau
    network.tau_nmda = np.clip(network.tau_nmda, 20.0, 300.0)

    # 7. Recompute caches
    with contextlib.redirect_stdout(io.StringIO()):
        network._build_csr()
    network._precompute_decay_factors(config['dt'])


# ============================================================
# STD MASK CONSTRUCTION
# Replicated from std_sanity_check.py::build_std_masks()
# ============================================================

def build_std_masks(network, zone_info):
    """Precompute per-neuron masks for recurrent-only STD.

    Returns:
        std_eligible: set of neuron IDs eligible for STD
            (excitatory neurons in the reservoir)
        recurrent_masks: dict {nid: bool array} — True for outgoing
            connections targeting reservoir neurons.
    """
    reservoir_set = set(int(x) for x in zone_info['reservoir_zone_indices'])
    is_exc = ~network.is_inhibitory

    std_eligible = set()
    recurrent_masks = {}

    for nid in reservoir_set:
        if not is_exc[nid]:
            continue
        std_eligible.add(nid)
        start, end = network._csr_indptr[nid], network._csr_indptr[nid + 1]
        targets = network._csr_targets[start:end]
        if len(targets) > 0:
            mask = np.array([int(t) in reservoir_set for t in targets])
            if mask.any():
                recurrent_masks[nid] = mask

    return std_eligible, recurrent_masks


# ============================================================
# STD-AWARE SIMULATION
# Replicated from std_sanity_check.py::run_sample_with_std()
# ============================================================

def run_sample_with_std(network, sample, zone_info, config,
                        std_u=0.0, std_tau_rec=200.0,
                        std_eligible=None, recurrent_masks=None,
                        record_vm_every_ms=None,
                        record_adapt_at_ms=None):
    """Run one sample with optional short-term synaptic depression.

    When record_vm_every_ms is set, snapshots network.v (membrane voltages)
    at that interval. Returns (activity_record, vm_snapshots) instead of
    just activity_record.  vm_snapshots is a list of 1-D numpy arrays,
    one per snapshot, each of length n_neurons.

    STD model (Tsodyks & Markram 1997, simplified per-neuron):
      x_i ∈ [0, 1] = available vesicle fraction (starts at 1.0)
      On spike from neuron i:
        x_i *= (1 - U)          # depletion
      Between spikes:
        x_i recovers toward 1.0 with time constant τ_rec

    STD is applied ONLY to excitatory recurrent connections within the
    reservoir (E→E and E→I where source ∈ reservoir). Input→reservoir
    synapses are left untouched.

    Implementation: lazy recovery — only recompute x when a neuron spikes
    again. Depression state is baked into outgoing weights at spike emission
    time, so the correct depleted weight travels through the delay buffer.
    Only the recurrent portion of outgoing weights is scaled; input and
    feedforward weights remain at their original values.

    When std_u == 0.0, no STD is applied (baseline).
    """
    network.reset_all()
    use_std = std_u > 0.0 and std_eligible is not None

    dt = config['dt']
    total_duration_ms = config['audio_duration_ms'] + config['post_stimulus_ms']
    n_steps = int(total_duration_ms / dt)
    stim_current = config['stimulus_current']
    input_mapping = zone_info['input_neuron_mapping']

    spike_times = sample['spike_times_ms']
    freq_bins = sample['freq_bin_indices']
    spike_steps = np.clip((spike_times / dt).astype(np.int64), 0, n_steps - 1)

    step_to_spike_indices = {}
    for i, step in enumerate(spike_steps):
        step = int(step)
        if step not in step_to_spike_indices:
            step_to_spike_indices[step] = []
        step_to_spike_indices[step].append(i)

    # --- V_m recording ---
    record_vm = record_vm_every_ms is not None
    vm_snapshots = []
    if record_vm:
        vm_interval_steps = max(1, int(record_vm_every_ms / dt))
        next_vm_step = 0

    # --- Adaptation snapshot at specific time ---
    record_adapt = record_adapt_at_ms is not None
    adapt_snapshot = None
    if record_adapt:
        adapt_snapshot_step = max(0, int(record_adapt_at_ms / dt))
        adapt_recorded = False

    # --- STD state ---
    n_neurons = network.n_neurons  # Use actual network size (may be compacted)
    if use_std:
        x_resource = np.ones(n_neurons)
        last_update_time = np.zeros(n_neurons)
        original_outgoing_weights = {}  # only store for eligible neurons
        for nid in recurrent_masks:
            start, end = network._csr_indptr[nid], network._csr_indptr[nid + 1]
            original_outgoing_weights[nid] = network._csr_weights[start:end].copy()
        dirty_neurons = set()

    activity_record = []
    for step in range(n_steps):
        current_time_ms = step * dt

        # Inject BSA input
        if step in step_to_spike_indices:
            spike_idx = step_to_spike_indices[step]
            active_freq_bins = freq_bins[spike_idx]
            neuron_indices = np.array([idx for fb in active_freq_bins
                                       for idx in input_mapping[int(fb)]])
            if len(neuron_indices) > 0:
                currents = np.full(len(neuron_indices), stim_current)
                network.stimulate_neurons(neuron_indices, currents)

        active = network.update_network(dt)
        activity_record.append(list(active))

        # Snapshot membrane voltages at regular intervals
        if record_vm and step >= next_vm_step:
            vm_snapshots.append(network.v.copy())
            next_vm_step += vm_interval_steps

        # Snapshot adaptation conductance at specified time
        if record_adapt and not adapt_recorded and step >= adapt_snapshot_step:
            adapt_snapshot = network.adaptation.copy()
            adapt_recorded = True

        if use_std and len(active) > 0:
            for nid in active:
                if nid not in std_eligible:
                    continue  # skip input neurons and inhibitory neurons

                mask = recurrent_masks.get(nid)
                if mask is None:
                    continue  # no recurrent outgoing connections

                # Lazy recovery since last update
                delta_t = current_time_ms - last_update_time[nid]
                if delta_t > 0 and x_resource[nid] < 1.0:
                    x_resource[nid] = 1.0 - (1.0 - x_resource[nid]) * np.exp(
                        -delta_t / std_tau_rec)
                    x_resource[nid] = min(x_resource[nid], 1.0)

                # Deplete
                x_resource[nid] *= (1.0 - std_u)
                last_update_time[nid] = current_time_ms

                # Scale ONLY recurrent outgoing weights; leave the rest intact
                orig = original_outgoing_weights[nid]
                w = orig.copy()
                w[mask] = orig[mask] * x_resource[nid]
                start, end = network._csr_indptr[nid], network._csr_indptr[nid + 1]
                network._csr_weights[start:end] = w
                dirty_neurons.add(nid)

    if use_std:
        for nid in dirty_neurons:
            start, end = network._csr_indptr[nid], network._csr_indptr[nid + 1]
            network._csr_weights[start:end] = original_outgoing_weights[nid].copy()

    if record_vm and record_adapt:
        return activity_record, vm_snapshots, adapt_snapshot
    elif record_vm:
        return activity_record, vm_snapshots
    elif record_adapt:
        return activity_record, adapt_snapshot
    return activity_record


# ============================================================
# NETWORK COMPACTION — remove non-arc input zone neurons
# ============================================================

# Every per-neuron array that must be sliced during compaction.
_PER_NEURON_ARRAYS = [
    # Neuron classification
    'is_inhibitory', 'is_slow_inhibitory',
    # Biophysical parameters
    'v_rest', 'v_threshold', 'v_reset',
    'tau_m', 'tau_ref', 'tau_e', 'tau_i', 'tau_i_slow', 'tau_nmda',
    'adaptation_increment', 'tau_adaptation',
    # Reversal potentials
    'e_reversal_arr', 'i_reversal_arr', 'k_reversal',
    # Noise
    'v_noise_amp_arr', 'i_noise_amp_arr',
    # Dynamic state
    'v', 'g_e', 'g_i', 'g_i_slow', 'g_nmda', 'adaptation', 't_since_spike',
    # Precomputed decay factors
    '_exp_decay_e', '_exp_decay_i', '_exp_decay_i_slow',
    '_exp_decay_adapt', '_exp_decay_nmda',
]


def compact_network(network, zone_info, config, quiet=False):
    """Remove non-arc input zone neurons, shrinking the network in-place.

    In the default connectivity regime the 272 non-arc surface neurons are
    completely inert (no input, no recurrence, no feedback).  Rather than
    simulating 1000 neurons with 272 dead ones, this function physically
    removes them — shrinking weight/delay matrices, per-neuron arrays,
    the graph, and all precomputed caches.

    Must be called AFTER dynamical overrides (Config B weight scaling etc.)
    since those reference the original indices.

    Returns the same network object (modified in-place) and a new zone_info
    with remapped indices.
    """
    arc_set = set(int(x) for x in zone_info['input_neuron_indices'])
    input_zone_set = set(int(x) for x in zone_info['input_zone_indices'])
    nonarc_set = input_zone_set - arc_set

    if len(nonarc_set) == 0:
        return network, zone_info

    n_old = network.n_neurons
    keep = sorted(set(range(n_old)) - nonarc_set)
    n_new = len(keep)
    keep_arr = np.array(keep, dtype=np.int64)
    old_to_new = {old: new for new, old in enumerate(keep)}

    if not quiet:
        print(f"  Compacting: {n_old} → {n_new} neurons "
              f"({len(nonarc_set)} non-arc removed)")

    # 1. Slice all per-neuron arrays
    for attr_name in _PER_NEURON_ARRAYS:
        arr = getattr(network, attr_name, None)
        if arr is not None and hasattr(arr, '__len__') and len(arr) == n_old:
            setattr(network, attr_name, arr[keep_arr])

    # 2. Slice weight and delay matrices
    network.weights = network.weights[np.ix_(keep_arr, keep_arr)]
    network.delays = network.delays[np.ix_(keep_arr, keep_arr)]

    # 3. Remap 3D positions
    new_positions = {}
    for new_idx, old_idx in enumerate(keep):
        new_positions[new_idx] = network.neuron_3d_positions[old_idx]
    network.neuron_3d_positions = new_positions

    # 4. Rebuild the networkx graph with new indices
    import networkx as nx
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(range(n_new))
    for new_i in range(n_new):
        for new_j in range(n_new):
            w = network.weights[new_i, new_j]
            if w != 0.0:
                new_graph.add_edge(new_i, new_j, weight=float(w))
    network.graph = new_graph

    # 5. Update n_neurons
    network.n_neurons = n_new

    # 6. Update neuron_params dict (if it exists)
    if hasattr(network, 'neuron_params') and network.neuron_params is not None:
        for key in list(network.neuron_params.keys()):
            arr = network.neuron_params[key]
            if hasattr(arr, '__len__') and len(arr) == n_old:
                network.neuron_params[key] = arr[keep_arr]

    # 7. Clear ring buffer state (will re-init on next update_network)
    network._ring_initialized = False

    # 8. Recompute CSR outgoing connections and decay factors
    with contextlib.redirect_stdout(io.StringIO()):
        network._build_csr()
    network._precompute_decay_factors(config['dt'])

    # 9. Remap zone_info indices
    def _remap(arr):
        return np.array([old_to_new[int(x)] for x in arr if int(x) in old_to_new],
                        dtype=np.int64)

    new_zone_info = {
        'input_zone_indices': _remap(zone_info['input_neuron_indices']),
        'reservoir_zone_indices': _remap(zone_info['reservoir_zone_indices']),
        'input_neuron_indices': _remap(zone_info['input_neuron_indices']),
        'y_threshold': zone_info['y_threshold'],
        'sphere_radius': zone_info['sphere_radius'],
        'connectivity_regime': zone_info.get('connectivity_regime', 'default'),
    }

    # Remap input_neuron_mapping (mel bin → list of neuron IDs)
    new_mapping = {}
    for mel_bin, old_ids in zone_info['input_neuron_mapping'].items():
        new_ids = [old_to_new[int(nid)] for nid in old_ids if int(nid) in old_to_new]
        new_mapping[mel_bin] = new_ids
    new_zone_info['input_neuron_mapping'] = new_mapping

    if not quiet:
        n_ff = int(np.count_nonzero(network.weights[
            np.ix_(new_zone_info['input_zone_indices'],
                   new_zone_info['reservoir_zone_indices'])]))
        n_nonzero = int(np.count_nonzero(network.weights))
        print(f"  Post-compact: {n_new} neurons, {n_nonzero} nonzero weights, "
              f"{n_ff} feedforward")

    return network, new_zone_info


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def build_full_network(config, quiet=False, dynamical_overrides=None,
                       connectivity_regime='default', remove_nonarc=False):
    """Build ring-zone network with dynamical overrides applied.

    Parameters
    ----------
    dynamical_overrides : dict, optional
        If provided, call apply_dynamical_overrides() with these kwargs
        instead of apply_config_b_overrides(). Keys: shell_core_mult,
        core_core_mult, adapt_inc, nmda_tau.
    connectivity_regime : str
        Surface shell pruning regime (see create_ring_zone_network).
    remove_nonarc : bool
        If True, physically remove non-arc input zone neurons after
        applying overrides, shrinking the network from 1000 to ~728.

    Returns:
        network: SphericalNeuronalNetworkVectorized instance
        zone_info: dict with zone indices and input neuron mapping
    """
    network, zone_info = create_ring_zone_network(
        config, quiet=quiet, connectivity_regime=connectivity_regime)
    if dynamical_overrides is not None:
        apply_dynamical_overrides(network, zone_info, config, **dynamical_overrides)
    else:
        apply_config_b_overrides(network, zone_info, config)

    if remove_nonarc:
        network, zone_info = compact_network(network, zone_info, config, quiet=quiet)

    return network, zone_info
