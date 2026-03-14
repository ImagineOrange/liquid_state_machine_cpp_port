#include "experiments.h"

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // Parse args
    std::string arms = "all";
    int n_workers = 8;
    std::string output_dir = "";

    std::string snapshot_path = "";
    bool no_snapshot = false;
    bool verify_only = false;
    std::string verify_output = "";
    std::string data_dir_override = "";
    int trace_neuron = -1;
    int trace_sample = 0;
    std::string trace_output = "";
    std::string trace_file = "";
    bool no_noise = false;
    bool no_input_nmda = false;
    double stim_current_override = -1.0;
    double input_tau_e_override = -1.0;
    double input_adapt_inc_override = -1.0;
    double input_std_u_override = -1.0;
    double adapt_inc_override = -1.0;
    double adapt_tau_override = -1.0;
    double tonic_conductance_override = -1.0;
    bool input_grid = false;
    std::string input_grid_output = "input_grid_results.csv";
    bool mi_refine = false;
    std::string mi_refine_input = "";
    int mi_refine_top = 50;
    int mi_refine_samples = 20;
    std::string mi_refine_output = "";
    std::string raster_dump = "";  // output dir for raster dump
    bool wm_sweep = false;
    bool serial_sweep = false;
    bool noisy_sweep = false;
    std::string noisy_taus_str = "";
    bool noisy_per_bin = false;
    bool mech_interp = false;
    bool mech_raster = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--arms" && i + 1 < argc) arms = argv[++i];
        else if (std::string(argv[i]) == "--n-workers" && i + 1 < argc) n_workers = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (std::string(argv[i]) == "--snapshot" && i + 1 < argc) snapshot_path = argv[++i];
        else if (std::string(argv[i]) == "--no-snapshot") no_snapshot = true;
        else if (std::string(argv[i]) == "--verify-only") verify_only = true;
        else if (std::string(argv[i]) == "--verify-output" && i + 1 < argc) verify_output = argv[++i];
        else if (std::string(argv[i]) == "--samples-per-digit" && i + 1 < argc) SAMPLES_PER_DIGIT = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--data-dir" && i + 1 < argc) data_dir_override = argv[++i];
        else if (std::string(argv[i]) == "--trace-neuron" && i + 1 < argc) trace_neuron = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--trace-sample" && i + 1 < argc) trace_sample = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--trace-output" && i + 1 < argc) trace_output = argv[++i];
        else if (std::string(argv[i]) == "--trace-file" && i + 1 < argc) trace_file = argv[++i];
        else if (std::string(argv[i]) == "--no-noise") no_noise = true;
        else if (std::string(argv[i]) == "--no-input-nmda") no_input_nmda = true;
        else if (std::string(argv[i]) == "--stim-current" && i + 1 < argc) stim_current_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--input-grid") input_grid = true;
        else if (std::string(argv[i]) == "--input-grid-output" && i + 1 < argc) input_grid_output = argv[++i];
        else if (std::string(argv[i]) == "--input-tau-e" && i + 1 < argc) input_tau_e_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--input-adapt-inc" && i + 1 < argc) input_adapt_inc_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--input-std-u" && i + 1 < argc) input_std_u_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--mi-refine") mi_refine = true;
        else if (std::string(argv[i]) == "--mi-refine-input" && i + 1 < argc) mi_refine_input = argv[++i];
        else if (std::string(argv[i]) == "--mi-refine-top" && i + 1 < argc) mi_refine_top = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--mi-refine-samples" && i + 1 < argc) mi_refine_samples = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--mi-refine-output" && i + 1 < argc) mi_refine_output = argv[++i];
        else if (std::string(argv[i]) == "--raster-dump" && i + 1 < argc) raster_dump = argv[++i];
        else if (std::string(argv[i]) == "--adapt-inc" && i + 1 < argc) adapt_inc_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--adapt-tau" && i + 1 < argc) adapt_tau_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--tonic-conductance" && i + 1 < argc) tonic_conductance_override = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--wm-sweep") wm_sweep = true;
        else if (std::string(argv[i]) == "--serial-sweep") serial_sweep = true;
        else if (std::string(argv[i]) == "--noisy-sweep") noisy_sweep = true;
        else if (std::string(argv[i]) == "--noisy-taus" && i + 1 < argc) noisy_taus_str = argv[++i];
        else if (std::string(argv[i]) == "--noisy-per-bin") noisy_per_bin = true;
        else if (std::string(argv[i]) == "--mech-interp") mech_interp = true;
        else if (std::string(argv[i]) == "--mech-raster") mech_raster = true;
    }

    // Default: use network_snapshot.npz next to the binary if it exists
    if (snapshot_path.empty() && !no_snapshot) {
        fs::path default_snap = fs::path(argv[0]).parent_path() / "network_snapshot.npz";
        if (fs::exists(default_snap)) {
            snapshot_path = default_snap.string();
            printf("  Using default snapshot: %s\n", snapshot_path.c_str());
        }
    }
    g_snapshot_path = snapshot_path;

    #ifdef _OPENMP
    omp_set_num_threads(n_workers);
    #endif

    // Resolve paths
    fs::path exe_dir = fs::path(argv[0]).parent_path();
    fs::path base_dir = exe_dir.parent_path();
    if (base_dir.empty()) base_dir = ".";
    std::string data_dir = data_dir_override.empty()
        ? (base_dir / "data").string() : data_dir_override;

    // --raster-dump mode
    if (!raster_dump.empty()) {
        if (trace_file.empty()) {
            fprintf(stderr, "ERROR: --raster-dump requires --trace-file <audio.npz>\n");
            return 1;
        }
        fs::create_directories(raster_dump);
        return run_raster_dump(g_snapshot_path, trace_file, raster_dump,
                               stim_current_override, input_tau_e_override,
                               input_adapt_inc_override,
                               adapt_inc_override, adapt_tau_override,
                               tonic_conductance_override);
    }

    // --input-grid mode
    if (input_grid) {
        if (input_grid_output == "input_grid_results.csv") {
            fs::path ig_out_dir = exe_dir / "results" / "input_grid_search";
            fs::create_directories(ig_out_dir);
            input_grid_output = (ig_out_dir / "input_grid_results.csv").string();
        }
        return run_input_grid(argc, argv, g_snapshot_path, data_dir,
                               n_workers, input_grid_output);
    }

    // --mi-refine mode
    if (mi_refine) {
        if (mi_refine_input.empty()) {
            mi_refine_input = (exe_dir / "results" / "input_grid_search" / "input_grid_results.csv").string();
        }
        if (mi_refine_output.empty()) {
            fs::path mr_out_dir = exe_dir / "results" / "input_grid_search";
            fs::create_directories(mr_out_dir);
            mi_refine_output = (mr_out_dir / "mi_refine_top50.csv").string();
        }
        return run_mi_refine(argc, argv, g_snapshot_path, data_dir,
                              n_workers, mi_refine_input, mi_refine_top,
                              mi_refine_samples, mi_refine_output);
    }

    // --serial-sweep mode
    if (serial_sweep) {
        if (output_dir.empty()) {
            output_dir = (base_dir / "results" / "serial_classification_adaptation_sweep").string();
        }
        fs::create_directories(output_dir);
        return run_serial_sweep(argc, argv, n_workers, output_dir, data_dir);
    }

    // --noisy-sweep mode
    if (noisy_sweep) {
        if (output_dir.empty()) {
            output_dir = (base_dir / "results" / "noisy_classification_sweep").string();
        }
        fs::create_directories(output_dir);
        // Parse tau values: default pilot (5000 only), or comma-separated
        std::vector<double> tau_values = {5000.0};
        if (!noisy_taus_str.empty()) {
            tau_values.clear();
            std::istringstream iss(noisy_taus_str);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                tau_values.push_back(std::atof(tok.c_str()));
            }
        }
        return run_noisy_sweep(argc, argv, n_workers, output_dir, data_dir, tau_values, noisy_per_bin);
    }

    // --mech-raster mode (quick: 2 trials only)
    if (mech_raster) {
        if (output_dir.empty()) {
            output_dir = (base_dir / "results" / "mechanistic_interp").string();
        }
        fs::create_directories(output_dir);
        return run_mech_raster(n_workers, output_dir, data_dir);
    }

    // --mech-interp mode
    if (mech_interp) {
        if (output_dir.empty()) {
            output_dir = (base_dir / "results" / "mechanistic_interp").string();
        }
        fs::create_directories(output_dir);
        return run_mechanistic_interp(argc, argv, n_workers, output_dir, data_dir);
    }

    // --wm-sweep mode
    if (wm_sweep) {
        if (output_dir.empty()) {
            output_dir = (base_dir / "results" / "wm_adaptation_sweep").string();
        }
        fs::create_directories(output_dir);
        return run_wm_sweep(argc, argv, n_workers, output_dir, data_dir);
    }

    // Default: classification adaptation sweep
    if (output_dir.empty()) {
        output_dir = (base_dir / "results" / "classification_adaptation_sweep").string();
    }
    fs::create_directories(output_dir);

    return run_classification_sweep(argc, argv, arms, n_workers, output_dir, data_dir,
                                     verify_only, verify_output,
                                     trace_neuron, trace_sample, trace_output, trace_file,
                                     no_noise, no_input_nmda,
                                     stim_current_override, input_tau_e_override,
                                     input_adapt_inc_override, input_std_u_override);
}
