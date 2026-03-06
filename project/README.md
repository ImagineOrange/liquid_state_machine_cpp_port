# Classification Adaptation Sweep

Runs the full 159-point grid for 5-class spoken digit classification across the SFA (spike-frequency adaptation) parameter space.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

## Directory structure

```
cls_sweep_package/
├── README.md
├── requirements.txt
├── experiments/
│   └── classification_adaptation_sweep.py    # main sweep script
├── utils/
│   ├── __init__.py
│   ├── network_builder.py                    # builds the reservoir network
│   └── simulation_utils.py                   # audio loading, stats helpers
├── LIF_objects/
│   ├── __init__.py
│   └── SphericalNeuronalNetworkVectorized.py # core spiking network simulator
├── data/
│   ├── preprocessing_config_bsa.json
│   └── spike_trains_bsa/                     # ~7 GB, 3000 .npz files (BSA-encoded audio)
└── results/
    └── classification_adaptation_sweep/      # output dir (created automatically)
```

## Run

```bash
python experiments/classification_adaptation_sweep.py \
  --arms all \
  --n-workers 4
```

- `--n-workers N` — number of parallel CPU workers (use ~half your cores)

Progress is checkpointed after every grid point, so you can kill and resume safely:

```bash
python experiments/classification_adaptation_sweep.py \
  --arms all \
  --n-workers 4 \
  --cached-calibration \
  --resume results/classification_adaptation_sweep/classification_adaptation_sweep_checkpoint.json
```

## Output

Results are written to `results/classification_adaptation_sweep/classification_adaptation_sweep_checkpoint.json`. Send this file back when done.

## Estimated time

~5-10 min per grid point at 4 workers. 159 points total ≈ 15-25 hours depending on CPU.
