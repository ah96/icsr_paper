# Adaptive Social Robot Explanations

This repository contains the supplementary material for the ICSR 2026 paper:

> **Adaptive Explanation Strategy Selection Under Uncertain Human Responses in Social Robotics**

The repository provides the simulation code, RDDL library-domain model, generated evaluation outputs, and user-study materials used for the camera-ready version of the paper. The central scenario is a social robot librarian that must decide how to communicate with library visitors under uncertain and user-dependent preferences.

## Recommended repository name

**`adaptive-social-robot-explanations`**

This name is more descriptive than `icsr_repository`, remains understandable outside the conference context, and captures the core contribution: adaptive explanation selection for social robots.

Alternative shorter names:

- `adaptive-robot-explanations`
- `social-robot-explanation-selection`
- `adaptive-explanation-selection`

## Repository structure

```text
.
├── domains/
│   ├── library_domain.rddl          # RDDL domain for the robot-librarian scenario
│   └── library_instance.rddl        # Example RDDL instance with two visitors and preference probabilities
├── eval/
│   ├── figures/                     # Camera-ready evaluation plots as PDF files
│   ├── results/                     # CSV files underlying the evaluation plots
│   └── run_metadata.csv             # Parameters used for the reported simulation run
├── user_study/
│   ├── Condition_1_Study.pdf        # User-study material for condition 1
│   └── Condition_2_Study.pdf        # User-study material for condition 2
└── simulate_icsr.py                 # Simulation script for adaptive explanation strategy selection
```

## Overview

The simulation evaluates explanation strategy selection under uncertain human responses. At each interaction step, the robot observes a situation and chooses one of three communication actions:

- **Explain**
- **Update**
- **Silent**

Human responses are modeled probabilistically. For each user type, the probability of a positive response depends on the current situation and the selected action. The adaptive policy uses Thompson sampling with Beta-Bernoulli belief updates to learn which communication action is most useful for a given situation.

The implemented situations are:

- **Delay**
- **Failure**
- **Reordering**

The evaluated policies are:

- Always Explain
- Always Update
- Always Silent
- Random
- Adaptive Thompson Sampling

The main output metrics are utility and cumulative regret.

## Requirements

The simulation script requires Python 3 and the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn
```

The RDDL files require an RDDL-compatible planner or simulator, such as PROST/RDDLSim, if you want to run the planning model directly.

## Running the simulation

From the repository root, run:

```bash
python3 simulate_icsr.py \
  --outdir eval \
  --T 300 \
  --runs 30 \
  --seed 42 \
  --switch_prob 0.03 \
  --switch_mode toggle \
  --dynamic_pair Type1_ExplanationOriented+Type2_Minimalist \
  --cost_explain 0.15 \
  --cost_update 0.05 \
  --cost_silent 0.00
```

This produces:

```text
eval/results/*.csv
eval/figures/*.pdf
eval/run_metadata.csv
```

The CSV files contain the mean and standard deviation over simulation runs for each policy. The PDF files visualize utility and cumulative regret for each user type.

## Simulation parameters

| Argument | Description | Default |
|---|---|---:|
| `--T` | Number of interaction steps per run | `300` |
| `--runs` | Number of simulation runs per user type | `30` |
| `--seed` | Base random seed | `42` |
| `--outdir` | Output directory | `out_icrs` |
| `--switch_prob` | Probability of switching user response model per step | `0.0` |
| `--switch_mode` | Switching mode for dynamic users: `toggle` or `random` | `toggle` |
| `--dynamic_pair` | Pair of user types used for dynamic switching | `Type1_ExplanationOriented+Type2_Minimalist` |
| `--cost_explain` | Communication cost for Explain | `0.15` |
| `--cost_update` | Communication cost for Update | `0.05` |
| `--cost_silent` | Communication cost for Silent | `0.00` |

## User types

The simulation includes three stationary user types and one optional dynamic user type:

1. **Type1_ExplanationOriented**: generally prefers explicit explanations.
2. **Type2_Minimalist**: generally prefers brief updates or no communication when possible.
3. **Type3_SituationSensitive**: preferences depend strongly on the interaction situation.
4. **Type4_DynamicSwitching**: optional non-stationary user model that switches between two user types when `--switch_prob > 0`.

## RDDL model

The `domains/` folder contains a robot-librarian planning model in RDDL. The model includes:

- a TIAGo-style robot,
- library visitors,
- books and waypoints,
- book-request and delivery actions,
- explanation-planning actions,
- probabilistic human explanation preferences over modality, scope, and detail.

The RDDL model is included as a formal domain representation of the paper scenario. The Python simulation in `simulate_icsr.py` is used for the reported adaptive strategy evaluation.

## User-study materials

The `user_study/` folder contains the two PDF materials used for the study conditions reported in the paper. These files are included to support transparency and reproducibility of the human-facing part of the work.

## Reproducing the camera-ready outputs

The checked-in `eval/` directory contains the outputs used for the camera-ready version:

- `eval/run_metadata.csv` records the simulation settings.
- `eval/results/` contains the numerical results as CSV files.
- `eval/figures/` contains the generated PDF plots.

To regenerate these outputs, run the command shown in [Running the simulation](#running-the-simulation). Existing files in `eval/` may be overwritten.

## Citation

If you use this repository, please cite the corresponding ICSR 2026 paper:

```bibtex
@inproceedings{halilovic2026adaptive,
  title     = {Adaptive Explanation Strategy Selection Under Uncertain Human Responses in Social Robotics},
  author    = {Halilovic, Amar and others},
  booktitle = {Proceedings of the International Conference on Social Robotics},
  year      = {2026}
}
```

Please update the BibTeX entry with the final author list, proceedings details, publisher, pages, and DOI once they are available.

## License

No license file is currently included. Before making the repository public, add an explicit license, for example MIT, BSD-3-Clause, or another license appropriate for the code and study materials.
