# MultiscaleDynamicsAI

Learning multiscale dynamics for AI in science and engineering.

## 1. Experiments and evaluation

### 1.a. Panda pilot

The `panda_pilot` data-generation run creates 32 reproducible trajectory pairs using seeds 0 through 31. Each seed produces one noise-free single-scale trajectory, one noise-free multiscale trajectory, and its fitted closure. The single-scale and multiscale trajectories within a seed start from the same slow state.

With the `lorenz96` Conda environment active, run from the repository root:

```bash
./scripts/panda_pilot.sh
```

The script runs the seeds sequentially and prints a start and completion message for each successful run. Outputs are organized as:

```text
data/panda_pilot/
├── seed_0/
│   ├── closure.joblib
│   ├── simulation_data_singlescale.npz
│   └── simulation_data_multiscale.npz
├── seed_1/
│   └── ...
└── seed_31/
    ├── closure.joblib
    ├── simulation_data_singlescale.npz
    └── simulation_data_multiscale.npz
```

This produces 32 matched pairs, or 64 trajectories in total. Process noise is explicitly set to zero in the batch script; the remaining generator settings use their defaults.

## 2. Core modules

### 2.a. Lorenz–96 data generator

The Lorenz–96 utility generates two trajectories per run:

- A learned single-scale trajectory containing the `K` slow variables.
- A full multiscale trajectory containing all `K` slow variables and all `K * J` fast variables.

The utility first spins up the full multiscale system. By default, it then generates a multiscale learning trajectory, fits a Gaussian-process closure, and uses that closure for the single-scale model. Both final trajectories start from the same slow state.

#### 2.a.i. Environment

Create and activate the Conda environment from the repository root:

```bash
conda env create --file environments/lorenz96.yml
conda activate lorenz96
```

#### 2.a.ii. Generate data

Run the generator with its defaults:

```bash
python scripts/generate_lorenz96.py --output-dir data/lorenz96
```

Process noise is disabled by default. To add independent Gaussian process noise with standard deviation `0.001` after every sampled transition, run:

```bash
python scripts/generate_lorenz96.py \
    --process-noise 0.001 \
    --output-dir data/lorenz96_noisy
```

Each run writes:

```text
closure.joblib
simulation_data_singlescale.npz
simulation_data_multiscale.npz
```

Each `.npz` file contains one array named `states`. With the defaults `K=9`, `J=8`, `dynamics-duration=50`, and `sampling-interval=0.001`, the shapes are:

```text
single-scale states: (9, 50000)
multiscale states:   (81, 50000)
```

The multiscale state layout is:

```text
states[:K, :]   slow variables
states[K:, :]   fast variables, grouped by slow-variable index
```

Load and separate the arrays with:

```python
import numpy as np

data = np.load("data/lorenz96/simulation_data_multiscale.npz")
states = data["states"]

K = 9
J = 8
slow_states = states[:K]
fast_states = states[K:].reshape(K, J, -1)
```

#### 2.a.iii. Important parameters

Model parameters:

```text
--K                 number of slow variables; default 9
--J                 fast variables per slow variable; default 8
--hx                fast-to-slow coupling; default -0.8
--hy                slow-to-fast coupling; default 1.0
--forcing           slow-variable forcing; default 10.0
--epsilon           fast/slow time-scale separation; default 2**-7
```

Integration and sampling parameters:

```text
--spinup-duration       spin-up duration; default 50
--learning-duration     closure-learning duration; default 300
--dynamics-duration     final trajectory duration; default 50
--sampling-interval     time between saved states; default 0.001
--spinup-max-step       maximum solver step during spin-up; default 0.01
--learning-max-step     maximum solver step during learning; default 0.001
--dynamics-max-step     maximum solver step during final generation;
                        defaults to sampling-interval when omitted
--solver-method         scipy.solve_ivp method; default RK45
```

Other commonly used parameters:

```text
--seed                  NumPy random seed; default 42
--process-noise         process-noise standard deviation; default 0
--stencil-left          left closure-stencil offset; default 0
--stencil-right         right closure-stencil offset; default 0
--closure-sample-size   maximum GP training pairs; default 800
```

List every available option with:

```bash
python scripts/generate_lorenz96.py --help
```

#### 2.a.iv. Reuse an existing closure

To skip closure training and load a fitted closure:

```bash
python scripts/generate_lorenz96.py \
    --skip-closure-training \
    --closure-path data/lorenz96/closure.joblib \
    --output-dir data/lorenz96_reused_closure
```

When closure training is skipped, generation begins from the end of spin-up. When a new closure is trained, generation begins from the end of the learning trajectory.
