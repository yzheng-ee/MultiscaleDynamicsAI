# MultiscaleDynamicsAI

Learning multiscale dynamics for AI in science and engineering.

## 1. Experiments and evaluation

### 1.a. Panda pilot

The Panda pilot evaluates the pretrained [Panda](https://github.com/abao1999/panda)
forecasting model on matched single-scale and multiscale Lorenz–96 trajectories.

#### 1.a.i. Data preparation

Data preparation has three sequential stages:

1. Generate densely sampled trajectories. These long, fine-resolution
   trajectories provide enough temporal information to estimate a
   characteristic period reliably; they are intermediate data rather than the
   final evaluation inputs.
2. Estimate a sampling interval separately for each dense single-scale and
   multiscale trajectory. The interval normalizes the final time grid by the
   trajectory's characteristic period.
3. Generate the target datasets using the estimated intervals. These
   target-sampled trajectories are the inputs used by the Panda evaluation.

Each epsilon value uses seeds 0 through 31. For every seed, dense generation
produces a noise-free single-scale trajectory, a noise-free multiscale
trajectory, and a fitted closure. The paired trajectories begin from the same
slow state. Target generation reuses that closure and the two estimated
sampling intervals.

Create and activate the unified environment from the repository root:

```bash
conda env create --file environments/panda_pilot.yml
conda activate panda_pilot
```

The standalone `timescales` environment uses `dysts==0.96`, whereas
`panda_pilot` uses `dysts==0.95` to match the version resolved by the original
Panda project. The timescale APIs used here are unchanged between these
versions, so this version difference does not affect the implementation or the
experimental results.

The entire pipeline can be launched with one command:

```bash
./scripts/panda_pilot/generate_panda_pilot.sh
```

This convenience script runs all three stages serially and stops if a stage
fails. It is generally not recommended for routine use because dense and
target generation are both computationally expensive, and the script processes
every epsilon value and seed sequentially in one long run.

For better control, run the stages separately. Dense generation is usually
best run one epsilon value at a time, for example:

```bash
./scripts/panda_pilot/generate_lorenz96_dense/generate_lorenz96_dense_eps_2neg7.sh
```

Each epsilon-specific script runs its 32 seeds sequentially. The aggregate
dense-generation script is also available when a single long run is desired:

```bash
./scripts/panda_pilot/generate_lorenz96_dense/generate_lorenz96_dense.sh
```

Dense outputs have the following structure:

```text
data/panda_pilot/lorenz96_dense/
├── eps_2neg5/
│   ├── seed_0/
│   │   ├── closure.joblib
│   │   ├── simulation_data_multiscale.npz
│   │   └── simulation_data_singlescale.npz
│   └── ...
├── eps_2neg6/
│   └── ...
└── eps_2neg10/
    └── ...
```

The sampling-interval JSON files are added in the second stage. Estimation is
relatively fast, so running it for all epsilon values at once is normally
reasonable:

```bash
./scripts/panda_pilot/estimate_lorenz96_sampling_interval/estimate_lorenz96_sampling_intervals.sh
```

This stage adds the following files to every corresponding dense seed
directory:

```text
sampling_interval_multiscale.json
sampling_interval_singlescale.json
```

Target generation is also usually best run one epsilon value at a time:

```bash
./scripts/panda_pilot/generate_lorenz96_target/generate_lorenz96_target_eps_2neg7.sh
```

To process every epsilon value serially instead, use:

```bash
./scripts/panda_pilot/generate_lorenz96_target/generate_lorenz96_target.sh
```

Target outputs follow a parallel structure:

```text
data/panda_pilot/lorenz96_target/
├── eps_2neg5/
│   ├── seed_0/
│   │   ├── simulation_data_multiscale.npz
│   │   └── simulation_data_singlescale.npz
│   └── ...
├── eps_2neg6/
│   └── ...
└── eps_2neg10/
    └── ...
```

#### 1.a.ii. Evaluation

The Panda evaluation workflow and all of its executable instructions are kept
in the accompanying Panda pilot notebook. Follow that notebook after completing
the data-preparation pipeline above.

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

Each `.npz` file contains one array named `states`. With the defaults `K=9`, `J=8`, both dynamics durations set to `50`, and both sampling intervals set to `0.001`, the shapes are:

```text
single-scale states: (9, 50000)
multiscale states:   (81, 50000)
```

The dynamics durations are exclusive upper sampling boundaries. A trajectory
with sampling interval `dt` contains observations at `0, dt, ...` strictly
below its dynamics duration. Consequently, `N` observations span physical
time `(N - 1) * dt` and can be requested with a dynamics duration of `N * dt`.

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
--single-scale-dynamics-duration
                        single-scale trajectory duration; default 50
--multiscale-dynamics-duration
                        multiscale trajectory duration; default 50
--single-scale-sampling-interval
                        time between saved single-scale states; default 0.001
--multiscale-sampling-interval
                        time between saved multiscale states; default 0.001
--spinup-max-step       maximum solver step during spin-up; default 0.01
--learning-max-step     maximum solver step during learning; default 0.001
--single-scale-dynamics-max-step
                        maximum solver step during single-scale generation;
                        defaults to single-scale-sampling-interval when omitted
--multiscale-dynamics-max-step
                        maximum solver step during multiscale generation;
                        defaults to multiscale-sampling-interval when omitted
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

### 2.b. Sampling-interval estimator

The sampling-interval utility converts a densely and uniformly sampled
trajectory into a target sampling interval normalized by the trajectory's
characteristic period. It uses the Fourier-based characteristic-timescale
estimator from Dysts 0.96 for each selected channel, reduces those channel
periods to one system-level period, and computes

```text
target sampling interval = num_periods * characteristic_period / (num_points - 1)
```

The `num_points - 1` denominator is intentional: the corresponding uniform
time grid includes both endpoints. The defaults produce 4096 observations
spanning 40 characteristic periods.

#### 2.b.i. Environment

Create and activate the dedicated Conda environment from the repository root:

```bash
conda env create --file environments/timescales.yml
conda activate timescales
```

#### 2.b.ii. Estimate a sampling interval

The command-line tool accepts `.npy` files or `.npz` files containing a dense
trajectory. For example, estimate an interval from the nine slow variables in
a Lorenz–96 trajectory:

```bash
python scripts/estimate_sampling_interval.py \
    --input data/lorenz96/simulation_data_multiscale.npz \
    --array-key states \
    --sampling-interval 0.001 \
    --time-axis 1 \
    --channels 0:9 \
    --reduction median \
    --num-periods 40 \
    --num-points 4096 \
    --output-dir data/lorenz96
```

`--sampling-interval` is the physical interval of the dense input, not the
interval being estimated. `--time-axis` identifies the time dimension. All
remaining dimensions are flattened into channels after the time axis is moved
to the end. `--channels` then selects flattened channel indices using either
slice syntax (`0:9` or `0:9:2`) or comma-separated list syntax (`0,2,4`).

Choose channels that represent the timescale of interest. In a slow-fast
system, use only the slow variables when the target interval should follow the
slow dynamics. If `--channels` is omitted, every flattened channel is used and
the tool prints a warning for multichannel input.

By default, channel periods are combined with their median. The other
command-line reductions are `mean`, `min`, and `max`. The result is written to
`sampling_interval.json` unless `--output-filename` specifies another name.
The JSON records the input shape and sampling interval, selected channels,
individual channel periods, reduction, aggregate characteristic period,
target grid settings, and estimated target sampling interval.

List every available option with:

```bash
python scripts/estimate_sampling_interval.py --help
```

#### 2.b.iii. Python API

The reusable API is in `src/data/timescales.py`:

```python
import numpy as np

from src.data.timescales import (
    estimate_channel_periods,
    estimate_characteristic_period,
    estimate_sampling_interval,
    make_period_normalized_times,
)

states = np.load("data/lorenz96/simulation_data_multiscale.npz")["states"]
slow_states = states[:9]

periods = estimate_channel_periods(slow_states, 0.001, time_axis=1)
period = estimate_characteristic_period(
    slow_states,
    0.001,
    time_axis=1,
    reduction="median",
)
target_dt = estimate_sampling_interval(
    slow_states,
    0.001,
    time_axis=1,
    reduction="median",
    num_periods=40,
    num_points=4096,
)
times = make_period_normalized_times(period, num_periods=40, num_points=4096)
```

The library API also accepts a callable as the channel-period reduction.
Inputs must contain at least two finite time points, and sampling intervals,
period spans, and estimated periods must be finite and positive.

## 3. References and acknowledgments

### 3.a. Lorenz–96 data generation

The Lorenz–96 data generator was developed based on the implementation in
[EnsembleKalmanMethods](https://github.com/EdoardoCalvello/EnsembleKalmanMethods/tree/main).
To acknowledge the repository and the work associated with it, please cite:

```bibtex
@article{Calvello2025Ensemble,
  title={Ensemble Kalman methods: A mean-field perspective},
  volume={34},
  journal={Acta Numerica},
  author={Calvello, Edoardo and Reich, Sebastian and Stuart, Andrew M.},
  year={2025},
  pages={123--291}
}
```

The upstream repository attributes its Lorenz–96 data-generation files to
Dmitry Burov and asks users of those files to also cite:

```bibtex
@article{Burov2021Kernel,
  title={Kernel analog forecasting: {M}ultiscale test problems},
  author={Burov, Dmitry and Giannakis, Dimitrios and Manohar, Krithika and Stuart, Andrew},
  journal={Multiscale Modeling \& Simulation},
  volume={19},
  number={2},
  pages={1011--1040},
  year={2021},
  publisher={SIAM},
  doi={10.1137/20M1338289}
}
```

### 3.b. Characteristic-timescale estimation

The timescale module wraps characteristic-timescale functionality provided by
[dysts](https://github.com/GilpinLab/dysts). The project asks users to consider
citing the following papers:

```bibtex
@inproceedings{Gilpin2021Chaos,
  title={Chaos as an interpretable benchmark for forecasting and data-driven modelling},
  author={Gilpin, William},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2021},
  url={https://arxiv.org/abs/2110.05266}
}
```

```bibtex
@article{Gilpin2023Model,
  title={Model scale versus domain knowledge in statistical forecasting of chaotic systems},
  author={Gilpin, William},
  journal={Physical Review Research},
  volume={5},
  number={4},
  pages={043252},
  year={2023},
  doi={10.1103/PhysRevResearch.5.043252}
}
```

### 3.c. Panda

The Panda pilot uses the pretrained model and evaluation methodology from
[Panda](https://github.com/abao1999/panda). Please cite:

```bibtex
@misc{lai2025panda,
  title={Panda: A pretrained forecast model for universal representation of chaotic dynamics},
  author={Jeffrey Lai and Anthony Bao and William Gilpin},
  year={2025},
  eprint={2505.13755},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.13755}
}
```
