# MultiscaleDynamicsAI

Learning multiscale dynamics for AI in science and engineering.

## 1. Experiments and evaluation

### 1.a. Panda pilot

The Panda pilot evaluates the pretrained [Panda](https://github.com/abao1999/panda)
forecasting model on matched single-scale and multiscale Lorenz–96 trajectories.
Follow [`notebooks/panda_pilot.ipynb`](notebooks/panda_pilot.ipynb) for the
environment setup, complete data-preparation workflow, and evaluation.

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

### 2.b. Characteristic-period estimator

The characteristic-period utility estimates a Fourier-based timescale for
each channel of a densely and uniformly sampled trajectory using Dysts 0.96.
It can then reduce those channel periods to one system-level period. Selecting
a target sampling interval from that period is an experiment-design decision
and is intentionally outside the scope of this module.

#### 2.b.i. Environment

Create and activate the dedicated Conda environment from the repository root:

```bash
conda env create --file environments/timescales.yml
conda activate timescales
```

#### 2.b.ii. Python API

The reusable API is in `src/data/characteristic_period`:

```python
import numpy as np

from src.data.characteristic_period import (
    estimate_channel_periods,
    estimate_characteristic_period,
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
```

The library API also accepts a callable as the channel-period reduction.
Inputs must contain at least two finite time points, and the dense input
sampling interval and estimated periods must be finite and positive.

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

### 3.b. Characteristic-period estimation

The characteristic-period module wraps characteristic-timescale functionality provided by
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
