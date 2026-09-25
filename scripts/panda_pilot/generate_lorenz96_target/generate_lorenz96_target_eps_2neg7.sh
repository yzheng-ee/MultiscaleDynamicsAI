#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
num_points=4096

for seed in {0..31}; do
    dense_seed_dir="${project_root}/data/panda_pilot/lorenz96_dense/eps_2neg7/seed_${seed}"
    target_seed_dir="${project_root}/data/panda_pilot/lorenz96_target/eps_2neg7/seed_${seed}"
    single_scale_sampling_interval="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["target_sampling_interval"])' "${dense_seed_dir}/sampling_interval_singlescale.json")"
    multiscale_sampling_interval="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["target_sampling_interval"])' "${dense_seed_dir}/sampling_interval_multiscale.json")"
    single_scale_dynamics_duration="$(python -c 'import sys; print(int(sys.argv[2]) * float(sys.argv[1]))' "${single_scale_sampling_interval}" "${num_points}")"
    multiscale_dynamics_duration="$(python -c 'import sys; print(int(sys.argv[2]) * float(sys.argv[1]))' "${multiscale_sampling_interval}" "${num_points}")"

    echo "Generating target-sampled Lorenz-96 trajectories with epsilon 2^-7 for seed ${seed}..."
    python "${project_root}/scripts/generate_lorenz96.py" \
        --seed "${seed}" \
        --epsilon 0.0078125 \
        --single-scale-dynamics-duration "${single_scale_dynamics_duration}" \
        --multiscale-dynamics-duration "${multiscale_dynamics_duration}" \
        --single-scale-sampling-interval "${single_scale_sampling_interval}" \
        --multiscale-sampling-interval "${multiscale_sampling_interval}" \
        --single-scale-dynamics-max-step 0.001 \
        --multiscale-dynamics-max-step 0.001 \
        --skip-closure-training \
        --closure-path "${dense_seed_dir}/closure.joblib" \
        --output-dir "${target_seed_dir}"
    echo "Finished target-sampled Lorenz-96 trajectories with epsilon 2^-7 for seed ${seed}."
done
