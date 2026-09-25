#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
panda_num_periods=40.15686274509804  # 4096 / (4096 // 40)

for seed in {0..31}; do
    seed_dir="${project_root}/data/panda_pilot/lorenz96_dense/eps_2neg10/seed_${seed}"

    echo "Estimating single-scale sampling interval for seed ${seed}..."
    python "${project_root}/scripts/estimate_sampling_interval.py" \
        --input "${seed_dir}/simulation_data_singlescale.npz" \
        --array-key states \
        --sampling-interval 0.001 \
        --time-axis 1 \
        --channels 0:9 \
        --num-periods "${panda_num_periods}" \
        --num-points 4096 \
        --output-dir "${seed_dir}" \
        --output-filename sampling_interval_singlescale.json

    echo "Estimating multiscale sampling interval for seed ${seed}..."
    python "${project_root}/scripts/estimate_sampling_interval.py" \
        --input "${seed_dir}/simulation_data_multiscale.npz" \
        --array-key states \
        --sampling-interval 0.001 \
        --time-axis 1 \
        --channels 0:9 \
        --num-periods "${panda_num_periods}" \
        --num-points 4096 \
        --output-dir "${seed_dir}" \
        --output-filename sampling_interval_multiscale.json

    echo "Finished sampling-interval estimation for seed ${seed}."
done
