#!/usr/bin/env bash
# Collect every training/validation observation trajectory for both families.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python}"
data_root="data/lorenz96_benchmark"
output_dir="outputs/panda_lorenz96/trajectory_paths"

for family in singlescale multiscale; do
    for split in training validation; do
        input_dir="$data_root/$split/$family"
        if [[ "$family" == "multiscale" ]]; then
            input_dir="$input_dir/slow_downsampled"
        fi
        "$python_bin" scripts/panda/collect_trajectory_paths.py \
            --input-dir "$input_dir" \
            --output-path "$output_dir/${family}_${split}.json"
    done
done
