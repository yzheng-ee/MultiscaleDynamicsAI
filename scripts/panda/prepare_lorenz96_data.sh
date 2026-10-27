#!/usr/bin/env bash
# Prepare windows from the four manifests created by collect_lorenz96_trajectory_paths.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python}"
manifest_dir="outputs/panda_lorenz96/trajectory_paths"
output_dir="outputs/panda_lorenz96/windows"

for family in singlescale multiscale; do
    for split in training validation; do
        "$python_bin" scripts/panda/prepare_data.py \
            --manifest-path "$manifest_dir/${family}_${split}.json" \
            --context-length 512 \
            --prediction-length 128 \
            --stride 128 \
            --output-dir "$output_dir/${family}_${split}"
    done
done
