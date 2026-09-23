#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in {0..31}; do
    echo "Generating panda_pilot trajectories for seed ${seed}..."
    python "${project_root}/scripts/generate_lorenz96.py" \
        --seed "${seed}" \
        --process-noise 0.0 \
        --output-dir "${project_root}/data/panda_pilot/seed_${seed}"
    echo "Finished panda_pilot trajectories for seed ${seed}."
done
