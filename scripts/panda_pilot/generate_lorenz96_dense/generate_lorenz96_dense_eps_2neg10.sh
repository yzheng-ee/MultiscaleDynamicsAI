#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

for seed in {0..31}; do
    echo "Generating dense Lorenz-96 trajectories with epsilon 2^-10 for seed ${seed}..."
    python "${project_root}/scripts/generate_lorenz96.py" \
        --seed "${seed}" \
        --epsilon 0.0009765625 \
        --output-dir "${project_root}/data/panda_pilot/lorenz96_dense/eps_2neg10/seed_${seed}"
    echo "Finished dense Lorenz-96 trajectories with epsilon 2^-10 for seed ${seed}."
done
