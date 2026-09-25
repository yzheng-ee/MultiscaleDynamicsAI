#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stage 1/3: Generating dense Lorenz-96 datasets..."
"${script_dir}/generate_lorenz96_dense/generate_lorenz96_dense.sh"
echo "Stage 1/3 complete."

echo "Stage 2/3: Estimating Lorenz-96 sampling intervals..."
"${script_dir}/estimate_lorenz96_sampling_interval/estimate_lorenz96_sampling_intervals.sh"
echo "Stage 2/3 complete."

echo "Stage 3/3: Generating target-sampled Lorenz-96 datasets..."
"${script_dir}/generate_lorenz96_target/generate_lorenz96_target.sh"
echo "Stage 3/3 complete."

echo "Panda pilot data generation complete."
