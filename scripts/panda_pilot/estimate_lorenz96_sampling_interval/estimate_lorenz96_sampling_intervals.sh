#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for exponent in {5..10}; do
    echo "Estimating Lorenz-96 sampling intervals with epsilon 2^-${exponent}..."
    "${script_dir}/estimate_lorenz96_sampling_intervals_eps_2neg${exponent}.sh"
    echo "Finished Lorenz-96 sampling intervals with epsilon 2^-${exponent}."
done
