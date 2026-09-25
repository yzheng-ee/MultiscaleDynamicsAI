#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for exponent in {5..10}; do
    echo "Generating dense Lorenz-96 dataset with epsilon 2^-${exponent}..."
    "${script_dir}/generate_lorenz96_dense_eps_2neg${exponent}.sh"
    echo "Finished dense Lorenz-96 dataset with epsilon 2^-${exponent}."
done
