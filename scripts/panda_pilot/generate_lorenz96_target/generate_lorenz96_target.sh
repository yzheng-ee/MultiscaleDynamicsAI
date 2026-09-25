#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for exponent in {5..10}; do
    "${script_dir}/generate_lorenz96_target_eps_2neg${exponent}.sh"
done
