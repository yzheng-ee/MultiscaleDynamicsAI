#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Extracting Lorenz-96 trajectories from the Panda base40 training split..."
python "${project_root}/scripts/panda_pilot/extract_panda_base40.py" \
    Lorenz96 \
    --data-root "${project_root}/data/panda_pilot/panda_base40" \
    --split train \
    --output-dir "${project_root}/data/panda_pilot/panda_base40_lorenz96"
echo "Finished extracting Panda base40 Lorenz-96 trajectories."
