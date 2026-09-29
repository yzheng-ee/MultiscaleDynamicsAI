#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Extracting Double Pendulum trajectories from the Panda base40 zero-shot split..."
python "${project_root}/scripts/panda_pilot/extract_panda_base40.py" \
    DoublePendulum \
    --data-root "${project_root}/data/panda_pilot/panda_base40" \
    --split test_zeroshot \
    --output-dir "${project_root}/data/panda_pilot/panda_base40_double_pendulum"
echo "Finished extracting Panda base40 Double Pendulum trajectories."
