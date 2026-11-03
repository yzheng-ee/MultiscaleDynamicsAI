#!/usr/bin/env bash
set -euo pipefail

# Run with the lorenz96_benchmark environment activated
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

python_command="${PYTHON:-python}"
input_dir="data/lorenz96_benchmark/calibration/multiscale"
output_dir="data/lorenz96_benchmark/closure"
settings=(a b_k b_j b_hx b_hy b_f)
completed=0
max_jobs=3
running=0
pids=()

# Each process gets one numerical-library thread to avoid oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
total=${#settings[@]}


# Wait for every job in the batch, including when one fails
wait_batch() {
    local pid status=0
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            completed=$((completed + 1))
            printf 'Completed %d/%d jobs\n' "$completed" "$total"
        else
            printf 'Job %s failed\n' "$pid" >&2
            status=1
        fi
    done
    pids=()
    running=0
    return "$status"
}

for setting in "${settings[@]}"; do
    printf 'Training closure: %s\n' "$setting"

    "$python_command" scripts/lorenz96_benchmark/train_closure.py \
        --input-path "$input_dir/$setting/trajectory_001.npz" \
        --sample-size 800 \
        --seed 2003 \
        --alpha 1.0 \
        --n-restarts-optimizer 2 \
        --output-prefix "$output_dir/$setting/closure" </dev/null &

    pids+=("$!")
    running=$((running + 1))
    if (( running == max_jobs )); then
        wait_batch
    fi
done

if (( running > 0 )); then
    wait_batch
fi
