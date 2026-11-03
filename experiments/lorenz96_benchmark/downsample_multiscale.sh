#!/usr/bin/env bash
set -euo pipefail

# Run with the lorenz96_benchmark environment activated
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

python_command="${PYTHON:-python}"
data_dir="data/lorenz96_benchmark"
completed=0
max_jobs=3
running=0
pids=()
total=76

# Each process gets one numerical-library thread to avoid oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Wait for every job in the batch, including when one fails
wait_batch() {
    local pid status=0
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            completed=$((completed + 1))
            printf 'Completed %d/%d trajectories\n' "$completed" "$total"
        else
            printf 'Job %s failed\n' "$pid" >&2
            status=1
        fi
    done
    pids=()
    running=0
    return "$status"
}

# Strides are copied from the calibrated multiscale subdivision factors
while read -r split setting count stride; do
    input_dir="$data_dir/$split/multiscale/$setting"

    for ((index = 1; index <= count; index++)); do
        printf -v trajectory_name 'trajectory_%03d' "$index"
        printf 'Downsampling multiscale trajectory: %s/%s/%s\n' "$split" "$setting" "$trajectory_name"

        "$python_command" scripts/lorenz96_benchmark/downsample_multiscale.py \
            --input-path "$input_dir/$trajectory_name.npz" \
            --stride "$stride" \
            --output-prefix "$input_dir/slow_downsampled/$trajectory_name" </dev/null &

        pids+=("$!")
        running=$((running + 1))
        if (( running == max_jobs )); then
            wait_batch
        fi
    done
done <<'COLLECTIONS'
training   a    32 128
validation a    8  128
test       a    16 128
test       b_k  4  128
test       b_j  4  128
test       b_hx 4  128
test       b_hy 4  128
test       b_f  4  128
COLLECTIONS

if (( running > 0 )); then
    wait_batch
fi
