#!/usr/bin/env bash
set -euo pipefail

# Run with the lorenz96_benchmark environment activated
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

python_command="${PYTHON:-python}"
output_dir="data/lorenz96_benchmark"
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

n_slow_samples=8192

# Intervals and subdivision factors are copied at full precision from the calibration results
while read -r split setting n_slow n_fast hx hy forcing count first_seed sample_dt subdivision_factor; do
    # Including both endpoints gives exactly n_slow_samples after downsampling by the factor
    n_samples=$(((n_slow_samples - 1) * subdivision_factor + 1))

    for ((index = 1; index <= count; index++)); do
        seed=$((first_seed + index - 1))
        printf -v trajectory_name 'trajectory_%03d' "$index"
        printf 'Generating multiscale trajectory: %s/%s/%s (seed %d)\n' "$split" "$setting" "$trajectory_name" "$seed"

        "$python_command" scripts/lorenz96_benchmark/generate_multiscale.py \
            --K "$n_slow" \
            --J "$n_fast" \
            --hx "$hx" \
            --hy "$hy" \
            --forcing "$forcing" \
            --epsilon 0.0078125 \
            --seed "$seed" \
            --low -5.0 \
            --high 10.0 \
            --sample-dt "$sample_dt" \
            --n-samples "$n_samples" \
            --burn-in 50.0 \
            --t0 0.0 \
            --method DOP853 \
            --rtol 1e-8 \
            --atol 1e-10 \
            --output-prefix "$output_dir/$split/multiscale/$setting/$trajectory_name" </dev/null &

        pids+=("$!")
        running=$((running + 1))
        if (( running == max_jobs )); then
            wait_batch
        fi
    done
done <<'COLLECTIONS'
training   a    9  8  -0.8 1.0 10.0 32 1011 0.000109480100896861   128
validation a    9  8  -0.8 1.0 10.0 8  2011 0.000109480100896861   128
test       a    9  8  -0.8 1.0 10.0 16 3011 0.000109480100896861   128
test       b_k  18 8  -0.8 1.0 10.0 4  4011 0.00010433360042735044 128
test       b_j  9  16 -0.8 1.0 10.0 4  5011 0.00010344941737288138 128
test       b_hx 9  8  -1.2 1.0 10.0 4  6011 0.00011570645734597157 128
test       b_hy 9  8  -0.8 1.4 10.0 4  7011 0.0001114797374429224  128
test       b_f  9  8  -0.8 1.0 14.0 4  8011 8.057446369636965e-05  128
COLLECTIONS

if (( running > 0 )); then
    wait_batch
fi
