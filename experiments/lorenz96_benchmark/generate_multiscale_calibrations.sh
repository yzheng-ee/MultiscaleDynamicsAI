#!/usr/bin/env bash
set -euo pipefail

# Run with the lorenz96_benchmark environment activated
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

python_command="${PYTHON:-python}"
output_dir="data/lorenz96_benchmark/calibration/multiscale"
completed=0
max_jobs=3
running=0
pids=()

# Each process gets one numerical-library thread to avoid oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
total=6


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

# Each recorded trajectory spans 99.9999 time units, including both endpoints
while read -r setting n_slow n_fast hx hy forcing; do
    printf 'Generating multiscale calibration: %s\n' "$setting"

    "$python_command" scripts/lorenz96_benchmark/generate_multiscale.py \
        --K "$n_slow" \
        --J "$n_fast" \
        --hx "$hx" \
        --hy "$hy" \
        --forcing "$forcing" \
        --epsilon 0.0078125 \
        --seed 2003 \
        --low -5.0 \
        --high 10.0 \
        --sample-dt 0.0001 \
        --n-samples 1000000 \
        --burn-in 50.0 \
        --t0 0.0 \
        --method DOP853 \
        --rtol 1e-8 \
        --atol 1e-10 \
        --output-prefix "$output_dir/$setting/trajectory_001" </dev/null &

    pids+=("$!")
    running=$((running + 1))
    if (( running == max_jobs )); then
        wait_batch
    fi
done <<'CONFIGURATIONS'
a    9  8  -0.8 1.0 10.0
b_k  18 8  -0.8 1.0 10.0
b_j  9  16 -0.8 1.0 10.0
b_hx 9  8  -1.2 1.0 10.0
b_hy 9  8  -0.8 1.4 10.0
b_f  9  8  -0.8 1.0 14.0
CONFIGURATIONS

if (( running > 0 )); then
    wait_batch
fi
