#!/usr/bin/env bash
#SBATCH --job-name=panda_lorenz96_smoke_4gpu
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit from the repository root:
# sbatch --account=<GPU_ACCOUNT> scripts/panda/smoke_lorenz96.slurm
# One Slurm task launches four GPU workers, matching the full training job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PANDA_ENV:-panda}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# Keep Hugging Face cache locks on the node-local filesystem.
export HF_HOME="/tmp/$USER/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE"

smoke_dir="outputs/panda_lorenz96/smoke/lorenz96_${SLURM_JOB_ID:?}"
mkdir -p outputs/panda_lorenz96/smoke
mkdir "$smoke_dir"

# Copy only 64 windows per split; preserve the full benchmark caches.
srun --ntasks=1 --cpu-bind=none --gpu-bind=none python - "$smoke_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np
import torch

assert torch.cuda.device_count() == 4, "Expected four visible CUDA GPUs"
assert torch.cuda.is_bf16_supported(), "This test requires BF16 support"
print(f"Testing on four GPUs: {[torch.cuda.get_device_name(i) for i in range(4)]}")
output = Path(sys.argv[1])
for family in ("multiscale", "singlescale"):
    family_dir = output / family
    family_dir.mkdir()
    for split in ("training", "validation"):
        source = Path("outputs/panda_lorenz96/windows") / f"{family}_{split}" / "windows.npy"
        windows = np.load(source, mmap_mode="r", allow_pickle=False)
        assert windows.shape[0] >= 64 and windows.shape[1:] == (640, 9), source
        np.save(family_dir / f"{split}.npy", windows[:64], allow_pickle=False)
PY

for family in multiscale singlescale; do
    test_dir="$smoke_dir/$family"

    # 64 windows / (8 per GPU * 4 GPUs) = two steps per epoch: checkpoint-2 and -4.
    # Reuse the identical arguments for resume, as required by train.py.
    train_args=(
        --train-windows-path "$test_dir/training.npy"
        --validation-windows-path "$test_dir/validation.npy"
        --context-length 512
        --prediction-length 128
        --checkpoint GilpinLab/panda
        --revision 0ede89c96f555749781978c5dea7f62e813a5ff6
        --output-dir "$test_dir/training"
        --seed 99
        --epochs 2
        --early-stopping-patience 10
        --batch-size 8
        --eval-batch-size 32
        --gradient-accumulation-steps 1
        --learning-rate 3e-5
        --weight-decay 0.01
        --max-grad-norm 1.0
        --lr-scheduler-type cosine
        --warmup-ratio 0.05
        --optim adamw_torch
        --num-workers 0
        --precision bf16
        --logging-steps 1
        --save-total-limit 2
        --run-name "panda_${family}_smoke_${SLURM_JOB_ID}"
        --report-to wandb
        --wandb-project multiscale-dynamics
        --wandb-group panda_smoke
        --wandb-mode offline
    )

    echo "$family: fresh training for two epochs on 64 windows using four GPUs"
    srun --ntasks=1 --cpu-bind=none --gpu-bind=none \
        python -u -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/panda/train.py "${train_args[@]}"
    cp "$test_dir/training/run.json" "$test_dir/fresh_run.json"

    echo "$family: resume from checkpoint-2 and repeat the second epoch"
    srun --ntasks=1 --cpu-bind=none --gpu-bind=none \
        python -u -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/panda/train.py "${train_args[@]}" \
        --resume-from-checkpoint "$test_dir/training/checkpoint-2"

    python - "$test_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
run = root / "training"
for record_path in (root / "fresh_run.json", run / "run.json"):
    record = json.loads(record_path.read_text())
    assert record["status"] == "finished", record_path
    assert record["optimizer_steps"] == 4, record_path
    assert record["world_size"] == 4, record_path
assert json.loads((run / "run.json").read_text())["polynomial_indices"] == "restored"
for step in (2, 4):
    for name in ("model.safetensors", "polynomial_indices.json", "optimizer.pt",
                 "scheduler.pt", "trainer_state.json"):
        assert (run / f"checkpoint-{step}" / name).is_file(), (step, name)
    for rank in range(4):
        assert (run / f"checkpoint-{step}" / f"rng_state_{rank}.pth").is_file(), (step, rank)
for name in ("model.safetensors", "config.json", "polynomial_indices.json"):
    assert (run / "model" / name).is_file(), name
assert list((run / "wandb").glob("offline-run-*/run-*.wandb")), "Missing offline W&B record"
print(f"PASS: four-GPU/BF16 training, validation, saving, best-model export, distributed resume, and offline W&B. Outputs: {root.resolve()}")
PY
done
