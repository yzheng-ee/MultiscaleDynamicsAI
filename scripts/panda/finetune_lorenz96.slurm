#!/usr/bin/env bash
#SBATCH --job-name=finetune_lorenz96
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit from the repository root: sbatch --account=<GPU_ACCOUNT> scripts/panda/finetune_lorenz96.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Use the project's Panda environment, with CUDA-enabled PyTorch installed.
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PANDA_ENV:-panda}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

python -c 'import torch; assert torch.cuda.device_count() == 4, "Expected four visible CUDA GPUs"'

# Run both families sequentially in the same allocation, using all four GPUs.
for family in multiscale singlescale; do
    echo "Fine-tuning Panda: $family"
    # One Slurm task launches four GPU workers; 8 examples each gives global batch 32.
    srun --ntasks=1 --cpu-bind=none --gpu-bind=none \
        python -u -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/panda/train.py \
        --train-windows-path outputs/panda_lorenz96/windows/${family}_training/windows.npy \
        --validation-windows-path outputs/panda_lorenz96/windows/${family}_validation/windows.npy \
        --context-length 512 \
        --prediction-length 128 \
        --checkpoint GilpinLab/panda \
        --revision 0ede89c96f555749781978c5dea7f62e813a5ff6 \
        --output-dir outputs/panda_lorenz96/training/${family}_seed99 \
        --seed 99 \
        --epochs 50 \
        --early-stopping-patience 10 \
        --batch-size 8 \
        --eval-batch-size 32 \
        --gradient-accumulation-steps 1 \
        --learning-rate 3e-5 \
        --weight-decay 0.01 \
        --max-grad-norm 1.0 \
        --lr-scheduler-type cosine \
        --warmup-ratio 0.05 \
        --optim adamw_torch \
        --num-workers 0 \
        --precision bf16 \
        --logging-steps 10 \
        --save-total-limit 2 \
        --run-name panda_lorenz96_${family}_seed99 \
        --report-to wandb \
        --wandb-project multiscale-dynamics \
        --wandb-group panda_lorenz96 \
        --wandb-mode offline \
        "$@"
done
