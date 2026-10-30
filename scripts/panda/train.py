r"""Fine-tune Panda on prepared NumPy window caches; launch directly or with torchrun.

Example from the repository root:
    python scripts/panda/train.py \
        --train-windows-path outputs/panda_lorenz96/windows/multiscale_training/windows.npy \
        --validation-windows-path outputs/panda_lorenz96/windows/multiscale_validation/windows.npy \
        --context-length 512 --prediction-length 128 \
        --output-dir outputs/panda_lorenz96/training/multiscale

Experiment settings belong in shell/Slurm arguments. Outputs are run.json,
checkpoint-* directories with training state, and model/ with the selected model.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input windows and history/target lengths.
    parser.add_argument("--train-windows-path", type=Path, required=True)
    parser.add_argument("--validation-windows-path", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--prediction-length", type=int, required=True)

    # Initial model weights and polynomial mapping.
    parser.add_argument("--checkpoint", help="Local model directory or HF repository; default: GilpinLab/panda.")
    parser.add_argument("--revision", help="HF checkpoint commit/tag; pin a commit for reproducibility.")
    parser.add_argument("--polynomial-indices-path", type=Path, help="Initial mapping file; omit to use seeded indices.")
    parser.add_argument("--local-files-only", action="store_true")

    # Output checkpoints and training resume.
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, help="Resume a checkpoint-* directory in this output.")
    parser.add_argument("--save-total-limit", type=int, default=2)

    # Reproducibility and training duration.
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--max-steps", type=int, help="Positive optimizer-step budget; overrides --epochs.")
    parser.add_argument("--early-stopping-patience", type=int, help="Stop after N validations without improvement.")

    # Batch sizes and gradient accumulation.
    parser.add_argument("--batch-size", type=int, default=8, help="Training examples per batch per process.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    # Optimizer and learning-rate schedule.
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine", help="Native HF scheduler name.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_torch", help="Native HF optimizer name.")

    # Data loading, numerical precision, and distributed execution.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)

    # Run naming, logging, and optional W&B tracking.
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--run-name", help="Defaults to the output directory name.")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="multiscale-dynamics")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")

    args = parser.parse_args()
    for name in (
        "context_length", "prediction_length", "epochs", "batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "logging_steps", "save_total_limit",
        "max_steps", "early_stopping_patience",
    ):
        value = getattr(args, name)
        if value is not None and not value > 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.num_workers < 0 or args.weight_decay < 0 or args.max_grad_norm < 0:
        parser.error("seed, num-workers, weight-decay, and max-grad-norm must be nonnegative")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup-ratio must be between 0 and 1")
    return args


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Replace the run record after its complete contents have been written."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    # Keep --help available without importing model/training dependencies.
    from transformers import TrainingArguments

    from src.models.baseline.panda.loader import DEFAULT_CHECKPOINT, build_panda_model
    from src.models.baseline.panda.polynomial_indices import load_polynomial_indices
    from src.training.baseline.panda import PandaTrainer, dataset_from_cache
    from src.training.baseline.panda.run_metadata import build_run_metadata, describe_cache
    from src.training.baseline.panda.trainer import PandaEarlyStoppingCallback

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    if Path(args.checkpoint).is_dir():
        args.checkpoint = str(Path(args.checkpoint).resolve())
    args.run_name = args.run_name or args.output_dir.name
    output, resume = args.output_dir, args.resume_from_checkpoint
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty training output directory or explicitly resume: {output}")
    if resume is not None:
        if resume.parent != output or not resume.name.startswith("checkpoint-"):
            raise ValueError("Resume from a checkpoint-* directory inside the original output directory")
        if not (resume / "trainer_state.json").is_file():
            raise FileNotFoundError(resume / "trainer_state.json")
    if args.train_windows_path == args.validation_windows_path:
        raise ValueError("Training and validation must use separate window files")

    windowing = dict(context_length=args.context_length, prediction_length=args.prediction_length)
    train_dataset = dataset_from_cache(args.train_windows_path, **windowing)
    validation_dataset = dataset_from_cache(args.validation_windows_path, **windowing)
    if train_dataset.windows.shape[2] != validation_dataset.windows.shape[2]:
        raise ValueError("Training and validation must share a channel count")
    data = {
        "training": describe_cache(args.train_windows_path, train_dataset.windows.shape),
        "validation": describe_cache(args.validation_windows_path, validation_dataset.windows.shape),
    }
    training_args = TrainingArguments(
        output_dir=str(output), run_name=args.run_name, seed=args.seed, data_seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type, warmup_ratio=args.warmup_ratio, optim=args.optim,
        dataloader_num_workers=args.num_workers, logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit, evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        use_cpu=args.cpu, fp16=args.precision == "fp16", bf16=args.precision == "bf16",
        local_rank=args.local_rank, report_to=["wandb"] if args.report_to == "wandb" else [], push_to_hub=False,
    )
    settings = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items() if name not in {"resume_from_checkpoint", "local_rank"}
    }
    previous = json.loads((output / "run.json").read_text()) if resume is not None else None
    if previous is not None and (
        previous["settings"] != settings or previous["data"] != data
        or previous["world_size"] != training_args.world_size
    ):
        raise ValueError("Resume requires the original settings, window files, and process count")

    model = build_panda_model(
        checkpoint=resume or args.checkpoint, seed=args.seed,
        revision=None if resume else args.revision, local_files_only=args.local_files_only,
    )
    indices_path = resume / "polynomial_indices.json" if resume else args.polynomial_indices_path
    indices_restored = False
    if indices_path is not None:
        indices_restored = load_polynomial_indices(model, polynomial_indices_path=indices_path, required=True)
    # Trainer validates cache lengths against the checkpoint's forecast settings.
    if args.report_to == "wandb":
        os.environ.update(
            WANDB_PROJECT=args.wandb_project, WANDB_MODE=args.wandb_mode, WANDB_DIR=str(output),
            WANDB_LOG_MODEL="false", WANDB_RUN_ID=hashlib.sha256(str(output).encode()).hexdigest()[:16],
            WANDB_RESUME="allow" if resume else "never",
        )
        if args.wandb_group is not None:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
    callbacks = []
    if args.early_stopping_patience is not None:
        callbacks.append(PandaEarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = PandaTrainer(model, training_args, train_dataset=train_dataset, eval_dataset=validation_dataset, callbacks=callbacks)
    record = build_run_metadata(
        settings=settings, training_args=training_args, data=data, model_config=model.config,
        polynomial_indices_restored=indices_restored, project_root=ROOT,
        resume_from_checkpoint=resume, previous=previous,
    )
    metrics = record["metrics"]
    if trainer.is_world_process_zero():
        write_json(output / "run.json", record)
        LOGGER.info("%d training windows; %d validation windows", len(train_dataset), len(validation_dataset))
    try:
        if resume is None:
            metrics.update(trainer.evaluate(metric_key_prefix="initial_eval"))
            if trainer.is_world_process_zero():
                write_json(output / "run.json", record)
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        metrics.update(result.metrics)
        metrics.update(trainer.evaluate())
        metrics["best_validation_loss"] = trainer.state.best_metric
        trainer.save_model(str(output / "model"))
        if trainer.is_world_process_zero():
            record.update(
                status="finished", optimizer_steps=trainer.state.global_step,
                best_checkpoint=trainer.state.best_model_checkpoint,
            )
            write_json(output / "run.json", record)
            LOGGER.info("Selected model exported to %s", output / "model")
    except Exception as error:
        if trainer.is_world_process_zero():
            write_json(output / "run.json", record | {"status": "failed", "error": str(error)})
        raise
    finally:
        if args.report_to == "wandb" and trainer.is_world_process_zero():
            import wandb

            wandb.finish(exit_code=0 if sys.exc_info()[0] is None else 1)


if __name__ == "__main__":
    main()
