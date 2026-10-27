"""A small Hugging Face Trainer specialization for Panda.

Pass standard TrainingArguments. Use train(resume_from_checkpoint=...), evaluate(),
and save_model() normally. Configure saving, best-model selection, and callbacks
through HF. Epoch checkpoints retain optimizer/scheduler/RNG state; save_model
exports weights only. Split provenance remains the experiment's responsibility.
"""

from contextlib import nullcontext
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from panda.patchtst.patchtst import PatchTSTForPrediction

from src.models.baseline.panda.polynomial_indices import (
    POLYNOMIAL_INDICES_FILE,
    load_polynomial_indices,
    save_polynomial_indices,
)

from .data import PandaForecastDataset


class PandaEarlyStoppingCallback(EarlyStoppingCallback):
    """Restore the patience counter from validation history when training resumes."""

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        # HF 4.40.2 does not persist this callback's patience counter. Rebuild it
        # from the restored scores, matching both HF's best metric and threshold.
        self.early_stopping_patience_counter = 0
        metric = args.metric_for_best_model
        metric = metric if metric.startswith("eval_") else f"eval_{metric}"
        better = np.greater if args.greater_is_better else np.less
        best = None
        for entry in state.log_history:
            if metric not in entry:
                continue
            value = entry[metric]
            if best is None or (better(value, best) and abs(value - best) > self.early_stopping_threshold):
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1
            if best is None or better(value, best):
                best = value


class PandaTrainer(Trainer):
    """Native HF interface with forecasting dataset checks and finite-loss checks.

    Set prediction_loss_only=False after construction if predictions are needed.
    polynomial_indices_filename names the companion file in every checkpoint
    and model export; use the same name when resuming training.
    """

    def __init__(
        self,
        model: PatchTSTForPrediction,
        args: TrainingArguments,
        *,
        train_dataset: PandaForecastDataset | None = None,
        eval_dataset: PandaForecastDataset | None = None,
        polynomial_indices_filename: str = POLYNOMIAL_INDICES_FILE,
        **kwargs: Any,
    ) -> None:
        if (
            not polynomial_indices_filename
            or Path(polynomial_indices_filename).name != polynomial_indices_filename
            or polynomial_indices_filename in {".", ".."}
        ):
            raise ValueError("polynomial_indices_filename must be a single file name, not a path")
        self.polynomial_indices_filename = polynomial_indices_filename

        if not isinstance(model, PatchTSTForPrediction):
            raise TypeError("model must be PatchTSTForPrediction")

        for dataset in (train_dataset, eval_dataset):
            if dataset is None:
                continue

            if not isinstance(dataset, PandaForecastDataset):
                raise TypeError("expected PandaForecastDataset")

            for name in ("context_length", "prediction_length"):
                if getattr(dataset, name) != getattr(model.config, name):
                    raise ValueError(f"dataset {name} differs from model config")

        args.label_names = ["future_values"]
        args.remove_unused_columns = False
        args.prediction_loss_only = True
        args.logging_nan_inf_filter = False

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        # Require complete, equally sized gradient-accumulation groups.
        # This avoids partial-group handling issues observed in older HF Trainers.
        # Accumulation=1 is unrestricted. Revisit this policy if partial groups
        # are needed after a dependency upgrade.
        group = (
            self.args.train_batch_size  # Examples per batch, per process.
            * self.args.gradient_accumulation_steps  # Batches per weight update.
            * self.args.world_size  # Number of distributed training processes.
        )
        if (self.args.gradient_accumulation_steps > 1 and len(self.train_dataset) % group):
            raise ValueError("window count must be divisible by batch_size * accumulation_steps * world_size")

        return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Panda averages errors over batch, time, and channels into a scalar tensor.
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if not torch.isfinite(loss).all():  # .all() is redundant for Panda's scalar loss.
            raise FloatingPointError("Panda returned a nonfinite loss")

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        was_training = self.model.training

        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self.model.train(was_training)

    def save_model(self, output_dir=None, _internal_call=False):
        """Include the feature map in both epoch checkpoints and model exports."""
        super().save_model(output_dir, _internal_call=_internal_call)  # Uses self.args.output_dir when output_dir is None.
        if self.args.should_save:
            save_polynomial_indices(
                self.model,
                self.args.output_dir if output_dir is None else output_dir,
                filename=self.polynomial_indices_filename,
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore the feature map as well as weights before training resumes."""
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        load_polynomial_indices(
            self.model if model is None else model,
            polynomial_indices_path=Path(resume_from_checkpoint) / self.polynomial_indices_filename,
            required=True,
        )

    def _load_best_model(self):
        """Keep validation-selected weights and their feature map together."""
        super()._load_best_model()
        load_polynomial_indices(
            self.model,
            polynomial_indices_path=Path(self.state.best_model_checkpoint) / self.polynomial_indices_filename,
            required=True,
        )

    def _load_rng_state(self, checkpoint):
        # Older HF Trainers restore RNG records through torch.load(), which can
        # reject their NumPy objects under newer PyTorch loading restrictions.
        # When safe_globals is available, temporarily allow the specific NumPy
        # functions/types needed for restoration; other restrictions remain active.
        # The API check below detects availability, not affected library versions.
        # This supports training resume, not ordinary inference. Recheck whether
        # this compatibility workaround is needed when upgrading dependencies.
        context = nullcontext()

        if hasattr(torch.serialization, "safe_globals"):
            context = torch.serialization.safe_globals(
                [
                    importlib.import_module("numpy.core.multiarray")._reconstruct,
                    np.ndarray,
                    np.dtype,
                    type(np.dtype("uint32")),
                ]
            )

        with context:
            super()._load_rng_state(checkpoint)
