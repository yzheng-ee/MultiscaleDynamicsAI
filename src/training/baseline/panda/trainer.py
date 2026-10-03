"""Hugging Face Trainer integration for Panda forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.models.baseline.panda.upstream.panda.patchtst.patchtst import (
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)

from .config import PandaTrainerConfig

PandaTask = Literal["forecast", "mlm"]
PandaModel = PatchTSTForPrediction | PatchTSTForPretraining


@dataclass(frozen=True)
class PandaTrainingResult:
    """Loss history and checkpoint information from a training run."""

    training_loss: list[float]
    validation_loss: list[float]
    optimizer_steps: int
    output_dir: Path | None


class PandaTrainer:
    """Train an already-constructed Panda model with HF Trainer."""

    def __init__(
        self,
        model: PandaModel,
        config: PandaTrainerConfig,
        *,
        task: PandaTask = "forecast",
    ) -> None:
        if task not in ("forecast", "mlm"):
            raise ValueError(f"unsupported Panda task: {task}")
        if task == "forecast" and not isinstance(model, PatchTSTForPrediction):
            raise TypeError("the forecast task requires PatchTSTForPrediction")
        if task == "mlm" and not isinstance(model, PatchTSTForPretraining):
            raise TypeError("the mlm task requires PatchTSTForPretraining")
        self.model = model
        self.config = config
        self.task = task
        self._trainer: Trainer | None = None

    def _validate_dataset(self, dataset: Dataset, name: str) -> None:
        if len(dataset) == 0:
            raise ValueError(f"{name} must contain at least one window")

        attributes = ["context_length"]
        if self.task == "forecast":
            attributes.append("prediction_length")

        for attribute in attributes:
            actual = getattr(dataset, attribute, None)
            expected = getattr(self.model.config, attribute)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"{name} {attribute} is {actual}, but the model requires "
                    f"{expected}"
                )

    def _make_training_arguments(
        self,
        output_dir: str | Path,
        *,
        evaluate_each_epoch: bool,
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=self.config.num_workers,
            seed=self.config.seed,
            data_seed=self.config.seed,
            use_cpu=self.config.use_cpu,
            logging_strategy="epoch",
            evaluation_strategy="epoch" if evaluate_each_epoch else "no",
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            label_names=[
                "future_values" if self.task == "forecast" else "past_values"
            ],
        )

    def _make_trainer(
        self,
        output_dir: str | Path,
        *,
        train_dataset: Dataset | None = None,
        validation_dataset: Dataset | None = None,
    ) -> Trainer:
        arguments = self._make_training_arguments(
            output_dir,
            evaluate_each_epoch=validation_dataset is not None,
        )
        return Trainer(
            model=self.model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )

    @staticmethod
    def _loss_history(trainer: Trainer, key: str) -> list[float]:
        return [
            float(entry[key])
            for entry in trainer.state.log_history
            if key in entry
        ]

    def evaluate(self, dataset: Dataset) -> float:
        """Calculate the mean Panda loss for a dataset."""

        self._validate_dataset(dataset, "evaluation dataset")
        with TemporaryDirectory(prefix="panda-evaluation-") as temporary_dir:
            trainer = self._make_trainer(
                temporary_dir,
                validation_dataset=dataset,
            )
            metrics = trainer.evaluate()
        return float(metrics["eval_loss"])

    def save_checkpoint(self, output_dir: str | Path) -> Path:
        """Save a Transformers-compatible Panda checkpoint."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self._trainer is None:
            self.model.save_pretrained(output_path, safe_serialization=True)
        else:
            self._trainer.save_model(str(output_path))
        return output_path

    def fit(
        self,
        train_dataset: Dataset,
        *,
        validation_dataset: Dataset | None = None,
        output_dir: str | Path | None = None,
    ) -> PandaTrainingResult:
        """Train the model for the configured task and optionally save it."""

        self._validate_dataset(train_dataset, "training dataset")
        if validation_dataset is not None:
            self._validate_dataset(validation_dataset, "validation dataset")

        if output_dir is None:
            with TemporaryDirectory(prefix="panda-training-") as temporary_dir:
                return self._fit(
                    train_dataset,
                    validation_dataset=validation_dataset,
                    trainer_output_dir=temporary_dir,
                    checkpoint_output_dir=None,
                )

        return self._fit(
            train_dataset,
            validation_dataset=validation_dataset,
            trainer_output_dir=output_dir,
            checkpoint_output_dir=output_dir,
        )

    def _fit(
        self,
        train_dataset: Dataset,
        *,
        validation_dataset: Dataset | None,
        trainer_output_dir: str | Path,
        checkpoint_output_dir: str | Path | None,
    ) -> PandaTrainingResult:
        self._trainer = self._make_trainer(
            trainer_output_dir,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        )
        self._trainer.train()

        saved_output_dir = (
            self.save_checkpoint(checkpoint_output_dir)
            if checkpoint_output_dir is not None
            else None
        )
        return PandaTrainingResult(
            training_loss=self._loss_history(self._trainer, "loss"),
            validation_loss=self._loss_history(self._trainer, "eval_loss"),
            optimizer_steps=self._trainer.state.global_step,
            output_dir=saved_output_dir,
        )
