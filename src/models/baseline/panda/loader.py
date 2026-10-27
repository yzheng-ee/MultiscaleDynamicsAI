"""Construct forecasting models and load pipelines from the installed Panda package."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from panda.patchtst.patchtst import PatchTSTForPrediction
from panda.patchtst.pipeline import PatchTSTPipeline
from transformers import PatchTSTConfig, set_seed

from .polynomial_indices import load_polynomial_indices

DEFAULT_CHECKPOINT = "GilpinLab/panda"


def build_panda_model(
    *,
    checkpoint: str | Path | None = None,
    model_config: PatchTSTConfig | None = None,
    seed: int = 42,
    **load_kwargs: Any,
) -> PatchTSTForPrediction:
    """Use exactly one of checkpoint/config; HF loading options pass through.

    Scratch configs must include Panda-specific fields (start from a Panda
    checkpoint config). Polynomial indices use seeded initialization; restore a
    saved mapping explicitly with load_polynomial_indices after construction.
    """
    if (checkpoint is None) == (model_config is None):
        raise ValueError("provide exactly one of checkpoint and model_config")

    set_seed(seed)

    if checkpoint is None:
        if load_kwargs:
            raise ValueError("checkpoint loading options do not apply to scratch models")
        return PatchTSTForPrediction(deepcopy(model_config))

    model, info = PatchTSTForPrediction.from_pretrained(
        str(checkpoint),
        output_loading_info=True,
        **load_kwargs,
    )

    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"checkpoint does not match PatchTSTForPrediction: {info}")

    return model


def restore_panda_model(
    *,
    checkpoint: str | Path,
    polynomial_indices_path: str | Path,
    **load_kwargs: Any,
) -> PatchTSTForPrediction:
    """Restore weights from checkpoint and a mapping from the exact local JSON path.

    The mapping is required for models with polynomial features. Restoring both
    weights and indices replaces their initialization, so no seed is needed.
    Additional checkpoint loading options pass through to build_panda_model.
    """
    model = build_panda_model(checkpoint=checkpoint, **load_kwargs)
    load_polynomial_indices(model, polynomial_indices_path=polynomial_indices_path, required=True)
    return model


def load_panda(
    checkpoint: str | Path | None = None,
    *,
    device_map: str = "cpu",
    revision: str | None = None,
    **from_pretrained_kwargs: Any,
) -> PatchTSTPipeline:
    """Load a forecasting pipeline without restoring a saved polynomial mapping.

    When ``checkpoint`` is omitted, use ``GilpinLab/panda``.
    Pin ``revision`` to a Hugging Face commit for reproducible
    experiments. Additional keyword arguments are forwarded to Transformers'
    ``from_pretrained`` implementation. Polynomial indices are generated during
    initialization; use restore_panda to restore a saved mapping as well.
    """
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT

    if revision is not None:
        from_pretrained_kwargs["revision"] = revision

    return PatchTSTPipeline.from_pretrained(
        mode="predict",
        pretrain_path=str(checkpoint),
        device_map=device_map,
        **from_pretrained_kwargs,
    )


def restore_panda(
    *,
    checkpoint: str | Path,
    polynomial_indices_path: str | Path,
    device_map: str = "cpu",
    revision: str | None = None,
    **from_pretrained_kwargs: Any,
) -> PatchTSTPipeline:
    """Load a forecasting pipeline and restore its required polynomial mapping.

    Supply checkpoint weights and the exact local polynomial JSON file path.
    No seed is needed because the saved weights and indices are restored.
    """
    pipeline = load_panda(
        checkpoint=checkpoint,
        device_map=device_map,
        revision=revision,
        **from_pretrained_kwargs,
    )
    load_polynomial_indices(pipeline.model, polynomial_indices_path=polynomial_indices_path, required=True)
    return pipeline
