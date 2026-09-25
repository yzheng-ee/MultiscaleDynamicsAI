"""Project-facing loading adapter for vendored Panda models."""

from typing import Any, Literal

from .._vendor.panda.panda.patchtst.pipeline import PatchTSTPipeline

PandaMode = Literal["predict", "pretrain"]

DEFAULT_CHECKPOINTS: dict[PandaMode, str] = {
    "predict": "GilpinLab/panda",
    "pretrain": "GilpinLab/panda_mlm",
}


def load_panda(
    mode: PandaMode = "predict",
    checkpoint: str | None = None,
    *,
    device_map: str = "cpu",
    revision: str | None = None,
    **from_pretrained_kwargs: Any,
) -> PatchTSTPipeline:
    """Load a Panda forecasting or masked-pretraining checkpoint.

    When ``checkpoint`` is omitted, the default checkpoint for ``mode`` is
    selected. Pin ``revision`` to a Hugging Face commit for reproducible
    experiments. Additional keyword arguments are forwarded to Transformers'
    ``from_pretrained`` implementation.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINTS[mode]

    if revision is not None:
        from_pretrained_kwargs["revision"] = revision

    return PatchTSTPipeline.from_pretrained(
        mode=mode,
        pretrain_path=checkpoint,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
