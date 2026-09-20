"""Adapter for PANDA (Patched Attention for Nonlinear DynAmics).

*****
mostly similar to PANDA pipeline, rollout loop reimplemented
*****

Reference: Lai, Bao & Gilpin, Panda: A pretrained forecast model for chaotic dynamics, ICLR 2026 (arXiv:2505.13755). 
Weights: GilpinLab/panda (21M) and GilpinLab/panda-72M. Code: https://github.com/abao1999/panda

Why not just use PatchTSTPipeline
PANDA's own pipeline class imports panda.utils, which transitively imports 'dysts' and 'gluonts' purely for dataset plumbing we do not use. 
We import PatchTSTForPrediction directly and reimplement the (short) rollout loop, which keeps inference dependency-light and, 
more importantly, gives us explicit control over the sliding-context policy 

Constraints inherited:
* context_length = 512 exactly The prediction head is a flatten-linear over a fixed patch count, so a shorter context is not merely worse, it is a shape error.
* prediction_length = 128 per forward pass; longer horizons are produced by autoregressive rollout, and the paper notes rollout quality degrades.
* Instance normalisation (scaling="std") is applied inside the model, per window and per channel. Do not standardise the input yourself.
* Trained on 3-dimensional systems. Channel attention is what lets it accept arbitrary channel counts, so the K = 9 slow variables 
    (or the full 81-dimensional multiscale state) are out-of-distribution in width as well as in dynamics
"""

from __future__ import annotations
import numpy as np
from msdyn.models.base import validate_forecast

DEFAULT_CHECKPOINT = "GilpinLab/panda"
LARGE_CHECKPOINT = "GilpinLab/panda-72M"

def _select_device(requested: str | None) -> str:
    import torch
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PandaForecaster:
    """Zero-shot forecasting with a pretrained PANDA checkpoint.

    checkpoint: HuggingFace model id or local path.
    device: cpu, mps, cuda, autodetect when None
    sliding_context: If True (PANDA's own default for rollout) the context window slides, keeping the most recent context_length steps. If Fals` the context grows, 
        which changes the patch count and is not supported by the fixed head, so this exists only to make the choice explicit.
    batch_size: Windows per forward pass.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str | None = None,
        sliding_context: bool = True,
        batch_size: int = 32,
        name: str | None = None,
    ) -> None:
        import torch
        from panda.patchtst.patchtst import PatchTSTForPrediction

        if not sliding_context:
            raise NotImplementedError("the pretrained head is built for a fixed patch count, so the context cannot grow; pass sliding_context=True")

        self.checkpoint = checkpoint
        self.device = _select_device(device)
        self.batch_size = batch_size
        self.sliding_context = sliding_context
        self.name = name or f"panda[{checkpoint.split('/')[-1]}]"

        self.model = PatchTSTForPrediction.from_pretrained(checkpoint).to(self.device).eval()
        self._torch = torch

        config = self.model.config
        self.context_length = int(config.context_length)
        self.prediction_length = int(config.prediction_length)

    def __repr__(self) -> str:
        return (f"PandaForecaster({self.checkpoint!r}, device={self.device!r}, context_length={self.context_length}, prediction_length={self.prediction_length})")

    # ------------------------------------------------------------------ 

    def _prepare_context(self, context: np.ndarray) -> np.ndarray:
        """Trim to the exact context length the head expects, or fail loudly."""
        if context.ndim != 3:
            raise ValueError(f"context must be (n_windows, context_length, n_channels), got {context.shape}")
        available = context.shape[1]
        if available < self.context_length:
            raise ValueError(f"{self.name} requires a context of exactly {self.context_length} steps (the prediction head is a fixed-size flatten-linear); got {available}. ")
        return context[:, -self.context_length :, :]

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Autoregressive zero-shot forecast.
        The horizon is produced in prediction_length-sized chunks; when horizon is not a multiple of it, the final chunk is truncated.
        """
        torch = self._torch
        trimmed = self._prepare_context(context)
        n_windows = trimmed.shape[0]
        outputs = []

        for start in range(0, n_windows, self.batch_size):
            batch = trimmed[start : start + self.batch_size]
            current = torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)

            chunks: list[np.ndarray] = []
            produced = 0
            with torch.no_grad():
                while produced < horizon:
                    generated = self.model.generate(current)
                    # sequences: (bs, n_samples, prediction_length, n_channels)
                    step = generated.sequences.median(dim=1).values
                    chunks.append(step.detach().cpu().numpy())
                    produced += step.shape[1]
                    if produced >= horizon:
                        break
                    current = torch.cat([current, step], dim=1)[:, -self.context_length :, :]

            outputs.append(np.concatenate(chunks, axis=1)[:, :horizon, :])

        prediction = np.concatenate(outputs, axis=0).astype(np.float64)
        validate_forecast(context, prediction, horizon)
        return prediction

    def rollout_chunks(self, horizon: int) -> int:
        """Number of autoregressive steps a horizon needs, the rollout depth."""
        return int(np.ceil(horizon / self.prediction_length))
