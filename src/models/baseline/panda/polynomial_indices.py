"""Preserve polynomial feature indices omitted from upstream Panda checkpoints."""

import json
from pathlib import Path
from typing import Any

import torch
from panda.patchtst.patchtst import PatchTSTForPrediction

POLYNOMIAL_INDICES_FILE = "polynomial_indices.json"


def get_polynomial_indices(model: PatchTSTForPrediction) -> dict[str, Any] | None:
    """Return a JSON-compatible feature record, or None if polynomial features are unused.

    Return structure::

        {
            "patch_length": int,
            "indices": [
                [[i, j], ...],       # Degree 2: [num_poly_feats, 2].
                [[i, j, k], ...],    # Degree 3: [num_poly_feats, 3].
                ...                 # config.poly_degrees groups in total.
            ],
        }

    Each index is an integer in [0, patch_length); repeated positions are allowed.
    """
    indices = getattr(model.model.encoder.embedder, "patch_indices", None)
    if indices is None:
        return None  # The ordinary linear embedding has no polynomial features.

    return {
        "patch_length": model.config.patch_length,
        "indices": [value.detach().cpu().tolist() for value in indices],
    }


def save_polynomial_indices(
    model: PatchTSTForPrediction,
    output_dir: str | Path,
    *,
    filename: str = POLYNOMIAL_INDICES_FILE,
) -> Path | None:
    """Save unregistered feature indices in output_dir using the supplied filename."""
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must be a single file name, not a path")

    payload = get_polynomial_indices(model)
    if payload is None:
        return None

    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return path


def load_polynomial_indices(
    model: PatchTSTForPrediction,
    polynomial_indices_path: str | Path,
    *,
    required: bool = False,
) -> bool:
    """Restore polynomial indices into the supplied model in place.

    Read the JSON file at polynomial_indices_path and validate the
    whole mapping before replacing the model's existing indices. No new model
    is created. Return True if restored, or False if the model has no polynomial
    features or the file is missing and required=False.

    Published upstream checkpoints have no companion file, so ordinary loading
    permits its absence. Training resume requires it for exact feature restoration.
    """
    embedder = model.model.encoder.embedder
    expected = getattr(embedder, "patch_indices", None)
    if expected is None:
        return False

    path = Path(polynomial_indices_path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing polynomial feature map: {path}")
        return False

    payload = json.loads(path.read_text())
    if payload["patch_length"] != model.config.patch_length or len(payload["indices"]) != len(expected):
        raise ValueError(f"Polynomial feature map does not match model configuration: {path}")

    restored = []
    for values, template in zip(payload["indices"], expected):
        indices = torch.as_tensor(values)
        if template.numel() == 0 and values == []:
            indices = torch.empty(template.shape, dtype=torch.long)
        if indices.dtype != torch.long or indices.shape != template.shape:
            raise ValueError(f"Invalid polynomial index shape or dtype: {path}")
        if indices.numel() and ((indices < 0).any() or (indices >= model.config.patch_length).any()):
            raise ValueError(f"Polynomial index outside the patch: {path}")
        restored.append(indices.to(model.device))

    embedder.patch_indices = restored
    return True
