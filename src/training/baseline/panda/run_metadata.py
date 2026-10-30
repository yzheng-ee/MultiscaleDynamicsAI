"""Build the run.json record shared by Panda training entry points."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig, TrainingArguments


def describe_cache(path: Path, shape: tuple[int, ...]) -> dict[str, Any]:
    """Fingerprint the actual window file, independently of preparation metadata."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "shape": list(shape), "sha256": digest}


def build_run_metadata(
    *,
    settings: dict[str, Any],
    training_args: "TrainingArguments",
    data: dict[str, Any],
    model_config: "PretrainedConfig",
    polynomial_indices_restored: bool,
    project_root: Path,
    resume_from_checkpoint: Path | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the initial record; callers update metrics/status and write the file.

    Resumed runs retain the original checkpoint revision and initial evaluation
    metrics. The metrics dictionary can be updated in place during training.
    """
    metrics = {
        name: value for name, value in (previous["metrics"] if previous else {}).items()
        if name.startswith("initial_eval_")
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    resolved_revision = previous["resolved_checkpoint_revision"] if previous else getattr(model_config, "_commit_hash", None)
    return {
        "status": "running", "settings": settings, "training_arguments": json.loads(training_args.to_json_string()),
        "data": data, "world_size": training_args.world_size, "device": str(training_args.device),
        "versions": {name: version(name) for name in ("panda", "torch", "transformers", "accelerate", "numpy")},
        "python_version": sys.version.split()[0], "project_commit": commit, "project_dirty": dirty,
        "resolved_checkpoint_revision": resolved_revision,
        "polynomial_indices": "restored" if polynomial_indices_restored else "generated_at_initialization",
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None, "metrics": metrics,
    }
