# Panda upstream provenance

This directory contains a minimal source snapshot from:

- Repository: https://github.com/abao1999/panda
- Upstream commit: `c229e7c8c49cbe294458c44160248bb17856a715`
- Imported: 2026-09-23
- Upstream license: MIT; see `LICENSE`

## Files retained

The snapshot contains the PatchTST implementation and inference pipeline used by
the published Panda checkpoints, plus the evaluation utility imported by that
pipeline. Dataset generation, training, plotting, Chronos, notebooks, scripts,
configuration files, and experimental assets are intentionally excluded.

## Local changes

1. In `panda/patchtst/pipeline.py`, the two absolute `panda.*` imports were
   changed to package-relative imports so the snapshot can live under this
   project's `_vendor` namespace.
2. `panda/patchtst/__init__.py` was added to make the package boundary explicit.
3. `panda/utils/__init__.py` was replaced with a minimal package initializer.
   Upstream eagerly imports all utility modules there, including training,
   plotting, dataset, and `dysts` functionality not included in this inference
   snapshot.

No model implementation or checkpoint-loading behavior was otherwise changed.
