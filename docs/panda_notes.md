# PANDA: what the checkpoint actually requires

Working notes on the constraints the pretrained model imposes, gathered from the config, the
source, and testing. All of these are enforced or documented in `src/msdyn/models/panda.py`.

**Paper:** [arXiv:2505.13755](https://arxiv.org/abs/2505.13755) · **Code:**
[abao1999/panda](https://github.com/abao1999/panda) · **Weights:** `GilpinLab/panda` (21M),
`GilpinLab/panda-72M` (72M); MLM variants `GilpinLab/panda_mlm`, `panda_mlm-66M`

## Checkpoint configuration

| Field | `panda` (21M) | `panda-72M` |
|---|---|---|
| `context_length` | 512 | 512 |
| `prediction_length` | 128 | 128 |
| `patch_length` / `patch_stride` | 16 / 16 | 16 / 16 |
| `d_model` / `ffn_dim` | 512 / 512 | 768 / 768 |
| `num_hidden_layers` / heads | 8 / 8 | 12 / 12 |
| `channel_attention` | true | true |
| `use_dynamics_embedding` | true | true |
| `num_poly_feats` / `num_rff` | 120 / 256 | 188 / 376 |
| `scaling` | `std` | `std` |
| `distribution_output` | null (deterministic, MSE loss) | null |

## Hard constraints

1. **Context must be exactly 512 steps.** `PatchTSTPredictionHead` is a flatten-linear over a
   fixed patch count, so a shorter context is a shape error rather than a quality loss. Our
   adapter raises instead of padding — padding with NaN or zeros is itself a distribution shift.
2. **128 steps per forward pass.** Longer horizons are autoregressive rollout with a sliding
   context. The paper reports rollout degradation (severe for the MLM checkpoint). Report the
   rollout depth alongside any horizon.
3. **Normalisation is internal.** `scaling="std"` applies instance normalisation per window and
   per channel inside the model. Standardising the input ourselves normalises twice.
4. **Sampling density is part of the training distribution.** Everything was generated with
   `dysts` at `4096 / 40 ≈ 102` points per dominant period. Not a convention — a distribution.
5. **Trained at `d = 3`.** Channel attention is the claimed mechanism for generalising to other
   widths; Lorenz '96 (9 or 81 channels) tests that claim well past where the paper does.

## Importing without `dysts`

`panda.patchtst.pipeline.PatchTSTPipeline` imports `panda.utils`, whose `__init__` pulls in
`dysts` and `gluonts` for dataset plumbing. For inference only:

```python
from panda.patchtst.patchtst import PatchTSTForPrediction   # no dysts, no gluonts
model = PatchTSTForPrediction.from_pretrained("GilpinLab/panda").to(device).eval()
generated = model.generate(context)          # context: (batch, 512, n_channels)
step = generated.sequences.median(dim=1).values   # (batch, 128, n_channels)
```

`sequences` is `(batch, num_samples, prediction_length, n_channels)`. With a deterministic head
`num_samples` is 1; the median contraction is what PANDA's own pipeline does and is kept for
parity.

## Measured on this machine (Apple M4, MPS)

- Load: ~5 s · 21.4M parameters
- Forecast, 4 windows × 9 channels × 256 steps (2 rollout chunks): ~0.2 s

## Dependency pins

`transformers==4.40.2` and `numpy<2` are hard PANDA requirements. `huggingface_hub` must be
`<0.26` — later releases removed APIs that transformers 4.40 still calls.
