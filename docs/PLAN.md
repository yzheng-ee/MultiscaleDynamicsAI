# Research plan: PANDA on multiscale dynamics

**Status:** PR-1 (foundation + first benchmark) implemented; its results have already
revised the central hypothesis -- see section 3. PR-2 onwards are proposals.
**Last updated:** 2026-09-17

---

## 1. The question

PANDA ([Lai, Bao & Gilpin, ICLR 2026](https://arxiv.org/abs/2505.13755)) is a 21M-parameter
encoder-only transformer pretrained on ~2x10^4 algorithmically discovered chaotic ODEs. Its
headline result is *zero-shot* forecasting of unseen systems, including emergent transfer to
high-dimensional PDEs it never saw in training.

Every system in its training set is **low-dimensional (d = 3) and single-scale**. Real
scientific and engineering systems mostly are not: they have fast variables slaved to slow
ones, and the interesting predictability lives on the slow manifold.

> **Central question.** How does a pretrained chaotic-dynamics model behave when the dynamics
> are multiscale, and can the classical single-scale closure be used to help it?

The Lorenz '96 multiscale system is the right test-bed because it has a knob. The
scale-separation parameter `eps` continuously interpolates between nearly-single-scale
(`eps ~ 1`) and strongly separated (`eps = 2^-7`), while `K`, `F` and the slow-variable
statistics stay essentially fixed. And averaging theory hands us a *principled* reduced model
to compare against, rather than an arbitrary baseline.

---

## 2. What the system actually looks like (measured, not assumed)

At the reference parameters `K=9, J=8, h_x=-0.8, h_y=1, F=10, eps=2^-7`
(Example B.1 of Calvello, Reich & Stuart 2025):

| Quantity | Value | Why it matters |
|---|---|---|
| Slow dominant period | **1.44-1.58** time units (estimator-dependent) | Sets the PANDA-native sampling rate: `sample_dt ~ 0.015`, giving 96-118 points/period against PANDA's 102.4 |
| `lambda_max`, full system | **53.3** at `eps = 2^-7` | The **fast** exponent. Scales like `1/eps`: 53.3, 22.1, 7.6, 1.70 across the sweep |
| `lambda_max`, single-scale | **0.80** (cubic closure) / 1.06 (GP) | The right yardstick for *slow-variable* forecast horizons |
| Slow Lyapunov time | **1.26** time units = 84 samples | A 512-step context spans 6.1 of these; a 128-step horizon spans 1.5 |
| Slow-variable std | ~3.6 per channel | |
| `ybar` std | ~1.4 per channel | The signal a closure has to reproduce |
| Closure residual std | **0.63** (cubic fit, 1.1M pairs) | The cubic explains ~80% of `ybar`'s **variance**; the residual std is ~45% of `ybar`'s std |

Three consequences that shape every experiment:

1. **Report horizons in slow Lyapunov times.** The full system's maximal Lyapunov exponent is
   dominated by the fast ring and says nothing about slow-variable predictability. Quoting
   "128 steps" or even "2 time units" hides the comparison. Note also that the GP and cubic
   closures give measurably *different* reduced dynamics (`lambda_max` 1.06 vs 0.80, dominant
   period 1.62 vs 1.77), the choice of closure is not cosmetic.
2. **Sampling rate is a first-class variable, and "the period" is not well defined.** PANDA was
   trained at ~102 points per dominant period. Feed it Lorenz '96 at `dt = 0.001` and we have
   changed the distribution before we have changed anything about the dynamics. But for
   broad-band chaos the estimate depends on the estimator: on eight `dysts` systems generated at
   a known 102.4, a naive spectral-peak estimator returns 51-195 while an amplitude-weighted
   centroid returns 63-113. We default to the centroid and treat the *empirical density sweep*,
   not the estimator, as the real control.
3. **A memoryless closure leaves ~20% of the forcing variance unexplained.** The fitted cubic is
   `m(x) = 0.187 + 0.402x + 0.0042x^2 - 0.0017x^3`. Its residual is *not white*,it is
   autocorrelated over several sampling intervals (notebook 02), so what the closure is missing
   *includes* temporal structure. Mori-Zwanzig memory is the natural candidate and is what PR-4
   targets, but autocorrelated residuals do not on their own establish it: state-dependent model
   error, an imperfect fit of `m`, and dependence on the unobserved fast state would all produce
   correlated residuals too. Distinguishing them is part of PR-4, not an assumption going in.

---

## 3. First results (PR-1), and what they do to the plan

All numbers below: 48-64 windows, context 512, horizon 128 (one forward pass, no rollout),
`sample_dt = 0.015`, the 21M `GilpinLab/panda` checkpoint. **Valid prediction time (VPT)** is
the lead time at which normalised RMSE first exceeds 0.3 of the climatological spread, quoted in
**slow Lyapunov times** (slow Lyapunov time = 1.26).

### The adapter is sound

On systems from PANDA's own held-out set, generated with `dysts` exactly as the authors do:

| System | sMAPE | RMSE / std | VPT (periods) |
|---|---|---|---|
| QiChen | 24.7 | 0.57 | 1.00 |
| Lorenz | 19.5 | 0.17 | 1.08 |
| Rossler | 50.7 | 0.15 | 0.83 |

The paper reports a median sMAPE of 27.6 at `L_pred = 128` across 9.3x10^3 held-out systems, so
this is in line. Climatology on the same windows scores sMAPE 121-168. Nothing is wrong with
the plumbing.

### On Lorenz '96, PANDA is close to the floor

| Forecaster | single-scale VPT | multiscale VPT |
|---|---|---|
| persistence | 0.042 | 0.041 |
| climatology | 0.000 | 0.000 |
| linear-AR (fitted on the context) | 0.161 | 0.140 |
| **PANDA (zero-shot)** | **0.129** | **0.128** |
| closure model (averaged physics) | 1.529 * | 1.381 |

\* on single-scale data the closure model *is* the generating process, so that entry is a
consistency check, not a baseline. Its multiscale entry is the meaningful one, and it is
excellent: classical averaging tracks the true two-scale slow variables for well over a
Lyapunov time.

PANDA reaches sMAPE ~114 and RMSE/std ~1.15, roughly three times persistence on VPT, but no
better than a *linear autoregression fitted on the context window itself*, and worse than
climatology on horizon-averaged RMSE. Against its own domain (sMAPE ~25, VPT ~1 period) this is
a collapse, not a degradation.

### The controls rule out the obvious explanations

| Control | Result | Verdict |
|---|---|---|
| Sampling density, 112 -> 56 -> 28 points/period | VPT 0.150, 0.142, 0.145 | **Flat across 4x.** Not a sampling artifact. |
| Channel count, 3 -> 5 -> 9 | VPT 0.219, 0.160, 0.150 | Real but modest; nowhere near enough to close the gap. |
| Adapter, on PANDA's own held-out systems | reproduces published sMAPE | Not a bug. |

And the direct test of the original hypothesis, the scale-separation sweep itself:

| `eps` | PANDA VPT | persistence VPT |
|---|---|---|
| 2^-7 | 0.128 | 0.041 |
| 2^-5 | 0.136 | 0.043 |
| 2^-3 | 0.135 | 0.045 |
| 2^-1 | 0.141 | 0.045 |

Skill is flat to within ~10% across a 64-fold change in scale separation, and what little trend
there is tracks persistence i.e. it is a property of the data, not of the model.

### This reorders the roadmap

**H1 as originally posed is not supported in this regime.** PANDA scores essentially identically
on the single-scale and multiscale slow variables (VPT 0.129 vs 0.128), and flat across the whole
`eps` sweep. More to the point, it already performs poorly on the corresponding *single-scale*
problem, which has no fast variables at all. So within this setup there is no evidence that
scale separation is what limits it. That is weaker than "PANDA is at a mathematical floor", which
we have not shown; it is simply that the experiment cannot attribute the failure to
multiscale structure.

The first-order effect is therefore **not** scale separation. The leading remaining hypothesis is
**attractor dimension**, and unlike the original H1 this one now has a measurement behind it.

The PANDA paper characterises its corpus by Grassberger-Procaccia correlation dimension:
`2.09 +/- 0.27` for the 129 founder systems, `2.11 +/- 0.23` for the evolved skew systems. (Table 2 in the ICLR 2026 PDF in `Literatures/`) We implemented *their* estimator (App. F gives the recipe: pairwise distances,
5th-50th-percentile scaling region, Clauset-Shalizi-Newman power-law MLE) and applied it to
both corpora:

| Attractor | GP dimension |
|---|---|
| PANDA founder systems - **as the paper reports** | 2.09 +/- 0.27 |
| PANDA founder systems - **our implementation, 10 systems** | **2.12 +/- 0.20** |
| L96 slow variables, `K = 9`, single-scale | **3.41 +/- 0.13** |
| L96 slow variables, `K = 9`, multiscale `eps = 2^-7` | 3.25 +/- 0.11 |
| L96 slow variables, `K = 9`, multiscale `eps = 2^-1` | 3.33 +/- 0.11 |
| L96 full multiscale state, 81 dimensions | 6.86 +/- 0.25 |
| L96 single-scale, first 3 channels only | 2.37 +/- 0.02 |

Our implementation reproduces their published figure, so the comparison is on the same footing.
Three things follow.

**Lorenz '96 sits far outside the corpus.** 3.41 +/- 0.13 against a founder-system distribution
of 2.09 +/- 0.27, about 4.9 founder-set standard deviations above its mean. Note what that
number is and is not: 0.27 is the spread *across systems*, not a standard error, so this
describes where L96 falls in their distribution and is not a significance test. And because the
estimator is *compressive* above `D2 ~ 2` (see `msdyn.diagnostics.dimension`: a true 6-cube reads
3.75), the gap is a **lower bound** on the true one.

**Dimension barely moves with `eps`** (3.25 -> 3.33 across a 64-fold change), which is exactly
what the flat skill-vs-`eps` curve would predict if dimension is what binds.

**The 3-channel slice is the suggestive one.** It is the only L96 configuration whose dimension
lands inside the corpus range (2.37, ~1 sigma above the founder mean), and it is also where PANDA
performs best (VPT 0.219 vs 0.150 at 9 channels). That is consistent with the hypothesis but does
not establish it: channel count and measured dimension are confounded here, and a 3-channel
projection of a 9-dimensional attractor underestimates the true dimension by construction.

**Revised H1.** Zero-shot skill is governed primarily by attractor dimension, not by scale
separation. Concretely: a sweep over `K` at fixed `eps` will produce a much steeper skill curve
than a sweep over `eps` at fixed `K`, and skill should track measured `D2` rather than `K` itself.

Note what is *not* yet established. We have measured correlation dimension, not the Lyapunov
spectrum, only `lambda_max`. The claim that Lorenz '96 at `K = 9` has many positive exponents
is standard in the literature but is not something this repository has computed. PR-2 should
either compute the spectrum (the single-scale model is a non-stiff 9-dimensional system with an
analytic Jacobian, so this is cheap) or drop the claim.

If that holds, the project's framing sharpens usefully: the interesting question stops being
"does PANDA handle multiscale dynamics?" and becomes "**the averaged single-scale model is a
low-dimensional surrogate for a high-dimensional system -- can a pretrained low-dimensional model
be made useful by working in that reduced space?**" That is a better fit for the reduced-data
direction in the project brief, and it makes the closure the centre of the project rather than a
baseline.

---

## 4. Roadmap

### PR-1 -- Foundation *(implemented)*

| Piece | Where |
|---|---|
| Vectorised Lorenz '96 (multiscale + single-scale), verified against the Burov reference | `src/msdyn/systems/l96.py` |
| Batched RK4 with a calibrated stiff step size, plus convergence diagnostics | `src/msdyn/systems/integrate.py` |
| Closures: GP (reference method), polynomial (Wilks), linear/G0, tabulated | `src/msdyn/closures/` |
| Dataset format with mandatory provenance; ensemble generation | `src/msdyn/data/` |
| Lyapunov exponents, power spectra, PANDA-density resampling | `src/msdyn/diagnostics/` |
| Forecaster interface; persistence / climatology / linear-AR / closure-model / perfect-model | `src/msdyn/models/` |
| PANDA adapter (dependency-light, explicit rollout control) | `src/msdyn/models/panda.py` |
| Weather **and** climate metrics; head-to-head harness | `src/msdyn/evaluation/` |

Notebooks `01`--`03` walk through generation, closure fitting, and the first zero-shot forecast.

### PR-2 -- What actually limits zero-shot skill: dimension, or scale separation?

Driven by the PR-1 result above. **Revised H1:** attractor dimension is the first-order
constraint; scale separation is second-order.

- **Primary experiment: sweep `K in {3, 4, 5, 6, 9, 12, 16}` at fixed `eps`.** `K = 3` is the
  key point, but be precise about why. The paper trains "exclusively on `d = 3`-dimensional
  dynamical systems" (S4.1) in the sense that it *randomly samples 3 channels from each
  multivariate trajectory to enable efficient batching*, processing full multivariate
  trajectories only at inference (Appendix, model-configuration section); its founder systems
  themselves span `d_min = 3` to `d_max = 10` and undergo channel-mixing and time-delay
  augmentations. So `K = 3` matches the **training-time observed channel count**, not
  necessarily PANDA's intrinsic attractor dimension. Both are worth separating.
- Measure the correlation (Grassberger-Procaccia) dimension of each attractor so the curve can
  be plotted against the same quantity the PANDA paper reports for its training set, rather than
  against `K`.
- Re-run the `eps` sweep *at the `K` where PANDA has skill*. Only there is there headroom to
  detect a multiscale effect at all. This is the corrected version of the original H1.
- Resample per `eps` rather than holding `sample_dt` fixed, so the sweep is not contaminated by
  the drift in dominant period.
- Bootstrap confidence intervals on everything; PR-1's window counts are too small to quote.
- Try `GilpinLab/panda-72M`: does scale help, or is this a data-coverage limit rather than a
  capacity limit?

**Deliverable:** skill against attractor dimension and against `eps`, with intervals, and a clear
statement of which one binds.

### PR-3 -- Partial observation

Project brief bullet [3]. The reference data-assimilation experiments observe 6 of 9 slow
variables (`H : R^9 -> R^6`, indices 0,1,3,4,6,7).

- Forecast the unobserved components from the observed ones.
- Compare against 3DVAR and the EnKF from `EnsembleKalmanMethods`, which solve the
  *same* problem with the single-scale model and a known observation operator.
- **Domain-expertise injection:** append the closure tendency `h_x * m(x)` as an extra input
  channel. Channel attention treats channels as an unordered set, so this is a cheap,
  architecturally natural way to hand the model physics, and it is directly the "inject
  physical domain expertise in the latent space" idea from the project brief.

### PR-4 -- PANDA as a closure

The single-scale model replaces `ybar_k` with `m(x_k)`: **memoryless** and a function of the
local slow variable only. The measured residual (~40% of `ybar`'s spread) is what that
approximation discards. Mori-Zwanzig says the exact closure has memory.

- **(a) PANDA-as-closure.** Use PANDA to forecast `ybar(t)` from the slow-variable *history*,
  and integrate the slow ODE with that forcing. This is a data-driven memory closure. Score it
  on reproduction of the **multiscale** slow-variable statistics, not on pointwise error.
- **(b) Hybrid residual.** Integrate the single-scale model, have PANDA forecast the residual
  against the true multiscale slow trajectory, and add them. Tests whether a physics prior plus
  a learned correction beats either alone.
- **(c) Stochastic baseline.** A Wilks-style AR(1) stochastic parameterisation using the
  fitted `residual_std`. Without this, any "memory helps" claim is confounded with "noise helps".

### PR-5 -- Diffusion transformer on the reduced data

Project brief bullet [2]. PANDA is explicitly a *weather* model: trained for short-horizon
pointwise accuracy, and the paper documents that it regresses to the mean at long horizons.
A diffusion model targets the conditional *distribution* instead, which is what long-horizon
multiscale forecasting actually needs. Same harness, same windows, same metrics, the
climate-side metrics (`invariant_measure_kl`, `spectral_hellinger`) are already implemented
and are where a diffusion model should win.

### PR-6 -- Transfer

Fine-tune PANDA on single-scale Lorenz '96 and evaluate on multiscale (and vice versa). Does
the reduced model make a *useful* pretraining corpus for the full one? This is the practical
version of the project's transfer-across-domains motivation.

---

## 5. Things that might bite

- **PANDA's context length is exactly 512.** The prediction head is a flatten-linear over a
  fixed patch count, so a shorter context is a shape error, not a quality loss. The adapter
  raises a clear error rather than padding, because padding is itself a distribution shift.
- **The native prediction length is 128.** Longer horizons are autoregressive rollout, and the
  paper reports rollout degrades (notably for the MLM checkpoint). Always report the rollout
  depth alongside the horizon.
- **Instance normalisation is internal** (`scaling="std"`, per window per channel). Do not
  standardise inputs (would be double normalisation).
- **Integration step.** The fast ring needs `dt ~ 5e-3 * eps` (`~3.9e-5` at `eps = 2^-7`), an
  order of magnitude below what linear stability suggests, because the quadratic terms make
  `|dy/dtau| ~ 60`. Validate with `self_convergence_error`, never by comparing long
  trajectories against an adaptive solver, at `eps = 2^-7` two correct solutions separate
  exponentially within a fraction of a time unit.
- **Burn-in is mandatory.** The reference initial condition (fast variables set equal to their
  parent slow variable) has `|dy/dt| ~ 10^4` and will blow up a fixed-step integrator.
- **The correlation-dimension estimator is compressive.** Reproducing the PANDA paper's recipe
  gives numbers comparable with their reported figures, but not absolute fractal dimensions: a
  uniform 6-cube reads 3.75, a 1-cube reads 1.61. It is near-unbiased at `D2 ~ 2`, which is where
  their corpus sits, so the comparison is fair but any gap we measure is a lower bound, the
  numbers should never be quoted as absolute dimensions, and the corpus spread is an
  across-system standard deviation rather than a standard error, so "N sigma" phrasing implies a
  significance test that has not been done.
- **Never integrate against a raw sklearn GP.** A GP posterior costs ~20 ms per call against an
  800-point training set; at four calls per RK4 step a single long run takes hours. Wrap it in
  `TabulatedClosure` (~100x faster, agrees to ~1e-7 on the attractor). The wrapper also
  extrapolates linearly instead of reverting to the GP's zero prior, a GP asked for `m(x)` far
  off the attractor would switch the fast forcing off entirely, which is both wrong and a route
  to a blown-up integration.

---

## 6. Attribution

- Lorenz '96 model equations and the GP closure methodology: D. Burov, via
  [EdoardoCalvello/EnsembleKalmanMethods](https://github.com/EdoardoCalvello/EnsembleKalmanMethods);
  Burov, Giannakis, Manohar & Stuart, *Kernel analog forecasting*, MMS 19(2), 2021.
- Averaging-principle framing and Example B.1: Calvello, Reich & Stuart,
  *Ensemble Kalman methods: a mean-field perspective*, Acta Numerica 34, 2025.
- PANDA: Lai, Bao & Gilpin, ICLR 2026, [arXiv:2505.13755](https://arxiv.org/abs/2505.13755);
  code [abao1999/panda](https://github.com/abao1999/panda), weights `GilpinLab/panda`.
