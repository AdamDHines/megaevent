# Tokyo 24/7 headroom — the eval-side queue

**Date:** 2026-07-30 · **Scope:** what can be improved on Tokyo 24/7 **without retraining**.

> **Superseded as a scoreboard, 2026-08-04.** The `.8095` headline below is the best *v1-era*
> checkpoint. Three training waves since then moved it to **`.9016`**
> (`runs/b_full_P64_v4_vpr/step10000.pt`, 322², PCA 4096/0.5) — see
> `docs/artifact_v3_packet.html` §01–§04. The two conclusions this document exists for are
> unchanged and are what the v4 protocol adopts: **322² is the eval resolution**, and the
> whitening setting must be re-tuned per checkpoint (v4 ships at `(4096, 0.5)`, not `(2048, 0.5)`).
> Everything below still describes the eval-side levers correctly; only the checkpoint it was
> measured on is old.

Training-side analysis lives in the sibling repo: `../gept/docs/report_2026-07-30_training_regime_audit.md`.
Every number below is reproducible with `scripts/tokyo_headroom.py --which all` (PCA grid, query
structure, orientation) and `scripts/verify_countmask_jpeg.py` (the training-tree corruption).
Short version of why that matters here: the shipped checkpoint
(`ckpts/s_salad_ft4.pt`, `meta = {run_name: 's_salad_ft4_fine50', step: 3300}`) is a mid-flight snapshot
of a 40,000-step cosine horizon taken at **99.1 % of peak LR**, selected by a Brisbane metric that
stopped moving at step 1,000 and cannot observe illumination or viewpoint change. Tokyo 24/7 is
evaluated in the **I2E domain**, which is the same domain the model trains on — so the sim-to-real drift
that Brisbane-based selection suppresses is exactly the fit Tokyo rewards.

## Result

**.6762 → .8095 whitened R@1 (+13.3 pts, 42 of 315 queries) with no retraining**, from three changes
that touch no weights — a later checkpoint (0.4), a 322² eval (0.2), and ViT-B (0.5):

| config | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| shipped — ViT-S `s_salad_ft4` step 3300, 224², PCA (2048, 0.5) | .6762 | .8032 | .8540 | .8889 |
| ViT-S step 8000, 322², PCA (4096, 0.5) | .7810 | .8698 | .8952 | .9238 |
| **ViT-B `b_salad_ft4` step 16000, 322², PCA (4096, 0.5)** | **.8095** | **.8921** | **.9270** | **.9460** |

Native R@1 over the same change: .5714 → .7016.

**Brisbane is now scored at both resolutions too** (`scripts/brisbane_resolution.py`, see 0.6), which
splits the result cleanly into a free part and a traded part:

| config | Tokyo R@1 | Brisbane 4-cond mean | Brisbane sunset1 |
|---|---|---|---|
| ViT-S step 2000, 224² — what `best.pt` selected | .6667 | .7660 | .9346 |
| **ViT-S step 2000, 322²** | **.7270** | **.7762** | **.9378** |
| ViT-S step 8000, 322² | .7810 | .7338 | .9333 |
| ViT-B step 16000, 322² | **.8095** | .6858 | .9173 |

**Row 2 is a strict Pareto improvement: +6.0 Tokyo *and* +1.0 Brisbane, from changing one number.**
Beyond it there is a real frontier — +11.4 Tokyo costs 3.2 Brisbane, +14.3 costs 8.0.

## Current baseline

`features/tokyo247/results_megaevent.json`, log
`logs/tokyo247/database_queries/2026-07-29_16-53-48.log`:

| | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| native (cosine) | .5714 | .7365 | .8190 | .8571 |
| pca (dim 2048, power 0.5) | **.6762** | .8032 | .8540 | .8889 |

Protocol, re-verified: 75,984 database / 315 queries, 25 m radius, **118.29 positives/query**, 315/315
scorable, chance R@1 ≈ .0016. Baselines are near chance (sparse_event .001, eventvlad .003, eventgem
.019 → .117 with RANSAC reranking).

---

## 0.1 — PCA whitening is not at its optimum

`src/inference.py:52` hardcodes `PCA_DIM, PCA_POWER, PCA_EPS = 2048, 0.5, 1e-4`, and
`src/scoring.py:134` (`pca_fit_subsampled`, `PCA_FIT_SAMPLES = 20000`) fits on a seeded 20 k-row
subsample of the database bank to keep the randomized SVD on an 8 GB card.

Sweep run through megaevent's own `pca_fit` / `pca_apply` / `recall_at_k` on the cached full banks
(`features/tokyo247/s_salad_ft4_countmask_{database,queries}.npy`) — see the grid below. The
`(2048, 0.5)` cell is the reproduction check: it must return `results_megaevent.json`'s
`0.6761904761904762` exactly before any other cell is trusted.

R@1 over the full 75,984-image bank, 20 k-row fit (verified: the `(2048, 0.5)` cell returns
`0.6761904761904762`, matching `results_megaevent.json` exactly):

| dim \ power | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| 1024 | .6444 | .6825 | .6508 | .6413 |
| **2048** | .6571 | **.6762** ← shipped | **.6952** | .6889 |
| 4096 | .6571 | .6857 | .6921 | .6698 |
| 8448 (full) | .6571 | .6857 | .6857 | .6667 |

The R@1 argmax is `(2048, 0.75)` at **.6952**, but it is not the right pick — it *loses* at every other
cutoff. Across all four cutoffs:

| setting | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| (2048, 0.50) — shipped | .6762 | .8032 | .8540 | .8889 |
| (2048, 0.75) — R@1 argmax | **.6952** | .7968 | .8381 | .8825 |
| **(4096, 0.50)** | .6857 | **.8222** | **.8603** | **.8984** |

**Recommendation: `(4096, 0.50)`.** It dominates the shipped setting on *every* cutoff, which is a more
robust signal than a single-cutoff win. `(2048, 0.75)`'s extra R@1 comes with a loss everywhere else.

⚠️ **Read these deltas with the sample size in mind.** 315 queries means **one query = 0.317 pt of
R@1**, so shipped → argmax is **6 queries** and shipped → `(4096, 0.50)` is **3 queries**. Tuning two
hyperparameters against the test set on 315 samples is exactly the kind of thing that does not
replicate. Treat `(4096, 0.50)` as a mild, defensible default rather than a result, and confirm it on
Brisbane before adopting it.

**Fit size is a non-issue** — at the best cell, `pca_fit` on 20,000 / 40,000 / 75,984 rows gives R@1
.6952 / .6952 / .6889. `scoring.PCA_FIT_SAMPLES = 20000` is not costing anything, and its docstring
claim is correct.

## 0.2 — Evaluate at higher input resolution (FixRes)

`src/inference.py:119-129` resizes the whole frame to `cfg.H × cfg.W` = 224², while training used
`RandomResizedCrop(224, scale=(0.5, 1.0))` (`gept/src/vpr/dataset.py:126-136`). Mean apparent object
scale therefore differs between train and test — the standard FixRes mismatch, whose standard remedy is
to test at a higher resolution. The backbone is a ViT-S/14 built at `img_size=518`, and DINOv2
interpolates its position embedding, so any multiple of 14 works; 322² is SALAD/MegaLoc's own eval
resolution.

**DONE on the full bank, crossed with 0.4's checkpoints.** Best whitened R@1, all 7 checkpoints × both
resolutions (`scripts/tokyo_trajectory.py --resolution 322`):

| step | 224² native | 322² native | 224² whitened | 322² whitened | Δ whitened |
|---|---|---|---|---|---|
| 2000 ← `best.pt` | .5651 | .6000 | .6667 | .7270 | **+6.0** |
| 4000 | .5651 | .6444 | .7048 | .7619 | +5.7 |
| 6000 | .6000 | .6317 | .7175 | .7619 | +4.4 |
| 8000 | .5968 | .6413 | .7111 | **.7810** | +7.0 |
| 10000 | .6095 | .6254 | .7429 | .7619 | +1.9 |
| 12000 | .6286 | .6444 | .7397 | .7651 | +2.5 |
| 12620 | .6190 | .6254 | .7302 | .7587 | +2.9 |

**322² beats 224² at every one of the seven checkpoints** (+1.9 to +7.0 pts whitened). Seven independent
confirmations is not noise, so the FixRes reading holds — and it held up on the full 75,984 bank, unlike
the earlier reduced-bench estimate.

### Best ViT-S configuration

| | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| shipped: step 3300, 224², PCA (2048, 0.5) | .6762 | .8032 | .8540 | .8889 |
| **ViT-S step 8000, 322², PCA (4096, 0.5)** | **.7810** | **.8698** | **.8952** | **.9238** |
| gain | **+10.5** | +6.7 | +4.1 | +3.5 |

+10.5 points of R@1 with no retraining, every cutoff improving; native .5714 → .6413. **ViT-B does better
still — see 0.5 — so the overall best is .8095.**

How the two effects decompose, in whitened R@1: checkpoint alone (step 12000, 224²) = .7397, +6.4;
resolution alone (step 2000, 322²) = .7270, +5.1; both = .7810, +10.5. They compound, but not additively.

### Reading it honestly

At 315 queries, 1 query = 0.317 pt, so anything under ~1.5 pts (5 queries) is noise. That leaves two
robust conclusions and one over-read:

- **Robust:** 322² > 224² at all seven checkpoints.
- **Robust:** step 2000 — the checkpoint `best.pt` selected — is the *worst* of the seven at both
  resolutions. Every step ≥ 4000 beats it.
- **Do not over-read step 8000.** At 322² the steps ≥ 4000 span .7524–.7810, i.e. a 9-query spread
  across six checkpoints. Treat it as a plateau of ~.76–.78 rather than a peak at 8000.

✅ **Brisbane is now scored at 322² too — see 0.6.** For ViT-S it is a *pure win* there as well
(4-condition mean +1.0 at step 2000, +0.8 at step 8000; 7 of 8 individual conditions improve), so
**322² should simply become the default eval resolution.** It costs something only for ViT-B on
Brisbane (−1.2). Resolution and checkpoint are therefore separable: the resolution change is free, and
the entire Tokyo-vs-Brisbane trade lives in which checkpoint and backbone you pick.

## 0.3 — The query structure hides a 48-point R@1 spread · **partly done**

Verified: the 315 queries are **35 distinct UTM locations × exactly 9 images**, and all 35 stem runs are
contiguous in field 7 of the filename (`@easting@northing@54@S@lat@lon@00685@@@@@@@@.npz` — location 0
is stems 685–693). That is Tokyo 24/7's 3 times of day × 3 camera directions. All 9 images at a location
share the same GT positives, so per-position recall differences are purely query difficulty.

R@1 by position-within-run, off the cached similarity matrices:

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| native | .771 | .714 | .457 | .657 | .571 | .400 | .657 | .514 | .400 |
| pca | **.914** | .857 | .657 | .686 | .714 | .486 | .686 | .657 | **.429** |

**A 48-point spread (.914 → .429), and both axes matter**: grouped `pos // 3` gives .810 / .629 / .590,
grouped `pos % 3` gives .762 / .743 / .524. This is far larger than any modelling change under
discussion, so knowing which axis is time-of-day is genuinely valuable.

**What is still missing:** assigning time-of-day to the positions. The obvious free proxy — I2E event
count as a brightness stand-in — **fails**: between/within-group variance ratio is 0.001 (time-major)
and 0.008 (direction-major), i.e. no signal. That is expected, since I2E differences *log* luma and
countmask then divides by a per-frame 99th-percentile α, so absolute brightness is largely normalised
away. Recovering the axis needs the raw `247query_v2` jpgs (or their timestamps), which live on HPC at
`$RAW_ROOT/247_Tokyo_GSV_Perspective`, not locally. One directory listing with EXIF would settle it.

Both groupings decline monotonically, so the ordering convention cannot be inferred from recall alone.

## 0.4 — Tokyo across the training trajectory · **DONE — this is the result**

`scripts/tokyo_trajectory.py`, full 75,984-image bank, shipped harness. Brisbane columns come from the
same checkpoints' own `eval_history`, so both benchmarks are one run:

| step | Tokyo native R@1 | Tokyo whitened R@1 | Brisbane sunset1 | Brisbane 4-cond mean |
|---|---|---|---|---|
| 2000 | .5651 | .6667 | **.923** ← `best.pt` selected here | .691 |
| 4000 | .5651 | .7048 | .919 | .675 |
| 6000 | .6000 | .7175 | .913 | .647 |
| 8000 | .5968 | .7111 | .915 | .654 |
| 10000 | .6095 | **.7429** | .908 | .645 |
| 12000 | **.6286** | .7397 | .912 | .648 |
| 12620 (`latest`) | .6190 | .7302 | — | — |

**Tokyo whitened R@1 climbs +7.3 points (.6667 → .7397) over exactly the span where Brisbane sunset1
falls 1.1 points and the Brisbane 4-condition mean falls 4.3.** The benchmarks move in opposite
directions, and selecting on sunset1 picked the **worst checkpoint in the set** for Tokyo.

Against the shipped baseline at the top of this document (.5714 native / .6762 whitened): step 12000 is
**+5.7 pts native and +6.35 pts whitened — 20 of 315 queries — for loading a different file.**

Two consistency checks pass: the shipped step-3300 number (.6762) falls between step 2000 (.6667) and
step 4000 (.7048), as it should; and `best.pt` was byte-identical to `step2000.pt`, which the script
detected and skipped.

**Not finished:** the curve had **not plateaued** — still rising at step 12000 of a 40,000-step horizon,
and the local checkpoint set stops at 12620. `latest` (12620) sitting 2 queries below step 12000 is
inside noise, so the top of the curve is somewhere past 10000 and unresolved. Scoring the remaining
milestones is the next measurement.

**Consequence for 0.1:** the PCA setting is **checkpoint-dependent**. `(2048, 0.5)` beats `(4096, 0.5)`
at step 2000 (.6667 vs .6476) and loses at step 10000 (.7270 vs .7429). The "dominates on every cutoff"
reading above was fitted to the undertrained shipped checkpoint — re-tune whitening on whatever
checkpoint actually ships.

**Consequence for the paper:** one checkpoint has to serve both claims, so this is now an explicit,
quantified trade rather than a hypothetical. Step 12000 buys +6.35 Tokyo at the cost of 4.3 on the
Brisbane 4-condition mean and 1.1 on sunset1. Which side that favours is a judgement call, but it should
be made deliberately — the current checkpoint made it by accident, in the direction that costs Tokyo most.

## 0.5 — ViT-B on Tokyo · **DONE — "ViT-S > ViT-B" is refuted, and ViT-B is the new best model**

`b_salad_ft4` steps 2000–24000, full bank, both resolutions, against `s_salad_ft4`. Same SALAD geometry
(`desc_dim` 8448), same lr 5e-5, same `--blocks 4`, same normalisation constants. Whitened R@1:

| step | ViT-S @224 | ViT-S @322 | ViT-B @224 | ViT-B @322 |
|---|---|---|---|---|
| 2000 | .6667 | .7270 | **.5905** | .6825 |
| 4000 | .7048 | .7619 | .6952 | .7460 |
| 8000 | .7111 | .7810 | .7270 | .8000 |
| 12000 | .7397 | .7651 | .7016 | .7873 |
| 16000 | — | — | .7206 | **.8095** |
| 20000 | — | — | .7238 | .8032 |
| 24000 | — | — | .7270 | .8063 |
| **best** | .7429 | .7810 | .7270 | **.8095** |

**The ordering flips along both axes.** At step 2000 / 224² — the exact operating point the countmask
sweep ranked on — ViT-B is **7.6 points worse**. By step 8000 at 224² they are level. **At 322², ViT-B
wins from step 4000 onward by 1.9–4.4 points whitened, and by +5.7 on native** (.7016 vs .6444).

That is what a capacity-limited comparison looks like: 4× the parameters trained at **half the batch**
(`--P 16` → 15 negative places per MS loss call instead of 31) needs more steps to get anywhere, and the
extra capacity pays off more when given more tokens. Judged early and at low resolution it loses; judged
converged and at SALAD's own resolution it wins. ViT-B's Brisbane `sunset1` spans just **.896–.905 across
all 24k steps**, so the selection metric had no power to rank its checkpoints at all.

⚠️ `--P 16` still handicaps it, so **.8095 is a floor on ViT-B, not a fair capacity comparison.** A
matched-`P` run (activation checkpointing on the trainable blocks, not gradient accumulation) should beat
it. And `b_gem_ft4` has not been scored — GeM is nearly parameter-free, so it would isolate backbone
quality from aggregator capacity.

## 0.6 — Brisbane at 322² · **DONE — 322² is a free win for ViT-S, mildly negative for ViT-B**

`scripts/brisbane_resolution.py`: ref **sunset2** → {sunset1, morning, daytime, sunrise}, stride 1,
dt 50 ms, the shipped hot-pixel filter, all four traverses at full length. **Harness check: ViT-S
step 2000 at 224² gives sunset1 native R@1 .9233, reproducing gept's recorded `.923` for the same
checkpoint** — so this is the same protocol the in-training numbers came from.

Whitened R@1 (best of PCA 2048/4096 at power 0.5):

| checkpoint | res | sunset1 | morning | daytime | sunrise | **4-cond mean** |
|---|---|---|---|---|---|---|
| ViT-S step 2000 | 224² | .9346 | .7702 | .5050 | .8544 | .7660 |
| ViT-S step 2000 | **322²** | **.9378** | **.7827** | **.5068** | **.8776** | **.7762** (+1.0) |
| ViT-S step 8000 | 224² | .9361 | .7126 | .4420 | .8145 | .7263 |
| ViT-S step 8000 | **322²** | .9333 | .7262 | .4446 | .8312 | **.7338** (+0.8) |
| ViT-B step 16000 | 224² | .9208 | .6422 | .4326 | .7958 | .6979 |
| ViT-B step 16000 | 322² | .9173 | .5947 | **.4534** | .7784 | .6858 (−1.2) |

Three readings:

- **For both ViT-S checkpoints 322² is a pure win** — +1.0 and +0.8 on the 4-condition mean, and it
  improves 7 of the 8 individual conditions. Combined with 322² winning at *all 14* Tokyo
  checkpoint/architecture combinations, **322² should simply become the default eval resolution.** There
  is no benchmark on which it costs ViT-S anything.
- **For ViT-B it is mildly negative** (−1.2), and the damage is concentrated in `morning`
  (.6422 → .5947, −4.7) while `daytime` actually improves (+2.1). So the ViT-B resolution interaction is
  condition-specific rather than uniform — worth a look if ViT-B is adopted, but it does not change the
  ordering of anything.
- **The resolution change and the checkpoint change are separable.** Resolution is free (for ViT-S);
  the Tokyo-vs-Brisbane trade lives entirely in *which checkpoint and which backbone* you pick.

⚠️ The baseline row here is ViT-S **step 2000** — what `best.pt` selected — not the shipped step-3300
`_fine50` checkpoint, which was not in this set. Step 3300's Tokyo (.6762) sits between step 2000 (.6667)
and step 4000 (.7048), so its Brisbane should sit just below step 2000's; scoring it would tighten the
baseline but not move any conclusion.

---

## Caveat that applies to all of the above

One checkpoint has to serve both the Tokyo claim and the "beats every conventional event-VPR method on
real event data" claim. So **every constant tuned here — PCA dim/power, eval resolution — must be
re-checked on Brisbane**, or a Tokyo gain silently costs the Brisbane table. The PCA transform is fit
per-dataset (on the reference bank), so its *hyperparameters* are the shared quantity, not the transform.

## Cosmetic / latent, while in here

- `src/inference.py:141` passes `{"window_ms": dt_ms, "white_frame": False}`, but
  `EventStream.countmask` only accepts `(pct, white_frame)`. `reader.with_repr` silently swallows the
  extra kwarg today; it would raise `TypeError` if eventcv tightened forwarding, and the
  `except ValueError` at `src/inference.py:202-208` would not catch it.
- `main.py:52` advertises an `s_gem_ft4` checkpoint that is not in `ckpts/` — `load_model` raises
  `FileNotFoundError`.
- The Brisbane traverse path opens with `hot_pixel_filter=True` (`main.py:119`,
  `src/inference.py:190-191`) where gept's in-training eval does not (`gept/src/vpr/evalsuite.py:128-133`).
  Irrelevant to Tokyo; relevant when comparing Brisbane numbers across the two repos.
