# Deep-dive working notes (2026-08-19 → )

Accumulator for the final artifact (`docs/artifact_v4_deepdive.html`). Each phase appends its
measured results here; every number cites its source file. Plan:
`~/.claude/plans/this-repository-contains-code-vectorized-galaxy.md`.

## Phase 0.4 — baseline snapshot

`docs/phase0_baseline_table.txt` (2026-08-19, `table_native.py` from cached JSONs).
Headline: best megaevent − best baseline R@1 = **+0.070 I2E sim (3/3 ahead), +0.004 recorded
(2/3 ahead)**.

## Phase 0.1 — NYC leave-one-session-out (CONCLUSION-CHANGING)

Source: `evaluations/nycevent/loso.json` / `loso_20260819.txt`
(`scripts/nycevent_all_sessions.py`, cached banks, 25 m native, no PCA).

**The published NYC protocol (random 10% split) is ~90% same-session near-duplicate retrieval.**
Under LOSO every method collapses from ~0.80 to ~0.10 mean R@1:

- Per-session R@1 ranges 0.00–0.44; the best cells are session 2022-12-07_15-46-32
  (MegaLoc **0.438** > ours 0.408) and 2022-12-09_13-59-10 (MegaLoc **0.395** > ours 0.377).
- MegaLoc wins or ties most of the matchable sessions; our NYC "+0.047" lead does not survive
  the protocol correction (exact means in loso.json once the clean rerun lands).
- Session 2022-12-09_14-41-29 has **zero** scorable queries (disjoint route — no other session
  within 25 m of any of its 3,854 frames). Reported as unscorable, excluded from means.
- Scorable fractions are low across the board (e.g. 3,621 queries → 1,578 scorable): the 15
  sessions only partially share routes. NYC-LOSO is a *sparse-overlap, cross-season* benchmark.
- Random-split reproduction check passed for all 10 models (max dev 6e-4 vs 3 dp transcription),
  so the LOSO numbers are scored from exactly the banks that produced the published cells.

Consequence for the ledger: recorded-set wins drop to Brisbane only (pending NSAVP whitened
parity). The corrected NYC row belongs in the artifact as primary, with the random-split row
beside it labeled near-duplicate retrieval.

## Phase 0.2 — whitened-space parity (NSAVP settled)

Source: `nsavp_pooled/results_route0_v5_reproduce.json`, `megaloc_pooled/nsavp.json`
(pre-overwrite snapshot in `json_snapshots_20260819/`).

- Machinery validated: shipped ViT-B MLoc NSAVP native 0.6467 / pca4096p0.5 **0.766**
  reproduced exactly from cached banks through the standard `score_configuration` path.
- **MegaLoc NSAVP whitened = 0.7846** (native 0.7287). So MegaLoc wins NSAVP in BOTH spaces:
  native +0.082, whitened +0.019. Whitening helps MegaLoc (+0.056) as well as us (+0.119).
  The missing-cell asterisk is removed; NSAVP is honestly lost.
- Brisbane + salad/mixvpr/cricavpr whitened cells in flight (phase0_logs/).

## Phase 0.2b — whitening moves EVERY control substantially (native-only convention is not neutral)

Native → pca4096p0.5 R@1 (25 m, from `{method}_pooled/*.json`, 2026-08-19):

| method | Brisbane | NSAVP |
|---|---|---|
| MegaLoc | .782 → **.845** | .729 → **.785** |
| SALAD | .640 → .762 | .654 → .730 |
| MixVPR | .794 → **.872** | .532 → .672 |
| CricaVPR | .722 → .832 | .608 → .750 |

**MixVPR whitened Brisbane (.872) exceeds our best native (.853).** The `tab:native` "no PCA"
convention materially flatters us; per-method-best-space is the honest frame. Our whitened
cells (batch 2) will complete it — NSAVP ours-whitened .766 already known (< MegaLoc .785).

## Phase 0.2c — best-space ledger (recorded sets, 25 m)

REPRODUCIBLE ASSEMBLER: `scripts/table_bestspace.py` (native + whitened + best columns for
Brisbane/NSAVP from cached JSONs, plus the NYC LOSO block). Full table 2026-08-19:

| method | Bris nat | Bris wht | NSAVP nat | NSAVP wht |
|---|---|---|---|---|
| ours B-MLoc s3500 | .820 | .874 | .647 | .766 |
| ours B-SALAD s8000 | .792 | .846 | .650 | .758 |
| ours S-MLoc s3500 | .853 | **.885** | .578 | .707 |
| ours S-SALAD s2000 | .848 | .876 | .590 | .706 |
| MegaLoc | .782 | .845 | .729 | **.785** |
| MixVPR | .794 | .872 | .532 | .672 |
| CricaVPR | .722 | .832 | .608 | .749 |
| SALAD | .640 | .762 | .654 | .730 |
| Event-GeM +rerank | .776 | .877 | .455 | .556 |

**Pre-rerank scoreboard: Brisbane +0.008 (win), NSAVP −0.019 (loss), NYC LOSO tie (~0.107
vs 0.108).** The L1 two-stage prototype is what could move this.

Brisbane (query sunset1): ours native .853 / whitened **.885** (S-MLoc s3500; all four ships
gain +0.03-0.05). Best baselines: Event-GeM pca128+rerank .877, MixVPR whitened .872, MegaLoc
whitened .845. **Best-vs-best lead: +0.008 over Event-GeM, +0.013 over the best RGB control —
the Brisbane lead survives whitened parity, narrowly.**
NSAVP: ours whitened .766 vs MegaLoc whitened .785 → **−0.019 best-vs-best** (was −0.078
native-vs-native). NYC (LOSO, native): MegaLoc .108 vs ours .107 — tie at a collapsed level.
Sources: brisbane_ship/ship_no_sunset2_pca.json, brisbane_v5/noise_projmegaloc_no_sunset2_pca.json,
{method}_pooled/*.json, nsavp_pooled/results_route0_*.json, evaluations/nycevent/loso.json.

## Phase 0.3 — masked sim2real arms (E12 closed)

`table_sim2real.py --masked` now populated (megaevent + Event-GeM; 40,939 × 14,228,
aps-aligned, night excluded). **The vignette control CONFIRMS the domain-advantage reading:**
MegaEvent-minus-Event-GeM margin shifts real→i2e by **+0.083 R@1 masked** (+0.055 unmasked)
and +0.129 R@10 — masking strengthens the shift rather than explaining it. Event-GeM's real
arm improves when the dead vignette is dropped (0.880→0.890) while its i2e arm worsens
(0.756→0.743), exactly as its keypoint stage predicts. Honest artifact line: the I2E-simulated
benchmark leads (+0.05..0.09 vs MegaLoc) carry a measured, vignette-controlled domain
advantage and must be labeled as in-domain results.

## Phase 2 (pulled forward) — H4 heading-erosion evidence

1. **Per-stream loss curves** (wandb output.log, b_noise_projmegaloc_P64_v5): all six streams
   train; SF-XL frontal is the SLOWEST (still ~0.90 at the step-3500 shipping point, 0.82 at
   9000; gsv reaches 0.60). Heading content is not failing to train — it is *under-trained at
   the shipping step*, and training longer (which learns it) is what real-event selection
   forbids. The recipe structurally trades heading selectivity for real-event fit.
2. **Linear heading probe** (scratchpad/heading_probe.py; train FA/RA, test FS/RS): everyone
   ~0.92-0.94 in-condition; cross-condition MegaLoc .627 / SALAD .644 / CricaVPR .568 vs ALL
   fine-tuned models .509-.564 (MixVPR .512 ≈ chance, matching its NSAVP collapse). Moderate
   support: fine-tuning reduces linearly-decodable heading. Behavioral forward-pick evidence
   (monotone decline with fine-tuning) remains the primary mechanism evidence.

## Phase 2 — H1/H3 mechanism results (edges_megaloc_s1.json, cka_megaloc_s1.json)

MegaLoc R@1, sim2real pooled protocol (sunset1 → daytime+morning+sunrise, aps-aligned):
APS photo **0.947** · Sobel magnitude of photo **0.858** · real-event countmask **0.788** ·
fixed-alpha countmask **0.791** · signed-Sobel-as-countmask 0.613.
**Mechanism established:** an edge image retains most of what MegaLoc needs, and events are a
noisy edge rendering (0.79 of a 0.95 ceiling). The per-frame adaptive alpha is IRRELEVANT
(0.788 vs 0.791 — kills that sub-hypothesis) and the countmask channel layout doesn't help
(fake polarity countmask is worst). CKA(photo, events)=0.38 with 52% top-1 agreement —
transfer at the metric level, not shared features. CKA(real_cm, real_cm_ga)=0.98 confirms
alpha-invariance at the representation level.

## Phase 2 — H2/E9 closed (megaloc_pooled/brisbane_event_cmstats.json)

MegaLoc with countmask-matched normalization on Brisbane: native 0.782→**0.812**, whitened
0.845→**0.855**. The ImageNet stats WERE handicapping the control by ~3 pts. Honest
strongest-MegaLoc Brisbane cell = 0.855 (our 0.885 still leads +0.030). MixVPR cmstats arm
queued (batch 4) for parity with the strongest RGB control.

## Phase 3 — L1 rerank RESULT (batch 4, 2026-08-19): two-stage MegaEvent works

With the activity mask (`--min-activity 0.1`, set once from the bonnet-failure diagnosis):

| dataset | global (space) | + rerank | best baseline |
|---|---|---|---|
| Brisbane | 0.8847 (pca4096) | **0.9144** | Event-GeM+rerank 0.877 |
| Brisbane | 0.8525 (native) | 0.9092 | |
| NSAVP | 0.7660 (pca4096) | **0.7804** | MegaLoc whitened 0.785 |
| NSAVP | 0.6467 (native) | 0.7143 (no mask) | |

**Best-vs-best: Brisbane +0.037 (was +0.008 single-stage); NSAVP −0.005 (was −0.078
native-vs-native).** Verified-pair fraction drops 100%→92% with the mask (the static-region
false matches are gone). Config = Event-GeM's own verification defaults (top-50, RANSAC 5.0,
w=0.05) + the mask; sensitivity sweep queued, reported as sensitivity, not for max-picking.
Caveat for the paper: prototype rerank throughput is ~2.8 q/s single-threaded (Event-GeM's
optimized rerank is 22 ms/query) — needs an engineering pass before latency claims.

## Phase 1.1 — T1 trajectory audit (brisbane_v5/trajectory_p*.json)

v5 run, reporting protocol (pooled sunset1, 322, 25 m), native/whitened R@1: **monotone
decline from the earliest checkpoint** — breal500 .846/.886 > s1000 .846/.882 > s3500
(shipped) .820/.874 > s10000 .806/.864. A better Brisbane model than shipped has been on
disk all along (+0.026 native / +0.012 whitened at step 500), but the NSAVP trade
(real-best loses −0.068 there) is confirmed, so re-shipping on Brisbane alone stays wrong.
The Brisbane-vs-training-progress frontier is monotone across the WHOLE run — the v7
selection arm should see its combined head pick differently at 322.

## Phase 2 — MixVPR cmstats (fairness arm completed)

Countmask stats HURT MixVPR (native .794→.782, whitened .872→.861): the normalization
effect is model-dependent (helps MegaLoc +3 pts, hurts MixVPR). Per-control best configs:
MegaLoc .855 (cmstats whitened), MixVPR .872 (ImageNet whitened) — both below our
single-stage .885 and far below two-stage .914.

## Phase 3 — L1 rerank, first pass (rerank/*/rerank_*_w0.05_a0.0.json)

- **NSAVP: +0.068** (global 0.647 → reranked **0.714**, pm_s3500, native space, no mask).
  The verification stage works on NSAVP out of the box.
- **Brisbane: −0.079** (0.8525 → 0.7732) with 100% of pairs "verified" — diagnosis: dense
  token grids match the STATIC bonnet/vignette between any two DAVIS frames, handing wrong
  candidates large-inlier homographies. Event-GeM avoids this via sparse keypoints.
  Retry with `--min-activity 0.1` (mask event-free patches) + `--whiten` (rerank the
  best-space .885/.766 rankings) in batch 4.

## Phase 1 findings from gept audit fold-in (2026-08-19)

- **Place-id collisions across streams are real and structural**: sf_xl_frontal/lateral ids
  (~2.7e10, minted from UTM cells) collide by construction for cells 2 km apart (offsets
  differ by exactly 1e8 = 100 easting cells × 1e6); GSV-Cities overruns its 100M block from
  city 10. Harmless under the per-stream loss, but **the shared-queue XBM run in v2 treated
  colliding ids as POSITIVES** — the "XBM measured negative" verdict is confounded and XBM
  deserves a rerun with re-minted ids (v7 candidate). Guard: `streams.check_id_ranges`
  (measures + warns at build). Tests: `gept/tests/test_streams_negatives.py`.
- **T5 fixed**: `Normalize` moved to the end of the train pipeline (augment counts, then
  describe). Countmask unaffected (R/B stats near-equal — why three sweeps trained fine);
  tencode/polmask were silently corrupted. Test proves both. `gept/tests/test_dataset_transforms.py`.
- **T7 guarded**: `dataset.MSLS_TEST_CITIES` — the six benchmark cities can never enter a
  training index regardless of flags. `gept/tests/test_msls_leakage.py`.
- **T2 instrumented**: clip-engagement counter + one-shot warning (>50% of steps clipped)
  + `clip_fraction` in wandb. **T1 warned**: trainer warns when selection resolution ≠ 322
  (`train.REPORTING_RESOLUTION`). **get_lr(0) fixed** (step 0 is no longer a no-op).
  `gept/tests/test_schedule.py`. gept suite: 24 tests green + vpr-smoke OK.

## Phase 1 guards landed so far (all suites green, 57 tests)

- `sim_matrix` refuses unnormalized banks (`_assert_l2_normalised`); EventVLAD call site passes
  `allow_unnormalized=True` explicitly. Test: `tests/test_inference_metrics.py`.
- `load_gt` announces its band resample and raises when it changes band density >2x.
- `inference.run` raises on zero scorable queries instead of reporting NaN.
- New tests: `test_inference_metrics.py`, `test_streamed_scoring_parity.py` (streamed pooled
  scorer ≡ reference scorer, incl. db_chunk merge), `test_pca_whitening.py`.

## 2026-08-20 — batch 5 relaunch + store-reuse bug fix

The sensitivity sweep (batch 5) died overnight to a broken SSH pipe during config 1's
query-token extraction (sunset1 14080/14280). Post-mortem found a real bug in
`scripts/megaevent_rerank.py`: the store-reuse check tested `<store>.tokens.npy` but
`TokenStore` writes `<store>.db.tokens.npy` / `<store>.q.tokens.npy`, so **every run rebuilt
its token store from scratch** (confirmed by mtimes: the native Brisbane store was rebuilt at
16:35 despite existing before the 13:17 run). No published number is affected — each run was
internally consistent (store + token-PCA built together per run) — it only wasted ~15 min per
run and made the sweep slow enough to be killed. Worse, the killed run left a *partial* q
store at the `_wht` path with valid-looking preallocated files full of zero rows.

Fix: reuse is now gated on `<store>.meta.json`, written only after BOTH stores finish,
carrying a sha256 of the shortlist-union indices (+ n_queries, token_dim). Mismatch or
absence → rebuild. The partial `_wht` store therefore rebuilds once, then configs 2–4 reuse
it; the NSAVP native store rebuilds once for config 5 and gains a manifest. Suite: 82 green.

Relaunched detached (`setsid nohup`, script + logs in
`/media/adam/vprdatasets/megaevent/phase0_logs/`, master log `batch5_run2.log`) so a dropped
SSH session can no longer kill it.

## 2026-08-20 — frozen-encoder rep probe + v8 built

**Frozen probe** (`scripts/frozen_rep_probe.py`, untouched gept/small.pt, pooled Brisbane
5-traverse db incl. sunset2, npz events hot-pixel-only): accumulate beats countmask in
every cell — whitened meanpatch R@1 **0.758 vs 0.579**, native 0.566 vs 0.317, CLS
whitened 0.730 vs 0.602. The GEPT diet mismatch (see memory) costs recall at init, not
just GPU time. Regenerated the missing `night` npz tree (39 s).

**v8 machinery** (user directives: accumulate only, no countmask control, noise stays,
no whitening anywhere, selection = pooled Brisbane in-loop):
* `gept/src/vpr/dataset.py`: `AccumulateDomainRandomise` — v5's gain/dropout/salt in
  accumulate's inverted space (dropout writes WHITE; salt darkens polarity-consistent
  channels); `--aug-domain-rand` dispatches per representation. Polarity swap was already
  always-on (`RandomSwapEventRedBlue` p=0.5), matching GEP's own alignment aug.
* `gept/src/vpr/evalsuite.py`: `run_pooled` — sunset1 vs pooled 5-traverse db, 25 m GT,
  native cosine; coords from `gept/data/brisbane_pooled_coords.npz` (exported by
  `megaevent/scripts/export_brisbane_coords.py` from the calibrated benchmark geometry);
  per-traverse raw-slice offsets from eval_offsets.json or count-diff fallback —
  verified by count cross-correlation, lag 0 on all six traverses (daytime −1 = <1 m).
* `gept/src/vpr/train.py` + `config.py`: `--eval-pooled` replaces the per-condition
  suite; selection real == main == pooled R@1; msls stays a monitor-only head.
* `gept/run_v8.sh`: 2 arms (b/s_v8_accum), 2000 steps, eval every 250, save every 500,
  parse-checked; corpus prerequisite = rerender_from_npz.py per stream (numpy trees
  confirmed kept on HPC). Tests: gept 32 green (6 new in test_accumulate_pooled.py).

## 2026-08-21 — v8 accumulate benchmark pass (full recorded-set table)

All native cosine, 25 m (whitening dropped per user directive). Sources:
`v8_bench/{brisbane,nsavp,brisbane_s,nsavp_s}/results.json`, 4-traverse rescores inline,
`evaluations/nycevent/nycevent/results_[bs]_v8*.json`, `evaluations/nycevent/loso.json`.

| native R@1        | best countmask | MegaLoc | v8-B (s750) | v8-S (s500) |
|-------------------|---------------|---------|-------------|-------------|
| Brisbane 4-trav   | 0.8525        | 0.8123  | 0.8597      | 0.8422      |
| NSAVP             | 0.6467        | 0.7287  | **0.7789**  | 0.7251      |
| NYC random-split  | 0.813         | 0.767   | **0.9048**  | 0.8877      |
| NYC LOSO mean     | 0.1074        | 0.1075  | **0.1409**  | 0.1157      |

NSAVP flips (first outright MegaLoc win, single-stage, native); NYC LOSO shows the first
separation on the corrected protocol (+31% rel); Brisbane single-stage stays ceiling-bound
(v8-B +0.007 = noise; v8-S ties the from-scratch-head s_salad ship at −0.006). v8-S ties
MegaLoc on NSAVP (−0.004) — B's extra margin is confounded (size × MegaLoc-head warm
start); v9 head-transplant arm would separate. Ship 0.8525 reproduced exactly through the
fresh path. Filter guard made rep-aware (white bg counted as active). Full details in
memory v8-accumulate-verdict.

## 2026-08-21/22 — Event-GeM 0.1.0 as-released campaign

Tag-0.1.0 worktree, pinned submodules, pr.pt sha-verified against paper-era copies.
Pooled/LOSO (native, 25 m): Brisbane 0.4952 · NSAVP 0.2347 · NYC 0.4354 · LOSO 0.0231.
Pairwise (its protocol): Brisbane 0.7675→0.9021 (14.6 ms/q rerank); NSAVP FS
0.3634→0.5912, FN 0.0798→0.0972. Reverse headings absent from the released protocol and
the original data (truncated R0_RA0 raw). Old-era feature caches proven NOT 0.1.0 output
(|Δsim| 0.98) — first pooled Brisbane number retracted and recomputed (0.4964→0.4952,
conclusion unchanged). Full 8-item errata in memory eventgem-010-as-released.
Artifacts: v8_bench/eg010_*, worktree feature tree, evaluations/nycevent/eg010_gem224_*.

### 2026-08-22 — the "0.495 vs 0.877" decomposition (user challenge)

Implementation validated: pooled harness reproduces the release's own saved
sunset2-sunset1 similarity to 8e-7; ship 0.8525 reproduced through the same scorer.
The ledger's 0.877 decomposes as: post-release GLOBAL native 0.5145 (+0.019 vs 0.1.0's
0.4952 — the versions' globals nearly match) + whitening pca128p0.5 +0.168 (0.6827)
+ rerank +0.194 (0.8772). Whitening on the 0.1.0 banks buys only +0.026 (0.5214):
the pr.pt+GeM space lacks the dominant nuisance directions whitening strips from the
post-release SuperEvent-global space. So "lower than before" = comparing a native global
against a whitened+reranked cell, plus a genuinely stronger post-release feature space
under whitening. Source: eventgem_pooled/brisbane_event.json + fresh checks.

### 2026-08-22 — 0.1.0 pooled + released rerank (the missing cell)

User: "Obviously I want the eventgem 0.1.0 result with the rerank mechanism." Built the
pooled+rerank cell by running the release's OWN loop (eventgem.utils.rerank_utils
.process_single_query, main.py defaults: top-50, USAC_FAST maxIters=100, thresh 5.0,
inlier_weight 0.05, ratio 0.8) inside the 0.1.0 pixi env (cv2 4.13 — megaevent's cv2 5.0
not trusted for USAC parity). Pooled db presented to the unmodified loop as a symlink
farm: pooled idx -> per-traverse kp file; a row with no kp store is a missing file, which
the release answers with 0 inliers / base distance kept (its own semantics).

Keypoint stores harvested through the release CLI (ref sunset2, query = each db traverse;
GT pairs exist so each run completed end-to-end and printed its own pairwise table —
per-traverse alignment proof). All count-verified vs MCTS. Bonus pairwise (base->rerank
R@1): daytime .1903->.3907, morning .2932->.6053, night .0325->.0396, sunrise
.4237->.7665 — the rerank roughly doubles cross-illumination pairs but cannot rescue
night. Fresh 0.1.0 MCTS generated for night/sunrise (Feb-era mcts_*_50 reused for
daytime/morning — same batch the validated sunset stores came from). NSAVP: R0_RN0
(4.3 GB, only RAM-feasible reverse recording) harvested via kp_only.py self-pair;
R0_RA0/R0_RS0 remain infeasible (erratum #6) -> farm rows uncovered by design.

Gates: (A) driver reruns the release's published pairwise protocol -> reproduces
0.7675/0.9021 (and R@5/R@10 rows) to all 4 dp. (B) pooled base recalls reproduce
eg010_pooled JSONs exactly.

**Pooled Brisbane 4-trav (native, 25 m): base 0.4952 -> reranked 0.7786** (R@5 .8048,
R@10 .8167, R@20 .8312), 19.6 ms/q, top-1 changed for 83% of queries (11,807/14,280),
100% db kp coverage. Post-release native+rerank was 0.7761 -> the generations are
EQUIVALENT at native+rerank; the old 0.877 was whitening-dependent. Still below MegaLoc
single-stage 0.8123 / ship 0.8525 / v8-B 0.8597. Top-1 provenance barely moves
(sunrise 8423->8446 of 14,280): rerank fixes which frame, not which traverse.
NSAVP pooled+rerank pending RN0 harvest. Results: v8_bench/eg010_rerank_pooled_*.json,
driver eventgem_010/pooled_rerank.py, geom workpack scripts/eg010_dump_geom.py.

**Pooled NSAVP (native, 25 m): base 0.2347 -> reranked 0.3632** (R@5 .4086, R@10 .4376,
R@20 .4786), 8.2 ms/q, top-1 changed for 38% of queries. Post-release native+rerank was
0.4550 (native 0.2937) -> unlike Brisbane, the post-release feature space is genuinely
stronger on NSAVP at BOTH stages (-0.059 global, -0.092 reranked). kp-coverage gap is NOT
the cause: a missing kp file and a 0-inlier verification are literally the same code path
(dist modified only when inliers > 0), so uncovered RA0/RS0 rows only differ from covered
ones by reverse-heading inlier yield (~0). Provenance is the heading story quantified:
rerank moves top-1 mass from reverse-daylight RA0 (14,747 -> 10,833) to forward-snow FS0
(3,421 -> 8,565) — geometric verification recovers the heading selectivity the global
stage lacks. Note the post-release campaign's matcher was mutual-NN (megaevent
reimplementation); the release's is ratio-test kNN — same params otherwise.

### 2026-08-22 — NYC-Event 0.1.0 + released rerank (ECDPT global stage confirmed)

User: "are you also doing the ecdpt global stage? that's critical" — yes: the base is the
parity-checked eg010_gem224 ECDPT+GeM banks; the rerank subtracts 0.05x inliers from
their top-50. NYC has no release support, so keypoints came from a fused
render->SuperEvent driver (nyc_kp_extract.py): mcts_numpy (the release's reference MCTS
implementation) per raw-event sample, t_ref = last event, main.py windows in the
samples' us clock; crop math + extraction = EventGeMMCTS/extract_superevent_features_
for_dir unmodified. Gates: mcts_numpy == gpu_mcts body (1.2e-7); SuperEvent stage
bit-reproduces kps_sunset1 at the campaign batch size (16/16 frames, desc diff 0.0;
batch-1 reorders NMS ties on 1/16 frames with identical SETS -> batch 1 safe for VRAM).
55,433 kp stores, 720x1280, top_k 640 by the release's max(Hc,Wc)//2 rule, 32 GB.

nyc_rerank.py: random split + LOSO, release loop verbatim; LOSO via ONE global farm +
own-session distances set to +inf (equivalent to gallery removal, loop untouched). Base
gates: random-split delta +0.0004 = 2 near-tie queries of CPU-vs-GPU rounding
(count-aware tolerance, delta printed); all LOSO per-session bases reproduce loso.json.

**Random split: 0.4357 -> 0.6528** (R@5 .7727, R@10 .8026, R@20 .8254). **LOSO mean:
0.0231 -> 0.0480.** 146 ms/q — NYC's 640 kpts/frame make matching ~7x Brisbane. Two
sessions REGRESS (02-14_18-20-40 .0078->.0054, 04-20_17-10-01 .0107->.0099): with a
near-random base, spurious homographies on wrong candidates outweigh rare correct ones.
Even reranked, LOSO 0.0480 < half of MegaLoc's single-stage 0.1075 and a third of v8-B's
0.1409; random-split 0.6528 < MegaLoc 0.767 < v8-B 0.9048.
Results: v8_bench/eg010_rerank_nyc.json; drivers nyc_kp_extract.py / nyc_rerank.py.

### 2026-08-22 — v8-S Brisbane 4-trav full curve (rescored from cached banks)

The verdict table's v8-S Brisbane 0.8422 had no persisted curve. Rescored from
v8_bench/brisbane_s per-traverse raw banks + eg010 geometry masks, gated by exact
reproduction of v8-B's row through the same path: **v8-S 4-trav R@1 0.8422, R@5 0.9174,
R@10 0.9320, R@20 0.9410** (54,620 db x 14,280 q, native, 25 m).

### 2026-08-23 — v8 on the I2E image sets (tokyo247 / msls / pitts250k, accumulate)

The v8 verdict previously covered only the real datasets. Extracted fresh accumulate
banks at 322 for both v8 checkpoints via the same main.py path as the NYC v8 runs
(rep read off the ckpt; batch 12 because concurrent Event-GeM jobs held ~3 GB of the
card — the 64 default OOMed in the SALAD head). All three trees verified raw-event npz
(x/y/t/p/resolution), so accumulate rendering is exact. Native cosine, 25 m:

| dataset | model | R@1 | R@10 |
|---|---|---|---|
| tokyo247 (76k db x 315 q) | v8-B | 0.8381 | 0.9460 |
| | v8-S | 0.6984 | 0.8698 |
| msls (36,207 x 1,094) | v8-B | 0.4607 | 0.6426 |
| | v8-S | 0.3537 | 0.5219 |
| pitts250k (83,952 x 8,280) | v8-B | 0.7803 | 0.9217 |
| | v8-S | 0.6446 | 0.8496 |

v8-B beats MegaLoc-on-countmask natively on all three (0.8159/0.4013/0.6893 R@1 ->
+2.2/+5.9/+9.1 points) and matches the v5-B shipping countmask cells where they exist
(msls 0.4598, pitts250k 0.7793): the accumulate switch costs nothing on I2E while
winning the real datasets. v8-S trails v8-B by 11–14 R@1 points here — a far larger
ViT-S penalty than on the real sets (1.7–5.4).

Caveats: both pitts250k runs were killed during the *post-native* PCA fit (RAM, most
likely — the same fit succeeded on NYC's 49,890-row bank with the GPU/RAM to itself),
so evaluations/pitts250k has cached v8 banks + log-line recalls but no results_*.json;
tokyo247 and msls JSONs are complete and match the logs to 4 dp. Logs:
v8_bench/{tokyo247,msls,pitts250k}_v8{b,s}.log.

### 2026-08-23 — v8 Brisbane sim2real arms (I2E re-run with accumulate)

The npz-arm render path only supplied countmask (deliberate guard in
brisbane_resolution.py); generalized to `tnpz.frame_loader(transform, representation)`
dispatching load_countmask/load_accumulate (gates: countmask byte-identical through the
new loader on i2e frames; test_accumulate_render + test_countmask_identity green).
Protocol identical to the v4/v5 runs: sunset1 -> daytime+morning+sunrise, aps-aligned,
r322, dt 50, native only; i2e arm filter-off (forced), real arm ba50 with banks reused
from v8_bench/brisbane{,_s} (existence cache, same ckpts). n_db 40,939 x 14,228 q.

| native 25 m R@1/R@10 | real (ba50, aligned) | i2e | delta R@1 |
|---|---|---|---|
| v4-B (P64_s10000) | 0.747/0.878 | 0.667/0.868 | -0.080 |
| v4-S (S_step10000) | 0.779/0.910 | 0.634/0.870 | -0.145 |
| v5-B (b_mloc_s3500) | 0.820/0.918 | 0.708/0.884 | -0.112 |
| v5-S (s_mloc_s3500) | 0.853/0.933 | 0.707/0.881 | -0.146 |
| **v8-B** | **0.860/0.943** | **0.725/0.912** | **-0.135** |
| **v8-S** | **0.842/0.932** | **0.700/0.899** | **-0.142** |

Reads: (1) v8-B's aligned real cell 0.860 ~= its unaligned 4-traverse 0.8597 — night's
presence/absence is score-neutral, coherent with the illumination-manifold finding.
(2) The accumulate switch moves the real arm (+4.0 v8-B over v5-B) more than the i2e
arm (+1.7); v8-S i2e ties v5-S. The real->i2e drop is stable across generations
(-0.11..-0.15 for every trained arm except v4-B) — no sign the newest models gain a
growing I2E-specific advantage; the i2e arm is simply harder (dusk APS source noise,
p90 top-1 distance 902 m vs 28 m real).
Results: sim2real/megaevent_v8_{i2e,real}.json, logs v8_bench/sim2real_v8_{i2e,real}.log.

### 2026-08-23 — Event-GeM 0.1.0 on the Brisbane sim2real arms (real vs I2E)

Same aligned protocol as the v8 cells (sunset1 -> daytime+morning+sunrise, APS-aligned
40,939 db x 14,228 q, native, 25 m). New drivers in the 0.1.0 worktree, everything
imported from the NYC-gated ones: brisbane_npz_extract.py (gem globals from the npz
trees, both arms), brisbane_npz_kps.py (i2e MCTS->SuperEvent stores, top_k 160 by the
release's crop rule), sim2real_rerank.py (pooled_rerank mechanics + a 4-trav ledger
gate). Geometry via eg010_dump_geom_sim2real.py, counts asserted against the v8 run.
Gates all green: 4-trav base == cached 0.4952 ledger; render parity 1.19e-07;
SuperEvent bit-reproduces kps_sunset1; loader parity vs EventGeMData.

| E-GeM 0.1.0, native 25 m | R@1 | R@10 |
|---|---|---|
| real base | 0.4954 | 0.7295 |
| real + released rerank | **0.7792** | 0.8175 |
| i2e base | 0.1958 | 0.4407 |
| i2e + released rerank | **0.3981** | 0.5458 |

Confound bound: realnpz (real events through the SAME npz drivers as i2e) base R@1
0.4925 vs release-pipeline 0.4954 = -0.0030 — the extraction path is score-neutral;
the real->i2e gap is the events. Rerank 50.5/37.7 ms/q, top-1 changed for 83%/87%.

Reads: (1) 0.1.0's real->i2e drop is -0.30 base / -0.38 reranked — vs v8-B's -0.135.
The real-event-trained baseline degrades ~2.5x more on I2E-converted frames than the
I2E-fine-tuned model does; the megaevent-vs-EventGeM margin is NOT arm-stable
(+0.081 real -> +0.327 i2e, reranked-vs-v8-B-global). The reviewer's domain-advantage
concern is partially supported and must be reported as such. (2) Post-release
comparison: real native+rerank 0.7792 ~= post-release 0.7769 (generations equivalent
again), but on i2e the post-release space was much stronger (0.629 vs 0.398 reranked)
— same pattern as NSAVP. Results: v8_bench/eg010_sim2real.json.

## 2026-09-10 — Why the event-trained model beats RGB VPR on event frames

Question: given we train on a classical VPR regime, where does the advantage come from —
are more features detected, are the features better, or are the descriptors more refined?

**Short answer: the features. Almost all of the advantage is present in the patch tokens
before any aggregation head exists, and it is not "more" features — it is the same 529
tokens carrying a representation that survives the illumination change.**

Protocol throughout: **single reference traverse**, native cosine, 25 m, no whitening
(`pairwise_sunset_ref_*.json`). Pooled cells are not used — a pooled gallery holds several
traverses of the same route, so a query needs only one of them to be an easy appearance
match, and the cell stops measuring cross-condition retrieval.

### 1. The size of the thing being explained

`scripts/table_margin.py` (new; reads the single-reference ledger, whole roster, not just
MegaLoc). R@1 is bounded, so the error ratio `(1-ours)/(1-theirs)` is reported beside the
absolute margin — it is comparable across cells of different difficulty.

| cell (ref -> query) | v8-B  | best control  | d R@1  | err ratio |
|---------------------|-------|---------------|--------|-----------|
| sunset1 -> daytime  | 0.706 | QAA    0.584  | +0.121 | 0.71      |
| sunset1 -> morning  | 0.780 | MixVPR 0.651  | +0.129 | 0.63      |
| sunset1 -> sunrise  | 0.882 | MixVPR 0.827  | +0.056 | 0.68      |
| NSAVP FS0 -> FA0    | 0.841 | QAA    0.821  | +0.020 | 0.89      |
| NSAVP RS0 -> RA0    | 0.806 | BoQ    0.757  | +0.049 | 0.80      |

v8-B leads **5/5**. Brisbane: a **near-constant one third** of the remaining error removed
across a 2.5x range of cell difficulty (0.63 / 0.68 / 0.71). NSAVP: 16%. The absolute margin
doubling from +0.056 to +0.129 is the R@1 bound, not a condition-specific effect — the gain
is uniform, which already points at the representation rather than at an illumination trick.

**Roster caveat:** MegaLoc is 4th-5th on these cells. Quoting the margin against MegaLoc
alone roughly doubles it (Brisbane daytime +0.121 vs QAA becomes +0.284 vs MegaLoc). MixVPR
and QAA are the bar.

### 2. Features or descriptor? — `scripts/backbone_probe.py` (new)

Strip the aggregation head off every model and pool the 529 patch tokens with a
**parameter-free** L2-normalised mean. Same frames, same rendering, same pooling, same
ranking, same GT; the only thing that varies is which encoder read the frame. Each RGB
backbone is run under both ImageNet and accumulate-matched statistics and keeps its better
arm. Stride 5 (all arms identical; absolute R@1 is not comparable to a full-protocol cell,
only the arms to each other).

| frozen backbone | pretraining      | daytime | morning | sunrise | mean   |
|-----------------|------------------|---------|---------|---------|--------|
| dinov2-s        | RGB              | 0.130   | 0.155   | 0.211   | 0.166  |
| gept-s          | **event**        | 0.165   | 0.162   | 0.267   | 0.198  |
| salad           | RGB + VPR        | 0.169   | 0.175   | 0.259   | 0.201  |
| megaloc         | RGB + VPR        | 0.256   | 0.214   | 0.344   | 0.271  |
| **v8-b**        | **event + event-VPR** | 0.299 | 0.368 | 0.493 | **0.387** |

With no head at all, our backbone leads the best RGB backbone by **1.43x** (0.387 / 0.271).
The full models, heads included, differ by **1.53x** (0.789 / 0.516 mean over the three
Brisbane cells). So the descriptor head accounts for almost none of the gap: **the advantage
is in the patch features.** That is the direct answer to the question as asked.

Two supporting reads:
* **It is not "more features detected".** Every backbone emits exactly the same 529 tokens
  on the same frames; nothing is detected or missed. What differs is what those tokens
  encode.
* **The head is worth little, and it is the same head the baselines use.** Inside our own
  family (v9 sweep, already trained), GeM -> SALAD is worth +0.039 on NYC, +0.007 on
  Brisbane, +0.040 on Springfield, and training only the last four blocks recovers nearly
  all of a full fine-tune (+0.001 on NYC, +0.008..+0.023 on Brisbane). Every architectural
  effect inside our family is 0.001-0.040 — an order of magnitude below the feature-level
  gap the probe measures.

### 3. Which part of the training buys it

Same probe, restricted to the **ViT-B row** so width is held fixed (all 768-d):

| frozen ViT-B backbone | pretraining | VPR training      | mean R@1 |
|-----------------------|-------------|-------------------|----------|
| dinov2-b (reg4)       | RGB         | none              | 0.198    |
| gept-b                | **event**   | none              | 0.214    |
| salad                 | RGB         | RGB (GSV-Cities)  | 0.201    |
| megaloc               | RGB         | RGB (large scale) | 0.271    |
| **v8-b**              | **event**   | **event (I2E)**   | **0.387**|

| step                                          | delta   | factor |
|-----------------------------------------------|---------|--------|
| event pretraining alone (dinov2-b -> gept-b)   | +0.016  | 1.08x  |
| RGB VPR training, capacity-matched (-> salad)  | +0.003  | 1.01x  |
| RGB VPR training, large scale (-> megaloc)     | +0.073  | 1.37x  |
| event VPR training on top (gept-b -> v8-b)     | +0.173  | 1.81x  |
| **total (dinov2-b -> v8-b)**                   | +0.189  | 1.95x  |

In log terms the total splits **12% event pretraining / 88% event VPR training**.

Three things follow, and they are the whole answer:

1. **GEPT's event pretraining is not what does it.** On its own it is worth 1.08x — real,
   but a twelfth of the effect. An RGB DINOv2 given large-scale *RGB* VPR training (megaloc,
   1.37x) improves its event-frame patch features more than event pretraining does.
2. **What does it is metric VPR training carried out on event frames.** 1.81x on top of the
   event-pretrained init, and it is the same *kind* of intervention that buys megaloc its
   1.37x — just performed in the target domain, where it is worth about 2.4x more.
3. **SALAD is the control that makes this legible.** Capacity-matched to ours (same DINOv2
   ViT-B/14 + SALAD, 88.0M) and VPR-trained, but on GSV-Cities photographs, its backbone is
   statistically indistinguishable from stock DINOv2 on event frames (0.201 vs 0.198). The
   architecture is not the variable; the training domain is.

### 4. What was wrong with how the margin had been read

Two defects, both fixed, neither large enough to change the conclusion above:

* **Normalisation.** `--norm-stats` offered `imagenet` and `countmask` only, but v8 evaluates
  on `accumulate` — a WHITE-background render that ImageNet centres **2.0-4.5 sigma** off
  (R 4.47, G 2.00, B 3.71) and overscales up to 2.2x. All 119 baseline `*_accum_*` banks on
  disk carry ImageNet constants. Fixed: `src/methods.py` gains `ACCUMULATE_MEAN/STD` read
  from `b_v8_accum_s750.pt`'s own `config.tencode_mean/std` (the same gept `--compute-stats`
  provenance as the countmask pair); `megaloc_transform` dispatches through `_NORM_STATS` and
  rejects unknown names; `rgb_pooled.py` / `springfield_baselines.py` gain
  `--norm-stats accumulate` with an `_acstats` tag and warn when stats and representation
  disagree. Tests: `tests/test_eval_transform.py::ControlNormStatsTests`.
  **Measured cost:** MegaLoc on the single-reference Brisbane cells, mean R@1
  0.516 -> **0.527** (+0.011; daytime -0.043, morning +0.040, sunrise +0.035). On the frozen
  backbone the matched stats *hurt* every RGB encoder (megaloc 0.271 -> 0.232) — the filters
  expect ImageNet-normalised input, and only the trained head prefers the matched arm.
* **Roster.** MegaLoc is 4th-5th on these cells; MixVPR and QAA set the bar. A margin quoted
  against MegaLoc alone roughly doubles. `scripts/table_margin.py` reports the whole roster
  and the error ratio.

Still open, and both would only narrow the margin, never widen it: MixVPR's and QAA's own
`_acstats` arms are not on disk, and Springfield has **MegaLoc as its only RGB control**
(v8-B 0.3122 vs 0.2323) — mixvpr/qaa/boq/salad/supervlad are now wired into
`scripts/springfield_baselines.py` but not yet run.

### 5. What this does not say

* The probe pools with a parameter-free mean, which is off-distribution for every trained
  head; it measures what the tokens carry, not what each model's own head would extract.
* Stride 5 shrinks the gallery, so its absolute R@1 is not a protocol number — only the arms
  are comparable to each other.
* "More features detected" was never a live option: every backbone emits the same 529 tokens
  on the same frames. The question only has two answers, and the answer is the features.
* No new training was run, so the crossed cell (RGB-DINOv2 init + our event recipe) is
  untested; event pretraining and event VPR training are separated only at the frozen end.
