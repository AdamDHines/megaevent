"""VPRConfig - configuration for the global (GeM) and SALAD VPR branch.

Standalone on purpose: it does NOT inherit ``src/config.py::Config`` (that base
class imports ``src/dataset.py`` and pulls in numba/h5py at construction time).
Everything the VPR pipeline needs is declared here.

Backbone specs are keyed off the checkpoint and were verified by inspecting the
weights (see ``PLAN.md`` / memory ``gept-architecture-facts``):

  small.pt : ViT-S/14, embed 384, 6 heads, 12 blocks, 0 register tokens, 3-ch
  base.pt  : ViT-B/14, embed 768, 12 heads, 12 blocks, 4 register tokens, 3-ch

The stock ``src/cls.py`` hardcodes ``num_register_tokens=4`` for both sizes, which
would fail to load ``small.pt``. Here the register count is derived per size.
"""

import os

# (embed_dim, n_head, num_register_tokens) per dinov2 size, matching the GEPT checkpoints.
_BACKBONE_SPECS = {
    "small": {"n_embed": 384, "n_head": 6, "num_register_tokens": 0, "ckpt": "small.pt"},
    "base": {"n_embed": 768, "n_head": 12, "num_register_tokens": 4, "ckpt": "base.pt"},
}


class VPRConfig:
    def __init__(self):
        # ---- paths -------------------------------------------------------
        self.repo_root = os.path.dirname(os.path.abspath(__file__))
        # tencode dataset root (I2E output: <data_root>/<dataset>/jpg/<relpath>.png)
        self.data_root = "/media/adam/vprdatasets/data"

        # ---- backbone ----------------------------------------------------
        # small first: it is the Jetson target and it won the sim-to-real evals (PLAN 1k).
        # base.pt only after MegaLoc multi-stream training is proven on small.
        self.vit = "small"                 # "small" | "base"
        self.vit_backbone = "dinov2"       # only dinov2 checkpoints exist for GEPT
        spec = _BACKBONE_SPECS[self.vit]
        self.n_embed = spec["n_embed"]
        self.n_head = spec["n_head"]
        self.num_register_tokens = spec["num_register_tokens"]
        self.n_layer = 12
        self.P = 14                        # patch size
        # Build the ViT at 518px so the pretrained pos_embed (1370 = 1 + 37^2) loads
        # strict=True; dinov2 interpolates pos-encoding for the real input size at runtime.
        self.backbone_img_size = 518
        # checkpoint to initialise the encoder from (event_encoder state_dict)
        self.ckpt_path = os.path.join(self.repo_root, spec["ckpt"])
        self.load_pretrained = True        # set False when restoring a full VPR checkpoint

        # ---- input -------------------------------------------------------
        self.image_ext = ".jpg"            # tree-layout image extension (gsvcities tencode = .jpg)
        self.compute_stats = False         # compute tencode mean/std from the data at startup
        self.stats_max_images = 5000       # cap for the stats pass (full GSV-Cities is huge)
        self.H, self.W = 224, 224          # multiples of 14 -> 16x16 = 256 tokens
        self.n_tokens_per_image = (self.H // self.P) * (self.W // self.P)
        # Fresh tencode normalization stats to be computed via compute_mean_std
        # (PLAN 1a). Placeholder prior = GEPT's white-bg event stats (NIMA_ME/SE);
        # REPLACE before real training - tencode's green=time channel differs.
        self.tencode_mean = [0.9673, 0.9297, 0.9624]
        self.tencode_std = [0.1204, 0.1674, 0.1265]
        self.white_frame = True            # GEPT aligned on white-bg frames (PLAN 1a)

        # Input representation (PLAN 1q; src/vpr/representations.py). The training frames
        # are rendered to disk in this rep (by repo/I2E), and the real-event eval renders
        # it on the fly from HDF5 — the two MUST match. Switching this does NOT re-render
        # training data — point --data at a tree rendered in the same rep. Options:
        #   tencode    — deployed default; R/B binary polarity, green = event TIME (black-bg)
        #   accumulate — GEPT-native src/utils.py::accumulate_to_rgb: white-bg, count-graded
        #                winner-take-all red/blue, green = inverse activity mask (no time)
        #   countmask  — GEPT paper Eq. 2: independent per-polarity counts R=Mr/B=Mb (joint
        #                percentile norm), green = plain binary activity mask (black-bg, no time)
        #   polmask    — ablation: tencode's R/B, green swapped time -> binary mask
        self.representation = "countmask"    # tencode | accumulate | countmask | polmask

        # ---- descriptor head --------------------------------------------
        self.aggregator = "salad"            # "gem" | "salad" (MegaLoc's optimal-transport head)
        self.desc_dim = 2048               # GeM descriptor dim; SALAD overrides via finalize()
        self.gem_p_init = 3.0              # learnable GeM exponent init
        self.gem_eps = 1e-6
        # SALAD (src/vpr/salad.py): output dim = clusters*cluster_dim + token_dim = 8448,
        # used directly (no projection). The aggregator is ALWAYS trained from scratch
        # (no SALAD weights in the GEPT checkpoint — only the backbone loads), so it takes
        # MegaLoc's from-scratch dropout 0.3, not the finetune-0 we (wrongly) used first.
        # With 0.0 the 0.82M-param head overfit the RGB source: sunset1 froze at ~0.36 from
        # step 1000 while train loss kept falling, ~2x behind GeM on the same frozen backbone
        # (features are fine — GeM proves it — so the gap is the aggregator overfitting).
        self.salad_clusters = 64
        self.salad_cluster_dim = 128
        self.salad_token_dim = 256
        self.salad_mlp_dim = 512
        self.salad_dropout = 0.3
        # Warm-start the SALAD head from MegaLoc's pretrained aggregator (verbatim-port keys
        # match ours). Only shape-compatible with ViT-B (num_channels 768); the helper asserts.
        # None -> random init. Path comes from GEPT_MEGALOC_WEIGHTS (a HF model.safetensors),
        # since compute nodes are offline; override with --megaloc-weights.
        self.salad_init = None             # None | "megaloc"
        # MegaLoc's head is SALAD *plus* a learned compression: a 16640-d aggregator followed
        # by Linear(16640 -> 8448) and an L2 norm. Ours is SALAD as published, emitting 8448
        # directly. Checkpoints trained with gept's --salad-proj carry the extra layer, so the
        # inference model has to be able to build it; these are restored from the checkpoint
        # (see inference._RESTORE), never inferred.
        self.salad_proj = False
        self.salad_proj_dim = 8448
        self.salad_out_dim = None          # aggregator width; None -> derived in the model
        self.megaloc_weights = os.environ.get("GEPT_MEGALOC_WEIGHTS", None)

        # ---- finetuning strategy (PLAN 1c / 1g / 1k) --------------------
        # DEPLOYMENT REGIME = ft4: unfreeze only the last 4 blocks, flat LR. PLAN 1k found
        # this beats the full-backbone LLRD finetune at *every* Brisbane eval stride
        # (+.05..+.09 R@1) at 3x less compute, because less drift off the pretrained GEPT
        # features means better sim-to-real. It is also MegaLoc's own recipe (last 4
        # transformer layers trainable, everything else frozen).
        # For the full-backbone variant: n_trainable_blocks=12, layerwise_lr_decay=0.75.
        self.transfer = "finetune"         # "finetune" | "linear" (linear = freeze whole encoder)
        self.n_trainable_blocks = 4        # unfreeze the last k transformer blocks (12 = all)
        self.train_norm = True             # unfreeze final encoder.norm
        self.train_patch_embed = True      # unfreeze patch_embed (tencode has a time channel)
        self.train_embeddings = False      # unfreeze cls/pos/mask/register tokens (needed for from-scratch)
        self.layerwise_lr_decay = None     # per-layer LR decay; None -> flat encoder_lr_mult

        # ---- optimisation ------------------------------------------------
        self.lr = 1e-4                     # head LR
        self.encoder_lr_mult = 0.1         # unfrozen-encoder LR = lr * this
        self.transformer_lr_mult = 1.0     # (no extra transformer here; kept for get_param_groups)
        self.wd = 1e-3                     # applied to 2D head weights only
        self.min_lr = 0.0
        self.warmup_steps = 1000
        # PLAN 1h: the 40k run peaked on real Brisbane at ~10k and decayed for the
        # remaining 30k while train loss kept falling. Default to a *cosine-annealed*
        # short schedule (4x cheaper) and let the real-eval hook pick `best.pt`.
        self.steps = 12000
        # Truncate the training *loop* at this step while leaving `steps` (the cosine
        # horizon) alone — a short run that re-treads the first N steps of a long one
        # rather than a differently-scheduled short run. 0 -> run all `steps`.
        self.stop_after = 0
        self.grad_clip = 1.0
        self.amp = True

        # ---- MegaLoc regime (PLAN 2) ------------------------------------
        self.P_places = 32                 # classes per sub-batch
        self.K_images = 4                  # images per class -> 128-image sub-batch
        # Active sub-batch streams (see src/vpr/streams.py::REGISTRY). MegaLoc runs all
        # six -- SF-XL contributes two, frontal and lateral -- for L = L1+...+L6.
        # Override with --streams; `--streams megaloc` expands to the full six.
        self.streams = ["gsv_cities"]
        self.refresh_index = False         # ignore cached place indices and rewalk the tree
        self.refresh_stats = False         # ignore cached tencode stats and recompute

        # SF-XL / EigenPlaces class generation (dataset.build_sf_xl_index).
        # groups=(0,) keeps one non-overlapping group: classes within a group are
        # M*N = 100 m apart, so no two classes in a sub-batch can be the same place
        # (which the multi-similarity loss would otherwise treat as a false negative).
        self.sfxl_M = 20                   # UTM cell side (metres) = one class
        self.sfxl_N = 5                    # class spacing within a group, in cells
        self.sfxl_focal_dist = 10          # metres from class centre to the focal point
        self.sfxl_groups = [0]             # which of the N*N groups to use (manual override)
        self.sfxl_min_images_per_class = 10
        # EigenPlaces-style group cycling: cover all N*N=25 groups across epochs, one group
        # per sub-batch. Classes within a group are M*N=100 m apart (no false positives) and
        # cross-group classes never co-occur in a batch (no false negatives). Fixes SF-XL
        # under-population (single group = ~3k places) with full coverage over the run.
        self.sfxl_cycle_groups = True
        # Cap on images kept per class for the directory-classed streams (MegaScenes,
        # ScanNet). Landmark/scan collections are wildly uneven — a famous cathedral can
        # have thousands of photos — and PlacesDataset draws K at random from whatever a
        # class holds, so an uncapped giant class would dominate its stream's sampling.
        self.max_img_per_place = 64
        # ScanNet sub-classing (dataset.build_scannet_index). 0 = one whole scan per place,
        # the original coarse organization. >0 splits each scan into contiguous runs of
        # `scannet_chunk` frames separated by `scannet_gap` dropped ones, so a place is a
        # sub-trajectory whose frames actually overlap — the thing MegaLoc gets from pose
        # co-visibility and we cannot (no poses in the converted tree). Also cures the
        # oversampling: 1006 scans = 31 batches/epoch = ~1290 epochs over a 40k run, vs
        # MegaScenes' ~17. Default OFF so it is an A/B against the countmask sweep, not a
        # silent change to the mixture every past number was measured on.
        self.scannet_chunk = 0
        self.scannet_gap = 4
        # MSLS: the converted tree has only opaque Mapillary keys, so the class
        # organization is joined in from the SOURCE distribution's train_val/ on the key.
        # postprocessed.csv gives easting/northing (already UTM), unique_cluster and
        # view_direction -> place = (unique_cluster, view-direction bin).
        self.msls_meta_root = "/work/qvpr/data/raw/Mapillary_Street_Level_Sequences/train_val"
        self.msls_view_bins = 4            # <=1 disables the view split (place = cluster)
        self.msls_merge_splits = "auto"
        # Multi-similarity loss hyperparameters (pytorch-metric-learning defaults).
        self.ms_alpha = 2.0
        self.ms_beta = 50.0
        self.ms_base = 0.5
        self.miner_epsilon = 0.1

        # ---- real-event eval / model selection (PLAN 1e, 1h) ------------
        # Brisbane-Event **tencode**: ref=sunset2, query=sunset1. This is the model-selection
        # signal — PLAN 1h found training loss is *anti-correlated* with real recall after
        # ~10k steps, so `best.pt` is chosen on R@1 here, never on loss.
        # NOTE: needs rendered 3-channel tencode frames. The 2-channel polarity-count `.npy`
        # accumulator dumps (LENSV2 `*-frames-50/`) are a different representation — not usable.
        self.brisbane_root = None          # None -> real eval disabled (e.g. --smoke)
        self.brisbane_ref_dir = "sunset2"
        self.brisbane_query_dir = "sunset1"
        self.brisbane_gt = "sunset2_sunset1_GT.npy"
        self.brisbane_ext = "*.png"
        # Eval gallery stride. R@1 is MONOTONE in density (same weights: .359 @5s ->
        # .686 @1s -> .833 @full-rate, because at full rate the nearest ref is 50ms /
        # ~0.5m away = a near-duplicate). A recall number is meaningless without its
        # stride — only compare like-for-like. 0.25s = 2896q/2565r, ~5s/eval, binomial
        # SE ~0.76% (vs ~1.7% at 1s) and still clear of saturation.
        self.brisbane_sec = 0.25
        self.brisbane_hz = 20.0            # 50ms tencode bins -> 20Hz
        self.eval_every = 500              # steps between real evals (0 -> off)
        self.eval_at_start = True          # baseline eval at step 0 (pretrained, untrained head)
        self.eval_batch_size = 64

        # ---- multi-condition eval suite (PLAN 1l; src/vpr/evalsuite.py) --
        # PLAN 1l: same-illumination retrieval is excellent (sunset1 .895) but
        # cross-illumination collapses (daytime .319, night .028). That is what the extra
        # MegaLoc streams are meant to fix, so every condition is tracked *during* training.
        # Reference is sunset2; each condition is scored against it.
        self.eval_root = None              # None -> suite disabled; <root>/<cond>/<cond>.hdf5
        self.eval_source = "hdf5"          # "hdf5" (eventcv slices raw events) | "png"
        self.eval_gt_dir = None            # None -> <eval_root>/ground_truth
        self.eval_ref = "sunset2"
        self.eval_conditions = ["sunset1", "morning", "daytime", "sunrise", "night"]
        # SELECTION IS ONE CONDITION ONLY. best.pt follows sunset1; the rest are monitors,
        # which is what keeps them valid as held-out evidence (PLAN 1e).
        self.eval_select = "sunset1"
        self.eval_dt_ms = 50.0             # slice duration = the 20 Hz tencode bin
        # Leading slices to skip per traverse, for eval_source="hdf5" only. A raw
        # recording can start BEFORE the published traverse: Brisbane sunset2 has 202
        # extra leading slices (13027 vs 12825 rendered frames), and because the GT is
        # resized proportionally onto the descriptor grid, ignoring that silently shifts
        # the whole GT band — measured cost on sunset1: R@1 .895 -> .608. None -> read
        # <eval_root>/eval_offsets.json (written by `evalsuite.py --derive-offsets`).
        # Baked-in defaults: these are physical leading-slice counts of the raw Brisbane
        # recordings (same files on HPC), validated to reproduce the measured offsets exactly
        # (sunset2 202, daytime 147, morning 288, sunset1 0); sunrise 190 / night 222 were
        # derived from the GT start timestamps. A local eval_offsets.json still overrides.
        self.eval_offsets = {"sunset1": 0, "sunset2": 202, "daytime": 147,
                             "morning": 288, "sunrise": 190, "night": 222}
        self.eval_sensor = [346, 260]      # DAVIS346; skips eventcv's resolution scan
        # Frame stride. R@1 is monotone in gallery density (PLAN 1j: .359 @5s -> .833 at
        # full rate for ONE unchanged model), so a recall number means nothing without it.
        # 5 = 0.25 s: ~2.9k queries, binomial SE ~0.76%, still clear of saturation.
        self.eval_stride = 5
        self.eval_suite_every = 2000       # steps between *full* suite runs (0 -> only select)
        # Early stop: N consecutive evals with no R@1 improvement. PLAN 1h's decay was
        # slow and monotonic, so keep this generous — it is a safety net, not the plan.
        self.early_stop_evals = 0          # 0 -> off (best.pt still tracked)

        # ---- eval-time PCA-whitening of the reference bank ---------------
        # Post-hoc whitening fit on the reference traverse's descriptors and applied to
        # both the reference gallery and every query (transductive: no query labels leak).
        # Default OFF so historical numbers stay comparable — it is an added monitor
        # (`eval/<cond>/R@1_pca`) and a persisted transform (`best.pca.npz`), never the
        # selection signal. power 0.5 + dim ~2048-4096 won the SALAD countmask probe by
        # ~+6pt mean R@1; power 0.25 was the *GeM* optimum and does NOT transfer to SALAD
        # (see evalsuite.pca_fit and the salad-pca-whiten-win note).
        self.eval_pca = False              # --eval-pca: also whiten, log R@1_pca, save transform
        self.pca_power = 0.5               # whitening exponent (0.5=full, 0.25=shrinkage)
        self.pca_dim = None                # None -> full rank; 2048-4096 = ~peak + 2-4x compression
        self.pca_eps = 1e-4                # per-axis variance floor

        # ---- wandb -------------------------------------------------------
        self.wandb = False                 # --wandb to enable (user is logged in on HPC)
        self.wandb_project = "gept-event-vpr"
        self.wandb_entity = None           # None -> default entity
        self.wandb_group = None            # set per sweep so runs group together
        self.wandb_mode = "online"
        self.wandb_tags = []

        # ---- runtime / logging ------------------------------------------
        self.device = "cuda"
        self.n_workers = 8
        self.seed = 0
        self.log_every = 50
        self.save_every = 2000
        self.save_dir = os.path.join(self.repo_root, "src", "vpr", "runs")

        # ---- checkpoint / resume / SLURM requeue ------------------------
        # run_name keys the run dir; default to SLURM_JOB_ID so a *requeued* job
        # (same id) auto-resumes into the same dir. Empty -> trainer uses a timestamp.
        self.run_name = os.environ.get("SLURM_JOB_ID", "")
        self.resume = "auto"               # "auto" (latest.pt in run dir) | path | None
        self.auto_requeue = True           # on SIGUSR1/SIGTERM: checkpoint + scontrol requeue
        self.ckpt_every_min = 30           # wall-time checkpoint cadence (beats the 48h cap)

    # convenience -------------------------------------------------------
    def apply_vit(self, size):
        """Switch backbone size, re-deriving embed/head/**register count** + ckpt path.

        Registers differ per checkpoint (small=0, base=4); setting ``vit`` alone would
        leave the old spec and fail the strict state_dict load.
        """
        spec = _BACKBONE_SPECS[size]
        self.vit = size
        self.n_embed = spec["n_embed"]
        self.n_head = spec["n_head"]
        self.num_register_tokens = spec["num_register_tokens"]
        self.ckpt_path = os.path.join(self.repo_root, spec["ckpt"])
        return self

    def finalize(self):
        """Resolve derived fields after all overrides are applied.

        SALAD emits its own descriptor (clusters*cluster_dim + token_dim) with no
        projection, so ``desc_dim`` must reflect that for logging, eval buffers, and the
        saved config. Call once after arg parsing, before building the model. Idempotent.
        """
        if self.salad_init == "megaloc":
            # MegaLoc's pretrained SALAD head geometry — its checkpoint uses cluster_dim 256
            # (not our 128 default), so warm-starting requires matching it. Descriptor becomes
            # 64*256 + 256 = 16640-d for these arms. Single source of truth so every entry
            # point (train.py, evalsuite) gets it before the model is built.
            self.salad_clusters, self.salad_cluster_dim = 64, 256
            self.salad_token_dim, self.salad_mlp_dim = 256, 512
        if self.aggregator == "salad":
            self.salad_out_dim = (self.salad_clusters * self.salad_cluster_dim
                                  + self.salad_token_dim)
            self.desc_dim = self.salad_proj_dim if self.salad_proj else self.salad_out_dim
        return self

    @property
    def h_tokens(self):
        return self.H // self.P

    @property
    def w_tokens(self):
        return self.W // self.P

    def summary(self):
        return (
            f"VPRConfig(vit={self.vit}, embed={self.n_embed}, regs={self.num_register_tokens}, "
            f"desc={self.desc_dim}, agg={self.aggregator}, ft_blocks={self.n_trainable_blocks}, "
            f"transfer={self.transfer}, streams={len(self.streams)}x"
            f"{self.P_places}x{self.K_images}={len(self.streams) * self.P_places * self.K_images}img/step, "
            f"lr={self.lr:g}, llrd={self.layerwise_lr_decay}, wd={self.wd:g}, steps={self.steps})"
        )
