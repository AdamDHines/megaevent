"""Shared checkpoint geometry and v8 training configuration."""

import os

_BACKBONE_SPECS = {
    "small": {"n_embed": 384, "n_head": 6, "num_register_tokens": 0, "ckpt": "small.pt"},
    "base": {"n_embed": 768, "n_head": 12, "num_register_tokens": 4, "ckpt": "base.pt"},
}


class VPRConfig:
    def __init__(self):
        self.repo_root = os.getcwd()
        self.data_root = "data"
        self.vit = "small"
        self.vit_backbone = "dinov2"
        spec = _BACKBONE_SPECS[self.vit]
        self.n_embed = spec["n_embed"]
        self.n_head = spec["n_head"]
        self.num_register_tokens = spec["num_register_tokens"]
        self.n_layer = 12
        self.P = 14
        self.backbone_img_size = 518
        self.ckpt_path = os.path.join(self.repo_root, spec["ckpt"])
        self.load_pretrained = True
        self.image_ext = ".jpg"
        self.force_ext = None
        self.compute_stats = False
        self.stats_max_images = 5000
        self.H, self.W = (224, 224)
        self.n_tokens_per_image = self.H // self.P * (self.W // self.P)
        self.eval_img_size = None
        self.tencode_mean = [0.9673, 0.9297, 0.9624]
        self.tencode_std = [0.1204, 0.1674, 0.1265]
        self.white_frame = True
        self.representation = "tencode"
        self.aggregator = "gem"
        self.desc_dim = 2048
        self.gem_p_init = 3.0
        self.gem_eps = 1e-06
        self.salad_clusters = 64
        self.salad_cluster_dim = 128
        self.salad_token_dim = 256
        self.salad_mlp_dim = 512
        self.salad_dropout = 0.3
        self.salad_init = None
        self.megaloc_weights = os.environ.get("GEPT_MEGALOC_WEIGHTS", None)
        self.salad_proj = False
        self.salad_proj_dim = 8448
        self.salad_out_dim = None
        self.transfer = "finetune"
        self.n_trainable_blocks = 4
        self.train_norm = True
        self.train_patch_embed = True
        self.train_embeddings = False
        self.layerwise_lr_decay = None
        self.grad_checkpoint = False
        self.lr = 0.0001
        self.encoder_lr_mult = 0.1
        self.transformer_lr_mult = 1.0
        self.wd = 0.001
        self.min_lr = 0.0
        self.warmup_steps = 1000
        self.steps = 12000
        self.stop_after = 0
        self.grad_clip = 1.0
        self.amp = True
        self.fp32_loss = False
        self.P_places = 32
        self.K_images = 4
        self.streams = ["gsv_cities"]
        self.refresh_index = False
        self.refresh_stats = False
        self.sfxl_M = 20
        self.sfxl_N = 5
        self.sfxl_focal_dist = 10
        self.sfxl_groups = [0]
        self.sfxl_min_images_per_class = 10
        self.sfxl_cycle_groups = True
        self.max_img_per_place = 64
        self.scannet_chunk = 0
        self.scannet_gap = 4
        self.msls_meta_root = None
        self.msls_view_bins = 4
        self.msls_merge_splits = "auto"
        self.msls_exclude_cities = []
        self.ms_alpha = 2.0
        self.ms_beta = 50.0
        self.ms_base = 0.5
        self.miner_epsilon = 0.1
        self.xbm_size = 0
        self.xbm_start = 2000
        self.aug_domain_rand = False
        self.aug_gain_jitter = [0.7, 1.4]
        self.aug_dropout_max = 0.15
        self.aug_salt_prob = 0.005
        self.aug_hflip = True
        self.brisbane_root = None
        self.brisbane_ref_dir = "sunset2"
        self.brisbane_query_dir = "sunset1"
        self.brisbane_gt = "sunset2_sunset1_GT.npy"
        self.brisbane_ext = "*.png"
        self.brisbane_sec = 0.25
        self.brisbane_hz = 20.0
        self.eval_every = 500
        self.eval_at_start = True
        self.eval_batch_size = 64
        self.eval_root = None
        self.eval_source = "hdf5"
        self.eval_gt_dir = None
        self.eval_ref = "sunset2"
        self.eval_conditions = ["sunset1", "morning", "daytime", "sunrise", "night"]
        self.eval_pooled = False
        self.eval_pooled_query = "sunset1"
        self.eval_pooled_db = ["sunset2", "daytime", "morning", "night", "sunrise"]
        self.eval_pooled_coords = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "brisbane_pooled_coords.npz",
        )
        self.eval_pooled_threshold_m = 25.0
        self.eval_select = "sunset1"
        self.eval_combined_w = 0.5
        self.select_on = "native"
        self.eval_dt_ms = 50.0
        self.eval_offsets = {
            "sunset1": 0,
            "sunset2": 202,
            "daytime": 147,
            "morning": 288,
            "sunrise": 190,
            "night": 222,
        }
        self.eval_sensor = [346, 260]
        self.eval_stride = 5
        self.eval_suite_every = 2000
        self.early_stop_evals = 0
        self.msls_val_cities = []
        self.msls_val_max_db = 10000
        self.msls_val_max_q = 2000
        self.msls_val_radius_m = 25.0
        self.msls_val_seed = 0
        self.eval_pca = False
        self.pca_power = 0.5
        self.pca_dim = None
        self.pca_eps = 0.0001
        self.wandb = False
        self.wandb_project = "gept-event-vpr"
        self.wandb_entity = None
        self.wandb_group = None
        self.wandb_mode = "online"
        self.wandb_tags = []
        self.device = "auto"
        self.n_workers = 0
        self.seed = 0
        self.log_every = 50
        self.save_every = 2000
        self.save_dir = "runs"
        self.run_name = os.environ.get("SLURM_JOB_ID", "")
        self.init_from = None
        self.resume = "auto"
        self.auto_requeue = False
        self.ckpt_every_min = 30

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

    def apply_img_size(self, n):
        """Set the train/eval input resolution, re-deriving the token count.

        Nothing structural depends on this: ``VPRModel.forward_encoder`` derives the patch
        grid from the input tensor and dinov2 interpolates its position embedding, so any
        multiple of ``P`` works and a checkpoint trained at one size records it (megaevent's
        inference reads ``H``/``W`` back off the checkpoint). ``n_tokens_per_image`` is only
        used for logging/buffers but must not go stale.
        """
        n = int(n)
        if n % self.P:
            raise ValueError(f"--img-size {n} is not a multiple of the patch size {self.P}")
        self.H = self.W = n
        self.n_tokens_per_image = self.H // self.P * (self.W // self.P)
        return self

    def finalize(self):
        """Resolve derived fields after all overrides are applied.

        SALAD emits its own descriptor (clusters*cluster_dim + token_dim) with no
        projection, so ``desc_dim`` must reflect that for logging, eval buffers, and the
        saved config. Call once after arg parsing, before building the model. Idempotent.
        """
        if self.salad_init == "megaloc":
            self.salad_clusters, self.salad_cluster_dim = (64, 256)
            self.salad_token_dim, self.salad_mlp_dim = (256, 512)
        if self.aggregator == "salad":
            self.salad_out_dim = self.salad_clusters * self.salad_cluster_dim + self.salad_token_dim
            self.desc_dim = self.salad_proj_dim if self.salad_proj else self.salad_out_dim
        return self

    @property
    def h_tokens(self):
        return self.H // self.P

    @property
    def w_tokens(self):
        return self.W // self.P

    def summary(self):
        return f"VPRConfig(vit={self.vit}, embed={self.n_embed}, regs={self.num_register_tokens}, desc={self.desc_dim}, agg={self.aggregator}, ft_blocks={self.n_trainable_blocks}, transfer={self.transfer}, streams={len(self.streams)}x{self.P_places}x{self.K_images}={len(self.streams) * self.P_places * self.K_images}img/step, {self.H}x{self.W}{(', ckpt-grad' if self.grad_checkpoint else '')}{(f', xbm={self.xbm_size}@{self.xbm_start}' if self.xbm_size else '')}{(', domain-rand' if self.aug_domain_rand else '')}, lr={self.lr:g}, llrd={self.layerwise_lr_decay}, wd={self.wd:g}, warmup={self.warmup_steps}, ms=({self.ms_alpha:g},{self.ms_beta:g},{self.ms_base:g}){('/fp32' if self.fp32_loss else '')}, steps={self.steps})"
