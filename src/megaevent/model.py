"""GEPT DINOv2 encoder with GeM or projected SALAD aggregation."""

import torch
import torch.nn.functional as F
from torch import nn

# NB: we deliberately subclass nn.Module rather than src/model.py's ``Transformer``
# base. That base is an empty placeholder (it just nulls four attrs), and our module
# is itself named ``model.py`` - importing the repo's ``model`` would self-shadow.


def build_encoder(cfg):
    """Build a dinov2 event ViT matching the GEPT checkpoint for ``cfg.vit``.

    Register count is derived from the size (0 for small, 4 for base) so the
    pretrained state_dict loads ``strict=True`` - unlike ``src/cls.py`` which
    hardcodes 4 and would fail on ``small.pt``.
    """
    from ._vendor.dinov2.models.vision_transformer import vit_base, vit_small

    builders = {"small": vit_small, "base": vit_base}
    if cfg.vit not in builders:
        raise ValueError(f"Unsupported vit size '{cfg.vit}'")
    return builders[cfg.vit](
        patch_size=cfg.P,
        img_size=cfg.backbone_img_size,
        block_chunks=0,
        init_values=1e-6,
        num_register_tokens=cfg.num_register_tokens,
    )


class GeMPool(nn.Module):
    """Generalized-Mean pooling with a learnable exponent ``p``.

    Operates on spatial feature maps ``[B, C, H, W]`` -> ``[B, C]``.
    Form follows EventGeM (``../Event-GeM/eventgem/feature_extraction.py:74``) but
    with ``p`` learnable (init 3.0) instead of a fixed 5.0.
    """

    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p_init))
        self.eps = eps

    def forward(self, x):  # x: [B, C, H, W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))  # [B, C, 1, 1]
        x = x.pow(1.0 / self.p)
        return x.flatten(1)  # [B, C]

    def extra_repr(self):
        return f"p_init(learnable), eps={self.eps}"


class VPRModel(nn.Module):
    """GEPT event ViT + GeM + linear projection -> L2-normalised descriptor."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        # ---- backbone ----------------------------------------------------
        self.encoder = build_encoder(cfg)
        if cfg.load_pretrained:
            sd = torch.load(cfg.ckpt_path, map_location="cpu")
            state = sd["event_encoder"] if "event_encoder" in sd else sd
            self.encoder.load_state_dict(state, strict=True)
            print(f"[VPRModel] loaded encoder from {cfg.ckpt_path} (strict)")

        # ---- aggregator + projection ------------------------------------
        # GeM -> linear proj to desc_dim. SALAD emits its own L2-normalised descriptor of
        # width clusters*cluster_dim + token_dim (default 64*128+256 = 8448) and by default
        # that IS the descriptor — SALAD-faithful, as published. --salad-proj adds the step
        # MegaLoc puts on top of SALAD instead: a wider aggregator learnedly compressed to
        # desc_dim, re-normalised after.
        self.is_salad = cfg.aggregator == "salad"
        if cfg.aggregator == "gem":
            self.aggregator = GeMPool(p_init=cfg.gem_p_init, eps=cfg.gem_eps)
            self.proj = nn.Linear(cfg.n_embed, cfg.desc_dim)
        elif cfg.aggregator == "salad":
            from .salad import FeatureAggregator

            self.aggregator = FeatureAggregator(
                num_channels=cfg.n_embed,
                num_clusters=cfg.salad_clusters,
                cluster_dim=cfg.salad_cluster_dim,
                token_dim=cfg.salad_token_dim,
                mlp_dim=cfg.salad_mlp_dim,
                dropout=cfg.salad_dropout,
            )
            # Optional warm-start from MegaLoc's pretrained SALAD head (ViT-B only). After the
            # backbone load so it only overwrites the aggregator, not the encoder.
            if getattr(cfg, "salad_init", None) == "megaloc":
                from .salad import load_megaloc_aggregator

                load_megaloc_aggregator(self.aggregator, getattr(cfg, "megaloc_weights", None))
            if getattr(cfg, "salad_proj", False):
                # cfg.finalize() has already made desc_dim the projection's output and
                # salad_out_dim the aggregator's, so these two cannot silently disagree.
                salad_out = cfg.salad_out_dim or (
                    cfg.salad_clusters * cfg.salad_cluster_dim + cfg.salad_token_dim
                )
                self.proj = nn.Linear(salad_out, cfg.desc_dim)
                if getattr(cfg, "salad_init", None) == "megaloc":
                    from .salad import load_megaloc_projection

                    load_megaloc_projection(self.proj, getattr(cfg, "megaloc_weights", None))
            else:
                self.proj = None  # SALAD emits the final descriptor itself
        else:
            raise ValueError(f"Unknown aggregator '{cfg.aggregator}'")

        self._apply_finetune_freeze()
        if getattr(cfg, "grad_checkpoint", False):
            self._enable_grad_checkpoint()

    # ------------------------------------------------------------------
    def _enable_grad_checkpoint(self):
        """Recompute encoder-block activations in backward instead of storing them.

        This is what lets ViT-B train at the same ``P_places`` as ViT-S. ``train_patch_embed``
        makes the patch-embedding output require grad, so autograd stores activations for
        **every** block regardless of which ones are trainable — peak memory scales with
        depth, not with trainable-parameter count. That, not the parameter count, is why the
        countmask sweep had to run ViT-B at ``--P 16``, which halved its negative pool.

        Patched onto the bound ``forward`` of each block rather than by wrapping the blocks
        in a new Module, so ``state_dict`` keys are untouched and checkpoints stay
        interchangeable with non-checkpointed runs. Inference (``no_grad``) and any input
        that does not require grad fall through to the original forward, so eval pays
        nothing and ``checkpoint`` never warns about a graph-free call.
        """
        from torch.utils.checkpoint import checkpoint

        def wrap(inner):
            def fwd(x, *args, **kwargs):
                if torch.is_grad_enabled() and torch.is_tensor(x) and x.requires_grad:
                    return checkpoint(inner, x, *args, use_reentrant=False, **kwargs)
                return inner(x, *args, **kwargs)

            return fwd

        for blk in self.encoder.blocks:
            blk.forward = wrap(blk.forward)
        print(
            f"[VPRModel] gradient checkpointing: {len(self.encoder.blocks)} encoder blocks "
            f"recomputed in backward (~30% slower/step, depth-proportional memory saving)"
        )

    # ------------------------------------------------------------------
    def _apply_finetune_freeze(self):
        """Freeze the backbone per the finetune strategy (PLAN 1c).

        finetune : train last ``n_trainable_blocks`` blocks + (optionally) norm +
                   patch_embed; freeze embeddings/cls/register/pos + earlier blocks.
        linear   : freeze the entire encoder (linear-probe sanity run).
        The aggregator (GeM ``p``) and projection are always trainable.
        """
        enc = self.encoder
        for p in enc.parameters():
            p.requires_grad = False

        if self.config.transfer == "linear":
            print("[VPRModel] linear-probe: encoder fully frozen")
            return

        k = self.config.n_trainable_blocks
        if k > 0:
            for blk in enc.blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
        if self.config.train_norm and hasattr(enc, "norm"):
            for p in enc.norm.parameters():
                p.requires_grad = True
        if self.config.train_patch_embed and hasattr(enc, "patch_embed"):
            for p in enc.patch_embed.parameters():
                p.requires_grad = True
        if getattr(self.config, "train_embeddings", False):
            # cls/pos/mask/register tokens — random init has to learn these, so a fair
            # from-scratch run must unfreeze them (they route to the encoder lr group).
            for attr in ("cls_token", "pos_embed", "mask_token", "register_tokens"):
                param = getattr(enc, attr, None)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = True

        n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in enc.parameters())
        print(
            f"[VPRModel] finetune: last {k} blocks"
            f"{' + norm' if self.config.train_norm else ''}"
            f"{' + patch_embed' if self.config.train_patch_embed else ''}"
            f"{' + embeddings' if getattr(self.config, 'train_embeddings', False) else ''}"
            f"  ({n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M encoder params trainable)"
        )

    # ------------------------------------------------------------------
    def forward_encoder(self, x):
        """[B,3,H,W] -> (spatial feature map [B, C, h, w], CLS token [B, C]).

        SALAD needs both the patch grid and the global CLS token; GeM uses only
        the grid. dinov2 ``forward_features`` returns both in one pass.
        """
        h, w = x.shape[-2] // self.config.P, x.shape[-1] // self.config.P
        out = self.encoder.forward_features(x)
        tokens = out["x_norm_patchtokens"]  # [B, N, C]
        cls = out["x_norm_clstoken"]  # [B, C]
        B, N, C = tokens.shape
        assert N == h * w, f"token count {N} != h*w {h * w} (input {x.shape[-2]}x{x.shape[-1]})"
        return tokens.transpose(1, 2).reshape(B, C, h, w), cls

    def forward(self, x):
        """[B,3,H,W] -> L2-normalised global descriptor."""
        feat, cls = self.forward_encoder(x)  # [B,C,h,w], [B,C]
        if self.is_salad:
            desc = self.aggregator((feat, cls))  # [B, salad_out_dim], already L2-normed
            if self.proj is None:
                return desc
        else:
            desc = self.aggregator(feat)  # GeM: [B, C]
        # MegaLoc normalises *after* its projection; an L2-normalised input to a linear map
        # does not come out normalised, and cosine retrieval requires that it is.
        return F.normalize(self.proj(desc), p=2, dim=1)
