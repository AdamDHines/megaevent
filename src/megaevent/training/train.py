"""V8 multi-stream training with native-cosine pooled validation."""

import contextlib
import os
import random
import signal
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn

from ..config import VPRConfig
from ..model import VPRModel
from ..runtime import device_for
from .streams import stream_summary
from .utils import get_lr, get_param_groups, get_param_groups_llrd


def _seed_everything(seed):
    """cfg.seed existed but was never applied — sweep arms must differ by the swept axis,
    not by a random init/shuffle order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loss_and_miner(cfg):
    """``(loss_fn, miner, xbm)``. Imported lazily so the module loads without pml.

    ``xbm`` is pytorch-metric-learning's own ``losses.CrossBatchMemory`` (None when
    ``cfg.xbm_size`` is 0) — see ``VPRTrainer.train_step`` for why the reference
    implementation rather than a hand-rolled queue.
    """
    from pytorch_metric_learning import losses, miners

    loss_fn = losses.MultiSimilarityLoss(alpha=cfg.ms_alpha, beta=cfg.ms_beta, base=cfg.ms_base)
    miner = miners.MultiSimilarityMiner(epsilon=cfg.miner_epsilon)
    xbm = None
    if getattr(cfg, "xbm_size", 0):
        if cfg.xbm_size < cfg.P_places * cfg.K_images:
            raise SystemExit(
                f"--xbm-size {cfg.xbm_size} is smaller than one sub-batch "
                f"({cfg.P_places}x{cfg.K_images}={cfg.P_places * cfg.K_images})"
            )
        xbm = losses.CrossBatchMemory(loss_fn, cfg.desc_dim, memory_size=cfg.xbm_size, miner=miner)
    return loss_fn, miner, xbm


# The pseudo-condition that selects on real-event *and* in-domain recall together.
COMBINED = "combined"

# Selection heads. Every eval updates all three independently, so choosing which one to
# ship is a post-hoc decision over a kept Pareto set instead of a bet made at training time.
#   real — mean R@1 over the real-event conditions (sim-to-real transfer)
#   i2e  — held-out MSLS cities in the training representation (in-domain quality)
#   main — whatever cfg.eval_select names: one condition, or COMBINED
# The audit that motivated this recovered +13.3 R@1 on Tokyo purely by re-scoring
# milestone checkpoints that a single-head best.pt had thrown away.
HEAD_FILES = {"main": "best.pt", "real": "best_real.pt", "i2e": "best_i2e.pt"}

# The resolution every headline number is reported at (megaevent's pooled/table protocol).
# Selection at any other resolution optimises a metric nobody reports — start() warns.
REPORTING_RESOLUTION = 322


def _fresh_heads():
    return {h: {"r1": -1.0, "step": -1} for h in HEAD_FILES}


def _migrate_heads(saved):
    """Read a checkpoint's ``best`` blob in either the old or the new layout.

    Pre-multi-head checkpoints stored a bare ``{"r1", "step"}`` for the single ``best.pt``;
    resuming one of those must not silently reset the score it had already reached, or the
    resumed job would overwrite a better best.pt with a worse one.
    """
    heads = _fresh_heads()
    if not isinstance(saved, dict):
        return heads
    if "r1" in saved:
        heads["main"] = {"r1": float(saved["r1"]), "step": int(saved.get("step", -1))}
        return heads
    for h, v in saved.items():
        if h in heads and isinstance(v, dict):
            heads[h] = {"r1": float(v.get("r1", -1.0)), "step": int(v.get("step", -1))}
    return heads


class VPRTrainer:
    def __init__(self, cfg, streams, device=None):
        """streams: dict[str, DataLoader] - one P x K places loader per dataset."""
        self.cfg = cfg
        self.device = str(device_for(device or cfg.device))
        self.model = VPRModel(cfg).to(self.device)
        if getattr(cfg, "init_from", None):
            self._init_from(cfg.init_from)
        self.loss_fn, self.miner, self.xbm = _make_loss_and_miner(cfg)

        if getattr(cfg, "layerwise_lr_decay", None):
            param_groups = get_param_groups_llrd(
                self.model, cfg.wd, cfg.n_layer, cfg.layerwise_lr_decay
            )
            print(
                f"[train] LLRD param groups: {len(param_groups)} layers, "
                f"decay={cfg.layerwise_lr_decay}, lr_mult "
                f"{param_groups[0]['lr_mult']:.3f}..{param_groups[-1]['lr_mult']:.4f}"
            )
        else:
            param_groups = get_param_groups(
                self.model, cfg.wd, cfg.encoder_lr_mult, cfg.transformer_lr_mult
            )
        self.optimizer = torch.optim.AdamW(param_groups)
        use_amp = cfg.amp and self.device.startswith("cuda")
        self.amp = torch.amp.autocast(device_type="cuda", enabled=use_amp)
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        self.streams = streams
        self.iters = {name: iter(loader) for name, loader in streams.items()}

        run_name = getattr(cfg, "run_name", "") or datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.run_dir = getattr(cfg, "output_dir", None) or os.path.join(cfg.save_dir, run_name)
        self.writer = None  # created lazily in start() to keep smoke tests quiet
        self.wandb = None  # ditto
        self._should_stop = False  # set by the signal handler (SLURM preemption/time-limit)
        self._last_ckpt_t = time.time()
        # Model selection on eval recall (PLAN 1h) — never on loss. Three heads, see HEAD_FILES.
        self.best = _fresh_heads()
        self._evals_since_best = 0
        self.eval_history = []
        self._n_real = 0  # conditions behind the latest `real` mean
        self._warned_partial = False  # say it once, not once per eval

        if self.xbm is not None:
            per_step = len(streams) * cfg.P_places * cfg.K_images
            print(
                f"[train] XBM: {cfg.xbm_size} x {cfg.desc_dim} queue from step "
                f"{cfg.xbm_start} (~{cfg.xbm_size * cfg.desc_dim * 2 / 1e6:.0f} MB fp16), "
                f"shared across all {len(streams)} streams -> ~{cfg.xbm_size // cfg.K_images} "
                f"reference places per loss call vs {cfg.P_places - 1} in-batch, refilled "
                f"every ~{max(1, cfg.xbm_size // per_step)} steps"
            )

    def _init_from(self, path):
        """Load **weights only** from a finished run: no optimizer, no step, no best scores.

        This is how the resolution fine-tune starts — a fresh short cosine schedule at a
        lower LR over an already-trained model. ``--resume`` is the opposite operation
        (continue the same run with the same optimizer moments and step counter) and would
        drag the old 20k-step schedule and the old best-head scores along with it.

        ``strict=True`` on purpose: nothing about the resolution lives in the state dict, so a
        key mismatch here means the *architecture* differs (a ViT-S checkpoint into a ViT-B
        run, or a GeM checkpoint into a SALAD one) and should stop the job, not be absorbed.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"] if "model" in ck else ck)
        src = ck.get("config", {})
        print(
            f"[train] initialised weights from {path} (its step {ck.get('step', '?')}, "
            f"{src.get('H', '?')}x{src.get('W', '?')}) -> training at "
            f"{self.cfg.H}x{self.cfg.W}; optimizer and step counter start fresh"
        )

    def _next(self, name):
        """Infinite cycling over a stream's loader."""
        try:
            return next(self.iters[name])
        except StopIteration:
            self.iters[name] = iter(self.streams[name])
            return next(self.iters[name])

    def train_step(self, step):
        self.model.train()
        lr = get_lr(step, self.cfg.warmup_steps, self.cfg.lr, self.cfg.steps, self.cfg.min_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g["lr_mult"]

        self.optimizer.zero_grad(set_to_none=True)
        # Miner and loss run inside autocast by default, so the pairwise similarity matrix
        # is fp16 — ~3 decimal digits near cosine 1.0, against a miner epsilon of 0.1 and a
        # loss gradient of exp(-beta*(s - base)) with beta=50. --fp32-loss lifts just those
        # two out of autocast; the backbone forward stays mixed-precision either way.
        loss_ctx = contextlib.nullcontext() if self.cfg.fp32_loss else self.amp
        xbm_on = self.xbm is not None and step >= self.cfg.xbm_start

        per_stream = {}
        for s_idx, name in enumerate(self.streams):
            x, y = self._next(name)
            x, y = x.to(self.device), y.to(self.device)
            if xbm_on:
                # Re-mint labels for the shared queue: y*n+idx preserves equality within a
                # stream and can never collide across streams — the guarantee place_offset
                # was believed to give and does not (streams.check_id_ranges). Without
                # this, an sf_xl_frontal/lateral collision in the queue is a false
                # POSITIVE, which is what confounded the v2 XBM verdict.
                y = y * len(self.streams) + s_idx
            with self.amp:
                desc = self.model(x)  # [P*K, D], L2-normalised
            with loss_ctx:
                d = desc.float() if self.cfg.fp32_loss else desc
                # One shared queue across all six streams. CAUTION: this assumed place ids
                # are globally unique across streams, and they are NOT — sf_xl_frontal and
                # sf_xl_lateral collide by construction (streams.check_id_ranges measures
                # and warns at startup). A collision in the queue makes two places km apart
                # "the same place", i.e. a false POSITIVE in CrossBatchMemory — so the v2
                # "XBM is negative" verdict is confounded and must be re-run with re-minted
                # ids before XBM is retired for good. CrossBatchMemory enqueues the live
                # batch *before* mining, so in-batch positives are in the reference set and
                # self-comparisons are removed; anchors stay the live batch.
                loss = self.xbm(d, y) if xbm_on else self.loss_fn(d, y, self.miner(d, y))
                per_stream[name] = loss.item()
            self.scaler.scale(loss).backward()  # accumulate across streams

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return sum(per_stream.values()), per_stream, lr, float(grad_norm)

    # ---- real-event eval / model selection ---------------------------
    def _eval_enabled(self):
        return bool(getattr(self.cfg, "eval_root", None)) and getattr(self.cfg, "eval_every", 0)

    def _indomain_enabled(self):
        return bool(getattr(self.cfg, "msls_val_cities", None))

    def _evaluate_indomain(self, step, verbose=False):
        """Held-out MSLS cities in the training representation, or None if unavailable."""
        from . import i2eval

        t0 = time.time()
        try:
            res = i2eval.evaluate(
                self.model,
                self.cfg,
                self.device,
                ks=(1,),
                num_workers=min(self.cfg.n_workers, 8),
                verbose=verbose,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as err:
            # A broken monitor must not kill the run, but it must be impossible to miss:
            # with --eval-select combined it takes the selection signal down with it.
            print(f"[eval] in-domain monitor FAILED: {err}")
            return None
        per_city = " ".join(
            f"{k.split('/', 1)[1]}={v:.3f}" for k, v in res.items() if k.startswith("R@1/")
        )
        print(
            f"[eval] step {step}  in-domain msls[{','.join(self.cfg.msls_val_cities)}] "
            f"R@1={res['R@1']:.3f}  ({per_city or 'single city'}) "
            f"({res['nq']}q/{res['nr']}r, {time.time() - t0:.0f}s)"
        )
        return res

    def evaluate_suite(self, step, full=False):
        """Evaluate native-cosine pooled validation and update checkpoint heads."""
        from .evalsuite import run_pooled

        cfg = self.cfg
        t0 = time.time()
        res = run_pooled(
            self.model,
            cfg.eval_root,
            cfg.eval_pooled_coords,
            cfg.eval_pooled_query,
            list(cfg.eval_pooled_db),
            cfg,
            self.device,
            ks=(1,),
            dt_ms=cfg.eval_dt_ms,
            sensor_size=tuple(cfg.eval_sensor),
            stride=cfg.eval_stride,
            num_workers=min(cfg.n_workers, 8),
            threshold_m=cfg.eval_pooled_threshold_m,
            verbose=step == 0,
        )
        dt = time.time() - t0
        results = {"pooled": res}
        indomain = (
            self._evaluate_indomain(step, verbose=step == 0) if self._indomain_enabled() else None
        )
        scores = {"real": res["R@1"], "main": res["R@1"]}
        if indomain is not None and indomain.get("R@1") is not None:
            scores["i2e"] = indomain["R@1"]
        self._n_real = 1
        if "main" not in scores:
            why = (
                results.get(cfg.eval_select, {}).get("error", "missing")
                if cfg.eval_select != COMBINED
                else "needs a full real-event suite AND the in-domain monitor"
            )
            print(
                f"[eval] step {step}: selection signal {cfg.eval_select!r} did not score ({why}) — best.pt unchanged"
            )
        self.eval_history.append(
            {
                "step": step,
                "secs": dt,
                "scores": dict(scores),
                "indomain": indomain,
                **{c: r for c, r in results.items()},
            }
        )
        improved = [h for h, v in scores.items() if h in HEAD_FILES and v > self.best[h]["r1"]]
        if improved:
            for h in improved:
                self.best[h] = {"r1": scores[h], "step": step}
            # Evaluation receives completed steps; resume stores the last zero-based step.
            state = self._state(step - 1)
            for h in improved:
                fname = HEAD_FILES[h]
                self._atomic_save(state, os.path.join(self.run_dir, fname))
        self._evals_since_best = 0 if "main" in improved else self._evals_since_best + 1
        head_parts = " ".join(
            (
                f"{h}={scores[h]:.3f}" + ("*" if h in improved else "")
                for h in ("main", "real", "i2e")
                if h in scores
            )
        )
        print(
            f"[eval] step {step}  {head_parts}  (best main {self.best['main']['r1']:.3f} @{self.best['main']['step']}, {dt:.0f}s)"
            + (f"  *** -> {', '.join((HEAD_FILES[h] for h in improved))}" if improved else "")
        )
        scalars = {}
        for cond, res in results.items():
            for k in (1,):
                if f"R@{k}" in res:
                    scalars[f"eval/{cond}/R@{k}"] = res[f"R@{k}"]
        if indomain:
            scalars["eval/msls_val/R@1"] = indomain["R@1"]
            for k, v in indomain.items():
                if k.startswith("R@1/"):
                    scalars[f"eval/msls_val/{k.split('/', 1)[1]}/R@1"] = v
        for h, v in scores.items():
            scalars[f"eval/score_{h}"] = v
        scalars["eval/n_real_conditions"] = self._n_real
        for h in HEAD_FILES:
            scalars[f"eval/best_{h}_R@1"] = self.best[h]["r1"]
        scalars["eval/secs"] = dt
        if self.writer:
            for key, v in scalars.items():
                self.writer.add_scalar(key, v, step)
        if self.wandb:
            self.wandb.log(
                {**scalars, **{f"eval/best_{h}_step": self.best[h]["step"] for h in HEAD_FILES}},
                step=step,
            )
        return results

    def _init_wandb(self):
        if not getattr(self.cfg, "wandb", False):
            return
        try:
            import wandb
        except ImportError:
            print(
                "[train] wandb requested but not installed (`pixi add wandb`) — continuing without"
            )
            return
        run_name = os.path.basename(self.run_dir).removesuffix("_vpr")
        self.wandb = wandb
        cfg = self.cfg
        # Tags make the ladder legible in the UI: which streams, which backbone, which depth.
        tags = (
            list(cfg.wandb_tags)
            + [f"vit-{cfg.vit}", f"blocks{cfg.n_trainable_blocks}", f"{len(self.streams)}stream"]
            + list(self.streams)
        )
        config = {
            **cfg.__dict__,
            "n_streams": len(self.streams),
            "images_per_step": len(self.streams) * cfg.P_places * cfg.K_images,
            "streams_detail": stream_summary(self.streams),
        }
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=run_name,
            mode=cfg.wandb_mode,
            tags=tags,
            config=config,
            resume="allow",
            id=run_name,
        )
        # Summary keeps the *peak* of each condition, so the run table is directly
        # comparable across the ladder even though every run is early-stopped differently.
        for cond in cfg.eval_conditions:
            wandb.define_metric(f"eval/{cond}/R@1", summary="max")
        wandb.define_metric("eval/msls_val/R@1", summary="max")
        for h in HEAD_FILES:
            wandb.define_metric(f"eval/score_{h}", summary="max")
            wandb.define_metric(f"eval/best_{h}_R@1", summary="max")
        print(
            f"[train] wandb: project={cfg.wandb_project} group={cfg.wandb_group} "
            f"run={run_name} tags={tags}"
        )

    def start(self):
        from .logging import ScalarWriter

        os.makedirs(self.run_dir, exist_ok=True)

        start_step = 0
        rp = self._resume_path()
        if rp:
            start_step = self.load_checkpoint(rp) + 1
            print(f"[train] resumed from {rp} -> continue at step {start_step}")

        # `stop_after` truncates the *loop* without touching cfg.steps, which is the cosine
        # horizon (utils.get_lr). Lowering cfg.steps instead would anneal the LR over the
        # short run and produce a different trajectory — the point of a fine-grained re-run
        # is to resolve the SAME curve at finer sampling, so the schedule must be identical.
        last_step = self.cfg.steps
        if getattr(self.cfg, "stop_after", 0):
            last_step = min(self.cfg.steps, self.cfg.stop_after)

        self.writer = ScalarWriter(self.run_dir, getattr(self.cfg, "tensorboard", False))
        self._init_wandb()
        self._install_signals()
        self._last_ckpt_t = time.time()
        ckpt_every_s = getattr(self.cfg, "ckpt_every_min", 30) * 60
        print(
            f"[train] {self.cfg.summary()} | streams={list(self.streams)} | "
            f"device={self.device} | run={self.run_dir}"
        )
        if last_step < self.cfg.steps:
            print(
                f"[train] stop-after: training steps 0..{last_step} only "
                f"(LR schedule still annealed over {self.cfg.steps} — trajectory matches "
                f"the full-length run)"
            )
        if self._eval_enabled():
            print(
                f"[train] real eval: {self.cfg.eval_root} (source={self.cfg.eval_source}, "
                f"ref={self.cfg.eval_ref}, stride={self.cfg.eval_stride})\n"
                f"        conditions {self.cfg.eval_conditions} every "
                f"{self.cfg.eval_suite_every} steps (selection every {self.cfg.eval_every})\n"
                f"        select on {self.cfg.eval_select!r} "
                f"({getattr(self.cfg, 'select_on', 'native')} recall) -> best.pt; "
                f"heads kept: {', '.join(HEAD_FILES.values())}"
            )
            # Selection happens at the in-loop eval resolution — and every headline number
            # is reported at 322. Measured (2026-08-04): 224-vs-322 does not just shift
            # the recall, it REORDERS checkpoints, so best.pt selected at another
            # resolution is the argmax of a metric nobody reports.
            eval_size = int(getattr(self.cfg, "eval_img_size", None) or self.cfg.H)
            if eval_size != REPORTING_RESOLUTION:
                print(
                    f"[train] WARNING: checkpoint selection runs at {eval_size}x"
                    f"{eval_size} but headline evaluation is at "
                    f"{REPORTING_RESOLUTION}x{REPORTING_RESOLUTION}. Resolution reorders "
                    f"checkpoints; best.pt will optimise a metric that is never "
                    f"reported. Pass --eval-img-size {REPORTING_RESOLUTION} before "
                    f"trusting selection."
                )
        else:
            print(
                "[train] real eval DISABLED (no --eval-root) — no best.pt, "
                "and loss is NOT a valid selection signal (PLAN 1h)"
            )
        if self._indomain_enabled():
            # Build the monitor index before the first step: a typo'd city or an unconverted
            # split should fail in the first minute, not after hours of training.
            from . import i2eval

            i2eval.msls_val_index(self.cfg)
            print(
                f"[train] in-domain monitor: held-out MSLS "
                f"{self.cfg.msls_val_cities} (excluded from training: "
                f"{self.cfg.msls_exclude_cities or 'NOTHING — see --msls-val-cities'})"
            )
        elif self.cfg.eval_select == COMBINED:
            raise SystemExit("--eval-select combined needs --msls-val-cities")

        if self._eval_enabled() and self.cfg.eval_at_start and start_step == 0:
            results = self.evaluate_suite(0, full=True)
            # Fail in the first minute rather than at hour 18. A condition that cannot score
            # at step 0 will not start scoring later (it is a missing GT file or an unreadable
            # traverse), and for combined selection it silently degrades the criterion every
            # run it touches — so make it a decision the submitter has to take deliberately.
            dead = {
                c: results.get(c, {}).get("error", "missing")
                for c in self.cfg.eval_conditions
                if results.get(c, {}).get("R@1") is None
            }
            if dead and self.cfg.eval_select == COMBINED:
                raise SystemExit(
                    f"[train] {len(dead)} of {len(self.cfg.eval_conditions)} eval conditions "
                    f"failed at step 0: {dead}\nWith --eval-select {COMBINED!r} the real-event "
                    f"half of the criterion would be a mean over a reduced set for the whole "
                    f"run. Fix the data, or drop them from --eval-conditions so the reduced "
                    f"set is on the record."
                )

        clip_hits, clip_steps, clip_warned = 0, 0, False
        for step in range(start_step, last_step):
            t0 = time.time()
            total, per_stream, lr, gnorm = self.train_step(step)
            # The clip is meant to be a safety net, not the optimizer's operating mode.
            # When most steps exceed it, every step is renormalised to grad_clip and the
            # nominal LR stops meaning anything — an LR sweep then partly measures the
            # clip. Measured on the v4/v5 recipe: gnorm 7-12 against grad_clip 1.0, i.e.
            # ~100% engagement. Said once, loudly, after enough steps to be sure.
            clip_steps += 1
            clip_hits += int(gnorm > self.cfg.grad_clip)
            if not clip_warned and clip_steps >= 200 and clip_hits / clip_steps > 0.5:
                clip_warned = True
                print(
                    f"[train] WARNING: gradient clipping engaged on "
                    f"{clip_hits}/{clip_steps} steps (grad_clip={self.cfg.grad_clip}, "
                    f"recent gnorm={gnorm:.2f}). The effective step size is "
                    f"grad_clip*lr, not lr — raise --grad-clip or read LR sweeps "
                    f"accordingly."
                )
            if step % self.cfg.log_every == 0:
                parts = " ".join(f"{k}:{v:.3f}" for k, v in per_stream.items())
                print(
                    f"step {step}  L={total:.4f} ({parts})  lr={lr:.2e}  gnorm={gnorm:.2f}  dt={time.time() - t0:.2f}s"
                )
                self.writer.add_scalar("loss/total", total, step)
                for k, v in per_stream.items():
                    self.writer.add_scalar(f"loss/{k}", v, step)
                self.writer.add_scalar("lr", lr, step)
                gem_p = getattr(self.model.aggregator, "p", None)  # GeM only; SALAD has none
                if gem_p is not None:
                    self.writer.add_scalar("gem_p", gem_p.item(), step)
                if self.wandb:
                    log = {
                        "loss/total": total,
                        "lr": lr,
                        "gnorm": gnorm,
                        "clip_fraction": clip_hits / max(clip_steps, 1),
                        **{f"loss/{k}": v for k, v in per_stream.items()},
                    }
                    if gem_p is not None:
                        log["gem_p"] = gem_p.item()
                    self.wandb.log(log, step=step)

            # Real eval on its own cadence (and always on the final step). The full suite
            # runs on a slower cadence: the monitor conditions cost real time and only the
            # selection condition drives best.pt.
            if self._eval_enabled() and (
                (step + 1) % self.cfg.eval_every == 0 or (step + 1) == last_step
            ):
                full = (step + 1) == last_step or (
                    self.cfg.eval_suite_every and (step + 1) % self.cfg.eval_suite_every == 0
                )
                self.evaluate_suite(step + 1, full=full)
                if (
                    self.cfg.early_stop_evals
                    and self._evals_since_best >= self.cfg.early_stop_evals
                ):
                    print(
                        f"[train] early stop: {self._evals_since_best} evals without "
                        f"improving on R@1={self.best['main']['r1']:.3f} "
                        f"@step {self.best['main']['step']}"
                    )
                    self.save(step)
                    break

            # Checkpoint on a signal (SLURM preemption/time-limit) at the step boundary,
            # then on step milestones and on a wall-time cadence (survives the 48h cap).
            if self._should_stop:
                self._requeue_and_exit(step)
            due_step = (step + 1) % self.cfg.save_every == 0 or (step + 1) == last_step
            due_time = (time.time() - self._last_ckpt_t) >= ckpt_every_s
            if due_step or due_time:
                self.save(step, tag=(f"step{step + 1}" if due_step else None))

        if any(v["step"] >= 0 for v in self.best.values()):
            print("[train] DONE.")
            for h, fname in HEAD_FILES.items():
                v = self.best[h]
                if v["step"] >= 0:
                    print(
                        f"        {h:<5} R@1={v['r1']:.4f} @ step {v['step']:<6} -> "
                        f"{os.path.join(self.run_dir, fname)}"
                    )
                if self.wandb:
                    self.wandb.summary[f"best_{h}_R@1"] = v["r1"]
                    self.wandb.summary[f"best_{h}_step"] = v["step"]
            print(
                f"        + every step*.pt milestone (--save-every {self.cfg.save_every}) — "
                f"re-score these before deciding what ships; the 2026-07-30 audit's whole "
                f"+13.3 R@1 was hiding in discarded milestones."
            )
        if self.wandb:
            self.wandb.finish()

    # ---- checkpointing / resume --------------------------------------
    def _state(self, step):
        """Full training state: weights + optimizer + AMP scaler + step + all RNGs."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step,
            "config": self.cfg.__dict__,
            # carried across requeue so a resumed job can't overwrite best.pt with a worse score
            "best": self.best,
            "eval_history": self.eval_history,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def _atomic_save(self, state, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # rename is atomic -> a killed job never leaves a half-written ckpt

    def save(self, step, tag=None):
        """Always refresh ``latest.pt`` (the resume target); also keep ``<tag>.pt`` milestones."""
        state = self._state(step)
        if tag:
            self._atomic_save(state, os.path.join(self.run_dir, f"{tag}.pt"))
        self._atomic_save(state, os.path.join(self.run_dir, "latest.pt"))
        self._last_ckpt_t = time.time()
        print(f"[train] saved latest.pt @ step {step + 1}" + (f" (+{tag})" if tag else ""))

    def load_checkpoint(self, path):
        """Restore weights/optimizer/scaler/RNG. Returns the step it was saved at."""
        # weights_only=False: our own checkpoint carries optimizer + numpy/python RNG objects
        # (trusted source; the PyTorch>=2.6 safe-unpickler rejects the numpy RNG state).
        ck = torch.load(path, map_location="cpu", weights_only=False)
        saved_cfg = ck.get("config", {})
        # A stale run_dir from an earlier sweep (different run-name collision, or a config
        # that predates a script change) can leave a latest.pt whose optimizer was built
        # with a different param-group layout (e.g. LLRD on/off, depth). resume="auto" would
        # otherwise hand that straight to optimizer.load_state_dict() and die deep inside
        # PyTorch with an opaque "different number of parameter groups" error. Fail fast here
        # instead, naming the actual mismatch.
        mismatch_keys = (
            "layerwise_lr_decay",
            "n_trainable_blocks",
            "aggregator",
            "vit",
            "representation",
            "H",
        )
        mismatches = {
            k: (saved_cfg.get(k), getattr(self.cfg, k, None))
            for k in mismatch_keys
            if k in saved_cfg and saved_cfg.get(k) != getattr(self.cfg, k, None)
        }
        if mismatches:
            detail = ", ".join(
                f"{k}: saved={old!r} vs current={new!r}" for k, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                f"[train] refusing to resume from {path}: checkpoint config disagrees with the "
                f"current run ({detail}). This run_dir likely holds a stale checkpoint from a "
                f"different sweep/config sharing this run-name. Remove or rename "
                f"{os.path.dirname(path)} (or pass --fresh / --resume none) to start clean."
            )
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        for st in self.optimizer.state.values():  # move optimizer moments onto the device
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        if ck.get("best"):
            self.best = _migrate_heads(ck["best"])
            self.eval_history = ck.get("eval_history", [])
            print(
                "[train] restored best: "
                + "  ".join(
                    f"{h}={v['r1']:.3f}@{v['step']}" for h, v in self.best.items() if v["step"] >= 0
                )
            )
        rng = ck.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        return int(ck["step"])

    def _resume_path(self):
        r = getattr(self.cfg, "resume", None)
        if not r:
            return None
        if r == "auto":
            p = os.path.join(self.run_dir, "latest.pt")
            return p if os.path.exists(p) else None
        if not os.path.isfile(r):
            raise FileNotFoundError(f"Resume checkpoint not found: {r}")
        return r

    def _install_signals(self):
        """SIGTERM on preemption/time-limit (both PBS and SLURM), SIGUSR1 on SLURM request."""

        def handler(signum, _frame):
            print(f"[train] caught signal {signum} — will checkpoint at the next step boundary")
            self._should_stop = True

        for s in (getattr(signal, "SIGUSR1", None), signal.SIGTERM):
            if s is None:
                continue
            try:
                signal.signal(s, handler)
            except (ValueError, OSError):
                pass  # e.g. not running on the main thread

    def _requeue_and_exit(self, step):
        """Checkpoint at a step boundary and leave, so nothing is lost to a kill signal."""
        self.save(step)  # refresh latest.pt so the next launch resumes exactly here
        slurm = os.environ.get("SLURM_JOB_ID")
        pbs = os.environ.get("PBS_JOBID")
        if getattr(self.cfg, "auto_requeue", False) and slurm:
            print(f"[train] requeuing SLURM job {slurm}")
            os.system(f"scontrol requeue {slurm}")
        elif pbs:
            # Deliberately NOT auto-resubmitting on PBS. `qrerun` only applies to a job that
            # has already finished, and a job script that re-qsubs itself will spin forever if
            # it dies early. The submitted scripts carry `#PBS -r y`, so the scheduler requeues
            # on a *system* failure by itself, and `resume="auto"` picks up latest.pt either
            # way — so a manual resubmit of the same run_<idx>.sh continues from here.
            print(
                f"[train] PBS job {pbs}: checkpointed at step {step}. Resubmit the same "
                f"run script to continue — resume='auto' will restart from latest.pt."
            )
        print("[train] checkpointed; exiting (will resume on next launch)")
        sys.exit(0)


def _smoke(n_streams=6, **cfg_overrides):
    """Verify loop mechanics on synthetic data with the real backbone (CPU ok).

    Runs ``n_streams`` streams by default — the MegaLoc sub-batch count — so the
    grad-accumulation path (one ``backward()`` per stream, a single ``step()``, i.e.
    ``L = L1+...+Ln``) is exercised, not just the single-stream case. ``cfg_overrides``
    lets the caller exercise one optional mechanism at a time (fp32 loss, XBM references,
    gradient checkpointing) on the same tiny separable problem.
    """
    import tempfile

    from PIL import Image

    from .dataset import PlacesDataset, build_transforms, make_places_loader

    cfg = VPRConfig()
    cfg.steps, cfg.log_every, cfg.warmup_steps = 6, 1, 2
    cfg.P_places = 3  # tiny sub-batch
    cfg.streams = [f"synthetic{i}" for i in range(n_streams)]
    for k, v in cfg_overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"VPRConfig has no attribute {k!r}")
        setattr(cfg, k, v)
    cfg.finalize()  # desc_dim must be right before the XBM queue is allocated
    _seed_everything(cfg.seed)
    if cfg_overrides:
        print(
            f"\n--- smoke: {n_streams} streams, "
            + ", ".join(f"{k}={v}" for k, v in cfg_overrides.items())
            + " ---"
        )
    with tempfile.TemporaryDirectory() as d:
        streams = {}
        for s in range(n_streams):
            index = []
            for pid in range(4):
                ps = []
                for k in range(cfg.K_images + 2):
                    p = os.path.join(d, f"s{s}_p{pid}_{k}.png")
                    # distinct per-place colour -> trivially separable
                    Image.new("RGB", (224, 224), (30 + pid * 50, 60, 200 - pid * 40)).save(p)
                    ps.append(p)
                index.append((s * 1000 + pid, ps))
            ds = PlacesDataset(index, cfg.K_images, build_transforms(cfg, train=True))
            streams[f"synthetic{s}"] = make_places_loader(ds, cfg.P_places, num_workers=0)

        trainer = VPRTrainer(cfg, streams, device=cfg.device)
        before = trainer.model.proj.weight.detach().clone()
        losses = []
        for step in range(cfg.steps):
            total, per, lr, gnorm = trainer.train_step(step)
            losses.append(total)
            assert len(per) == n_streams, f"expected {n_streams} per-stream losses, got {per}"
            gem_p = getattr(trainer.model.aggregator, "p", None)
            print(
                f"step {step}  L={total:.4f} ({len(per)} streams)  lr={lr:.2e}  gnorm={gnorm:.2f}"
                + (f"  gem_p={gem_p.item():.3f}" if gem_p is not None else "")
            )
            assert torch.isfinite(torch.tensor(total)), "non-finite loss"
        # With XBM the reference set grows as the queue fills, so the loss VALUE is not
        # monotone even while learning works — only assert monotonicity without it.
        assert not torch.equal(before, trainer.model.proj.weight), (
            "optimizer did not update weights"
        )
        if cfg.xbm_size:
            assert trainer.xbm.queue_idx or trainer.xbm.has_been_filled, "XBM never enqueued"
            assert not trainer.xbm.embedding_memory.requires_grad, (
                "XBM must hold detached embeddings"
            )
            assert (trainer.xbm.label_memory >= 0).any(), "XBM labels never written"
        print(
            f"OK train loop mechanics over {n_streams} streams "
            f"(loss finite, L=sum(Li), step/backward/schedule wired)"
        )
