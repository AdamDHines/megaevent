#!/usr/bin/env bash
# The two GeM rows of tab:ftagg on Springfield: v9 s_gem_full and b_gem_full (step 2000).
#
#   setsid nohup bash scripts/springfield_gem_arms.sh >/dev/null 2>&1 &
#
# Checkpoints come from arch_ablation_v9/ckpts, where link_ckpts.sh staged each arm under its
# LABEL. That is load-bearing: springfield_full.py builds the bank tag from the checkpoint
# basename, so pointing at the run dirs' step2000.pt would tag both arms "step2000". The
# staged links were provenance-checked on 2026-08-25 (depth, aggregator, LR, no llrd).
#
# Same protocol as every megaevent row already in the Springfield table: 322^2, batch 12,
# dt 50 ms, hot-pixel on, NO BA filter, the full 132,569-row gallery and all 84 query
# sessions. Per arm: extract + score (~3 h — render-bound at ~14 slices/s, so ViT-S and
# ViT-B cost about the same), then the diag `dump` (top-20 per query slice, minutes off the
# cached banks), which is what arch_ablation_table.py computes the paper cell from
# (day+dawn, |psi| < 135 deg, 5,557 slices). Resumable: an arm whose results json exists is
# skipped, a dump that exists is skipped, and banks cache per partition.
#
# NEVER edit this file while it runs. bash reads a script incrementally and executes whatever
# lands at the next read offset — springfield_vits.sh's "line 18: ckpt: command not found"
# was exactly that, and it painted a finished run as FAILED.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=output/springfield_full
DIAG=output/springfield_diag
CK=/media/adam/vprdatasets/megaevent/arch_ablation_v9/ckpts
mkdir -p "$OUT"
note() { echo "$(date +%H:%M) $*" | tee -a "$OUT/RUN_STATUS.txt"; }

# Single instance, enforced. Two copies racing on the same bank paths tear the files
# (cached_array is a bare np.save) and halve the GPU each can use — the 2026-08-25 incident.
exec 9>"$OUT/.gem_arms.lock"
flock -n 9 || { echo "another springfield_gem_arms.sh holds $OUT/.gem_arms.lock — exiting"; exit 1; }
# Our own pid: fishing it out of `ps` afterwards once caught a transient wrapper.
echo $$ > "$OUT/gem_arms.pid"

note "############ ftagg GeM arms (s_gem_full, b_gem_full) starting ############"
for a in s_gem_full b_gem_full; do
  ck="$CK/$a.pt"
  [ -e "$ck" ] || { note "!!! $a: missing $ck — run arch_ablation_v9/link_ckpts.sh"; continue; }
  # The tag carries the step (0-indexed: step2000.pt -> _s1999_) and the sha, so glob it.
  if compgen -G "$OUT/results_${a}_s*_accumulate_r322_hp1_baoff.json" >/dev/null; then
    note "=== $a: results present, extraction skipped ==="
  else
    note "=== $a: extract + score ==="
    pixi run python3 scripts/springfield_full.py --allow-unrepacked --no-event-filter \
        --ckpt "$ck" --topk-per-session 0 >> "$OUT/gem_arms.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      note "!!! $a FAILED (exit $rc) — see $OUT/gem_arms.log; continuing with the next arm"
      continue
    fi
  fi
  res=$(compgen -G "$OUT/results_${a}_s*_accumulate_r322_hp1_baoff.json" | head -1)
  tag=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "$res")
  if [ -e "$DIAG/dump_${tag}.npz" ]; then
    note "=== $a: dump present, skipped ==="
  else
    note "=== $a: diag dump (paper-cell input) ==="
    pixi run python3 scripts/springfield_diag.py --stage dump --results "$res" \
        >> "$OUT/gem_arms.log" 2>&1 \
      || note "!!! $a: dump FAILED — no paper cell for this arm; see $OUT/gem_arms.log"
  fi
  note "$a done"
done

note "=== tables ==="
{ pixi run python3 scripts/arch_ablation_table.py;
  pixi run python3 scripts/arch_ablation_table.py --latex --ftagg; } > "$OUT/ftagg_table.txt" 2>&1 \
  || note "!!! arch_ablation_table.py failed — see $OUT/ftagg_table.txt"
pixi run python3 scripts/springfield_table.py 2>&1 | tee -a "$OUT/table.txt" >> "$OUT/gem_arms.log"
note "############ ftagg GeM arms complete — $OUT/ftagg_table.txt ############"
notify-send "megaevent" "Springfield ftagg GeM arms complete" 2>/dev/null || true
