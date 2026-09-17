#!/usr/bin/env bash
# The v8 ViT-S row: same wave, representation and aggregator as the ViT-B already in the
# table, so backbone capacity is the only variable between the two megaevent rows.
# springfield_full.py has no reversal flag, so its query set is scored whole; the paper
# cell (day+dawn, |psi|<135) is recomputed from its per-slice dump exactly as the ViT-B's
# was, which is also what keeps the two megaevent rows built the same way.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp
OUT=output/springfield_full
note() { echo "$(date +%H:%M) $*" | tee -a "$OUT/RUN_STATUS.txt"; }

# Write our own pid: picking it out of `ps` afterwards caught a transient wrapper once
# and the watchdog then reported the run dead while it was still extracting.
echo $$ > "$OUT/vits.pid"
note "=== vit-s (s_v8_accum_s500) starting ==="
pixi run python3 scripts/springfield_full.py --allow-unrepacked --no-event-filter \
    --ckpt /media/adam/vprdatasets/megaevent/v8_bench/ckpts/s_v8_accum_s500.pt \
    --topk-per-session 0 >> "$OUT/vits.log" 2>&1 \
  || { note "!!! vit-s FAILED — see $OUT/vits.log"; exit 1; }
note "=== vit-s complete ==="
notify-send "megaevent" "Springfield ViT-S complete" 2>/dev/null || true
