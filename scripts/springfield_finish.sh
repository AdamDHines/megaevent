#!/usr/bin/env bash
# Everything still owed after springfield_suite.sh: the EventVLAD re-run (its first pass
# produced a 100% NaN bank under fp16 autocast — now forced to fp32) and Event-GeM 0.1.0.
#
# A separate script rather than an edit to the running one: bash reads a script incrementally
# as it executes, so editing a live suite makes it run whatever lands at the next read offset.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp
OUT=output/springfield_full
note() { echo "$(date +%H:%M) $*" | tee -a "$OUT/RUN_STATUS.txt"; }
COMMON=(--allow-unrepacked --no-event-filter --db-stride-m 0 --batch-size 16 --workers 4)

while pgrep -f "springfield_suite.sh" >/dev/null 2>&1; do sleep 120; done
note "=== suite finished; finish-up starting ==="

for m in eventvlad eg010; do
  for rev in 0 135; do
    note "$m: extract + score (reversals gate ${rev})"
    pixi run python3 scripts/springfield_baselines.py --method "$m" "${COMMON[@]}" \
        --exclude-reversals-deg "$rev" >> "$OUT/$m.log" 2>&1 \
      || { note "!!! $m FAILED at gate ${rev} — see $OUT/$m.log"; break; }
  done
done

note "eg010 rerank (released process_single_query, top_k 50)"
pixi run python3 scripts/springfield_eg010.py --stage rerank >> "$OUT/eg010.log" 2>&1 \
  || note "!!! eg010 rerank FAILED — the global row must NOT be quoted as Event-GeM, which is a two-stage method"

note "=== final table ==="
pixi run python3 scripts/springfield_table.py | tee -a "$OUT/table.txt"
note "=== finish-up complete ==="
notify-send "megaevent" "Springfield baselines: all rows complete" 2>/dev/null || true
