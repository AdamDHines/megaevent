#!/usr/bin/env bash
# Event-GeM 0.1.0 (released global + its own rerank) on the full Springfield protocol.
#
# Runs AFTER springfield_suite.sh, not inside it: bash reads a script incrementally as it
# executes, so appending a stage to a suite that is already running would make it execute
# whatever happens to land at the file offset it next reads. Waiting on the process is the
# safe way to chain.
#
# The global row goes through the ordinary suite machinery (--method eg010), so it inherits
# the same extraction, caching, protocol and scoring as every other row. The rerank is the
# release's own process_single_query on top of those banks.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp
OUT=output/springfield_full
STATUS=$OUT/RUN_STATUS.txt
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }

while pgrep -f "springfield_suite.sh" >/dev/null 2>&1; do sleep 120; done
note "=== suite finished; eg010 0.1.0 starting ==="

COMMON=(--allow-unrepacked --no-event-filter --db-stride-m 0 --batch-size 16 --workers 4)
for rev in 0 135; do
  note "eg010 global: extract + score (reversals gate ${rev})"
  pixi run python3 scripts/springfield_baselines.py --method eg010 "${COMMON[@]}" \
      --exclude-reversals-deg "$rev" >> "$OUT/eg010.log" 2>&1 \
    || { note "!!! eg010 global FAILED — see $OUT/eg010.log"; exit 1; }
done

note "eg010 rerank (released process_single_query, top_k 50)"
pixi run python3 scripts/springfield_eg010.py --stage rerank >> "$OUT/eg010.log" 2>&1 \
  || note "!!! eg010 rerank FAILED — see $OUT/eg010.log; the global row stands alone and "\
"must NOT be quoted as Event-GeM, which is a two-stage method"

note "=== table (with eg010) ==="
pixi run python3 scripts/springfield_table.py | tee -a "$OUT/table.txt"
note "=== eg010 chain complete ==="
notify-send "megaevent" "Springfield: Event-GeM 0.1.0 complete" 2>/dev/null || true
