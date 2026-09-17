#!/usr/bin/env bash
# Score the `i2e_gopro` arm with every method that already has paired `real` and `i2e` cells
# in sim2real/, so each new number lands beside a baseline with no extra runs.
#
# Sequential on purpose: the volume is a spinning disk and these passes are data-loading
# bound (a 1080p .npz is ~1.4 MB and renders to a 1080x1920x3 frame before the resize), so
# two runs at once cost more than they save. Every stage is resumable -- extract() skips a
# traverse whose bank file already exists.
set -uo pipefail

ROOT=/media/adam/vprdatasets/megaevent
NPZ=$ROOT/brisbane_npz/brisbane_event
OUT=$ROOT/sim2real
CKPTS=$ROOT/v8_bench/ckpts
LOG=/home/adam/repo/megaevent/logs/gopro
SRC=${SRC:-i2e_gopro}
DB=(daytime morning sunrise)          # sim2real's database: no night (APS ran at 1.73 Hz), no sunset2
Q=sunset1

mkdir -p "$LOG"
note() { echo "$*" | tee -a "$LOG/RUN_STATUS.txt"; }

# Single instance. Two of these racing is not a slow run, it is a corrupt one: extract()
# writes each bank through np.lib.format.open_memmap and treats *existence* as "cached and
# complete", so a second run reads a half-written bank as finished. It also puts two models
# on an 8 GB card at once, which is how the first attempt died -- a restart landed while the
# previous run still held 6.2 GB, and CUDA OOM'd during model load.
exec 9>"$LOG/.suite.lock"
if ! flock -n 9; then
  note "another gopro_suite.sh holds $LOG/.suite.lock — refusing to start a second"
  exit 1
fi

# --- guard: never score a half-converted tree -------------------------------------------
declare -A EXPECT=([daytime]=14318 [morning]=13453 [sunrise]=13724 [sunset1]=14478)
for s in "${!EXPECT[@]}"; do
  have=$(ls "$NPZ/$SRC/$s"/*.npz 2>/dev/null | wc -l)
  if [ "$have" -ne "${EXPECT[$s]}" ]; then
    note "ABORT: $SRC/$s has $have .npz, expected ${EXPECT[$s]} — conversion incomplete"
    exit 1
  fi
done
note "=== $SRC: all four traverses complete, scoring $(date +%H:%M:%S) ==="

run() {                                # run <name> <command...>
  local name=$1; shift
  note "--- $name $(date +%H:%M:%S)"
  "$@" > "$LOG/score_${name}.log" 2>&1
  note "--- $name exit $? $(date +%H:%M:%S)"
}

P=(env CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u)

run megaevent "${P[@]}" scripts/brisbane_pooled.py \
    --ckpt v8b="$CKPTS/b_v8_accum_s750.pt" v8s="$CKPTS/s_v8_accum_s500.pt" \
    --source "$SRC" --query "$Q" --database "${DB[@]}" \
    --resolutions 322 --pca none --out-dir "$OUT" --out-json "megaevent_v8_${SRC}.json"

# --workers 3 --batch-size 16, well below the 8/32 defaults. Event-GeM's loader is the
# heavy one at this resolution: load_mcts goes through src.npzdata._stream, which builds an
# int64 [N,4] array -- 40 MB per frame at the 1.27M events/frame this arm averages, against
# megaevent's load_accumulate which is pure numpy and allocates a fraction of that. At the
# defaults, 8 workers x prefetch 2 put ~16 of those in flight and systemd-oomd killed the
# entire terminal scope (106 processes) 4 minutes in, on 2026-09-09 22:17.
run eventgem "${P[@]}" scripts/eventgem_pooled.py \
    --dataset brisbane_event --source "$SRC" --query "$Q" --database "${DB[@]}" \
    --workers 3 --batch-size 16 \
    --out-dir "$OUT" --out-json "eventgem_${SRC}.json"

run eventvlad "${P[@]}" scripts/eventvlad_pooled.py \
    --dataset brisbane_event --source "$SRC" --query "$Q" --database "${DB[@]}" \
    --out-dir "$OUT"

run spikevpr "${P[@]}" scripts/spikevpr_pooled.py \
    --dataset brisbane_event --source "$SRC" --model nsavp --query "$Q" --database "${DB[@]}" \
    --out-dir "$OUT"

note "=== $SRC scoring done $(date +%H:%M:%S) ==="
