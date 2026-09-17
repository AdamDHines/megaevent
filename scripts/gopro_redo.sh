#!/usr/bin/env bash
# The gopro sim2real redo, end to end: frames -> I2E -> banks -> Event-GeM chain -> score.
#
#   setsid nohup bash scripts/gopro_redo.sh > /dev/null 2>&1 &
#
# Prerequisite (NOT automated here): scripts/gopro_align.py has been run for sunset1 and
# morning and scripts/gopro_timewarp.py has fitted a PASSing warp from it
# (logs/gopro2/warp_<seq>.json). Every guard below checks trees against the warp's
# fingerprint, so a stale or re-fitted warp stops the run rather than silently mixing
# time bases.
#
# Every stage has a completion guard, so a restart after an oomd scope kill re-runs only
# what is missing. Status lines land in logs/gopro2/RUN_STATUS.txt; per-stage logs sit
# next to it. GPU stages are strictly sequential (8 GB card), and the whole thing holds
# a flock so a second launch is a no-op.
set -uo pipefail

REPO=/home/adam/repo/megaevent
EG=/media/adam/vprdatasets/megaevent/eventgem_010
ROOT=/media/adam/vprdatasets/megaevent
B=$ROOT/brisbane_npz/brisbane_event
V8=$ROOT/v8_bench
S2R=$ROOT/sim2real
CK=$V8/ckpts
LOG=$REPO/logs/gopro2
KPROOT=$EG/eventgem/keypoints/brisbane_event
EGBANKS=$EG/eventgem/features/brisbane_npz
SEQS=(sunset1 morning)
declare -A NSLICES=([sunset1]=14478 [morning]=13453)

mkdir -p "$LOG"
exec 9>"$LOG/.suite.lock"
flock -n 9 || { echo "gopro_redo.sh already running"; exit 1; }

note() { echo "$(date '+%F %H:%M:%S') $*" >> "$LOG/RUN_STATUS.txt"; echo "== $*"; }
die()  { note "ABORT: $*"; exit 1; }

mpixi() ( cd "$REPO" && CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u "$@" )
egpixi() ( cd "$EG" && CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u "$@" )

# ---------------------------------------------------------------- 0. warp precondition
for seq in "${SEQS[@]}"; do
    mpixi -c "
import json, sys
w = json.load(open('$LOG/warp_$seq.json'))
sys.exit(0 if w.get('pass') else 1)" >/dev/null 2>&1 \
        || die "no PASSing time warp at logs/gopro2/warp_$seq.json — run scripts/gopro_align.py then scripts/gopro_timewarp.py --seq $seq first"
done
note "warp precondition OK (both traverses)"

# ---------------------------------------------------------------- 1. frames
guard_extract() {
    mpixi -c "
import glob, json, sys
seq = '$1'; d = '$B/gopro/' + seq
w = json.load(open('$LOG/warp_' + seq + '.json'))
try:
    r = json.load(open(d + '/select.json'))
except FileNotFoundError:
    sys.exit(1)
tw = r.get('timewarp') or {}
ok = (r['n_slices'] == ${NSLICES[$1]}
      and tw.get('fingerprint') == w['fingerprint']
      and len(glob.glob(d + '/frame_*.png')) == r['n_slices'])
sys.exit(0 if ok else 1)" >/dev/null 2>&1
}
for seq in "${SEQS[@]}"; do
    if guard_extract "$seq"; then
        note "extract $seq: guard OK, skipping"
    else
        note "extract $seq: start"
        mpixi scripts/extract_gopro.py --seq "$seq" --timewarp "$LOG/warp_$seq.json" \
            >> "$LOG/extract_$seq.log" 2>&1 \
            || die "extract $seq failed (logs/gopro2/extract_$seq.log)"
        guard_extract "$seq" || die "extract $seq finished but guard still fails"
        note "extract $seq: done"
    fi
done

# ---------------------------------------------------------------- 2. I2E conversion
guard_convert() {
    local seq=$1
    [ "$(find "$B/i2e_gopro/$seq" -name 'frame_*.npz' 2>/dev/null | wc -l)" -eq "${NSLICES[$seq]}" ] \
        && [ -z "$(find "$B/i2e_gopro/$seq" -name '*.npz' -size 0 2>/dev/null)" ] \
        && mpixi -c "
import numpy as np, sys
with np.load('$B/i2e_gopro/$seq/frame_007000.npz') as z:
    sys.exit(0 if z['resolution'].tolist() == [260, 494] else 1)" >/dev/null 2>&1
}
if guard_convert sunset1 && guard_convert morning; then
    note "i2e convert: guard OK, skipping"
else
    # Zero-byte npz from a killed run would be skipped by i2e's resume forever.
    find "$B/i2e_gopro" -name '*.npz' -size 0 -delete 2>/dev/null
    note "i2e convert: start"
    bash "$REPO/scripts/gopro_i2e.sh" >> "$LOG/i2e_convert_suite.log" 2>&1 \
        || die "i2e conversion failed (logs/gopro2/i2e_convert.log)"
    for seq in "${SEQS[@]}"; do
        guard_convert "$seq" || die "i2e convert finished but $seq guard fails"
    done
    note "i2e convert: done"
fi

# ---------------------------------------------------------------- 3. megaevent banks
if [ -f "$S2R/r322_i2e_gopro_sunset1_v8b.npy" ] && [ -f "$S2R/r322_i2e_gopro_morning_v8b.npy" ] \
   && [ -f "$S2R/r322_i2e_gopro_sunset1_v8s.npy" ] && [ -f "$S2R/r322_i2e_gopro_morning_v8s.npy" ]; then
    note "megaevent banks: guard OK, skipping"
else
    note "megaevent banks: start"
    # batch 8: both checkpoints are resident at once and the default batch OOMs the
    # 8 GB card (hit 2026-09-11; the pass is npz-render bound anyway).
    mpixi scripts/brisbane_pooled.py \
        --ckpt v8b="$CK/b_v8_accum_s750.pt" v8s="$CK/s_v8_accum_s500.pt" \
        --source i2e_gopro --query morning --database sunset1 \
        --resolutions 322 --pca none --batch-size 8 --out-dir "$S2R" \
        --out-json megaevent_v8_i2e_gopro_s1morning.json \
        >> "$LOG/banks_megaevent.log" 2>&1 || die "megaevent banks failed"
    note "megaevent banks: done"
fi

# ---------------------------------------------------------------- 4. EventVLAD + SpikeVPR banks
if [ -f "$S2R/eventvlad_i2e_gopro_sunset1_eventvlad.npy" ] && [ -f "$S2R/eventvlad_i2e_gopro_morning_eventvlad.npy" ]; then
    note "eventvlad banks: guard OK, skipping"
else
    note "eventvlad banks: start"
    mpixi scripts/eventvlad_pooled.py --dataset brisbane_event --source i2e_gopro \
        --query morning --database sunset1 --out-dir "$S2R" \
        >> "$LOG/banks_eventvlad.log" 2>&1 || die "eventvlad banks failed"
    note "eventvlad banks: done"
fi
if [ -f "$S2R/spikevpr_nsavp_i2e_gopro_sunset1.npy" ] && [ -f "$S2R/spikevpr_nsavp_i2e_gopro_morning.npy" ]; then
    note "spikevpr banks: guard OK, skipping"
else
    note "spikevpr banks: start"
    mpixi scripts/spikevpr_pooled.py --dataset brisbane_event --model nsavp \
        --source i2e_gopro --query morning --database sunset1 --out-dir "$S2R" \
        >> "$LOG/banks_spikevpr.log" 2>&1 || die "spikevpr banks failed"
    note "spikevpr banks: done"
fi

# ---------------------------------------------------------------- 5. eg010 geometry pack
# Always re-run: cheap, and its internal asserts re-validate mask + time base each time.
note "eg010 geom dump: start"
mpixi scripts/eg010_dump_geom_gopro.py >> "$LOG/eg010_geom.log" 2>&1 \
    || die "eg010 geometry dump failed"
note "eg010 geom dump: done"

# ---------------------------------------------------------------- 6. eg010 globals + kps
if [ -f "$EGBANKS/eg010_gem224_i2e_gopro_sunset1.npy" ] && [ -f "$EGBANKS/eg010_gem224_i2e_gopro_morning.npy" ]; then
    note "eg010 globals: guard OK, skipping"
else
    note "eg010 globals: start"
    egpixi brisbane_npz_extract.py --sources i2e_gopro --seqs sunset1 morning \
        --batch-size 16 >> "$LOG/eg010_globals.log" 2>&1 || die "eg010 globals failed"
    note "eg010 globals: done"
fi
guard_kps() {
    local n
    n=$(find "$KPROOT/kps_i2e_gopro_$1" -name 'mcts_*.feat.npz' 2>/dev/null | wc -l)
    [ "$n" -ge "$((${NSLICES[$1]} - 2))" ]
}
if guard_kps sunset1 && guard_kps morning; then
    note "eg010 kps: guard OK, skipping"
else
    note "eg010 kps: start"
    egpixi brisbane_npz_kps.py --source i2e_gopro --seqs sunset1 morning \
        --batch-size 16 --num-workers 3 >> "$LOG/eg010_kps.log" 2>&1 \
        || die "eg010 kps failed"
    note "eg010 kps: done"
fi

# ---------------------------------------------------------------- 7. score (globals, gate)
note "score pass 1: start"
mpixi scripts/gopro_pairwise.py >> "$LOG/score_pass1.log" 2>&1 \
    || die "scorer failed its ledger gate (logs/gopro2/score_pass1.log)"
note "score pass 1: done (gate PASSED)"

# ---------------------------------------------------------------- 8. rerank both arms
if [ -f "$V8/eg010_rerank_gopro_pairwise.json" ]; then
    note "rerank: guard OK, skipping"
else
    note "rerank: start"
    egpixi gopro_rerank.py >> "$LOG/rerank.log" 2>&1 || die "rerank failed"
    note "rerank: done"
fi

# ---------------------------------------------------------------- 9. final fold
note "score final: start"
mpixi scripts/gopro_pairwise.py >> "$LOG/score_final.log" 2>&1 \
    || die "final scorer run failed"
note "score final: done -> $V8/gopro_pairwise_brisbane_event.json"
command -v notify-send >/dev/null && notify-send "gopro_redo" "finished: gopro_pairwise_brisbane_event.json" || true
