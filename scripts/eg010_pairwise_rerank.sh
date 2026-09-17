#!/bin/bash
# Event-GeM 0.1.0's released keypoint rerank on the single-reference pairings that
# scripts/pairwise_sunset_ref.py scores the global stage under.
#
#     setsid nohup bash scripts/eg010_pairwise_rerank.sh > /dev/null 2>&1 &
#
# Runs in the 0.1.0 worktree's own pixi env — the rerank must execute on the release's
# dependency stack (cv2 4.13 USAC_FAST), not a lookalike. Each cell writes one entry into
# v8_bench/eg010_rerank_pairwise_<dataset>.json, gated on reproducing megaevent's cached
# global recalls for the same membership before any rerank number is kept.
#
# setsid + a status file rather than a foreground run: this box's systemd-oomd has killed
# long jobs out from under their watcher before. Progress is $ST; per-cell logs sit beside it.
set -u
cd /media/adam/vprdatasets/megaevent/eventgem_010 || exit 1
S=/media/adam/vprdatasets/megaevent/v8_bench
ST=$S/eg010_pairwise_rerank.status
: > "$ST"

run() {  # dataset reference query
    echo "START $1 $2->$3 $(date -Is)" >> "$ST"
    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u pooled_rerank.py \
        --dataset "$1" --database "$2" --query "$3" > "$S/eg010_pairwise_$1_$3.log" 2>&1
    echo "EXIT $? $1 $2->$3 $(date -Is)" >> "$ST"
}

run brisbane_event sunset1 daytime
run brisbane_event sunset1 morning
run brisbane_event sunset1 sunrise
run nsavp R0_FS0 R0_FA0
echo "ALL_DONE $(date -Is)" >> "$ST"
