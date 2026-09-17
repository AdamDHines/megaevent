#!/bin/bash
# Event-GeM 0.1.0's released keypoint rerank on NSAVP's reverse cell (ref R0_RS0 -> R0_RA0).
# Run only after scripts/eg010_reverse_kps.sh reports ALL_DONE — it needs kps_R0_RS0 (gallery)
# and kps_R0_RA0 (query), neither of which existed before that harvest.
#
#     setsid nohup bash scripts/eg010_reverse_rerank.sh > /dev/null 2>&1 &
#
# The driver gates its own base recalls against megaevent's cached global cell for the same
# membership (pairwise_sunset_ref_nsavp.json -> "R0_RS0->R0_RA0"), so two independent
# implementations must agree before any reranked number is written.
set -u
cd /media/adam/vprdatasets/megaevent/eventgem_010 || exit 1
S=/media/adam/vprdatasets/megaevent/v8_bench
ST=$S/eg010_reverse_rerank.status
: > "$ST"

echo "START nsavp R0_RS0->R0_RA0 $(date -Is)" >> "$ST"
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u pooled_rerank.py \
    --dataset nsavp --database R0_RS0 --query R0_RA0 \
    > "$S/eg010_pairwise_nsavp_R0_RA0.log" 2>&1
echo "EXIT $? nsavp R0_RS0->R0_RA0 $(date -Is)" >> "$ST"
echo "ALL_DONE $(date -Is)" >> "$ST"
