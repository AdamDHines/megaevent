#!/bin/bash
# MCTS generation + SuperEvent keypoint harvest for NSAVP's two reverse traverses, which the
# single-reference reverse cell (ref R0_RS0 -> query R0_RA0) needs before Event-GeM can rerank.
#
#     setsid nohup bash scripts/eg010_reverse_kps.sh > /dev/null 2>&1 &
#
# kp_only.py's docstring says these two (8-9 GB) do not fit MCTS generation in this box's 31 GB
# (0.1.1 erratum #6) and that R0_RN0 at 4.3 GB was the only reverse recording that did. Reading
# the code disagrees — utils/generate_mcts.py:249 gpu_mcts is per-frame with ~12 MB allocations
# and streamutils/stream.py:444 reads the h5 in chunk_size blocks — so this measures rather than
# assumes. Smaller file first; peak RSS is sampled every 10 s into the status file so an OOM is
# distinguishable from a crash after the fact.
set -u
cd /media/adam/vprdatasets/megaevent/eventgem_010 || exit 1
S=/media/adam/vprdatasets/megaevent/v8_bench
ST=$S/eg010_reverse_kps.status
: > "$ST"

run() {  # traverse
    local t=$1 log="$S/eg010_kps_$t.log" peak=0
    echo "START $t $(date -Is)  free_gb=$(free -g | awk '/^Mem:/{print $7}')" >> "$ST"
    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 -u kp_only.py \
        --dataset nsavp --traverse "$t" \
        --data-root /media/adam/vprdatasets/eventlab \
        --keypoint-batch-size 4 > "$log" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        # Whole process tree: pixi execs a child python, so the parent's own RSS is not it.
        local rss
        rss=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END{print s+0}')
        [ "$rss" -gt "$peak" ] && peak=$rss
        sleep 10
    done
    wait $pid; local rc=$?
    echo "EXIT $rc $t $(date -Is)  peak_rss_gb=$(awk -v k=$peak 'BEGIN{printf "%.1f", k/1048576}')" >> "$ST"
    return $rc
}

for t in R0_RS0 R0_RA0; do
    if ! run "$t"; then
        echo "ABORT after $t — not attempting the rest" >> "$ST"
        exit 1
    fi
    m=$(ls "/media/adam/vprdatasets/eventlab/nsavp/$t/mcts_${t}_50" 2>/dev/null | wc -l)
    k=$(ls "eventgem/keypoints/nsavp/kps_$t" 2>/dev/null | wc -l)
    echo "COUNTS $t mcts=$m kps=$k" >> "$ST"
    # extract_keypoints treats directory existence as completion (erratum #5), so a crashed run
    # would otherwise leave a short store that silently poisons every later rerank.
    [ "$m" = "$k" ] && [ "$m" != "0" ] || { echo "COUNT_MISMATCH $t" >> "$ST"; exit 1; }
done
echo "ALL_DONE $(date -Is)" >> "$ST"
