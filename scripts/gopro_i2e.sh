#!/usr/bin/env bash
# I2E over the co-recorded Brisbane windscreen-camera frames -> the `i2e_gopro` arm.
#
#   scripts/extract_gopro.py --seq <seq>      writes gopro/<seq>/frame_%06d.png
#   THIS SCRIPT                               writes i2e_gopro/<seq>/frame_%06d.npz
#   scripts/brisbane_pooled.py --source i2e_gopro   scores it
#
# The frames go in at 1920x1080, less the burnt-in date/GPS overlay the camera writes
# across rows 1018-1045 (--crop-bottom 70 leaves 1920x1010 with an 8-row margin for glyph
# antialiasing). That overlay sits at the same pixels in every frame of every traverse, so
# I2E would fire on its edges every time -- a constant nuisance pattern, identical in the
# database and the query, that carries no discriminative signal and only dilutes the real
# events.
#
# After the crop the frame is resized to --short-side 260 (494x260). The saccade radius is
# in PIXELS, so the pixel scale is part of the simulation: the trusted `i2e` arm ran the
# DAVIS APS frames at their native 346x260, and matching its 260-px vertical scale is what
# makes the two synthetic arms' event statistics comparable. It also keeps Event-GeM's
# MCTS/keypoint stage near the regime its top_k rule was tuned in. (Native 1920x1010 was
# tried on 2026-09-09/10 and repudiated along with its time base; at ~1.27 M events/frame
# it was also 8x the pixels for nothing the 322/240-px model transforms could see.)
# All other parameters are I2E's defaults, stated explicitly rather than relied on, because
# the ORIGINAL Brisbane APS conversion was run with no script, no log and no --report, and
# had to be reconstructed from the timestamps inside the .npz files:
#
#   --saccade-ms 30 --steps 24 --radius 2.0 --C 0.15 --resolution 0
#
# An absolute --npz-dir overrides --out, so the events land in i2e_gopro/ while the
# rendered QC frames go to i2e_gopro_render/. That render tree has no consumer -- it exists
# to be looked at once and deleted.
#
# Run under I2E's own env; it is CPU-only numpy/scipy/PIL and needs nothing from megaevent.
set -euo pipefail

I2E_ROOT=${I2E_ROOT:-/home/adam/repo/I2E}
NPZ_ROOT=${NPZ_ROOT:-/media/adam/vprdatasets/megaevent/brisbane_npz/brisbane_event}
WORKERS=${WORKERS:-7}          # 8 cores; one left so the box stays responsive
LOG_DIR=${LOG_DIR:-/home/adam/repo/megaevent/logs/gopro2}

mkdir -p "$LOG_DIR" "$NPZ_ROOT/i2e_gopro"

# One converter at a time: the 2026-09-10 run launched a second converter over a live one
# (two overlapping "start" lines in RUN_STATUS, workers fighting over the same tree).
exec 9>"$LOG_DIR/.i2e.lock"
flock -n 9 || { echo "gopro_i2e.sh already running (lock $LOG_DIR/.i2e.lock)"; exit 1; }

cd "$I2E_ROOT"

pixi run python3 -u i2e_infer.py \
    --dataset-root "$NPZ_ROOT/gopro" \
    --npz-dir      "$NPZ_ROOT/i2e_gopro" \
    --out          "$NPZ_ROOT/i2e_gopro_render" \
    --layout mirror \
    --representation accumulate \
    --saccade-ms 30 --steps 24 --radius 2.0 --C 0.15 --resolution 0 \
    --short-side 260 \
    --crop-bottom 70 \
    --img-format png \
    --workers "$WORKERS" \
    --report "$NPZ_ROOT/i2e_gopro/convert.json" \
    2>&1 | tee -a "$LOG_DIR/i2e_convert.log"
