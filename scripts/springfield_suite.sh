#!/usr/bin/env bash
# Springfield baseline suite — every method, one protocol, run overnight under tmux.
#
#   tmux new -s sf 'bash scripts/springfield_suite.sh 2>&1 | tee -a output/springfield_full/suite.log'
#
# Full rate: the whole 132,569-row gallery and all 15,886 query slices, 25 m, three lighting
# conditions kept. Reversals are a SCORING mask, so each method is scored twice off the same
# banks — once keeping them, once excluding |psi| >= 135 deg.
#
# Sequential on purpose. The recordings live on a spinning disk and concurrent readers thrash
# it: a measured 0.3 slices/s contended against 18.3 uncontended. Every stage is resumable —
# banks cache per partition, so re-running skips whatever already finished.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp

OUT=output/springfield_full
STATUS=$OUT/RUN_STATUS.txt
mkdir -p "$OUT"
COMMON=(--allow-unrepacked --no-event-filter --db-stride-m 0 --batch-size 16 --workers 4)

note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }

run_method() {   # $1 = method name, $2 = extra args (may be empty)
  local m=$1; shift
  note "=== $m: extract + score (reversals kept) ==="
  pixi run python3 scripts/springfield_baselines.py --method "$m" "${COMMON[@]}" \
      --exclude-reversals-deg 0 "$@" >> "$OUT/$m.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then note "!!! $m FAILED (exit $rc) — see $OUT/$m.log; continuing"; return 1; fi
  note "=== $m: score again, reversals excluded ==="
  pixi run python3 scripts/springfield_baselines.py --method "$m" "${COMMON[@]}" \
      --exclude-reversals-deg 135 "$@" >> "$OUT/$m.log" 2>&1
  note "$m done"
}

note "############ Springfield baseline suite starting ############"

# 1-2. the two bounded, certain rows first, so a partial night still answers the question
run_method megaloc
run_method eventvlad

# 3. SpikeVPR — whole-partition bridge into its own conda env. Health-check the bank before
#    believing it: its own docstring warns a collapsed near-constant descriptor looks fine
#    from the outside.
run_method spikevpr && {
  note "spikevpr health check"
  # It takes bank paths; called bare it exits 2 on argparse and the "problem" reported is
  # its own usage error, not a verdict on the bank. Feed it real banks.
  SVB=$(ls /media/adam/vprdatasets/megaevent/evaluations/springfield/*/spikevpr_*.npy 2>/dev/null \
        | grep -v '\.rows\.npy' | head -6)
  if [ -n "$SVB" ]; then
    pixi run python3 scripts/spikevpr_health.py $SVB >> "$OUT/spikevpr.log" 2>&1 \
      || note "!!! spikevpr health check reported a problem — treat its row as unverified"
  else
    note "!!! no spikevpr banks found to health-check"
  fi
}

# Event-GeM 0.1.0 is deliberately NOT in tonight's run. Its released env has no eventcv
# (verified: h5py/torch/numpy present, eventcv missing), so it cannot slice Springfield's h5
# recordings at all — every 0.1.0 driver in that worktree reads pre-sliced npz. It needs a
# materialisation shim plus its three parity gates before any number from it is trustworthy,
# and starting that unattended would only fail in the small hours. It is the next job, run
# with the gates watched, not overnight.

# 5. the table
note "=== table ==="
pixi run python3 scripts/springfield_table.py | tee -a "$OUT/table.txt"
note "############ suite complete ############"
notify-send "megaevent" "Springfield baseline suite complete" 2>/dev/null || true
