#!/bin/bash
# Build the solver, run it on every dataset, verify with scorer.py, keep the
# better of the new and existing outputs/<name>.out, and log to scores.log.
# Usage: ./run_all.sh "<note>" [seconds per dataset, default 120]   (FRESH=1 to ignore saved outputs)
set -e
cd "$(dirname "$0")"
note=${1:-""}
secs=${2:-120}
# Crate downloads stay in this folder; CMake (for HiGHS) comes from .venv.
export CARGO_HOME="$PWD/.cargo-home" PATH="$PWD/.venv/bin:$PATH"
(cd solver && cargo build --release -q)
mkdir -p outputs
tmp=$(mktemp -d)
total=0
line="$(date '+%Y-%m-%d %H:%M') secs=$secs"
for f in input_files/*.in; do
  name=$(basename "$f" .in)
  [ "$name" = example ] && continue
  # Continue from the best saved output, if any (set FRESH=1 to start from scratch).
  if [ -f "outputs/$name.out" ] && [ -z "$FRESH" ]; then
    SEED="outputs/$name.out" solver/target/release/solver "$f" "$tmp/$name.out" "$secs"
  else
    solver/target/release/solver "$f" "$tmp/$name.out" "$secs"
  fi
  new=$(python3 scorer.py "$f" "$tmp/$name.out" | tail -1); new=${new##* }
  old=0
  if [ -f "outputs/$name.out" ]; then
    o=$(python3 scorer.py "$f" "outputs/$name.out" | tail -1); [[ $o == score:* ]] && old=${o##* }
  fi
  if [[ $new =~ ^[0-9]+$ ]] && (( new > old )); then
    cp "$tmp/$name.out" "outputs/$name.out"; kept=$new; tag="new best"
  else
    kept=$old; tag="kept previous $old"
  fi
  echo "  verified: run=$new ($tag)"
  total=$((total + kept))
  line="$line $name=$new"
done
rm -rf "$tmp"
echo "TOTAL of best outputs: $total"
echo "$line BEST_TOTAL=$total | $note" >> scores.log
