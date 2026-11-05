#!/bin/bash
# Copy a candidate output to outputs/<map>.out if scorer.py says it is valid
# and better, and log it in scores.log.
# Usage: tools/save_if_better.sh <map> <candidate.out> "<note>"
cd "$(dirname "$0")/.."
m=$1; f=$2; note=$3
new=$(python3 scorer.py "input_files/$m.in" "$f" | tail -1)
old=$(python3 scorer.py "input_files/$m.in" "outputs/$m.out" | tail -1)
if [[ $new == score:* ]] && [ "${new##* }" -gt "${old##* }" ]; then
  cp "$f" "outputs/$m.out"
  echo "$(date '+%Y-%m-%d %H:%M') $m=${new##* } | $note" >> scores.log
  echo "$m: saved ${new##* } (was ${old##* })"
else
  echo "$m: kept ${old##* } (candidate: $new)"
fi
