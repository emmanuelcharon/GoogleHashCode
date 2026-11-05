#!/bin/bash
# Lower bound on the backbone size of each saved output, from GeoSteiner's
# exact rectilinear Steiner minimal tree on the rotated router positions
# (see tools/rotated_points.py). GeoSteiner (geosteiner.com, CC BY-NC) is not
# included: download it, build it with ./configure && make, and set
# GEOSTEINER=<its folder> (default: third_party/geosteiner-5.3).
# Usage: tools/tree_lower_bound.sh [map ...]   (default: all four maps)
set -e
cd "$(dirname "$0")/.."
gs=${GEOSTEINER:-third_party/geosteiner-5.3}
[ -x "$gs/rfst" ] && [ -x "$gs/bb" ] || { echo "build GeoSteiner first (see the header of this script)"; exit 1; }
mkdir -p tools/lower_bound
maps=${*:-"charleston_road rue_de_londres opera lets_go_higher"}
for m in $maps; do
  python3 tools/rotated_points.py "input_files/$m.in" "outputs/$m.out" > "tools/lower_bound/$m.pts"
  start=$(date +%s)
  status=0
  set -o pipefail
  "$gs/rfst" < "tools/lower_bound/$m.pts" | "$gs/bb" > "tools/lower_bound/$m.ps" 2>/dev/null || status=$?
  set +o pipefail
  secs=$(( $(date +%s) - start ))
  if [ $status -eq 0 ]; then
    len=$(grep -a -o "length = [0-9.]*" "tools/lower_bound/$m.ps" | tail -1 | awk '{print $3}')
    what="rectilinear SMT $len (exact)"
  else
    # bb's bundled LP solver (lp_solve 2.3) can fail on these lattice point
    # sets. Every LP it solved before failing is a relaxation of the exact
    # problem (fewer constraints), so the largest LP value is still a valid
    # lower bound, just a weaker one.
    len=$(grep -a -o "Node 0 LP [0-9]* Solution, length = [0-9.]*" "tools/lower_bound/$m.ps" | awk '{print $NF}' | sort -n | tail -1)
    [ -n "$len" ] || { echo "$m: GeoSteiner failed before solving any LP (${secs}s)"; continue; }
    what="exact solve failed, best LP relaxation $len"
  fi
  ours=$(sed -n 1p "outputs/$m.out")
  bound=$(python3 -c "import math; print(math.ceil($len / 2 - 1e-9))")
  gap=$(python3 -c "print(f'{100 * ($ours - $bound) / $bound:.2f}')")
  echo "$m: points $(wc -l < tools/lower_bound/$m.pts | tr -d ' '), $what, lower bound $bound cells, ours $ours cells, gap at most $gap% (${secs}s)"
done
