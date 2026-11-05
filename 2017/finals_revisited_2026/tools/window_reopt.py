#!/usr/bin/env python3
"""Shorten a solution's backbone window by window with GeoSteiner.

Repeatedly: pick a random rectangle of the grid. Inside it, throw away the
backbone and replace it with GeoSteiner's exact rectilinear Steiner tree
(on rotated coordinates, see tools/rotated_points.py) through
  - the routers in the rectangle (and the initial cell, if inside), and
  - the "portals": backbone cells in the rectangle next to a backbone cell
    outside it, so every piece outside stays connected.
Convert that tree back to grid cells, polish the whole backbone with the
Rust solver's tree search, and keep the result if the backbone is shorter.

GeoSteiner's bundled LP solver fails on whole maps the size of opera but is
fast on a few hundred points, hence the windows.

Usage: python3 tools/window_reopt.py <problem.in> <solution.out> <out.out> [iterations] [workers]
The best solution so far is written to <out.out> after every improvement.
"""

import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from scorer import read_problem, read_solution  # noqa: E402
from geosteiner_tree import parse_ps, tree_cells, write_solution  # noqa: E402

GS = os.environ.get("GEOSTEINER", os.path.join(ROOT, "third_party", "geosteiner-5.3"))
SOLVER = os.path.join(ROOT, "solver", "target", "release", "solver")
SCALE = 1000
MAX_TERMS = 260
NEIGH = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def try_window(p, problem_path, cells, routers, win, seed, tmp):
    """Candidate backbone size and solution path for one window, or None."""
    r0, r1, c0, c1 = win

    def inside(x):
        return r0 <= x[0] <= r1 and c0 <= x[1] <= c1

    start = (p.br, p.bc)
    terms = {x for x in routers if inside(x)}
    if inside(start):
        terms.add(start)
    for x in cells:
        if inside(x) and any((x[0] + dr, x[1] + dc) in cells and not inside((x[0] + dr, x[1] + dc))
                             for dr, dc in NEIGH):
            terms.add(x)
    if len(terms) < 3 or len(terms) > MAX_TERMS:
        return None
    rng = random.Random(seed)
    j = SCALE // 100
    pts = "".join(f"{(r + c) * SCALE + rng.randrange(j)} {(r - c + p.W) * SCALE + rng.randrange(j)}\n"
                  for r, c in sorted(terms))
    try:
        fst = subprocess.run([os.path.join(GS, "rfst")], input=pts, capture_output=True, text=True,
                             timeout=60, check=True).stdout
        ps = subprocess.run([os.path.join(GS, "bb")], input=fst, capture_output=True, text=True,
                            timeout=120, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    ps_path = os.path.join(tmp, f"w{seed}.ps")
    with open(ps_path, "w") as f:
        f.write(ps)
    t, segs = parse_ps(ps_path)
    local = tree_cells(t, segs, p.W, SCALE)
    new = {x for x in cells if not inside(x)} | local | set(routers) | {start}
    cand = os.path.join(tmp, f"w{seed}.out")
    if write_solution(p, new, routers, cand) is None:
        return None
    # Polish the whole tree (key-path exchange, key-vertex elimination and
    # a short rip-up-and-reconnect search), single-threaded.
    polished = os.path.join(tmp, f"w{seed}.pol.out")
    env = dict(os.environ, TREE_ILS="1.5", SEED=cand)
    r = subprocess.run([SOLVER, problem_path, polished], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    s = read_solution(polished)
    return len(s.backbone), polished


def main(argv):
    problem_path, sol_path, out_path = argv[1], argv[2], argv[3]
    iters = int(argv[4]) if len(argv) > 4 else 100
    workers = int(argv[5]) if len(argv) > 5 else 8
    p = read_problem(problem_path)
    sol = read_solution(sol_path)
    routers = list(sol.routers)
    cells = set(sol.backbone) | {(p.br, p.bc)}
    best = len(sol.backbone)
    print(f"start: backbone {best}", flush=True)
    rng = random.Random(1)
    tmp = tempfile.mkdtemp(prefix="wreopt_", dir=os.path.join(ROOT, "tools", "lower_bound"))
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        while done < iters:
            # A batch of windows on the same tree; keep the best result.
            wins = []
            for _ in range(workers):
                cr, cc = routers[rng.randrange(len(routers))]
                hr, hc = rng.randint(25, 70), rng.randint(25, 70)
                wins.append((max(0, cr - hr), min(p.H - 1, cr + hr), max(0, cc - hc), min(p.W - 1, cc + hc)))
            seeds = [done + k for k in range(len(wins))]
            results = list(pool.map(lambda a: try_window(p, problem_path, cells, routers, a[0], a[1], tmp),
                                    zip(wins, seeds)))
            done += len(wins)
            ok = [r for r in results if r is not None]
            if ok:
                n, path = min(ok)
                if n < best:
                    best = n
                    s = read_solution(path)
                    cells = set(s.backbone) | {(p.br, p.bc)}
                    with open(path) as f, open(out_path, "w") as g:
                        g.write(f.read())
            print(f"{done} windows tried ({len(ok)} solved, best of batch {min(ok)[0] if ok else None}), backbone {best}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
