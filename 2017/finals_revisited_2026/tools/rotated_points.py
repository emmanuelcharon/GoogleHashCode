#!/usr/bin/env python3
"""Write a solution's routers and the initial backbone cell as points for an
exact rectilinear Steiner tree solver (GeoSteiner).

A backbone step to any of the 8 neighbours is one unit of Chebyshev
distance. Turning the grid by 45 degrees, (r, c) -> (u, v) = (r + c, r - c),
turns Chebyshev distance into half the Manhattan distance. So a backbone
tree with E cells (not counting the initial cell) gives a rectilinear tree
of length 2E through the same points, and therefore

    E >= (length of the rectilinear Steiner minimal tree) / 2.

Usage: python3 tools/rotated_points.py <problem.in> <solution.out> [scale] > points.txt

With `scale`, coordinates are multiplied by it and get a small random offset
(less than 1/100 of a grid unit), which breaks the exact ties that make
GeoSteiner's bundled LP solver fail on some maps. Read the tree back with
the same scale (tools/geosteiner_tree.py).
"""

import random
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from scorer import read_problem, read_solution  # noqa: E402


def main(argv):
    p = read_problem(argv[1])
    s = read_solution(argv[2])
    scale = int(argv[3]) if len(argv) > 3 else 1
    rng = random.Random(0)
    cells = {(p.br, p.bc)} | set(s.routers)
    for r, c in sorted(cells):
        # Shift v so every coordinate is non-negative.
        u, v = r + c, r - c + p.W
        if scale == 1:
            print(u, v)
        else:
            j = max(1, scale // 100)
            print(u * scale + rng.randrange(j), v * scale + rng.randrange(j))


if __name__ == "__main__":
    main(sys.argv)
