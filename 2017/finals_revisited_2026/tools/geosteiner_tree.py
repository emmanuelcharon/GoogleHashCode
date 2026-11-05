#!/usr/bin/env python3
"""Turn GeoSteiner's rectilinear Steiner tree (bb PostScript output, on the
points from tools/rotated_points.py) back into a backbone on the grid.

Rotated point (u, v) is the cell r = (u + v - W) / 2, c = (u - v + W) / 2.
Steiner points with u + v - W odd fall between cells; each is moved to the
neighbouring lattice point closest to the points it is joined to. Every
segment then becomes a king-move path between its two cells.

Writes a solution with the same routers and the new backbone, listed in a
valid connection order (BFS from the initial cell).

Usage: python3 tools/geosteiner_tree.py <problem.in> <solution.out> <bb.ps> <out.out> [scale]
(`scale` as given to tools/rotated_points.py.)
"""

import re
import sys
from collections import deque

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from scorer import read_problem, read_solution  # noqa: E402


def parse_ps(path):
    """Terminal list and segments (each endpoint ('T', k) or ('P', u, v))."""
    terms, segs = [], []
    num = r"(-?\d+(?:\.\d+)?)"
    end = rf"(?:(\d+) T|{num}\s+{num})"
    seg_re = re.compile(rf"^\s*{end}\s+{end}\s+S\s*$")
    dt_re = re.compile(rf"^\s*{num}\s+{num}\s+DT\s*$")
    for line in open(path, errors="replace"):
        m = dt_re.match(line)
        if m:
            terms.append((float(m.group(1)), float(m.group(2))))
            continue
        m = seg_re.match(line)
        if m:
            g = m.groups()
            a = ("T", int(g[0])) if g[0] is not None else ("P", float(g[1]), float(g[2]))
            b = ("T", int(g[3])) if g[3] is not None else ("P", float(g[4]), float(g[5]))
            segs.append((a, b))
    return terms, segs


def king_path(a, b):
    """Cells from a to b (both included), diagonal moves first."""
    (r, c), (r1, c1) = a, b
    out = [(r, c)]
    while (r, c) != (r1, c1):
        r += (r1 > r) - (r1 < r)
        c += (c1 > c) - (c1 < c)
        out.append((r, c))
    return out


def tree_cells(terms, segs, W, scale=1):
    """Grid cells of a GeoSteiner tree (terminals, segments from parse_ps)."""
    if scale != 1:
        # Back to grid units; terminals snap to their exact lattice point.
        terms = [(round(u / scale), round(v / scale)) for u, v in terms]
        segs = [tuple(e if e[0] == "T" else ("P", e[1] / scale, e[2] / scale) for e in s) for s in segs]

    # Nodes: terminals by index, Steiner points by coordinates.
    pos = {("T", k): (u, v) for k, (u, v) in enumerate(terms)}
    adj = {}
    for a, b in segs:
        for e in (a, b):
            if e[0] == "P":
                pos[e] = (e[1], e[2])
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def on_lattice(u, v):
        return float(u).is_integer() and float(v).is_integer() and (int(u) + int(v) - W) % 2 == 0

    def dist(a, b):  # rotated Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Move off-lattice Steiner points to the best neighbouring lattice point.
    for node, (u, v) in list(pos.items()):
        if node[0] == "T" or on_lattice(u, v):
            continue
        cands = []
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                cu, cv = round(u) + du, round(v) + dv
                if on_lattice(cu, cv):
                    cost = sum(dist((cu, cv), pos[o]) for o in adj.get(node, []))
                    cands.append((cost, cu, cv))
        _, cu, cv = min(cands)
        pos[node] = (cu, cv)

    def cell(node):
        u, v = (int(round(x)) for x in pos[node])
        return ((u + v - W) // 2, (u - v + W) // 2)

    cells = set()
    for a, b in segs:
        cells.update(king_path(cell(a), cell(b)))
    for k in range(len(terms)):
        cells.add(cell(("T", k)))
    return cells


def write_solution(p, cells, routers, path):
    """Write cells (must include the routers) as a solution, in BFS order
    from the initial cell. Returns the backbone size, or None if some
    router is not connected or a cell is outside the grid."""
    if any(not (0 <= r < p.H and 0 <= c < p.W) for r, c in cells):
        return None
    start = (p.br, p.bc)
    order, seen, q = [], {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if n in cells and n not in seen:
                    seen.add(n)
                    order.append(n)
                    q.append(n)
    if any(x not in seen for x in routers):
        return None
    with open(path, "w") as f:
        f.write(f"{len(order)}\n")
        for r, c in order:
            f.write(f"{r} {c}\n")
        f.write(f"{len(routers)}\n")
        for r, c in routers:
            f.write(f"{r} {c}\n")
    return len(order)


def main(argv):
    p = read_problem(argv[1])
    sol = read_solution(argv[2])
    terms, segs = parse_ps(argv[3])
    scale = int(argv[5]) if len(argv) > 5 else 1
    cells = tree_cells(terms, segs, p.W, scale) | {(p.br, p.bc)} | set(sol.routers)
    n = write_solution(p, cells, sol.routers, argv[4])
    if n is None:
        sys.exit("tree does not connect every router, or leaves the grid")
    print(f"backbone cells {n}")


if __name__ == "__main__":
    main(sys.argv)
