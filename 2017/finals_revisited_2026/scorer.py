#!/usr/bin/env python3
"""Validator and scorer for Hash Code 2017 Final Round: Router placement.

Usage:
    python3 scorer.py <problem.in> <solution.out>

Prints the score if the solution is valid. Otherwise prints the reason it
is invalid and exits with status 1.
"""

import sys
from dataclasses import dataclass

import numpy as np

WALL, TARGET, VOID = "#", ".", "-"


class ValidationError(Exception):
    pass


@dataclass
class Problem:
    H: int
    W: int
    R: int
    Pb: int
    Pr: int
    B: int
    br: int
    bc: int
    grid: np.ndarray  # (H, W) array of single characters


@dataclass
class Solution:
    backbone: list  # list of (r, c), in output order, without the initial cell
    routers: list   # list of (r, c)


def read_problem(path):
    with open(path) as f:
        lines = f.read().split("\n")
    H, W, R = map(int, lines[0].split())
    Pb, Pr, B = map(int, lines[1].split())
    br, bc = map(int, lines[2].split())
    rows = lines[3:3 + H]
    if len(rows) != H or any(len(row) != W for row in rows):
        raise ValueError(f"{path}: grid does not match declared size {H}x{W}")
    grid = np.array([list(row) for row in rows])
    if not np.isin(grid, [WALL, TARGET, VOID]).all():
        raise ValueError(f"{path}: grid contains unexpected characters")
    return Problem(H, W, R, Pb, Pr, B, br, bc, grid)


def read_solution(path):
    """Parse a submission file. Raises ValidationError on format problems."""
    with open(path) as f:
        lines = f.read().split("\n")
    # Tolerate trailing blank lines only.
    while lines and lines[-1].strip() == "":
        lines.pop()

    pos = 0

    def next_ints(expected):
        nonlocal pos
        if pos >= len(lines):
            raise ValidationError(f"unexpected end of file at line {pos + 1}")
        parts = lines[pos].split()
        pos += 1
        if len(parts) != expected:
            raise ValidationError(
                f"line {pos}: expected {expected} integer(s), got {lines[pos - 1]!r}")
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValidationError(f"line {pos}: not an integer: {lines[pos - 1]!r}")

    (n,) = next_ints(1)
    backbone = [tuple(next_ints(2)) for _ in range(max(n, 0))]
    (m,) = next_ints(1)
    routers = [tuple(next_ints(2)) for _ in range(max(m, 0))]
    if pos != len(lines):
        raise ValidationError(f"line {pos + 1}: unexpected extra content")
    if n < 0 or m < 0:
        raise ValidationError("N and M must be non-negative")
    return Solution(backbone, routers)


def validate(p, s):
    """Check all validity rules. Raises ValidationError on the first violation."""
    H, W = p.H, p.W
    N, M = len(s.backbone), len(s.routers)

    if not N < W * H:
        raise ValidationError(f"N={N} must be < W*H={W * H}")
    if not M <= W * H:
        raise ValidationError(f"M={M} must be <= W*H={W * H}")

    connected = np.zeros((H, W), dtype=bool)
    connected[p.br, p.bc] = True
    for i, (r, c) in enumerate(s.backbone):
        if not (0 <= r < H and 0 <= c < W):
            raise ValidationError(f"backbone cell #{i + 1} {(r, c)} is out of bounds")
        if connected[r, c]:
            if (r, c) == (p.br, p.bc):
                raise ValidationError(
                    f"backbone cell #{i + 1} {(r, c)} is the initial backbone cell")
            raise ValidationError(f"backbone cell #{i + 1} {(r, c)} is repeated")
        if not connected[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any():
            raise ValidationError(
                f"backbone cell #{i + 1} {(r, c)} has no previously connected neighbor")
        connected[r, c] = True

    seen = set()
    for i, (r, c) in enumerate(s.routers):
        if not (0 <= r < H and 0 <= c < W):
            raise ValidationError(f"router #{i + 1} {(r, c)} is out of bounds")
        if (r, c) in seen:
            raise ValidationError(f"router #{i + 1} {(r, c)} is repeated")
        seen.add((r, c))
        if p.grid[r, c] == WALL:
            raise ValidationError(f"router #{i + 1} {(r, c)} is placed on a wall")
        if not connected[r, c]:
            raise ValidationError(
                f"router #{i + 1} {(r, c)} is not connected to the backbone")

    cost = N * p.Pb + M * p.Pr
    if cost > p.B:
        raise ValidationError(f"budget exceeded: cost {cost} > budget {p.B}")


def coverage(p, routers, chunk=2048):
    """Return an (H, W) bool array of target cells covered by the routers."""
    H, W, R = p.H, p.W, p.R
    is_wall = p.grid == WALL
    is_target = p.grid == TARGET
    covered = np.zeros((H, W), dtype=bool)
    if not routers:
        return covered

    # S[i, j] = number of walls in grid[0:i, 0:j]
    S = np.zeros((H + 1, W + 1), dtype=np.int32)
    S[1:, 1:] = is_wall.cumsum(0).cumsum(1)

    off = np.arange(-R, R + 1)
    pts = np.array(routers, dtype=np.int64)
    for k in range(0, len(pts), chunk):
        a = pts[k:k + chunk, 0][:, None, None]
        b = pts[k:k + chunk, 1][:, None, None]
        X = a + off[None, :, None]
        Y = b + off[None, None, :]
        X, Y = np.broadcast_arrays(X, Y)
        inside = (X >= 0) & (X < H) & (Y >= 0) & (Y < W)
        Xc, Yc = np.clip(X, 0, H - 1), np.clip(Y, 0, W - 1)
        r0, r1 = np.minimum(a, Xc), np.maximum(a, Xc)
        c0, c1 = np.minimum(b, Yc), np.maximum(b, Yc)
        walls = S[r1 + 1, c1 + 1] - S[r0, c1 + 1] - S[r1 + 1, c0] + S[r0, c0]
        ok = inside & (walls == 0) & is_target[Xc, Yc]
        covered[Xc[ok], Yc[ok]] = True
    return covered


def coverage_simple(p, routers):
    """Same result as `coverage`, written with plain loops for readability.

    It uses the same wall test (a prefix-sum table, so the number of walls
    in any rectangle comes from 4 values) but checks one router and one
    cell at a time. About 10-20x slower, still under a second on every map.
    """
    H, W, R = p.H, p.W, p.R
    is_wall = (p.grid == WALL).tolist()      # plain Python lists of bools
    is_target = (p.grid == TARGET).tolist()

    # walls_before[i][j] = number of walls in rows 0..i-1 and columns 0..j-1.
    # The extra row and column of zeros avoid special cases at the edges.
    walls_before = [[0] * (W + 1) for _ in range(H + 1)]
    for i in range(H):
        for j in range(W):
            walls_before[i + 1][j + 1] = (
                walls_before[i][j + 1]      # everything above this cell
                + walls_before[i + 1][j]    # everything to its left
                - walls_before[i][j]        # counted twice, remove once
                + is_wall[i][j]             # the cell itself
            )

    def walls_in(top, left, bottom, right):
        """Number of walls in rows top..bottom, columns left..right (inclusive)."""
        return (
            walls_before[bottom + 1][right + 1]
            - walls_before[top][right + 1]      # strip above the rectangle
            - walls_before[bottom + 1][left]    # strip left of the rectangle
            + walls_before[top][left]           # removed twice, add back once
        )

    covered = np.zeros((H, W), dtype=bool)
    for (a, b) in routers:
        # Every cell in the (2R+1) x (2R+1) square around the router,
        # cut to the grid.
        for x in range(max(0, a - R), min(H, a + R + 1)):
            for y in range(max(0, b - R), min(W, b + R + 1)):
                if not is_target[x][y]:
                    continue
                top, bottom = min(a, x), max(a, x)
                left, right = min(b, y), max(b, y)
                if walls_in(top, left, bottom, right) == 0:
                    covered[x, y] = True
    return covered


def coverage_per_router(p, routers):
    """Same result as `coverage`, halfway between the two other versions.

    Like `coverage`, it uses NumPy: the wall table comes from `cumsum`, and
    each router's square is checked with array operations. Like
    `coverage_simple`, it handles one router at a time with a plain Python
    loop, so there are no batches, and each square is cut to the grid.
    """
    H, W, R = p.H, p.W, p.R
    is_wall = p.grid == WALL
    is_target = p.grid == TARGET
    covered = np.zeros((H, W), dtype=bool)

    # S[i, j] = number of walls in rows 0..i-1 and columns 0..j-1.
    S = np.zeros((H + 1, W + 1), dtype=np.int32)
    S[1:, 1:] = is_wall.cumsum(0).cumsum(1)

    for (a, b) in routers:
        # The router's (2R+1) x (2R+1) square, cut to the grid:
        # rows x0..x1-1 and columns y0..y1-1.
        x0, x1 = max(0, a - R), min(H, a + R + 1)
        y0, y1 = max(0, b - R), min(W, b + R + 1)
        X = np.arange(x0, x1)[:, None]  # row numbers, as a column (kx, 1)
        Y = np.arange(y0, y1)[None, :]  # column numbers, as a row (1, ky)

        # Rectangle between the router and each cell. Its top and bottom
        # depend only on the cell's row, its left and right only on its column.
        top, bottom = np.minimum(a, X), np.maximum(a, X)
        left, right = np.minimum(b, Y), np.maximum(b, Y)

        # Walls in each rectangle, from 4 values of S (one per corner).
        # Combining a (kx, 1) index with a (1, ky) index gives (kx, ky).
        walls = (S[bottom + 1, right + 1]
                 - S[top, right + 1]
                 - S[bottom + 1, left]
                 + S[top, left])

        ok = (walls == 0) & is_target[x0:x1, y0:y1]
        # `|=` keeps cells that earlier routers already covered.
        covered[x0:x1, y0:y1] |= ok
    return covered


def coverage_staircase(p, routers, chunk=2048):
    """Same result as `coverage`, using the "staircase" method instead of S.

    The rectangle between router (a, b) and cell (x, y), with y >= b, has no
    wall if and only if, in every row between a and x, the first wall at or
    right of column b is beyond column y. So for each row of the square we
    only need how far it reaches right (and left) of column b, and then a
    running minimum of that reach moving away from the router row. The
    reachable area shrinks row by row, like a staircase.

    Routers are handled in batches, like `coverage`. About 1.8-2.6x faster than
    `coverage`, because it works per row of each square, not per cell.
    """
    H, W, R = p.H, p.W, p.R
    is_wall = p.grid == WALL
    is_target = p.grid == TARGET
    covered = np.zeros((H, W), dtype=bool)
    if not routers:
        return covered

    # next_wall[x, y]: column of the first wall at or right of (x, y), W if none.
    # prev_wall[x, y]: column of the last wall at or left of (x, y), -1 if none.
    cols = np.arange(W)
    next_wall = np.minimum.accumulate(
        np.where(is_wall, cols, W)[:, ::-1], axis=1)[:, ::-1]
    prev_wall = np.maximum.accumulate(np.where(is_wall, cols, -1), axis=1)

    off = np.arange(-R, R + 1)  # offsets -R..R, K = 2R+1 values; R is the middle
    pts = np.array(routers, dtype=np.int64)
    for k in range(0, len(pts), chunk):
        a = pts[k:k + chunk, 0][:, None]  # (n, 1)
        b = pts[k:k + chunk, 1][:, None]

        # Rows of each router's square, (n, K). Rows outside the grid are
        # clamped for indexing and made empty below.
        X = a + off
        row_ok = (X >= 0) & (X < H)
        Xc = np.clip(X, 0, H - 1)

        # How far each row alone reaches right / left of column b, capped at R.
        # -1 means nothing (a wall in column b itself, or outside the grid).
        # Bounded by the walls and the grid edge, so columns stay in the grid.
        right = np.minimum(R, next_wall[Xc, b] - b - 1)
        left = np.minimum(R, b - prev_wall[Xc, b] - 1)
        right[~row_ok] = -1
        left[~row_ok] = -1

        # The staircase: running minimum from the router row outwards, down
        # (indices R..K-1) and up (indices R..0). Once a row reaches -1,
        # every row beyond it stays empty.
        for reach in (right, left):
            reach[:, R:] = np.minimum.accumulate(reach[:, R:], axis=1)
            reach[:, R::-1] = np.minimum.accumulate(reach[:, R::-1], axis=1)

        # Cell at column offset dy in a row is covered if -left <= dy <= right
        # (and it is a target). Shapes: (n, K, 1) against (1, 1, K).
        dy = off[None, None, :]
        ok = (dy <= right[:, :, None]) & (-dy <= left[:, :, None])
        Xb, Yb = np.broadcast_arrays(
            Xc[:, :, None], np.clip(b[:, :, None] + dy, 0, W - 1))
        ok &= is_target[Xb, Yb]
        covered[Xb[ok], Yb[ok]] = True
    return covered


def score(p, s):
    """Validate and return (score, details dict). Raises ValidationError."""
    validate(p, s)
    t = int(coverage(p, s.routers).sum())
    cost = len(s.backbone) * p.Pb + len(s.routers) * p.Pr
    return 1000 * t + (p.B - cost), {
        "covered": t,
        "targets": int((p.grid == TARGET).sum()),
        "backbone": len(s.backbone),
        "routers": len(s.routers),
        "cost": cost,
        "budget": p.B,
    }


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    problem = read_problem(argv[1])
    try:
        solution = read_solution(argv[2])
        total, d = score(problem, solution)
    except ValidationError as e:
        print(f"INVALID: {e}")
        return 1
    print(f"covered {d['covered']}/{d['targets']} targets, "
          f"{d['backbone']} backbone cells, {d['routers']} routers, "
          f"cost {d['cost']}/{d['budget']}")
    print(f"score: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
