//! Problem parsing, wall prefix sums, router coverage and solution output.

use std::fs;
use std::io::Write;

pub const WALL: u8 = b'#';
pub const TARGET: u8 = b'.';

pub struct Problem {
    pub h: usize,
    pub w: usize,
    pub r: usize,
    pub pb: i64,
    pub pr: i64,
    pub budget: i64,
    pub start: usize, // index of the initial backbone cell
    pub grid: Vec<u8>, // h * w, row-major
    /// Wall prefix sums: s[i * (w + 1) + j] = walls in grid[0..i][0..j].
    s: Vec<u32>,
    /// Per cell: column of the first wall at or right of it in its row (w if none).
    next_wall: Vec<i32>,
    /// Per cell: column of the last wall at or left of it in its row (-1 if none).
    prev_wall: Vec<i32>,
}

impl Problem {
    pub fn read(path: &str) -> Problem {
        let text = fs::read_to_string(path).expect("cannot read input");
        let mut lines = text.lines();
        let nums = |line: Option<&str>| -> Vec<i64> {
            line.unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect()
        };
        let l1 = nums(lines.next());
        let l2 = nums(lines.next());
        let l3 = nums(lines.next());
        let (h, w, r) = (l1[0] as usize, l1[1] as usize, l1[2] as usize);
        let mut grid = Vec::with_capacity(h * w);
        for _ in 0..h {
            let row = lines.next().unwrap().as_bytes();
            assert_eq!(row.len(), w);
            grid.extend_from_slice(row);
        }
        let mut s = vec![0u32; (h + 1) * (w + 1)];
        for i in 0..h {
            let mut row_sum = 0;
            for j in 0..w {
                row_sum += (grid[i * w + j] == WALL) as u32;
                s[(i + 1) * (w + 1) + j + 1] = s[i * (w + 1) + j + 1] + row_sum;
            }
        }
        let mut next_wall = vec![0i32; h * w];
        let mut prev_wall = vec![0i32; h * w];
        for i in 0..h {
            let mut last = -1;
            for j in 0..w {
                if grid[i * w + j] == WALL {
                    last = j as i32;
                }
                prev_wall[i * w + j] = last;
            }
            let mut next = w as i32;
            for j in (0..w).rev() {
                if grid[i * w + j] == WALL {
                    next = j as i32;
                }
                next_wall[i * w + j] = next;
            }
        }
        Problem {
            h,
            w,
            r,
            next_wall,
            prev_wall,
            pb: l2[0],
            pr: l2[1],
            budget: l2[2],
            start: l3[0] as usize * w + l3[1] as usize,
            grid,
            s,
        }
    }

    #[inline]
    pub fn rc(&self, idx: usize) -> (usize, usize) {
        (idx / self.w, idx % self.w)
    }

    /// Number of walls in the inclusive rectangle [r0..=r1] x [c0..=c1].
    #[inline]
    fn walls(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> u32 {
        let w1 = self.w + 1;
        self.s[(r1 + 1) * w1 + c1 + 1] + self.s[r0 * w1 + c0]
            - self.s[r0 * w1 + c1 + 1]
            - self.s[(r1 + 1) * w1 + c0]
    }

    /// Calls `f` with the index of every target cell a router at `idx` covers.
    ///
    /// Each quadrant's covered area is a staircase: walking rows away from
    /// the router, the reachable column span can only shrink, and it shrinks
    /// to the nearest wall in that row on the router's side.
    #[inline]
    pub fn for_each_covered(&self, idx: usize, mut f: impl FnMut(usize)) {
        let (a, b) = self.rc(idx);
        if self.grid[idx] == WALL {
            return;
        }
        let (r, w) = (self.r as i32, self.w);
        let (bi, wi) = (b as i32, w as i32);
        let x0 = a.saturating_sub(self.r);
        let x1 = (a + self.r).min(self.h - 1);
        // The router's own row bounds every other row's span too.
        let right0 = r.min(self.next_wall[idx] - bi - 1);
        let left0 = r.min(bi - self.prev_wall[idx] - 1);
        // Downwards (including the router row), then upwards.
        for (lo, hi, down) in [(a, x1, true), (x0, a.wrapping_sub(1), false)] {
            if !down && a == 0 {
                continue;
            }
            let (mut right, mut left) = (right0, left0); // reachable span right/left of b
            let mut x = if down { lo } else { hi };
            loop {
                let row = x * w;
                right = right.min(self.next_wall[row + b] - bi - 1);
                left = left.min(bi - self.prev_wall[row + b] - 1);
                if right < 0 {
                    break; // wall in column b: nothing further is reachable
                }
                let c0 = (bi - left).max(0) as usize;
                let c1 = (bi + right).min(wi - 1) as usize;
                for j in row + c0..=row + c1 {
                    if self.grid[j] == TARGET {
                        f(j);
                    }
                }
                if down {
                    if x == hi {
                        break;
                    }
                    x += 1;
                } else {
                    if x == lo {
                        break;
                    }
                    x -= 1;
                }
            }
        }
    }

    /// Reference implementation of `for_each_covered` using prefix sums.
    #[allow(dead_code)]
    pub fn for_each_covered_slow(&self, idx: usize, mut f: impl FnMut(usize)) {
        let (a, b) = self.rc(idx);
        let r = self.r;
        let x0 = a.saturating_sub(r);
        let x1 = (a + r).min(self.h - 1);
        let y0 = b.saturating_sub(r);
        let y1 = (b + r).min(self.w - 1);
        for x in x0..=x1 {
            let (rlo, rhi) = if x < a { (x, a) } else { (a, x) };
            for y in y0..=y1 {
                let j = x * self.w + y;
                if self.grid[j] != TARGET {
                    continue;
                }
                let (clo, chi) = if y < b { (y, b) } else { (b, y) };
                if self.walls(rlo, clo, rhi, chi) == 0 {
                    f(j);
                }
            }
        }
    }

    /// Compare the fast and reference coverage on every non-wall cell.
    pub fn self_check(&self) {
        for i in 0..self.h * self.w {
            if self.grid[i] == WALL {
                continue;
            }
            let (mut a, mut b) = (vec![], vec![]);
            self.for_each_covered(i, |j| a.push(j));
            self.for_each_covered_slow(i, |j| b.push(j));
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "coverage mismatch at cell {:?}", self.rc(i));
        }
        eprintln!("coverage self-check passed");
    }

    pub fn num_targets(&self) -> usize {
        self.grid.iter().filter(|&&c| c == TARGET).count()
    }
}

#[derive(Clone)]
pub struct Solution {
    pub backbone: Vec<usize>, // in connection order, excluding the start cell
    pub routers: Vec<usize>,
}

impl Solution {
    /// Read a submission file (assumed valid, e.g. checked by scorer.py).
    pub fn read(p: &Problem, path: &str) -> Solution {
        let text = fs::read_to_string(path).expect("cannot read solution");
        let mut it = text.split_whitespace().map(|x| x.parse::<usize>().unwrap());
        let cells = |it: &mut dyn Iterator<Item = usize>| {
            let n = it.next().unwrap();
            (0..n)
                .map(|_| {
                    let r = it.next().unwrap();
                    r * p.w + it.next().unwrap()
                })
                .collect::<Vec<usize>>()
        };
        let backbone = cells(&mut it);
        let routers = cells(&mut it);
        Solution { backbone, routers }
    }

    pub fn cost(&self, p: &Problem) -> i64 {
        self.backbone.len() as i64 * p.pb + self.routers.len() as i64 * p.pr
    }

    /// Returns (score, covered targets). Assumes the solution is valid.
    pub fn score(&self, p: &Problem) -> (i64, usize) {
        let mut covered = vec![false; p.h * p.w];
        let mut t = 0;
        for &r in &self.routers {
            p.for_each_covered(r, |j| {
                if !covered[j] {
                    covered[j] = true;
                    t += 1;
                }
            });
        }
        (1000 * t as i64 + p.budget - self.cost(p), t)
    }

    pub fn write(&self, p: &Problem, path: &str) {
        let mut out = String::new();
        out += &format!("{}\n", self.backbone.len());
        for &c in &self.backbone {
            let (r, c) = p.rc(c);
            out += &format!("{r} {c}\n");
        }
        out += &format!("{}\n", self.routers.len());
        for &c in &self.routers {
            let (r, c) = p.rc(c);
            out += &format!("{r} {c}\n");
        }
        fs::File::create(path)
            .and_then(|mut f| f.write_all(out.as_bytes()))
            .expect("cannot write output");
    }
}
