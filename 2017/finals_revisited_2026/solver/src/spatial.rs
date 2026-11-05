//! Coarse bucket grid for nearest-neighbour queries in Chebyshev distance.

use crate::problem::Problem;

/// Routers bucketed on a coarse grid for nearest-neighbour queries.
pub struct Buckets {
    size: usize,
    bh: usize,
    bw: usize,
    w: usize,
    start: Option<usize>,
    cells: Vec<Vec<usize>>,
}

pub fn cheb(w: usize, a: usize, b: usize) -> usize {
    (a / w).abs_diff(b / w).max((a % w).abs_diff(b % w))
}

impl Buckets {
    pub fn new(p: &Problem, size: usize, start: Option<usize>) -> Buckets {
        let (bh, bw) = (p.h.div_ceil(size), p.w.div_ceil(size));
        Buckets { size, bh, bw, w: p.w, start, cells: vec![vec![]; bh * bw] }
    }
    fn bucket(&self, x: usize) -> usize {
        (x / self.w / self.size) * self.bw + (x % self.w) / self.size
    }
    pub fn insert(&mut self, x: usize) {
        let b = self.bucket(x);
        self.cells[b].push(x);
    }
    pub fn remove(&mut self, x: usize) {
        let b = self.bucket(x);
        let v = &mut self.cells[b];
        let k = v.iter().position(|&y| y == x).unwrap();
        v.swap_remove(k);
    }
    /// Chebyshev distance from `x` to the nearest point other than `exclude`
    /// (or to the start cell, if given, when that is closer). usize::MAX if none.
    pub fn nearest(&self, x: usize, exclude: usize) -> usize {
        let mut best = self.start.map_or(usize::MAX, |s| cheb(self.w, x, s));
        let (br, bc) = ((x / self.w / self.size) as isize, ((x % self.w) / self.size) as isize);
        for k in 0..=(self.bh.max(self.bw) as isize) {
            // Every cell in ring k is at least (k - 1) * size + 1 away.
            if k >= 1 && ((k - 1) as usize) * self.size >= best {
                break;
            }
            for r in (br - k)..=(br + k) {
                if r < 0 || r >= self.bh as isize {
                    continue;
                }
                let edge = r == br - k || r == br + k;
                let step = if edge { 1 } else { (2 * k).max(1) };
                let mut c = bc - k;
                while c <= bc + k {
                    if c >= 0 && c < self.bw as isize {
                        for &y in &self.cells[r as usize * self.bw + c as usize] {
                            if y != exclude {
                                best = best.min(cheb(self.w, x, y));
                            }
                        }
                    }
                    c += step;
                }
            }
        }
        best
    }
}
