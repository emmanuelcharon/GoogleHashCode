//! Greedy construction: repeatedly place the router with the best
//! (newly covered targets) / (router + backbone cost) ratio, connecting it
//! to the nearest existing backbone cell along a Chebyshev shortest path.

use crate::problem::{Problem, Solution, WALL};
use crate::spatial::Buckets;
use std::collections::{BinaryHeap, VecDeque};

pub struct State<'a> {
    pub p: &'a Problem,
    pub covered: Vec<bool>,
    pub connected: Vec<bool>,
    pub is_router: Vec<bool>,
    /// Chebyshev distance to the nearest backbone cell, and that cell.
    pub dist: Vec<u32>,
    pub src: Vec<u32>,
    pub sol: Solution,
    pub remaining: i64,
    /// Cells whose `dist` decreased during the last `place` call.
    pub improved: Vec<usize>,
}

impl<'a> State<'a> {
    pub fn new(p: &'a Problem) -> State<'a> {
        let n = p.h * p.w;
        let (sr, sc) = p.rc(p.start);
        let mut dist = vec![0u32; n];
        for i in 0..n {
            let (r, c) = p.rc(i);
            dist[i] = r.abs_diff(sr).max(c.abs_diff(sc)) as u32;
        }
        let mut connected = vec![false; n];
        connected[p.start] = true;
        State {
            p,
            covered: vec![false; n],
            connected,
            is_router: vec![false; n],
            dist,
            src: vec![p.start as u32; n],
            sol: Solution { backbone: vec![], routers: vec![] },
            remaining: p.budget,
            improved: vec![],
        }
    }

    /// State for an existing valid solution (backbone in connection order).
    pub fn from_solution(p: &'a Problem, sol: &Solution) -> State<'a> {
        let n = p.h * p.w;
        let mut st = State::new(p);
        let mut queue = VecDeque::new();
        queue.push_back(p.start);
        st.dist.fill(u32::MAX);
        st.dist[p.start] = 0;
        for &c in &sol.backbone {
            st.connected[c] = true;
            st.dist[c] = 0;
            st.src[c] = c as u32;
            queue.push_back(c);
        }
        // Multi-source BFS: distance to the nearest backbone cell.
        let w = p.w as isize;
        while let Some(u) = queue.pop_front() {
            let (ur, uc) = ((u / p.w) as isize, (u % p.w) as isize);
            for k in 0..9 {
                let (vr, vc) = (ur + k / 3 - 1, uc + k % 3 - 1);
                if vr < 0 || vc < 0 || vr >= p.h as isize || vc >= w {
                    continue;
                }
                let v = (vr * w + vc) as usize;
                if st.dist[v] == u32::MAX {
                    st.dist[v] = st.dist[u] + 1;
                    st.src[v] = st.src[u];
                    queue.push_back(v);
                }
            }
        }
        debug_assert!(st.dist.iter().take(n).all(|&d| d != u32::MAX));
        for &r in &sol.routers {
            st.is_router[r] = true;
            let covered = &mut st.covered;
            p.for_each_covered(r, |j| covered[j] = true);
        }
        st.sol = Solution { backbone: sol.backbone.clone(), routers: sol.routers.clone() };
        st.remaining = p.budget - st.sol.cost(p);
        st
    }

    pub fn gain(&self, idx: usize) -> u32 {
        let mut g = 0;
        self.p.for_each_covered(idx, |j| g += !self.covered[j] as u32);
        g
    }

    pub fn router_cost(&self, idx: usize) -> i64 {
        self.p.pr + self.p.pb * self.dist[idx] as i64
    }

    /// Connect `idx` to the backbone via its nearest backbone cell. Among
    /// the equally short paths, each step goes to the neighbour closest to
    /// the nearest router in `guide` (routers still to be connected), so
    /// later routers can share this wire.
    fn connect(&mut self, idx: usize, guide: Option<&Buckets>) {
        let p = self.p;
        let w = p.w as isize;
        let target = self.src[idx] as usize;
        let (tr, tc) = p.rc(target);
        let (tr, tc) = (tr as isize, tc as isize);
        // Walk from the backbone cell to the router; the path excludes the
        // backbone cell and ends at the router, already in connection order.
        let (gr, gc) = p.rc(idx);
        let (gr, gc) = (gr as isize, gc as isize);
        let (mut r, mut c) = (tr, tc);
        let mut path = vec![];
        while (r, c) != (gr, gc) {
            let d = (gr - r).abs().max((gc - c).abs());
            let mut best = (usize::MAX, r + (gr - r).signum(), c + (gc - c).signum());
            if let Some(guide) = guide {
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let (nr, nc) = (r + dr, c + dc);
                        if (gr - nr).abs().max((gc - nc).abs()) != d - 1 {
                            continue;
                        }
                        let near = guide.nearest((nr * w + nc) as usize, usize::MAX);
                        if near < best.0 {
                            best = (near, nr, nc);
                        }
                    }
                }
            }
            (r, c) = (best.1, best.2);
            path.push((r * w + c) as usize);
        }
        let mut queue = VecDeque::new();
        self.improved.clear();
        for &j in &path {
            debug_assert!(!self.connected[j]);
            self.connected[j] = true;
            self.dist[j] = 0;
            self.src[j] = j as u32;
            self.sol.backbone.push(j);
            queue.push_back(j);
            self.improved.push(j);
        }
        self.remaining -= p.pb * path.len() as i64;
        // Incremental multi-source BFS: only cells whose distance improves.
        let (h, wu) = (p.h as isize, w);
        while let Some(u) = queue.pop_front() {
            let d = self.dist[u] + 1;
            let s = self.src[u];
            let (ur, uc) = (u as isize / wu, u as isize % wu);
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let (vr, vc) = (ur + dr, uc + dc);
                    if vr < 0 || vc < 0 || vr >= h || vc >= wu {
                        continue;
                    }
                    let v = (vr * wu + vc) as usize;
                    if d < self.dist[v] {
                        self.dist[v] = d;
                        self.src[v] = s;
                        queue.push_back(v);
                        self.improved.push(v);
                    }
                }
            }
        }
    }

    pub fn place(&mut self, idx: usize) {
        self.place_guided(idx, None);
    }

    pub fn place_guided(&mut self, idx: usize, guide: Option<&Buckets>) {
        self.connect(idx, guide);
        self.is_router[idx] = true;
        self.sol.routers.push(idx);
        self.remaining -= self.p.pr;
        let p = self.p;
        let covered = &mut self.covered;
        p.for_each_covered(idx, |j| covered[j] = true);
    }
}

/// Initial gain of every cell (number of targets it covers), in parallel.
pub fn initial_gains(p: &Problem) -> Vec<u32> {
    let n = p.h * p.w;
    let threads = std::thread::available_parallelism().map_or(4, |x| x.get());
    let chunk = n.div_ceil(threads);
    let mut gains = vec![0u32; n];
    std::thread::scope(|scope| {
        for (t, out) in gains.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                for (k, g) in out.iter_mut().enumerate() {
                    let i = t * chunk + k;
                    if p.grid[i] != WALL {
                        let mut cnt = 0;
                        p.for_each_covered(i, |_| cnt += 1);
                        *g = cnt;
                    }
                }
            });
        }
    });
    gains
}

pub fn solve(p: &Problem, gains: &[u32]) -> Solution {
    let mut st = State::new(p);
    fill(&mut st, gains);
    st.sol
}

/// Connect the given routers with a Prim-style Steiner tree (always attach
/// the router closest to the current backbone next). Ignores the budget.
pub fn connect_all<'a>(p: &'a Problem, routers: &[usize]) -> State<'a> {
    let mut st = State::new(p);
    connect_into(&mut st, routers);
    st
}

/// Connect `routers` to the backbone of `st`, closest first, each path
/// guided towards the routers still waiting. Ignores the budget.
pub fn connect_into(st: &mut State, routers: &[usize]) {
    let p = st.p;
    let mut pending = Buckets::new(p, 2 * p.r + 1, None);
    for &c in routers {
        pending.insert(c);
    }
    let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize)>> = routers
        .iter()
        .map(|&c| std::cmp::Reverse((st.dist[c], c)))
        .collect();
    while let Some(std::cmp::Reverse((d, c))) = heap.pop() {
        if st.dist[c] != d {
            heap.push(std::cmp::Reverse((st.dist[c], c)));
            continue;
        }
        pending.remove(c);
        st.place_guided(c, Some(&pending));
    }
}

/// Like `connect_all`, but reuse the backbone of `base`: keep its branches
/// to the routers that did not move, and connect only the others.
pub fn connect_from<'a>(p: &'a Problem, base: &Solution, routers: &[usize]) -> State<'a> {
    let mut wanted = vec![false; p.h * p.w];
    for &r in routers {
        wanted[r] = true;
    }
    let kept: Vec<usize> = base.routers.iter().copied().filter(|&r| wanted[r]).collect();
    let mut was = vec![false; p.h * p.w];
    for &r in &kept {
        was[r] = true;
    }
    let moved: Vec<usize> = routers.iter().copied().filter(|&r| !was[r]).collect();
    let backbone = crate::lns::prune_backbone(p, &base.backbone, &kept);
    let mut st = State::from_solution(p, &Solution { backbone, routers: kept });
    // A moved router may sit on a kept backbone cell: then it needs no path.
    let (on_tree, off_tree): (Vec<usize>, Vec<usize>) =
        moved.into_iter().partition(|&r| st.connected[r]);
    for r in on_tree {
        st.is_router[r] = true;
        st.sol.routers.push(r);
        st.remaining -= p.pr;
        let covered = &mut st.covered;
        p.for_each_covered(r, |j| covered[j] = true);
    }
    connect_into(&mut st, &off_tree);
    st
}

/// Greedily add routers while the budget allows. `gains` are upper bounds on
/// each cell's gain (e.g. its gain with nothing covered).
pub fn fill(st: &mut State, gains: &[u32]) {
    fill_ext(st, gains, None, None);
}

/// `fill`, optionally restricted to the candidate cells `cands` (and to
/// cells whose cost drops while filling), and with a fixed random factor in
/// [1 - eps, 1] per cell (`noise = (seed, eps)`) to randomize the greedy.
pub fn fill_ext(st: &mut State, gains: &[u32], cands: Option<&[usize]>, noise: Option<(u64, f64)>) {
    let p = st.p;
    let factor = |i: usize| match noise {
        None => 1.0,
        Some((seed, eps)) => {
            let mut x = (i as u64 ^ seed).wrapping_mul(0x9E3779B97F4A7C15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58476D1CE4E5B9);
            x ^= x >> 32;
            1.0 - eps * ((x >> 11) as f64 / (1u64 << 53) as f64)
        }
    };
    let key = |i: usize, g: u32, cost: i64| (g as f64 / cost as f64 * factor(i)).to_bits();

    let mut heap: BinaryHeap<(u64, usize)> = match cands {
        None => (0..p.h * p.w)
            .filter(|&i| gains[i] > 0 && !st.is_router[i])
            .map(|i| (key(i, gains[i], st.router_cost(i)), i))
            .collect(),
        Some(c) => c
            .iter()
            .filter(|&&i| gains[i] > 0 && !st.is_router[i])
            .map(|&i| (key(i, st.gain(i), st.router_cost(i)), i))
            .collect(),
    };
    // Cost each cell had when it was last pushed (cells never pushed keep
    // their current cost, so they are pushed only if it drops).
    let mut pushed_cost: Vec<i64> = (0..p.h * p.w).map(|i| st.router_cost(i)).collect();
    let key = |g: u32, cost: i64, i: usize| key(i, g, cost);

    while let Some((_, c)) = heap.pop() {
        if st.is_router[c] {
            continue;
        }
        let g = st.gain(c);
        let cost = st.router_cost(c);
        if g == 0 || cost > st.remaining {
            continue;
        }
        let k = key(g, cost, c);
        if let Some(&(top, _)) = heap.peek() {
            if k < top {
                heap.push((k, c));
                continue;
            }
        }
        st.place(c);
        // Their cost dropped, so their stored keys are now too low.
        for k in 0..st.improved.len() {
            let i = st.improved[k];
            let cost = st.router_cost(i);
            // Any drop matters: stale low keys hurt greedy quality noticeably.
            if gains[i] > 0 && !st.is_router[i] && cost < pushed_cost[i] {
                pushed_cost[i] = cost;
                heap.push((key(gains[i], cost, i), i));
            }
        }
    }
}
