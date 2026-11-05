//! Local search on router positions, then rebuild a valid solution.

use crate::greedy::{State, connect_all, connect_from, fill};
use crate::steiner::Tree;
use crate::problem::{Problem, Solution, TARGET, WALL};
use crate::spatial::Buckets;
use std::time::Instant;

pub struct Rng(u64);

/// Per-run salt mixed into every seed, so repeated runs from the same
/// solution explore differently. Set SALT to reproduce a run.
fn salt() -> u64 {
    static SALT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SALT.get_or_init(|| {
        std::env::var("SALT").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        })
    })
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = seed ^ salt().wrapping_mul(0xD1B54A32D192ED03);
        Rng(s.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Simulated annealing on router positions with the router count fixed.
/// Objective: number of covered targets. Each move shifts one router by at
/// most `d` cells in each direction. The backbone is ignored here and rebuilt
/// afterwards by `complete`.
pub fn anneal(
    p: &Problem,
    routers: &mut [usize],
    secs: f64,
    d: usize,
    (t0, t1): (f64, f64),
    lambda: f64,
    rng: &mut Rng,
) {
    if routers.is_empty() {
        return;
    }
    let n = p.h * p.w;
    let mut cnt = vec![0u16; n];
    let mut is_router = vec![false; n];
    for &r in routers.iter() {
        is_router[r] = true;
        p.for_each_covered(r, |j| cnt[j] += 1);
    }
    // `lambda`: covered cells one budget unit is worth.
    let lambda = lambda * p.pb as f64;
    let mut buckets = Buckets::new(p, 2 * p.r + 1, Some(p.start));
    for &r in routers.iter() {
        buckets.insert(r);
    }
    let targets: Vec<usize> = (0..n).filter(|&i| p.grid[i] == TARGET).collect();
    let start = Instant::now();
    let mut temp = t0;
    let d = d as isize;
    let mut iter = 0u64;
    loop {
        if iter % 256 == 0 {
            let frac = start.elapsed().as_secs_f64() / secs;
            if frac >= 1.0 {
                break;
            }
            temp = t0 * (t1 / t0).powf(frac);
        }
        iter += 1;
        let i = rng.below(routers.len());
        let a = routers[i];
        // Usually a local shift; sometimes a teleport next to an uncovered
        // target, so routers can leave over-covered areas.
        let (cr, cc, dd) = if rng.below(10) == 0 {
            let t = targets[rng.below(targets.len())];
            if cnt[t] > 0 {
                continue;
            }
            let (tr, tc) = p.rc(t);
            (tr, tc, p.r as isize)
        } else {
            let (ar, ac) = p.rc(a);
            (ar, ac, d)
        };
        let br = cr as isize + rng.below(2 * dd as usize + 1) as isize - dd;
        let bc = cc as isize + rng.below(2 * dd as usize + 1) as isize - dd;
        if br < 0 || bc < 0 || br >= p.h as isize || bc >= p.w as isize {
            continue;
        }
        let b = br as usize * p.w + bc as usize;
        if is_router[b] || p.grid[b] == WALL {
            continue;
        }
        // Backbone proxy: distance to the nearest other router (or start).
        let wire = buckets.nearest(b, a) as f64 - buckets.nearest(a, a) as f64;
        let mut loss = 0i32;
        p.for_each_covered(a, |j| {
            cnt[j] -= 1;
            loss += (cnt[j] == 0) as i32;
        });
        let mut gain = 0i32;
        p.for_each_covered(b, |j| gain += (cnt[j] == 0) as i32);
        let delta = (gain - loss) as f64 - lambda * wire;
        if delta >= 0.0 || rng.unit() < (delta / temp).exp() {
            p.for_each_covered(b, |j| cnt[j] += 1);
            buckets.remove(a);
            buckets.insert(b);
            is_router[a] = false;
            is_router[b] = true;
            routers[i] = b;
        } else {
            p.for_each_covered(a, |j| cnt[j] += 1);
        }
    }
}

/// Turn a set of router positions into a valid solution: build the backbone
/// (Prim-style Steiner tree, then key-path exchange), drop the least useful
/// routers while over budget, then greedily spend whatever budget is left,
/// re-optimizing the tree while that frees enough budget for another router.
pub fn complete(p: &Problem, routers: &[usize], gains: &[u32], base: Option<&Solution>) -> Solution {
    let mut routers = prune(p, routers);
    loop {
        let mut sol = tidy_tree(p, connect_all(p, &routers).sol);
        // Also try reusing the tree of `base` (e.g. the polished best), and
        // keep the shorter of the two.
        if let Some(b) = base {
            let alt = tidy_tree(p, connect_from(p, b, &routers).sol);
            if alt.backbone.len() < sol.backbone.len() {
                sol = alt;
            }
        }
        if sol.cost(p) <= p.budget {
            let mut sol = sol;
            for _ in 0..4 {
                let mut st = State::from_solution(p, &sol);
                fill(&mut st, gains);
                let before = st.sol.backbone.len();
                sol = tidy_tree(p, st.sol);
                let saved = before - sol.backbone.len();
                if (saved as i64) * p.pb < p.pr {
                    break;
                }
            }
            return sol;
        }
        // Over budget: remove the routers with the least unique coverage.
        // Each removal saves at least one router price.
        let k = ((sol.cost(p) - p.budget) as usize).div_ceil(p.pr as usize).max(1);
        let mut cnt = vec![0u16; p.h * p.w];
        for &r in &routers {
            p.for_each_covered(r, |j| cnt[j] += 1);
        }
        let mut by_loss: Vec<(u32, usize)> = routers
            .iter()
            .map(|&r| {
                let mut l = 0;
                p.for_each_covered(r, |j| l += (cnt[j] == 1) as u32);
                (l, r)
            })
            .collect();
        by_loss.sort_unstable();
        routers = by_loss[k.min(by_loss.len())..].iter().map(|&(_, r)| r).collect();
    }
}

/// Shorten a solution's backbone with key-path exchange.
pub fn tidy_tree(p: &Problem, sol: Solution) -> Solution {
    let mut tree = Tree::new(p, &sol.routers, &sol.backbone);
    tree.optimize();
    Solution { backbone: tree.backbone(), routers: sol.routers }
}

/// Remove routers that cover nothing on their own, least useful first.
pub fn prune(p: &Problem, routers: &[usize]) -> Vec<usize> {
    let mut cnt = vec![0u16; p.h * p.w];
    for &r in routers {
        p.for_each_covered(r, |j| cnt[j] += 1);
    }
    let unique = |cnt: &[u16], r: usize| {
        let mut l = 0;
        p.for_each_covered(r, |j| l += (cnt[j] == 1) as u32);
        l
    };
    let mut order: Vec<(u32, usize)> = routers.iter().map(|&r| (unique(&cnt, r), r)).collect();
    order.sort_unstable();
    let mut keep = vec![];
    for (_, r) in order {
        if unique(&cnt, r) == 0 {
            p.for_each_covered(r, |j| cnt[j] -= 1);
        } else {
            keep.push(r);
        }
    }
    keep
}

/// Tree-only iterated local search on a fixed router set, then spend the
/// budget it frees on more routers.
pub fn polish(p: &Problem, sol: &Solution, gains: &[u32], secs: f64, seed: u64) -> Solution {
    let mut tree = Tree::new(p, &sol.routers, &sol.backbone);
    let mut rng = Rng::new(seed);
    tree.perturb_search(&mut rng, secs);
    let sol = Solution { backbone: tree.backbone(), routers: sol.routers.clone() };
    let mut st = State::from_solution(p, &sol);
    fill(&mut st, gains);
    tidy_tree(p, st.sol)
}

/// Remove the `k` routers with the least unique coverage.
pub fn drop_least_useful(p: &Problem, routers: &[usize], k: usize) -> Vec<usize> {
    let mut cnt = vec![0u16; p.h * p.w];
    for &r in routers {
        p.for_each_covered(r, |j| cnt[j] += 1);
    }
    let mut by_loss: Vec<(u32, usize)> = routers
        .iter()
        .map(|&r| {
            let mut l = 0;
            p.for_each_covered(r, |j| l += (cnt[j] == 1) as u32);
            (l, r)
        })
        .collect();
    by_loss.sort_unstable();
    by_loss[k.min(by_loss.len())..].iter().map(|&(_, r)| r).collect()
}

/// `optimize::polish` from `sol` in `threads` threads; the best result.
pub fn polish_parallel(p: &Problem, sol: &Solution, gains: &[u32], secs: f64, threads: usize, seed: u64) -> Solution {
    let cands: Vec<Solution> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|k| scope.spawn(move || polish(p, sol, gains, secs, seed * 1000 + k as u64 + 7)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    cands.into_iter().max_by_key(|c| c.score(p).0).unwrap()
}
