//! Ruin and recreate (large neighbourhood search) on complete solutions.
//!
//! Idea from the user's own Hash Code solution (remove ~10% of the routers,
//! then greedily add routers back), here mostly with spatial ruins: remove
//! every router in a random square, drop the backbone branches that no longer
//! lead to a router, and refill that area greedily. Candidates are judged on
//! the real score, with simulated-annealing acceptance.

use crate::greedy::{State, fill_ext};
use crate::optimize::{Rng, tidy_tree};
use crate::problem::{Problem, Solution};
use crate::spatial::cheb;
use std::time::Instant;

/// Remove backbone cells that no longer lead to a router. Returns the
/// remaining cells (without the start) in a valid connection (BFS) order.
pub fn prune_backbone(p: &Problem, backbone: &[usize], routers: &[usize]) -> Vec<usize> {
    let n = p.h * p.w;
    const NONE: u32 = u32::MAX;
    let mut in_tree = vec![false; n];
    let mut keep = vec![false; n];
    let mut parent = vec![NONE; n];
    in_tree[p.start] = true;
    for &c in backbone {
        in_tree[c] = true;
    }
    for &r in routers {
        keep[r] = true; // terminals
    }
    // BFS tree from the start over the backbone cells.
    let mut order = Vec::with_capacity(backbone.len() + 1);
    order.push(p.start);
    parent[p.start] = p.start as u32;
    let (h, w) = (p.h as isize, p.w as isize);
    let mut i = 0;
    while i < order.len() {
        let u = order[i];
        i += 1;
        let (ur, uc) = ((u / p.w) as isize, (u % p.w) as isize);
        for k in 0..9 {
            let (vr, vc) = (ur + k / 3 - 1, uc + k % 3 - 1);
            if vr < 0 || vc < 0 || vr >= h || vc >= w {
                continue;
            }
            let v = (vr * w + vc) as usize;
            if in_tree[v] && parent[v] == NONE {
                parent[v] = u as u32;
                order.push(v);
            }
        }
    }
    // Children come after their parents in BFS order, so one reverse pass
    // marks every cell on a path from the start to a router.
    for &u in order.iter().rev() {
        if keep[u] {
            keep[parent[u] as usize] = true;
        }
    }
    order.into_iter().skip(1).filter(|&u| keep[u]).collect()
}

pub struct Stats {
    pub iters: u64,
    pub accepted: u64,
    pub improved: u64,
}

/// Ruin and recreate from `sol` for `secs` seconds; returns the best found.
pub fn lns(
    p: &Problem,
    sol: Solution,
    gains: &[u32],
    secs: f64,
    (t0, t1): (f64, f64),
    rng: &mut Rng,
) -> (Solution, Stats) {
    let r = p.r;
    let mut cur_score = sol.score(p).0;
    let mut cur = sol;
    let mut best = cur.clone();
    let mut best_score = cur_score;
    let mut stats = Stats { iters: 0, accepted: 0, improved: 0 };
    let start = Instant::now();
    loop {
        let frac = start.elapsed().as_secs_f64() / secs;
        if frac >= 1.0 || cur.routers.is_empty() {
            break;
        }
        // Temperature in targets (score / 1000).
        let temp = t0 * (t1 / t0).powf(frac);
        stats.iters += 1;

        // Ruin: all routers in a square around a random router (shifted a
        // little), or occasionally a random 5% of all routers.
        let random_ruin = rng.below(20) == 0;
        let (kept, cands): (Vec<usize>, Option<Vec<usize>>) = if random_ruin {
            let kept = cur.routers.iter().copied().filter(|_| rng.below(20) != 0).collect();
            (kept, None)
        } else {
            let c = cur.routers[rng.below(cur.routers.len())];
            let (cr, cc) = p.rc(c);
            let rho = r + rng.below(2 * r + 1);
            let kept = cur.routers.iter().copied().filter(|&x| cheb(p.w, x, c) > rho).collect();
            // Candidate cells: the ruined square plus a router radius.
            let m = rho + r;
            let (x0, x1) = (cr.saturating_sub(m), (cr + m).min(p.h - 1));
            let (y0, y1) = (cc.saturating_sub(m), (cc + m).min(p.w - 1));
            let mut v = Vec::with_capacity((x1 - x0 + 1) * (y1 - y0 + 1));
            for x in x0..=x1 {
                for y in y0..=y1 {
                    v.push(x * p.w + y);
                }
            }
            (kept, Some(v))
        };
        let backbone = prune_backbone(p, &cur.backbone, &kept);
        let mut st = State::from_solution(p, &Solution { backbone, routers: kept });
        let noise = Some((rng.next(), 0.2));
        fill_ext(&mut st, gains, cands.as_deref(), noise);
        let mut cand = st.sol;
        let mut score = cand.score(p).0;
        // Shorten the tree only for promising candidates (it is costly), then
        // spend the budget it frees.
        if (score - cur_score) as f64 / 1000.0 > -3.0 * temp - 2.0 {
            cand = tidy_tree(p, cand);
            let mut st = State::from_solution(p, &cand);
            fill_ext(&mut st, gains, cands.as_deref(), None);
            cand = st.sol;
            score = cand.score(p).0;
        }
        let delta = (score - cur_score) as f64 / 1000.0;
        if delta >= 0.0 || rng.unit() < (delta / temp).exp() {
            stats.accepted += 1;
            cur = cand;
            cur_score = score;
            if cur_score > best_score {
                stats.improved += 1;
                best = cur.clone();
                best_score = cur_score;
            }
        }
    }
    (best, stats)
}
