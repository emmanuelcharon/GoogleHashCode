//! Exact re-placement of the routers in a window, with an integer program
//! solved by HiGHS.
//!
//! Pick a random rectangle. Keep every router outside it fixed. Choose the
//! same number of routers inside it so as to cover as many as possible of the
//! targets the outside routers leave uncovered (a maximum coverage problem):
//!
//!   maximise  sum_t y_t
//!   subject to  y_t <= sum of x_c over the cells c whose square covers t
//!               sum_c x_c = k          (k = routers now in the window)
//!               x_c in {0, 1},  0 <= y_t <= 1
//!
//! The backbone is ignored here. A placement that covers more is turned into
//! a full solution by `complete` (reusing the current tree, fitting the
//! budget) and a short tree polish, and kept only if the real score rises.
//!
//! With `cover` (for fully covered maps) the program instead finds the fewest
//! routers that still cover every target the window's routers cover now.

use crate::optimize::{self, Rng};
use crate::problem::{Problem, Solution, WALL};
use highs::{ColProblem, HighsModelStatus, Sense};
use std::time::Instant;

/// A window and what the integer program found in it.
pub struct WindowResult {
    /// Targets gained (or, with `cover`, routers saved) by the program.
    pub gain: i64,
    /// The full router set with the window's routers replaced.
    pub routers: Vec<usize>,
    /// Routers in the window, and whether the answer is proven optimal.
    pub k: usize,
    pub optimal: bool,
    /// Time spent building and solving the program.
    pub secs: f64,
}

pub struct Params {
    pub win_min: usize,
    pub win_max: usize,
    pub time_limit: f64,
    pub cover: bool,
    pub workers: usize,
    pub polish_secs: f64,
    /// Stop starting new batches after this many seconds.
    pub max_secs: f64,
    /// Skip windows with fewer or more routers than this.
    pub min_k: usize,
    pub max_k: usize,
}

/// The integer program for one window (r0, r1, c0, c1), inclusive.
pub fn solve_window(p: &Problem, sol: &Solution, win: (usize, usize, usize, usize), prm: &Params) -> Option<WindowResult> {
    let (r0, r1, c0, c1) = win;
    let w = p.w;
    let in_win = |x: usize| {
        let (r, c) = p.rc(x);
        r >= r0 && r <= r1 && c >= c0 && c <= c1
    };
    let (inside, outside): (Vec<usize>, Vec<usize>) = sol.routers.iter().partition(|&&x| in_win(x));
    let k = inside.len();
    if k < prm.min_k.max(2) || k > prm.max_k {
        return None;
    }
    let started = Instant::now();
    // Coverage by the outside routers, only where it matters: routers that
    // can reach the window's cells (within 2R of the window).
    let reach = 2 * p.r;
    let near = |x: usize| {
        let (r, c) = p.rc(x);
        r + reach >= r0 && r <= r1 + reach && c + reach >= c0 && c <= c1 + reach
    };
    let mut cnt = std::collections::HashMap::<usize, u16>::new();
    for &x in outside.iter().filter(|&&x| near(x)) {
        p.for_each_covered(x, |j| *cnt.entry(j).or_insert(0) += 1);
    }
    let free = |j: usize| !cnt.contains_key(&j);

    // Candidate cells and the free targets each would cover.
    let mut tid = std::collections::HashMap::<usize, usize>::new();
    let mut cands: Vec<(usize, Vec<usize>)> = vec![];
    for r in r0..=r1 {
        for c in c0..=c1 {
            let x = r * w + c;
            if p.grid[x] == WALL {
                continue;
            }
            let mut list = vec![];
            p.for_each_covered(x, |j| {
                if free(j) {
                    let n = tid.len();
                    list.push(*tid.entry(j).or_insert(n));
                }
            });
            if !list.is_empty() {
                cands.push((x, list));
            }
        }
    }
    if cands.len() < k {
        return None;
    }
    let nt = tid.len();
    // What the current routers cover of those targets.
    let mut cur = vec![false; nt];
    for &x in &inside {
        p.for_each_covered(x, |j| {
            if let Some(&t) = tid.get(&j) {
                cur[t] = true;
            }
        });
    }
    let cur_cov = cur.iter().filter(|&&b| b).count() as i64;

    let mut pb = ColProblem::default();
    let chosen: Vec<usize>;
    let optimal;
    let gain;
    if !prm.cover {
        let rows: Vec<_> = (0..nt).map(|_| pb.add_row(..=0.0)).collect();
        let card = pb.add_row(k as f64..=k as f64);
        for (_, list) in &cands {
            let factors = list.iter().map(|&t| (rows[t], -1.0)).chain(std::iter::once((card, 1.0)));
            pb.add_integer_column(0.0, 0.0..=1.0, factors);
        }
        for &row in &rows {
            pb.add_column(1.0, 0.0..=1.0, [(row, 1.0)]);
        }
        let mut model = pb.optimise(Sense::Maximise);
        model.make_quiet();
        model.set_option("time_limit", prm.time_limit);
        model.set_option("threads", 1i32);
        let solved = model.solve();
        optimal = solved.status() == HighsModelStatus::Optimal;
        let x = solved.get_solution();
        let cols = x.columns();
        if cols.len() < cands.len() {
            return None;
        }
        chosen = (0..cands.len()).filter(|&i| cols[i] > 0.5).map(|i| cands[i].0).collect();
        if chosen.len() != k {
            return None;
        }
        let mut covered = vec![false; nt];
        for i in (0..cands.len()).filter(|&i| cols[i] > 0.5) {
            for &t in &cands[i].1 {
                covered[t] = true;
            }
        }
        gain = covered.iter().filter(|&&b| b).count() as i64 - cur_cov;
    } else {
        // Every target covered now must stay covered; use as few routers as possible.
        let rows: Vec<Option<highs::Row>> = (0..nt).map(|t| cur[t].then(|| pb.add_row(1.0..))).collect();
        if rows.iter().all(|r| r.is_none()) {
            return None;
        }
        for (_, list) in &cands {
            let factors: Vec<_> = list.iter().filter_map(|&t| rows[t].map(|r| (r, 1.0))).collect();
            pb.add_integer_column(1.0, 0.0..=1.0, factors);
        }
        let mut model = pb.optimise(Sense::Minimise);
        model.make_quiet();
        model.set_option("time_limit", prm.time_limit);
        model.set_option("threads", 1i32);
        let solved = model.solve();
        optimal = solved.status() == HighsModelStatus::Optimal;
        let x = solved.get_solution();
        let cols = x.columns();
        if cols.len() < cands.len() {
            return None;
        }
        chosen = (0..cands.len()).filter(|&i| cols[i] > 0.5).map(|i| cands[i].0).collect();
        // A time-limited answer may not cover everything: check.
        let mut covered = vec![false; nt];
        for &x in &chosen {
            p.for_each_covered(x, |j| {
                if let Some(&t) = tid.get(&j) {
                    covered[t] = true;
                }
            });
        }
        if (0..nt).any(|t| cur[t] && !covered[t]) {
            return None;
        }
        gain = k as i64 - chosen.len() as i64;
    }
    let mut routers = outside;
    routers.extend(chosen);
    Some(WindowResult { gain, routers, k, optimal, secs: started.elapsed().as_secs_f64() })
}

/// Random windows from `start` for `prm.max_secs` seconds, on `prm.workers`
/// threads that never wait for each other: each one takes the current best,
/// solves a window, and if the program gains, rebuilds the full solution on
/// the latest best and keeps it when the real score is higher. Writes every
/// new best to `out`; returns the best solution.
pub fn run(p: &Problem, start: Solution, gains: &[u32], prm: &Params, out: &str, rng: &mut Rng) -> Solution {
    use std::sync::{Arc, Mutex};
    let t0 = Instant::now();
    let score0 = start.score(p).0;
    eprintln!("start: score {score0}, {} workers", prm.workers);
    let best = Mutex::new((score0, Arc::new(start)));
    // (windows solved, proven optimal, solve times, programs with a gain)
    let stats = Mutex::new((0usize, 0usize, Vec::<f64>::new(), 0usize));
    let seed = rng.next();
    std::thread::scope(|scope| {
        for wk in 0..prm.workers {
            let (best, stats) = (&best, &stats);
            scope.spawn(move || {
                let mut rng = Rng::new(seed ^ (wk as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
                while t0.elapsed().as_secs_f64() < prm.max_secs {
                    let base = best.lock().unwrap().1.clone();
                    // A window around a random router, with min_k..=max_k routers.
                    let mut win = None;
                    for _ in 0..50 {
                        let (cr, cc) = p.rc(base.routers[rng.below(base.routers.len())]);
                        let span = prm.win_max - prm.win_min + 1;
                        let (hr, hc) = (prm.win_min + rng.below(span), prm.win_min + rng.below(span));
                        let wnd = (cr.saturating_sub(hr), (cr + hr).min(p.h - 1), cc.saturating_sub(hc), (cc + hc).min(p.w - 1));
                        let k = base.routers.iter().filter(|&&x| in_window(p, x, wnd)).count();
                        if k >= prm.min_k && k <= prm.max_k {
                            win = Some(wnd);
                            break;
                        }
                    }
                    let Some(wnd) = win else { continue };
                    let Some(r) = solve_window(p, &base, wnd, prm) else { continue };
                    {
                        let mut s = stats.lock().unwrap();
                        s.0 += 1;
                        s.1 += r.optimal as usize;
                        s.2.push(r.secs);
                        s.3 += (r.gain > 0) as usize;
                    }
                    if r.gain <= 0 {
                        continue;
                    }
                    // Apply the window's new routers to the latest best.
                    let cur = best.lock().unwrap().1.clone();
                    let mut routers: Vec<usize> = cur.routers.iter().copied().filter(|&x| !in_window(p, x, wnd)).collect();
                    routers.extend(r.routers.iter().copied().filter(|&x| in_window(p, x, wnd)));
                    let cand = optimize::complete(p, &routers, gains, Some(&cur));
                    let cand = optimize::polish(p, &cand, gains, prm.polish_secs, rng.next());
                    let s = cand.score(p).0;
                    let mut g = best.lock().unwrap();
                    if s > g.0 {
                        eprintln!(
                            "[{:6.1}s] window of {} routers, program gain {} -> new best {s} (+{})",
                            t0.elapsed().as_secs_f64(),
                            r.k,
                            r.gain,
                            s - g.0
                        );
                        cand.write(p, out);
                        *g = (s, Arc::new(cand));
                    }
                }
            });
        }
    });
    let (solved, proven, mut times, with_gain) = stats.into_inner().unwrap();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times.get(times.len() / 2).copied().unwrap_or(0.0);
    let max = times.last().copied().unwrap_or(0.0);
    let total: f64 = times.iter().sum();
    eprintln!(
        "{solved} windows solved ({proven} proven optimal, {with_gain} with a gain); solve median {med:.1}s, max {max:.1}s; {:.0}% of worker time in solves",
        100.0 * total / (prm.workers as f64 * t0.elapsed().as_secs_f64())
    );
    let (_, sol) = best.into_inner().unwrap();
    Arc::try_unwrap(sol).unwrap_or_else(|a| (*a).clone())
}

fn in_window(p: &Problem, x: usize, (r0, r1, c0, c1): (usize, usize, usize, usize)) -> bool {
    let (r, c) = p.rc(x);
    r >= r0 && r <= r1 && c >= c0 && c <= c1
}
