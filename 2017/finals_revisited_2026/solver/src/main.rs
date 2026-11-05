mod greedy;
mod lns;
mod optimize;
mod problem;
mod spatial;
mod steiner;
mod wilp;

use problem::{Problem, Solution};
use std::time::Instant;

fn report(p: &Problem, sol: &Solution, label: &str, t0: &Instant) {
    let (score, covered) = sol.score(p);
    eprintln!(
        "  [{:6.1}s] {label}: score {score} | covered {covered}/{} | routers {} backbone {} | cost {}/{}",
        t0.elapsed().as_secs_f64(),
        p.num_targets(),
        sol.routers.len(),
        sol.backbone.len(),
        sol.cost(p),
        p.budget,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: solver <problem.in> <solution.out> [seconds=10]");
        std::process::exit(2);
    }
    let secs: f64 = args.get(3).map_or(10.0, |s| s.parse().unwrap());
    let t0 = Instant::now();
    let p = Problem::read(&args[1]);
    eprintln!("{}", args[1]);
    if std::env::var("CHECK").is_ok() {
        p.self_check();
        return;
    }

    let gains = greedy::initial_gains(&p);
    // WILP=<seconds> with SEED=<solution>: exact window re-placement of the
    // routers with HiGHS (see wilp.rs). Writes the best solution found.
    if let Ok(v) = std::env::var("WILP") {
        let seed = Solution::read(&p, &std::env::var("SEED").unwrap());
        let threads = std::thread::available_parallelism().map_or(4, |x| x.get());
        let prm = wilp::Params {
            win_min: env_f0("WIN_MIN", 32.0) as usize,
            win_max: env_f0("WIN_MAX", 55.0) as usize,
            time_limit: env_f0("ILP_SECS", 120.0),
            cover: std::env::var("COVER").is_ok(),
            workers: env_f0("WORKERS", threads as f64) as usize,
            polish_secs: env_f0("POLISH_SECS", 2.0),
            max_secs: v.parse().unwrap(),
            min_k: env_f0("MIN_K", 10.0) as usize,
            max_k: env_f0("MAX_K", 60.0) as usize,
        };
        let mut rng = optimize::Rng::new(5);
        let best = wilp::run(&p, seed, &gains, &prm, &args[2], &mut rng);
        report(&p, &best, "best", &t0);
        best.write(&p, &args[2]);
        return;
    }
    if let Ok(v) = std::env::var("TREE_ILS") {
        let secs: f64 = v.parse().unwrap();
        let seed = Solution::read(&p, &std::env::var("SEED").unwrap());
        let tidy = optimize::tidy_tree(&p, seed.clone());
        report(&p, &tidy, "seed + tidy", &t0);
        let mut tree = steiner::Tree::new(&p, &tidy.routers, &tidy.backbone);
        let mut rng = optimize::Rng::new(1);
        let saved = tree.perturb_search(&mut rng, secs);
        let sol = Solution { backbone: tree.backbone(), routers: tidy.routers.clone() };
        report(&p, &sol, &format!("after tree ILS ({saved} saved)"), &t0);
        sol.write(&p, &args[2]);
        return;
    }
    if std::env::var("DIAG").is_ok() {
        diag(&p, &gains, &std::env::var("SEED").unwrap());
        return;
    }
    let mut best = greedy::solve(&p, &gains);
    report(&p, &best, "greedy", &t0);
    // Optionally start from an existing (valid) solution instead.
    if let Ok(path) = std::env::var("SEED") {
        let seed = Solution::read(&p, &path);
        report(&p, &seed, "seed", &t0);
        let rebuilt = greedy::connect_all(&p, &seed.routers).sol;
        report(&p, &rebuilt, "seed routers, my backbone", &t0);
        let tidy = optimize::tidy_tree(&p, Solution { backbone: seed.backbone.clone(), routers: seed.routers.clone() });
        report(&p, &tidy, "seed + key-path exchange", &t0);
        let completed = optimize::complete(&p, &seed.routers, &gains, None);
        report(&p, &completed, "seed routers, complete()", &t0);
        for cand in [seed, tidy, completed] {
            if cand.score(&p).0 > best.score(&p).0 {
                best = cand;
            }
        }
    }
    let mut best_score = best.score(&p).0;

    // Backbone penalty weight for annealing. When greedy already covers every
    // target the budget is not binding, so wire is (almost) free.
    let full = best.score(&p).1 == p.num_targets();
    let lambda: f64 = std::env::var("LAMBDA")
        .map_or(if full { 0.0 } else { 0.3 }, |v| v.parse().unwrap());
    let env_f = |k: &str, d: f64| std::env::var(k).map_or(d, |v| v.parse().unwrap());
    let use_lns = std::env::var("LNS").is_ok();
    let lns_temps = (env_f("LNS_T0", 1.0), env_f("LNS_T1", 0.05));
    let threads = std::thread::available_parallelism().map_or(4, |x| x.get());
    // The last part of the time goes to the tree-only search (`polish`).
    let polish_frac = env_f("POLISH", 0.25);
    let search_secs = secs * (1.0 - polish_frac);
    // Share of each round spent polishing its best candidate's tree.
    let round_polish = env_f("ROUND_POLISH", 0.2);
    let rounds = 5;
    for round in 0..rounds {
        let remaining = search_secs - t0.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            break;
        }
        let t = remaining / (rounds - round) as f64;
        let t_sa = t * (1.0 - round_polish);
        // The last round is a near-greedy hill-climb from the best solution.
        let temps = if round + 1 == rounds { (0.05, 0.01) } else { (2.0, 0.05) };
        // Independent chains from the current best, one per thread.
        let cands: Vec<Solution> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|k| {
                    let (p, gains, start, best_ref) = (&p, &gains, &best.routers, &best);
                    scope.spawn(move || {
                        let mut rng = optimize::Rng::new((round * 1000 + k + 1) as u64);
                        if use_lns {
                            let (s, st) = lns::lns(p, best_ref.clone(), gains, t_sa, lns_temps, &mut rng);
                            if k == 0 {
                                eprintln!("    lns thread 0: {} iters, {} accepted, {} improved", st.iters, st.accepted, st.improved);
                            }
                            return s;
                        }
                        let mut routers = start.clone();
                        // Full coverage: the only gain left is cost, so start
                        // each chain with a few routers fewer (least unique
                        // coverage first) and let the annealing re-cover.
                        if full && k > 0 {
                            let drop = 1 + rng.below((routers.len() / 100).max(1));
                            routers = optimize::drop_least_useful(p, &routers, drop);
                        }
                        optimize::anneal(p, &mut routers, t_sa, p.r, temps, lambda, &mut rng);
                        optimize::complete(p, &routers, gains, Some(best_ref))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut cand = cands.into_iter().max_by_key(|c| c.score(&p).0).unwrap();
        report(&p, &cand, &format!("round {round}"), &t0);
        // The best solution had its tree polished, the candidate did not:
        // polish the candidate too before comparing.
        if cand.score(&p).0 <= best_score && round_polish > 0.0 {
            cand = optimize::polish_parallel(&p, &cand, &gains, t * round_polish, threads, round as u64);
            report(&p, &cand, &format!("round {round} polished"), &t0);
        }
        let s = cand.score(&p).0;
        if s > best_score {
            best = cand;
            best_score = s;
        }
    }
    // Tree-only iterated local search in parallel; each saved budget unit
    // may let `fill` add a router. Two passes, half the time each.
    for pass in 0..2 {
        let remaining = secs - t0.elapsed().as_secs_f64();
        if remaining <= 0.5 {
            break;
        }
        let t = remaining / (2 - pass) as f64;
        let cand = optimize::polish_parallel(&p, &best, &gains, t, threads, 100 + pass as u64);
        report(&p, &cand, &format!("polish {pass}"), &t0);
        if cand.score(&p).0 > best_score {
            best_score = cand.score(&p).0;
            best = cand;
        }
    }
    report(&p, &best, "best", &t0);
    best.write(&p, &args[2]);
}

/// Timing of the completion phases and the value of the marginal routers.
fn diag(p: &Problem, gains: &[u32], seed_path: &str) {
    let seed = Solution::read(p, seed_path);
    let t = Instant::now();
    let st = greedy::connect_all(p, &seed.routers);
    eprintln!("connect_all: {:.1}ms, backbone {}", t.elapsed().as_secs_f64() * 1e3, st.sol.backbone.len());
    let t = Instant::now();
    let tidy = optimize::tidy_tree(p, st.sol);
    eprintln!("tidy_tree: {:.1}ms, backbone {}", t.elapsed().as_secs_f64() * 1e3, tidy.backbone.len());
    let t = Instant::now();
    let tidy2 = optimize::tidy_tree(p, Solution { backbone: seed.backbone.clone(), routers: seed.routers.clone() });
    eprintln!("tidy_tree on seed: {:.1}ms, backbone {} -> {}", t.elapsed().as_secs_f64() * 1e3, seed.backbone.len(), tidy2.backbone.len());
    let t = Instant::now();
    let mut s2 = greedy::State::from_solution(p, &tidy2);
    eprintln!("from_solution: {:.1}ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    greedy::fill(&mut s2, gains);
    eprintln!("fill: {:.1}ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let c = optimize::complete(p, &seed.routers, gains, None);
    eprintln!("complete: {:.1}ms score {} (seed {})", t.elapsed().as_secs_f64() * 1e3, c.score(p).0, seed.score(p).0);
    // Unique coverage of each router: what removing it alone would lose.
    let mut cnt = vec![0u16; p.h * p.w];
    for &r in &seed.routers {
        p.for_each_covered(r, |j| cnt[j] += 1);
    }
    let mut uniq: Vec<u32> = seed.routers.iter().map(|&r| {
        let mut l = 0;
        p.for_each_covered(r, |j| l += (cnt[j] == 1) as u32);
        l
    }).collect();
    uniq.sort_unstable();
    eprintln!("lowest unique coverage: {:?}", &uniq[..uniq.len().min(30)]);
    eprintln!("median unique coverage: {}", uniq[uniq.len() / 2]);
    // Best gain of adding one more router, ignoring budget, by wire distance.
    let st = greedy::State::from_solution(p, &seed);
    let mut best: Vec<(f64, u32, i64)> = (0..p.h * p.w)
        .filter(|&i| gains[i] > 0 && !st.is_router[i])
        .map(|i| { let g = st.gain(i); let c = st.router_cost(i); (g as f64 / c as f64, g, c) })
        .collect();
    best.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eprintln!("best additions (gain, cost): {:?}", best.iter().take(10).map(|x| (x.1, x.2)).collect::<Vec<_>>());
    eprintln!("remaining budget {}", p.budget - seed.cost(p));
}


fn env_f0(k: &str, d: f64) -> f64 {
    std::env::var(k).map_or(d, |v| v.parse().unwrap())
}
