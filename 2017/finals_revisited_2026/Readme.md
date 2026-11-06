# HashCode 2017 Finals: Router placement, revisited with Claude (Nov 4-5, 2026)

In 2017 I worked on this problem on my own, in Python, and stopped after a couple of weeks at **548.08M**, just above
the best team of the 1-day contest (AIM Tech, 548.07M; see the [final standings](https://cphof.org/standings/hashcode/2017)). In 2026 I took the problem again, this time working with Claude Opus 5.5 (High), Anthropic's model, in
Claude Code. I spent a few hours on it over two days. I mostly gave directions, pointed to my 2017 repo and its
literature, and asked questions; Claude wrote and ran the code.

The steps were roughly:
* a small validator/scorer in Python : I wanted a readable, independant thing to check score and correctness)
* a solver in Rust : this seems the most efficient and AI-coding Agent friendly language at the moment
* Claude quickly found a greedy approach plus simulated annealing: **547.8M**
* After pointing it to ideas from my 2017 work and from the Steiner tree literature: **550.3M**, then **550.9M**;
* After exact re-optimisation of small windows with an integer programming solver (HiGHS): **551.04M**.

A big part of the gain comes from speed: the solver is written in Rust, in a very optimised way, and runs on 16 cores,
so it tries far more candidates than my single-threaded Python did. Scores were still going up slowly when I stopped.

| 2026 (Claude + me)    | charleston_road | opera          | rue_de_londres | lets_go_higher | total        |
| -----                 | -----           | ------         | -------        | -----------    | -----        |
| targets covered       | 21942/21942     | 178880/196899  | 59899/64426    | 288108/288108  |              |
| num routers used      | 78              | 857            | 191            | 3535           |              |
| num backbone cells    | 859             | 9158           | 2489           | 21874          |              |
| score                 | 21,963,248      | 178,880,002    | 59,899,045     | 290,299,807    | **551,042,102** |
| my 2017 score         | ~21.96M         | ~176.37M       | ~59.50M        | ~290.24M       | 548,079,795  |

Almost all the progress is on `opera` and `rue_de_londres`, the two maps where the budget does not cover all targets.

## I. The method

### I.A. Greedy and annealing

Same start as in 2017: add routers one by one, choosing the best ratio of new targets covered over cost (router +
backbone cells to reach it). One fix mattered a lot: when the backbone grows, many cells become cheaper to connect,
and their gains must be updated, otherwise the greedy keeps stale values. Then simulated annealing moves routers
around (small shifts, sometimes a jump next to an uncovered target), with 16 independent runs in parallel.

### I.B. Backbone: a Steiner tree, not a spanning tree

Given the routers, the backbone is a Steiner tree on the grid where each cell has 8 neighbours. The construction is
Prim-like, like my "constructive" approach of 2017: connect the routers one by one, closest first, to the nearest
backbone cell, and among equally short paths prefer the one heading towards routers still waiting. Then local search
from the literature (Uchoa & Werneck, *Fast local search for Steiner trees in graphs*):
* key-path exchange: remove a branch between two branching points and reconnect the two parts more cheaply;
* key-vertex elimination: remove a branching point with its branches and reconnect the pieces;
* "rip-up and reconnect": empty a random window of the tree, reconnect, re-run the two searches, keep if shorter.

This replaced my B1S approach (24h on `opera` in Python) and runs in seconds. To check how good the trees are, I used
the Chebyshev distance trick: rotated by 45°, it becomes the Manhattan distance, so the exact rectilinear Steiner solver
GeoSteiner gives a lower bound. My trees are within 10 cells of it on `charleston_road`, 19 on `rue_de_londres`, and at
most 2% on `opera` (where the bound is weaker).

### I.C. Exact windows: maximum coverage on a subset

This is what took me from 550.9M to 551M, and it is close to the decomposition I proposed in 2017: solve (A1) the
maximum coverage problem, then (A2) the Steiner tree problem. Back then I tried to solve (A1) for the whole map at once,
and the result was too rigid to connect well.

Here, instead of solving the whole problem in two big steps, every step takes a **subset**:
* choose a random rectangle around a router (up to ~40 routers inside);
* freeze everything outside it;
* find the **optimal** placement of the same number of routers inside, i.e. the one covering the most targets that
  the outside routers leave uncovered. This is a small maximum coverage problem, written as an integer program and
  solved exactly by HiGHS (an open source solver), usually in seconds;
* rebuild the backbone around the new routers (reusing the existing tree, then the tree search of I.B), fit the budget,
  and keep the change only if the real score goes up.

So the window step is exact, and the rest of the map gives it a good context. It is a mix of my two 2017 ideas: the
"remove some routers and rebuild" improvement loop, but rebuilding optimally instead of greedily. Small windows
(6-15 routers) rarely found anything (1-3 targets now and then): the annealing had already found almost every small
improvement. Windows of 30-40 routers still solve to proven optimality and gained up to 11 targets each. I also
tried bigger windows, up to 50-60 routers: they were too slow (a minute per window on average, a third stopped by the
2-minute time limit without a proof) and did not bring enough improvement, so 30-45 routers is the sweet spot.

### I.D. What did not help

My "remove 10% and greedily rebuild" loop on top of the annealing, GPUs (the inner loops are small, sequential and
branchy), and simply running the annealing longer.

## II. Code

| | 2017 (Python) | 2026 (Rust) |
| ----- | ----- | ----- |
| solver lines of code | 2713 (4 files) | 2275 (8 files) |
| run time per map | minutes to hours (24h for B1S on `opera`) | a few minutes on 16 threads, plus ~1h of window runs on `opera` |

Plus a Python scorer that validates every output (355 lines) and small helper scripts. `./run_all.sh "<note>" <seconds>`
runs every map and keeps the best outputs; `scores.log` has the score history.

To build and run: install Rust and CMake (e.g. `pip install cmake`; the `highs` crate compiles HiGHS from source), then
`./run_all.sh "<note>" <seconds>` from this folder. Inputs are read from `../finals/input` (through the `input_files` link).
The exact window search runs with `WILP=<seconds> SEED=outputs/opera.out solver/target/release/solver
input_files/opera.in <out.out>`. The tree lower-bound tools need GeoSteiner (geosteiner.com, CC BY-NC), which is not
included: download and build it, then set `GEOSTEINER=<its folder>`.
