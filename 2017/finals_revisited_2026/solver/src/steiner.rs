//! Local search on the backbone tree for a fixed set of routers.
//!
//! The backbone is a set of cells, 8-connected, containing the start cell and
//! every router. We view it as a tree via a BFS from the start cell. Key nodes
//! are terminals (routers, start) and branch cells (2+ children). A key path
//! is the chain of non-key cells between a key node and its key ancestor.
//!
//! Move (key-path exchange): remove a key path of m cells, which splits the
//! tree into the key node's subtree and the rest; find the shortest new
//! connection between the two parts with a BFS capped at depth m; keep it if
//! it needs fewer than m new cells.

use crate::optimize::Rng;
use crate::problem::Problem;
use std::collections::VecDeque;

pub struct Tree<'a> {
    p: &'a Problem,
    pub in_tree: Vec<bool>,
    terminal: Vec<bool>,
    // BFS tree, rebuilt by `rebuild`.
    parent: Vec<u32>,
    order: Vec<usize>,
    children: Vec<Vec<usize>>,
    // Scratch arrays with generation stamps (avoid clearing).
    side: Vec<u32>,  // == side_gen: in the detached subtree
    seen: Vec<u32>,  // == bfs_gen: visited by the reconnection BFS
    from: Vec<u32>,  // BFS predecessor
    rem: Vec<u32>,   // == stamp: removed by the current key-vertex elimination
    grp: Vec<u8>,    // component id during key-vertex elimination
    stamp: u32,
    /// If set, local searches only try moves near this window (r0, r1, c0, c1).
    focus: Option<(usize, usize, usize, usize)>,
}

const NONE: u32 = u32::MAX;

impl<'a> Tree<'a> {
    pub fn new(p: &'a Problem, routers: &[usize], backbone: &[usize]) -> Tree<'a> {
        let n = p.h * p.w;
        let mut in_tree = vec![false; n];
        let mut terminal = vec![false; n];
        in_tree[p.start] = true;
        terminal[p.start] = true;
        for &c in backbone {
            in_tree[c] = true;
        }
        for &r in routers {
            assert!(in_tree[r], "router not on backbone");
            terminal[r] = true;
        }
        Tree {
            p,
            in_tree,
            terminal,
            parent: vec![NONE; n],
            order: vec![],
            children: vec![vec![]; n],
            side: vec![0; n],
            seen: vec![0; n],
            from: vec![NONE; n],
            rem: vec![0; n],
            grp: vec![0; n],
            stamp: 0,
            focus: None,
        }
    }

    fn neighbours(&self, u: usize) -> impl Iterator<Item = usize> + use<> {
        let (h, w) = (self.p.h as isize, self.p.w as isize);
        let (r, c) = ((u / self.p.w) as isize, (u % self.p.w) as isize);
        (0..9).filter_map(move |k| {
            let (nr, nc) = (r + k / 3 - 1, c + k % 3 - 1);
            if (nr, nc) == (r, c) || nr < 0 || nc < 0 || nr >= h || nc >= w {
                None
            } else {
                Some((nr * w + nc) as usize)
            }
        })
    }

    /// BFS spanning tree from the start cell; also prunes non-terminal leaves.
    fn rebuild(&mut self) {
        loop {
            for &u in &self.order {
                self.children[u].clear();
                self.parent[u] = NONE;
            }
            self.order.clear();
            let s = self.p.start;
            self.parent[s] = s as u32;
            self.order.push(s);
            let mut i = 0;
            while i < self.order.len() {
                let u = self.order[i];
                i += 1;
                for v in self.neighbours(u) {
                    if self.in_tree[v] && self.parent[v] == NONE {
                        self.parent[v] = u as u32;
                        self.children[u].push(v);
                        self.order.push(v);
                    }
                }
            }
            debug_assert!(self.order.iter().all(|&u| self.in_tree[u]));
            // Remove dangling non-terminal leaves (repeat until none).
            let mut pruned = false;
            for k in (0..self.order.len()).rev() {
                let u = self.order[k];
                if !self.terminal[u] && self.children[u].is_empty() {
                    self.in_tree[u] = false;
                    let par = self.parent[u] as usize;
                    self.children[par].retain(|&x| x != u);
                    pruned = true;
                }
            }
            if !pruned {
                break;
            }
            // Reset parents of removed cells too.
            for &u in &self.order {
                if !self.in_tree[u] {
                    self.parent[u] = NONE;
                    self.children[u].clear();
                }
            }
            self.order.retain(|&u| self.in_tree[u]);
        }
    }

    fn in_focus(&self, u: usize) -> bool {
        match self.focus {
            None => true,
            Some((r0, r1, c0, c1)) => {
                let (r, c) = (u / self.p.w, u % self.p.w);
                r >= r0 && r <= r1 && c >= c0 && c <= c1
            }
        }
    }

    fn is_key(&self, u: usize) -> bool {
        self.terminal[u] || self.children[u].len() >= 2
    }

    /// Key paths as (key node, chain cells from the key node's parent upwards).
    fn key_paths(&self) -> Vec<(usize, Vec<usize>)> {
        let mut out = vec![];
        for &k in &self.order {
            if k == self.p.start || !self.is_key(k) {
                continue;
            }
            let mut chain = vec![];
            let mut u = self.parent[k] as usize;
            while !self.is_key(u) {
                chain.push(u);
                u = self.parent[u] as usize;
            }
            if !chain.is_empty()
                && (self.in_focus(k) || chain.iter().any(|&c| self.in_focus(c)))
            {
                out.push((k, chain));
            }
        }
        out
    }

    /// Try to replace the chain above `k` with a shorter connection.
    /// Returns the number of cells saved (0 if not applied).
    fn try_exchange(&mut self, k: usize, chain: &[usize]) -> usize {
        let m = chain.len();
        // Mark the subtree of k.
        self.stamp += 1;
        let g = self.stamp;
        let mut stack = vec![k];
        let mut sub = vec![];
        while let Some(u) = stack.pop() {
            self.side[u] = g;
            sub.push(u);
            stack.extend(self.children[u].iter().copied());
        }
        let rest_size = self.order.len() - sub.len() - m;
        // BFS from the smaller side towards the other.
        let from_sub = sub.len() <= rest_size;
        let is_source = |t: &Tree, u: usize| {
            t.in_tree[u] && (t.side[u] == g) == from_sub && !(!from_sub && chain.contains(&u))
        };
        let is_goal = |t: &Tree, u: usize| {
            t.in_tree[u] && (t.side[u] == g) != from_sub && !chain.contains(&u)
        };
        let sources: Vec<usize> = if from_sub {
            sub.clone()
        } else {
            self.order.iter().copied().filter(|&u| is_source(self, u)).collect()
        };
        let mut queue = VecDeque::new();
        for &s in &sources {
            self.seen[s] = g;
            self.from[s] = NONE;
            queue.push_back((s, 0usize));
        }
        // New cells needed = BFS depth of the goal cell - 1; need < m.
        let mut found = None;
        'bfs: while let Some((u, d)) = queue.pop_front() {
            if d >= m {
                break; // reaching a goal now would need >= m new cells
            }
            for v in self.neighbours(u) {
                if self.seen[v] == g {
                    continue;
                }
                if is_goal(self, v) {
                    self.from[v] = u as u32;
                    found = Some(v);
                    break 'bfs;
                }
                // Path cells must be free cells (or the chain being removed).
                if self.in_tree[v] && !chain.contains(&v) {
                    continue;
                }
                self.seen[v] = g;
                self.from[v] = u as u32;
                queue.push_back((v, d + 1));
            }
        }
        let Some(goal) = found else { return 0 };
        // Collect the new path cells (excluding both endpoints).
        let mut path = vec![];
        let mut u = self.from[goal] as usize;
        while self.from[u] != NONE {
            path.push(u);
            u = self.from[u] as usize;
        }
        if path.len() >= m {
            return 0;
        }
        for &c in chain {
            self.in_tree[c] = false;
        }
        for &c in &path {
            self.in_tree[c] = true;
        }
        m - path.len()
    }

    /// Key-vertex elimination: remove the Steiner branch cell `v` together
    /// with its key paths (up to its key ancestor and down to each key child),
    /// then reconnect the pieces with shortest paths, smallest piece first.
    /// Applied only if it needs fewer cells than it removed; returns the
    /// number of cells saved (0 if not applied).
    fn try_kve(&mut self, v: usize) -> usize {
        self.stamp += 1;
        let g = self.stamp;
        // Removed cells: v, the chain above it, the chain below each child.
        let mut removed = vec![v];
        let mut u = self.parent[v] as usize;
        while !self.is_key(u) {
            removed.push(u);
            u = self.parent[u] as usize;
        }
        let mut roots = vec![];
        for &c in &self.children[v] {
            let mut x = c;
            while !self.is_key(x) {
                removed.push(x);
                x = self.children[x][0];
            }
            roots.push(x);
        }
        if roots.len() + 1 > 250 {
            return 0;
        }
        let m = removed.len();
        for &x in &removed {
            self.rem[x] = g;
        }
        // Pieces: 0 is the rest of the tree (with the start), 1.. the subtrees.
        let mut cells: Vec<Vec<usize>> = vec![vec![]];
        for (i, &rt) in roots.iter().enumerate() {
            let mut list = vec![];
            let mut stack = vec![rt];
            while let Some(x) = stack.pop() {
                self.side[x] = g;
                self.grp[x] = (i + 1) as u8;
                list.push(x);
                stack.extend(self.children[x].iter().copied());
            }
            cells.push(list);
        }
        let k = cells.len();
        let mut uf: Vec<usize> = (0..k).collect();
        fn find(uf: &mut [usize], mut x: usize) -> usize {
            while uf[x] != x {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            x
        }
        let mut added: Vec<usize> = vec![];
        let mut used = 0usize;
        let mut ok = true;
        loop {
            // Smallest group not yet joined to the rest.
            let r0 = find(&mut uf, 0);
            let mut best: Option<(usize, usize)> = None;
            let mut sizes = vec![0usize; k];
            for i in 1..k {
                let ri = find(&mut uf, i);
                sizes[ri] += cells[i].len();
            }
            for i in 1..k {
                let ri = find(&mut uf, i);
                if ri != r0 && best.map_or(true, |(_, s)| sizes[ri] < s) {
                    best = Some((ri, sizes[ri]));
                }
            }
            let Some((gi, _)) = best else { break };
            if used + 1 >= m {
                ok = false;
                break;
            }
            let cap = m - used - 1; // a path may add at most cap cells
            self.stamp += 1;
            let sg = self.stamp;
            let mut queue = std::collections::VecDeque::new();
            for i in 1..k {
                if find(&mut uf, i) == gi {
                    for &x in &cells[i] {
                        self.seen[x] = sg;
                        self.from[x] = NONE;
                        queue.push_back((x, 0usize));
                    }
                }
            }
            let mut found = None;
            'bfs: while let Some((x, d)) = queue.pop_front() {
                for y in self.neighbours(x) {
                    if self.seen[y] == sg {
                        continue;
                    }
                    let in_tree = self.in_tree[y] && self.rem[y] != g;
                    if in_tree {
                        let gy = if self.side[y] == g { self.grp[y] as usize } else { 0 };
                        if find(&mut uf, gy) != gi {
                            self.from[y] = x as u32;
                            found = Some((y, gy));
                            break 'bfs;
                        }
                        continue; // own group: already a source
                    }
                    if d + 1 > cap {
                        continue;
                    }
                    self.seen[y] = sg;
                    self.from[y] = x as u32;
                    queue.push_back((y, d + 1));
                }
            }
            let Some((goal, gy)) = found else {
                ok = false;
                break;
            };
            // New path cells, from the goal back to the source group.
            let mut x = self.from[goal] as usize;
            let mut path = vec![];
            while self.from[x] != NONE {
                path.push(x);
                x = self.from[x] as usize;
            }
            used += path.len();
            if used >= m {
                ok = false;
                break;
            }
            for &c in &path {
                added.push(c);
                self.in_tree[c] = true;
                self.rem[c] = 0;
                self.side[c] = g;
                self.grp[c] = gi as u8;
            }
            cells[gi].extend(path.iter().copied());
            let ry = find(&mut uf, gy);
            uf[gi] = ry;
        }
        if !ok {
            // Undo: cells added on free ground leave the tree again; removed
            // cells that were reused stay in it.
            for &c in &added {
                self.in_tree[c] = removed.contains(&c);
            }
            return 0;
        }
        for &x in &removed {
            if self.rem[x] == g {
                self.in_tree[x] = false;
            }
        }
        m - used
    }

    /// Run key-path exchange passes until a pass finds no improvement.
    /// Returns total cells saved.
    pub fn optimize(&mut self) -> usize {
        self.rebuild();
        let base = self.order.len();
        let kve = std::env::var("KVE").map_or(true, |v| v != "0");
        loop {
            let mut improved = self.kpe_pass();
            if kve {
                improved |= self.kve_pass();
            }
            if !improved {
                break;
            }
        }
        self.check();
        base - self.order.len()
    }

    /// One sweep of key-path exchange over all key paths.
    fn kpe_pass(&mut self) -> bool {
        let mut improved = false;
        let mut next_key = 0usize;
        loop {
            let mut paths = self.key_paths();
            paths.retain(|(k, _)| *k >= next_key);
            paths.sort_by_key(|(k, _)| *k);
            let mut applied = false;
            for (k, chain) in &paths {
                next_key = *k + 1;
                if self.try_exchange(*k, chain) > 0 {
                    applied = true;
                    improved = true;
                    break;
                }
            }
            if !applied {
                return improved;
            }
            self.rebuild();
        }
    }

    /// One sweep of key-vertex elimination over all Steiner branch cells.
    fn kve_pass(&mut self) -> bool {
        let mut improved = false;
        let mut next = 0usize;
        loop {
            let mut cands: Vec<usize> = self
                .order
                .iter()
                .copied()
                .filter(|&u| {
                    u >= next && !self.terminal[u] && self.children[u].len() >= 2 && self.in_focus(u)
                })
                .collect();
            cands.sort_unstable();
            let mut applied = false;
            for v in cands {
                next = v + 1;
                if self.try_kve(v) > 0 {
                    applied = true;
                    improved = true;
                    break;
                }
            }
            if !applied {
                return improved;
            }
            self.rebuild();
        }
    }

    /// Iterated local search on the tree for `secs` seconds: remove every
    /// non-terminal cell in a random window, reconnect the pieces by shortest
    /// paths (random tie-breaks), run the local searches near the window, and
    /// keep the result if the tree is no larger. Returns cells saved.
    pub fn perturb_search(&mut self, rng: &mut Rng, secs: f64) -> usize {
        let t0 = std::time::Instant::now();
        self.rebuild();
        let base = self.order.len();
        let mut cur = base;
        // Temperature in cells (0: accept only trees that are no larger).
        let temp: f64 = std::env::var("TREE_T").map_or(0.0, |v| v.parse().unwrap());
        let mut best_len = base;
        let mut best_cells: Option<Vec<usize>> = None;
        let r = self.p.r.max(3);
        while t0.elapsed().as_secs_f64() < secs {
            let saved = self.order.clone();
            let c = saved[rng.below(saved.len())];
            let (cr, cc) = (c / self.p.w, c % self.p.w);
            let rad = 2 + rng.below(2 * r);
            let win = (
                cr.saturating_sub(rad),
                (cr + rad).min(self.p.h - 1),
                cc.saturating_sub(rad),
                (cc + rad).min(self.p.w - 1),
            );
            self.focus = Some(win);
            let mut removed = 0;
            for &u in &saved {
                if !self.terminal[u] && self.in_focus(u) {
                    self.in_tree[u] = false;
                    removed += 1;
                }
            }
            let ok = removed > 0 && self.reconnect(&saved, rng, 8 * rad + 8);
            if ok {
                let m = 2 * r + 2;
                self.focus = Some((
                    win.0.saturating_sub(m),
                    (win.1 + m).min(self.p.h - 1),
                    win.2.saturating_sub(m),
                    (win.3 + m).min(self.p.w - 1),
                ));
                self.rebuild();
                self.kpe_pass();
                self.kve_pass();
                self.kpe_pass();
            }
            self.focus = None;
            self.rebuild();
            let len = self.order.len();
            let accept = ok
                && (len <= cur || (temp > 0.0 && rng.unit() < (-((len - cur) as f64) / temp).exp()));
            if accept {
                cur = len;
                if cur < best_len {
                    best_len = cur;
                    best_cells = None;
                } else if best_cells.is_none() && cur > best_len {
                    // Leaving the best tree: remember it (its cells are `saved`).
                    best_cells = Some(saved.clone());
                }
            } else {
                for &u in &self.order {
                    self.in_tree[u] = false;
                }
                for &u in &saved {
                    self.in_tree[u] = true;
                }
                self.rebuild();
            }
        }
        if let Some(cells) = best_cells {
            if cur > best_len {
                for &u in &self.order {
                    self.in_tree[u] = false;
                }
                for &u in &cells {
                    self.in_tree[u] = true;
                }
                self.rebuild();
                cur = self.order.len();
            }
        }
        self.check();
        base - cur
    }

    /// Connect every piece of the (broken) tree to the start's piece with
    /// shortest paths, smallest piece first. `cells` holds every cell that
    /// may still be in the tree. Paths are at most `cap` cells long.
    fn reconnect(&mut self, cells: &[usize], rng: &mut Rng, cap: usize) -> bool {
        // Label the pieces (flood fill over tree cells).
        self.stamp += 1;
        let g = self.stamp;
        let mut pieces: Vec<Vec<usize>> = vec![];
        let mut stack = vec![];
        let mut start_piece = 0;
        for &s in cells {
            if !self.in_tree[s] || self.side[s] == g {
                continue;
            }
            let id = pieces.len();
            let mut list = vec![];
            self.side[s] = g;
            stack.push(s);
            while let Some(x) = stack.pop() {
                self.from[x] = id as u32; // piece id (reused array)
                list.push(x);
                if x == self.p.start {
                    start_piece = id;
                }
                for y in self.neighbours(x) {
                    if self.in_tree[y] && self.side[y] != g {
                        self.side[y] = g;
                        stack.push(y);
                    }
                }
            }
            pieces.push(list);
        }
        let k = pieces.len();
        let mut uf: Vec<usize> = (0..k).collect();
        fn find(uf: &mut [usize], mut x: usize) -> usize {
            while uf[x] != x {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            x
        }
        // Piece id of every tree cell (path cells join their source piece).
        let mut id_of = std::collections::HashMap::with_capacity(cells.len());
        for (i, p) in pieces.iter().enumerate() {
            for &x in p {
                id_of.insert(x, i);
            }
        }
        let dirs: [(isize, isize); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
        let (h, w) = (self.p.h as isize, self.p.w as isize);
        loop {
            let root = find(&mut uf, start_piece);
            let mut sizes = vec![0usize; k];
            for i in 0..k {
                let ri = find(&mut uf, i);
                sizes[ri] += pieces[i].len();
            }
            let mut best: Option<usize> = None;
            for i in 0..k {
                let ri = find(&mut uf, i);
                if ri != root && best.map_or(true, |b| sizes[ri] < sizes[b]) {
                    best = Some(ri);
                }
            }
            let Some(gi) = best else { return true };
            self.stamp += 1;
            let sg = self.stamp;
            let mut queue = VecDeque::new();
            for i in 0..k {
                if find(&mut uf, i) == gi {
                    for &x in &pieces[i] {
                        self.seen[x] = sg;
                        self.from[x] = NONE;
                        queue.push_back((x, 0usize));
                    }
                }
            }
            let rot = rng.below(8);
            let mut found = None;
            'bfs: while let Some((x, d)) = queue.pop_front() {
                let (xr, xc) = ((x / self.p.w) as isize, (x % self.p.w) as isize);
                for t in 0..8 {
                    let (dr, dc) = dirs[(t + rot) % 8];
                    let (yr, yc) = (xr + dr, xc + dc);
                    if yr < 0 || yc < 0 || yr >= h || yc >= w {
                        continue;
                    }
                    let y = (yr * w + yc) as usize;
                    if self.seen[y] == sg {
                        continue;
                    }
                    if self.in_tree[y] {
                        let py = id_of[&y];
                        if find(&mut uf, py) != gi {
                            self.from[y] = x as u32;
                            found = Some((y, py));
                            break 'bfs;
                        }
                        continue;
                    }
                    if d + 1 > cap {
                        continue;
                    }
                    self.seen[y] = sg;
                    self.from[y] = x as u32;
                    queue.push_back((y, d + 1));
                }
            }
            let Some((goal, py)) = found else { return false };
            let mut x = self.from[goal] as usize;
            let mut path = vec![];
            while self.from[x] != NONE {
                path.push(x);
                x = self.from[x] as usize;
            }
            for &c in &path {
                self.in_tree[c] = true;
                id_of.insert(c, gi);
            }
            pieces[gi].extend(path);
            let ry = find(&mut uf, py);
            uf[gi] = ry;
        }
    }


    /// Every terminal must still be reachable from the start.
    fn check(&mut self) {
        self.rebuild();
        let reached = self.order.iter().filter(|&&u| self.terminal[u]).count();
        let total = self.terminal.iter().filter(|&&t| t).count();
        assert_eq!(reached, total, "backbone tree lost a terminal");
    }

    /// Backbone cells (excluding the start) in a valid connection order.
    pub fn backbone(&mut self) -> Vec<usize> {
        self.rebuild();
        self.order[1..].to_vec()
    }
}
