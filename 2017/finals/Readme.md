# HashCode 2017 Finals: Router placement.

Code is in `finals2017_routers.py`.

I set out to work on this problem to improve my programming and algorithmic skills. The objective was to beat the score of the best performing team of the contest: just above 548M.

One of my main motivations is to get better at solving these integer programming problems. I do love the hashcode competition, because I like this type of problem and because of the short timeframe given to solve them. This allows to participate without sacrificing too much time. I've participated in a few Kaggle contests, but my experience is that the score difference between top performing teams is so thin that the final ranking is quite random, for algorithms which perform evenly (machine learning is statistical after all).

## I. Greedy approach with basic random exploration

In a limited time frame such as a few hours or one day, it is rare to be able to formulate the problem in rigorous terms and to find an optimal or provably good solution to a problem. So I started out with just common sense, and tried to come up with a good greedy solution to the problem.

### I.A. Greedy Idea

Here is a very simple greedy approach idea:
* For each potential router cell in the grid compute the score gain if we added a router to it (and the cost to connect it to the backbone).
* Choose the potential router cell with the best gain, add the corresponding backbone cells and router.
* Repeat this until we cannot add any router or until no router addition improves the score.

### I.B. Implementation

In order to implement this idea, we need to be able to:
* read the problem statement
* add backbones, add routers
* write our solution
* find/compute the backbones required to connect a cell
* find/compute the gain if a router is added to a cell

There are many ways to implement these requirements. My advice is to go for an object oriented approach: I used a class
`Building` and a class `Cell` to represent the two main objects. I advise against implementing everything as array for
speed reasons (in python for instance, you could implement every field of a cell as a numpy.array of the size of the grid):
this gets confusing quickly and we can always improve performance in classes after we found a decent solution.

You can check the implementation details in `routers_basic.py`, in method `Greedy.greedy_solve`. Here are some implementation remarks.

In this problem, one can see that adding a router on a cell only has an impact for the score of a nearby geographic are
(cells within distance `2R`). However, adding a backbone cell potentially has effects on a large part of the building.
So I chose to update the scores only in an area close to the newly added router. Hence the gains we compute are approximate,
because the not-updated gains vary a bit compared to their true value.

One thing that accelerated the program a lot was to cache, for each potential router cell A,
the set of targets that would be covered if a router was placed on A.
You can also see in the code a naive `O(R^4)` and a linear programming approach `O(R^2)` to compute this set for each cell.
Since this step actually took a substantial amount of time and never changed
(it follow directly from the problem statement), we saved the result into files so we
just need to read these files and not do the computation every time
(see the folder `input/covered_targets` and functions Utils.).

### I.C. Variation, random improvements, and router swaps

#### Variations

There are always possible variations on greedy methods, and in this case, we can for instance think of a way
to differentiate 2 router candidates that have the same score. I added a simple variation: take the router with
maximum gain per budget used instead of simply the candidate with the best gain. This led to minor score improvements
but selected solutions with more routers.

#### Random improvements

In general, there is no guarantee that greedy algorithms will find a good solution
(i.e. a solution with score close to the global maximum). Here is an easy and general way to improve a
greedy solution (but not always possible): remove random parts of the solution, then use that as a
starting point to reconstruct a greedy solution. This allows to explore neighbor solutions and usually to
improve the score of the solution.

This is applicable here: we will repeatedly remove about 10% of the routers (and associated backbones),
and then add routers like in the simple greedy approach. See method: `Greedy.greedy_solve_with_random_improvements`.

In order to do so, we implemented methods to:
* remove routers, remove backbones
* backtrack a backbone line to remove (backtrack until we find an intersection, another router or the initial backbone)

This approach has the advantage of exploring the set of solutions varying all parameters:
the router positions, the backbone positions, and also the number of routers used.

Remark that sometimes, the remaining budget goes below 0, because the distance to backbone is approximate.
This usually does not last long and the solutions become legal again after a few iterations.

#### Swaps

Another way to improve our greedy solution is to change router positions until there is no gain in moving
any router to any position.

This is implemented in `Greedy.swap_until_local_maximum`. It uses functions already implemented,
so it was quite fast to code. Here is the pseudo code:
```
while any router moved:
  for each router:
    remove router from its position
    add the router on the best possible spot of the grid
```

### I.D. Analysing results

The results for our greedy (but carefully implemented) approach are:

| Greedy                   | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                    | -------- | -----           | ------         | -------            | -----------          |
| targets covered          | 54/66    | 21942/21942     |  171696/196899 | 57004/64426        | 288108/288108        |
| score in millions        | ~0.054M  | ~21.96M         |  ~171.70M      | ~57.01M            | ~290.10M             |

Greedy approach total score: **540.77M**.


The results for the greedy solution transformed with 10 to 100 loops of random improvements, followed by swaps, are:

| Greedy + random impr. and swaps   | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                             | -------- | -----           | ------         | -------            | -----------          |
| targets covered                   | 54/66    | 21942/21942     |  174101/196899 | 59110/64426        | 288108/288108        |
| num routers used                  | 2        | 84              |  847           | 188                | 4253                 |
| score in millions                 | 0        | ~21.96M         |  ~174.10M      | ~59.11M            | ~290.22M             |


Greedy with random improvements and swaps total score: **545.40M**.

The *runtime* is about 5-10 minutes per sample, and the python code is fully single thread and single process.

*Memory* used can go up to 2.5-3GB for the biggest sample (I used activity monitor on Mac OS X to view that).

For information, here is the list of scores of the competition: https://hashcode.withgoogle.com/hashcode_2017.html .
Most teams managed to get above 520M (but remember they all managed to qualify, so they are all super good)
and the best team reached 548M. Our greedy approach looks ok compared to other teams scores, and would have ranked 36th.
The solution with random improvements and swaps would have ranked 9th. We could run more iterations but there are
diminishing returns.

It took me more than a day to think, code in a readable way, and run this solution. But for a team of 4 people,
probably very smart (because they got qualified for the finals) and not caring about readability,
it seems achievable in one day.

Note that:
* there are about 23K uncovered targets on `opera`, that is ~23M potential points (bound)
* there are about 5K uncovered targets on `rue_de_londres`, that is ~5M potential points (bound)
* we covered all targets for `charleston_road` and `lets_go_higher`, so we did not have to run too
many random improvement or swap loops
* we can holy hope to gain a few 100Ks points maximum by optimizing budget for `lets_go_higher` and `charleston_road`,
so we should not focus too much on these samples.
* running more iterations of random improvements yields diminishing score returns. However, with maybe a more optimized
code and with a more powerful machine, we could have run a lot more of them and continued improving the score.
* we could also vary the number of routers removed in each iteration: more routers removed is better for exploration,
less routers removed is better for finding a local maximum. This is a simple parameter to adjust
the exploration/maximisation trade-off.

Note that the greedy algorithm with random improvements is a generic meta-heuristic and can be applied to
many different problems.

### I.E. About running a lot more random improvements using multi-processing

One can follow the greedy algorithm with random improvements and obtain better scores: it just requires trying a lot
more combinations and adjust the number of routers removed. More optimised code can use multi-processing
(multi-threading is useless here since the bottleneck is the CPU), use GPU computations, or simply be faster
depending on the programming language and implementation optimizations.

Our solution is written in python 3. One must be extremely careful when multi-processing in python as there are many
pitfalls one can fall into. One must remember that basically no memory is shared, and that every object passed to a
new process must be fully "pickable". The best way to ensure that is to pass basic types only.

In order to multi-process the random improvements, we could:
- have each process loop infinitely, reading the solution in a text file at the beginning of every loop
- when a process finds a better solution, it can save it
- catch any concurrency read/write exception (which should be rarer and rarer when the score improves) inside the loop

I did not implement this multi-process speed up yet.

## II. Maximum coverage problem and Steiner trees:

### II.A. Problem decomposition

Lets try and solve the problem with a different method. Consider this sub-problem, noted (A):

| (A) Given a fixed number of routers N, find a good solution to the original problem using exactly N routers. |
| ------------- |


To solve the original problem, we will solve (A) for a few well chosen values of N and take the solution with the best score.

Now, in order to solve (A), we can try to solve 2 sequential sub-problems:

| (A1) Place N routers without considering the backbone, so that we maximize the number of targets covered. |
| ------------- |

| (A2) Given the positions of N routers and the initial backbone cell, find a backbone tree that connects them minimizing the number of backbone cells created. |
| ------------- |

If we manage to find a backbone tree that costs less than B - Pr * N, then we found a legal solution to the problem.

Remark that this might miss an optimal solution: suppose we find a solution to (A1), and we do not find a legal solution to (A2).
There probably are solutions with N routers covering less targets than the solution we find for (A1),
but for which we could find a backbone tree leading to a legal solution in (A2).
There are good chances that this method will miss the optimal solution, but in practice the scores for N and N+1
routers are close, and I hope to gain score by better optimizing step (A1) than what we found in the greedy approach.

The advantage of dividing our problem into (A1) and (A2) is that the new sub-problems look a lot more generic.
And indeed, after looking for similar problems on the web, I found that **both are classic problems**:

(A1) is called the *Maximum Coverage Problem*. The Wikipedia page is a good starting point to read about it:
https://en.wikipedia.org/wiki/Maximum_coverage_problem .

(A2) is called the *Steiner Tree problem*, https://en.wikipedia.org/wiki/Steiner_tree_problem .

We are dealing with special cases of these 2 problems:
* our problem is in the **2-dimensional plane**
* the **distance metric** we have is the **Chebyshev distance**.
(diagonal costs 1, cf https://en.wikipedia.org/wiki/Chebyshev_distance)

Note that we can improve an existing solution using only (A2):
* perform (A2) on the routers positions we computed
* replace the backbone with the solution form (A2)
* greedily place new routers with the remaining budget just freed
* repeat until freed budget is not enough to add a single helpful router
* could further improve this solution with random greedy improvements

Bonus remark: for the charleston_road and lets_go_higher examples, we saw that the budget is enough to cover
all the targets even with a the simple greedy algorithm. In these cases, instead of solving the (A1), the Maximum
Coverage Problem for a given N, we instead should solve the Minimum Set Cover problem
(see https://en.wikipedia.org/wiki/Set_cover_problem). We should try to find the smallest number of routers that can
cover all the targets. Since this would bring only minor score improvements, I did not investigate further.



### II.B. Maximum Coverage Problem

The maximum coverage problem (A1) is well studied in literature.
After a bit of reading, I realised that a greedy approach (placing routers one by one) is a good starter.
Then we can improve it by looking for router swaps, with one or several routers.
I used the same approach as for the general problem:
* exploration step: delete random routers (10%) and greedily add routers back
* maximisation step: place routers one by one on the best possible spot until convergence

I focusing on `opera` and `rue_de_londres`, the results I got managed to cover:
* for `opera`: 175975 targets using 854 routers, 176531 targets using 857 routers
* for `rue_de_londres` 59163 targets using 189 routers, 59526 targets using 191 routers
* see file `maximum_coverage.py` and the folder `/max_coverage`

### II.C. Steiner tree with Chebyshev distance

The Steiner tree problem (A2) is also well studied in the literature, notably the
ESTP (Euclidian STP) and the Rectilinear STP, using different distance functions.
Our distance is a bit different: the Chebyshev distance is not very famous.

In file `steiner_tree.py`, I considered we were trying to build a highway connecting cities,
with highway intersections possibly out of cities (called steiner points). It is exactly the same
problem as connecting routers using backbone.

I found and implemented 4 approaches to find a good Steiner tree:

#### Minimum spanning tree

This is simply connecting the different cities using our distance and
a classic MST algorithm (I used Prim's) without using highway intersections
outside cities. This gives pretty bad results on this problem but was useful
in order to have ideas about better algorithms.

Prim's algorithm connects cities 1 by 1, connecting the next closest city
to the set of cities already connected.

#### Constructive Steiner tree

I improved on the simple MST procedure by simply taking the next closest
city but computing its distance to the closest highway point instead of
simply the cities. In practice I kept this distance in a field and updated
it for each remaining city every time a new highway point was added.

This gave great improvements but we could still do better.

#### Batched iterative 1-Steiner greedy approach with dynamic maintenance

Some smart people have thought of another way to use MST algorithms to solve
the Steiner tree problem. Here is the idea: consider a Steiner tree solution.
We call "Steiner points" the highway intersections that are not cities.
They are the highway points with 3 or more highway neighbors.

Now, if a Steiner tree is an optimal solution, then it is a minimum spanning tree
over its set of cities and Steiner points (else we could find a better solution).

The idea is to find good steiner points in order to build a good overall solution,
which is a MST over cities and Steiner points. We will greedily add Steiner points
to a solution.

Here is a naive approach to do this:
* take as candidate Steiner points all the points in the grid that are not a city or
already a Steiner point in our current solution
* each candidate's score is the difference in cost of the MST with and without it
(given the cities and Steiner points already added to our solution).
* at each iteration, compute the score of each candidate and add the best one,
until no candidate is helpful

This can take a long time: number of candidates is the size of the grid,
the MST computation for each candidate is O(n^2) where n = number of cities
(there will not be more useful Steiner points than cities), and we will have n iterations.

A lot of improvements are mentioned in article *Closing the Gap: Near-Optimal
Steiner Trees in Polynomial Time*. They include:
* at several candidates in each iteration if they seem "independent"
* restrict the number of candidates using "Hannan" candidates
* compute the candidate scores in O(n) time using geometric properties and dynamic MST maintenance.
This reduced the cost of each iteration to about O(n^3) and there are usually less than 5 iterations.

Remark than with our problem we have a different distance function:
for the geometric considerations, I simply used 8 quadrant instead of 4.

This still takes along time for instance on `opera` with about 850 cities (something like 24h),
but it gave very good results, better than the 2 preceding approaches, but in a much longer time.


#### Local Steiner tree iterative improvements

The idea here is to use the previous approach ("B1S_DMSTM") to local
sub-trees of an existing tree.
I used clusters of 5-15 cities and at least one cluster centered around each city.
This helped greatly improve the run-time with little score concessions.



#### Steiner tree scores:

|londres (191 routers)| MST      | Constructive  | B1S_DMSTM  | Local B1S_DMSTM |
| -----               | -------- | -----         | ------     | -------         |
| score               | 3128     | 2669          |  2621      | 2614            |
| run time            | 0.3s     | 1s            |  4h        | 8min            |


|opera (857 routers)| MST      | Constructive  | B1S_DMSTM  | Local B1S_DMSTM |
| -----             | -------- | -----         | ------     | -------         |
| score             | 12075    | 9716          |  9200      | 9420            |
| run time          | 1s       | 7s            |  24h       | 20min           |


* the constructive approach worked well but the B1S_DMSTM worked better
* the run-times however have nothing in common: B1S_DMSTM is prohibitively
slow for big instances
* Local B1S_DMSTM ended up being the best compromise better score and speed,
even beating B1S_DMSTM on small instances


### II.D. Final Results

The implementation if (A1) allowed to find good router configurations
must faster than in part 1.
However, by combining solutions to (A1) and to (A2) I managed to get good results but
not as good as the best of the competition. The reason is: good solutions to (A1)
were too rigid, and finding a corresponding (A2) solution did not bring flexibility in the result.

I ended up not using (A1) at all, but simply inserting (A2) in our algorithm of part 1
(greedy with random improvements and swaps), in order to periodically reduce the
cost of the backbone tree.

Here are the final results, corresponding to files in folder `two_step_outputs`:

| Greedy + rand impr. + swaps + Local B1S_DMSTM   | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                 | -------- | -----           | ------         | -------            | -----------          |
| targets covered       | 54/66    | 21942/21942     |  176374/196899 | 59498/64426        | 288108/288108        |
| num routers used      | 2        | 79              |  857           | 190                | 4107                 |
| num backbone cells    | 11       | 812             |  9160          | 2581               | 21486                |
| score in millions     | 0        | ~21.96M         |  ~176.37M      | ~59.50M            | ~290.24M             |

I stopped when my score reached: 548079795 = **548.08M**.
This was just enough to beat the best score of the competition (but remember:
they all made it in 1 day).

If you worked on the problem, I hope this helped. Make sure to read the
references found in the literature folder. If you want to talk about it
with me, just open an issue on this repo.