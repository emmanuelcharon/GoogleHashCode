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

There are many ways to implement these requirements. My advice is to go for an object oriented approach: I used a class Building and a class Cell to represent the two main objects. I advise against implementing everything as array for speed reasons (in python for instance, you could implement every field of a cell as a numpy.array of the size of the grid): this gets confusing quickly and we can always improve performance in classes after we found a decent solution.

You can check the implementation details in routers_basic.py, in method Greedy.greedy_solve. Some implementation remarks:

In this problem, one can see that adding a router on a cell only has an impact for the score of a nearby geographic area. However, adding a backbone cell potentially has effects on a large part of the building. So I chose to update the scores only in an area close to the newly added router, hence we the scores that do not get updated might vary a bit compared to their true value.

One thing that accelerated the program a lot was to cache, for each potential router cell A, the set of targets that would be covered if a router was placed on A. You can also see in the code a naive O(R^4) and a linear programming approach O(R^2) to compute this set for each cell.

### I.C. Variation and random improvements

There are always possible variations on greedy methods, and in this case, we can for instance think of a way differentiate 2 router candidates that have the same score.
I added a simple variation, which is to thake the router with maximum gain per budget used instead of simply the candidate with the best gain. This led to minor score improvements but selected solutions with more routers.

In general, there is no guarantee that greedy algorithms will find a good solution (i.e. a solution with score close to the global maximum). Here is an easy and general way to improve a greedy solution (but not always possible): remove random parts of the solution, then use that as a starting point to reconstruct a greedy solution. This allows to explore neighbor solutions and usually to improve the score of the solution.

This is applicable here: we will repeatedly remove about 10% of the routers (and associated backbones), and then add routers like in the simple greedy approach. See method: Greedy.greedy_solve_with_random_improvements.

In order to do so, we implemented methods to:
* remove routers, remove backbones
* backtrack a backbone line to remove (backtrack until we find an intersection, another router or the initial backbone)

This approach has the advantage of exploring the set of solutions varying all parameters: the router positions, the backbone positions, and also the number of routers used.

Remark that sometimes, the remaining budget goes below 0, because the distance to backbone is approximate. This usually does not last long and the solutions become legal again after a few iterations.

### I.D. Analysing results

The results for the greedy (but carefully implemented and using ) approach are:

| Greedy                   | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                    | -------- | -----           | ------         | -------            | -----------          |
| targets covered          | 54/66    | 21942/21942     |  171696/196899 | 57004/64426        | 288108/288108        |
| score                    | 54 009   | 21 963 262      |  171 696 018     | 57 006 834       | 290 103 972          |
| score in millions        | ~0.054M  | ~21.96M         |  ~171.70M      | ~57.01M            | ~290.10M             |

The results for the greedy solution transformed with 20 to 100 loops of random improvements are:

| Greedy + random impr.    | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                    | -------- | -----           | ------         | -------            | -----------          |
| targets covered          | 54/66    | 21942/21942     |  173018/196899 | 58988/64426        | 288108/288108        |
| score                    | 54 009   | 21 962 554      |  173 018 079   | 58 988 015         | 290 194 257          |
| score in millions        | 0        | ~21.96M         |  ~173.02M      | ~58.99M            | ~290.19M             |

Greedy approach score: **540.77M**.
Greedy with random improvements: **544.16M**.
The runtime is about 5-10 minutes per sample, and the python code is fully single thread and single process. Memory used can go up to 2.5-3GB for the biggest sample.

For information, here is the list of scores of the competion: https://hashcode.withgoogle.com/hashcode_2017.html .
Most teams managed to get above 520M (but remember they all managed to qualify, so they are all super good) and the best team reached 548M. Our greedy approach looks ok compared to other teams scores, and would have ranked 36th. The solution with random improvements would have ranked 14th. We could run more iterations but there are diminishing returns.

It took me more than a day to think, code in a readable way, and run this solution. But for a team of 4 people, probably very smart (because they got qualified for the finals) and not caring about readability, it seems achievable in one day.

Note that:
* there are about 23K uncovered targets on `opera`, that is ~23M potential points (bound)
* there are about 5K uncovered targets on `rue_de_londres`, that is ~5M potential points (bound)
* we covered all targets for `charleston_road` and `lets_go_higher`
* we can holy hope to gain a few 100Ks points maximum by optimizing budget for `lets_go_higher` and `charleston_road`, so we should not focus too much on these samples.
* running more iterations of random improvements yields diminishing score returns. However, with maybe a more optimized code and with a more powerful machine, we could have run a lot more of them and continued improving the score.
* we could also vary the number of routers removed in each iteration: more routers removed is better for exploration, less routers removed is better for finding a local maximum. This is a simple parameter to adjust the exploration/maximisation tradeoff.

Note that the greedy algorithm with random improvements is a generic meta-heuristic and can be applied to many different problems.

### I.E. Bonus: Running a lot more random improvements using multi-processing

One can follow the greedy algorithm with random improvements and obtain better scores: it jsut requires to trying a lot more combinations and adjust the number of routers removed. More optimised code can use multi-processing (multi-threading is not usefull here since the bottleneck is the CPU), use GPU computations, or simply be faster depending on the programming language and implementation optimizations.

Our solution is written in python 3. One must be extremely careful when multi-processing in python as there are many pitfalls one can fall into. One must remember that basically no memery is shared, and that every object passed to a new process must be fully "pickable". The best way to ensure that is to pass basic types only.

In order to multi-process the random improvements, we could:
- have each process loop infinitely, reading the solution in a text file at the beginning of every loop
- save the "targetsCoveredIfRouter" (for all cells) in a text file, since this is what takes most time when computing initial gains.
- when a process finds a better solution, it can save it
- catch any concurrency read/write exception (which should be rarer and rarer when the score improves) inside the loop

## II. Maximum coverage problem and Steiner trees:

### II.A. Problem decomposition

Lets try and solve the problem with a different method. Consider this sub-problem, noted (A):

(A) Given a fixed number of routers N, find a good solution to the original problem using exactly N routers.

To solve the original problem, we will solve (A) for a few well chosen values of N and take the solution with the best score.

Now, in order to solve (A), we can try to solve 2 sequential sub-problems:

(A1) Place N routers without considering the backbone, so that we maximize the number of targets covered.

(A2) Given the positions of N routers and the initial backbone cell, find a backbone tree that connects them minimizing the number of backbone cells created.

If we manage to find a backbone tree that costs less than B - Pr * N, then we found a legal solution to the problem.

Remark: suppose we find a solution to (A1), and that we do not find a legal solution to (A2). There may solutions with N routers covering less targets than the solution we find for (A1), but for which we could find a backbone tree leading to a legal solution in (A2). There are good chances that this method will miss the optimal solution, but in practice the scores for N and N+1 routers are close, and hope to gain score by better optimizing step (A1) than what we found in the greedy approach.

The advantage of dividing our problem into (A1) and (A2) is that the new sub-problems look a lot more generic. And indeed, after looking for similar problems on the web, I found that **both are classic problems**:

(A1) is called the *Maximum Coverage Problem*. The Wikipedia page is a good starting point to read about it: https://en.wikipedia.org/wiki/Maximum_coverage_problem .

(A2) is called the *Steiner Tree problem*, https://en.wikipedia.org/wiki/Steiner_tree_problem .

### II.B. Maximum Coverage Problem

Coming soon.

### II.C. Steiner tree with Chebyshev distance

Coming soon.

### II.D. Results

Coming soon.