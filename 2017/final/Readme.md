# HashCode 2017 Finals: Router placement.

Code is in `final2017.py`.

The code here reaches a score of ~**549.46M**.
It computes solutions in less than 5 minutes but uses a lot of caching: you can expect about 4.5GB RAM
used for the biggest input.

## Phase 1: Greedy Solution

I used a greedy approach for this: add routers one by one, selecting the best possible place to add
a router at each step. The difficulty is to implement this fast and in a tractable way.

* compute a `gain` for each cell
  * this is the score gain we would get by adding a router on it (-1 if wall or already router)
  * take into account the number of backbone cells we would have to add and the associated cost
* take the cell with the `best_gain` and add a router to it
  * we determine which backbones we need to add to reach the new router
  * we must remember, for each covered cell, that covering it again will not add score
* we update the gains for cells only around the new router, in a range 2R around it.
  * this covers all potential routers that now have a potential target already covered (big gain difference)
  * `best_gain` is now an **approximation**: many other cell `gains` will be affected by the additional backbone cells.
  We assume this gain difference is small and we do not compute it.


I tried to **optimize** the steps that are repeated many times:

* use a numpy array to store cell `gains`, because the numpy `argmax` function is faster than python loops
* cache, for each cell, the number of routers covering it
  * so we know if covering it adds score or not
* cache, for each possible router position (i.e cell), the targets that would be covered
  * this allows to compute the score gain much faster, because it saves the naive O(R^3) operations many times
* note that because of caching, the implementation required 4.5GB of RAM for the biggest example (lets_go_higher)

One operation that is repeated a lot is **finding the closest backbone cell** to a cell.
I tried to make this function stay "local", so I go in circles around the cell, looking for a backbone cell.
* we call this function only on cells around a new router (except initially), so we know a backbone is not too far
* initially, when no backbone cell is added, we shortcut to the unique backbone cell (see variable `building.initialState`)

Note: before implementing the `building.initialState` trick, I simply computed the initial gains for a sparse grid of cells.

Another operation is to find the backbones cells to add when we decided where we will add the next router.
My function is very simple: it finds the closest backbone cell and then finds a **path going in diagonal first**,
staring from the backbone cell.

The results for the greedy (but carefully implemented) approach are:

|                          | example  | charleston_road | opera          | rue_de_londres     |    lets_go_higher    |
| -----                    | -------- | -----           | ------         | -------            | -----------          |
| backbone cells           | 15       |    945          |  11323         | 3043               | 23661                |
| routers                  |  2       |    77           |  835           | 185                | 3604                 |
| targets covered          | 54/66    | 21942/21942     |  175655/196899 | 61561/64426        | 288108/288108        |
| targets covered 2+ times | 0        | 8385            |  5840          | 12737              | 101864               |
| remaining budget         | 9/220    | 21262/29907     |  37/94860      | 91/21634           | 2175972/2654677      |
| score                    | 54009    | 21963262        |  175655037     | 61561091           | 290283972            |
| score in millions        | 0        | ~22.0M          |  ~175.6M       | ~61.6M             | ~290.3M              |

Total score (without example): 549463362 which is about **549.46M**.
This beats the best score of the competition (548.1M) !! https://hashcode.withgoogle.com/hashcode_2017.html .

Of course, I had more time and less stress than they had during the actual competition.

Note: instead of counting the number of routers on each cell, I saved a set of routers.
I figured it did not hurt my RAM too much and that it could be useful later.

## Phase 2: Improvement Ideas

Now looking back at the results in the table of phase 1:

* there are ~20,000 uncovered targets on `opera`, that is ~20M potential points (bound)
* there are ~3,000 uncovered targets on `rue_de_londres`, that is ~3M potential points (bound)
* we covered all targets for `charleston_road` and `lets_go_higher`
* we can holy hope to gain a few 100,000s points maximum by optimizing budget for `lets_go_higher`

**How could we improve our score?**

First, once in a while, we can re-compute the actual `gains` for all cells. This may improve our score.


The best other option seems to improve target coverage on the `opera` and `rue_de_londres` examples.
We may find some more points if we remove some routers that have become useless,
or which removal would us cost the least score. And then we could use the freed budget for more routers.


We used a greedy approach and we are most likely stuck in a local optimum. So in order to find more solutions,
we can destroy routers and find other cells to put them. If we destroy one router and then greedily optimize,
we will obtain little gains because we will stay close to the existing solution. Instead,
we could **remove many routers simultaneously, before greedily optimising again**.

Here are the steps we would need to implement this approach:

* write a function to remove a router
  * also remove corresponding backbones that were necessary only for this router
  (basically remove them until we touch an intersection of the backbones)
  * update local `gains`
  * must update all cached values
* maybe code a function to compute the score lost by removing a router
* when the optimisation step gives 0 improvement, destroy many servers simultaneously
  * for instance 10% of routers randomly selected
  * or all routers in an area of the grid
  * or select routers with the least decrease in score if we removed them
* then keep optimising with the same strategy, overwrite the output file only if score improved
