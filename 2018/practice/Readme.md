# HashCode 2018 Practice: Pizza Cutter.

The code is in file *practice2018.py* .

The problem statement, input and output files are joined. The scores for the output files are:

|                              | example  | small | medium | big     |
| -----------------            | -------- | ----- | ------ | ------- |
| bound                        | 15       |    42 |  50000 | 1000000 |
| tomato proportion            | 0.8      |  0.57 |  0.50  | 0.50    |
| number of slices in solution | 2        |     7 |  4106  | 63694   |
| score                        | 12       |    35 |  49255 | 891638  |


The total score without counting the example is **940928**. The program can read solutions to the problem and improve
them greedily (but not a lot).

## Algorithm

I call "potential slice" a legal slice we could put given existing slices (respects all conditions of size and availability).

The greedy solution I found is the following: for each cell, compute a "maximum area" potential slice to add,
where the cell is the top left corner of the potential slice.

Then, at each step, we take the maximum potential slice among all cells and we add it.

The algorithm is very simple but the difficulty is in the fast implementation details: every-time we add a slice,
we don't want to recompute the best potential slice for each cell of the pizza. We can restrict updates to cells
affected by the last slice added. See the *potential_slices_now_illegal* and the *cells_affected* variables in the *solve* function.

When we compute the best potential slice for each cell, in many cases there are several options possible, c.f.
function *best_slice_for_cell* .
For faster runtime, we do an early termination when a potential slice of size H is found (maximal size slice).

## Variations and improvements:

At first I had started with a completely random approach: find a random potential slice and add it if it improves the score,
removing slices in their way if necessary. This led to 750K to 820K scores and was not a good method to solve this problem.

I then implemented the greedy algorithm described above.
In the first version of the code in function *best_slice_for_cell*, I went through potential slices starting with the 1-row slices.
This led to create a solution with a lot of horizontal slices.
I tried with other configurations, prioritizing vertical slices, square slices and selecting in random order,
but this did not affect the score significantly.

I also added a way to optimize on the existing solution: destroy slices in an area of the pizza (like a quarter of the pizza)
an rebuild ones with the greedy optimizer. This did not give significant score improvements either.

