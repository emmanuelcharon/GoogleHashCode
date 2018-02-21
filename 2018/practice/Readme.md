# HashCode 2018 Practice: Pizza Cutter.

The code is in file *practice2018.py* .

The problem statement, input and output files are joined. The scores for the output files are:

|       | example  | small | medium | big     |
| ----- | -------- | ----- | ------ | ------- |
| bound | 15       |    42 |  50000 | 1000000 |
| tomato proportion | 0.2  |    0.43 |  0.50 | 0.50 |
| score | 8        |    30 |  44100 | 762871  |


The total score without counting the example is 807001. The program can read solutions to the problem and improve
them greedily (but slowly).

## Algorithm

I did not have a good intuition for how to find good slices among all possible slices in a reasonable amount of time.
So I went directly for a random approach, see the *solve* function:

* randomly select a legal pizza slice
** must have enough mushrooms, tomatoes, and not too big
** here we select a size first, then a spot in the pizza
* compute the score gain if we put it
** we count the loss of score when removing slices which take any cell on the new slice's spot.
* if we gain something, we perform the operation

## Implementation

The goal here is to try as many random options as possible: speed is of the essence.
So I tried to use numpy as much as possible, especially the numpy array slices,
cf https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html .

We need to remember for each cell the slice it belongs to, and to be able to quickly look them up.
So I used a numpy array of integers for that, and I made a "hash" function to represent each slice
uniquely as an integer given the total proportions of the pizza. See the * __hash__ * function of a slice.

After that, it was quick to compute the score lost and the score gained by adding a slice,
and I could launch the computation. It still requires a lot of time. The scores given correspond to about 30min
of computation using a single CPU of my Macbook Pro and about 1GB of RAM.

Another option could have been to create a custom numpy dtype for a slice, more info here:
https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html .

## Takeaways:

There are plenty of useful numpy functions for many operations, here I discovered: numpy slices, np.count_nonzeros,
np.unique, np.all, np.any.

One trick that helped a lot: I performed the replacement of the slice even if the score gain was 0.
This allowed to cover a larger space of solutions and led to a significant improvement in the scores.
