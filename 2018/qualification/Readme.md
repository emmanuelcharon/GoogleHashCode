# HashCode 2018 Qualification: Self-driving rides.

Code is in `qualif2018.py`.

With a friend we reached a score of 49,163,727 during the 3h45 that the competition lasted.

|                   | a. example  | b. should be easy | c. no hurry   |  d. metropolis  |. e high bonus  |
| ----------------- | --------    | -----             | ------        | -------         | ------         |
| score             | 10          |    176877         | 15792582      | 11728313        | 21465945       |


It is hard to find a function that optimizes a global score at each step, because it seems it would only benefit long
rides, and because it is hard to compute in advance if the bonus will be awarded.

The idea of our algorithm is to start each ride as soon as possible. So a each step, we select the ride that can start
the earliest. To do so, we save for each car when and where it will be available next.
If several rides can start at the same time, we select the one with the best score.


