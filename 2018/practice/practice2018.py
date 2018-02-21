"""
Created on 7 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import os
import numpy as np
import logging
import random
import time

def read_pizza(input_file_path):
  """ read input file and return a pizza (which is a problem statement instance) """
  pizza = None
  grid = []
  # efficient for large files
  with open(input_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        ls = [int(a) for a in l.split(' ')]
        pizza = Pizza(ls[0], ls[1], ls[2], ls[3])
      else:
        ls = [0 if a == 'T' else 1 for a in l]
        grid.append(ls)
      line_count += 1
  pizza.grid = np.array(grid)
  return pizza


def write_solution(solution, output_file_path):
  with open(output_file_path, 'w') as f:
    f.write(str(len(solution.get_slices())))
    f.write("\n")
    for s in solution.get_slices():
      f.write("{} {} {} {}".format(s.fromX, s.fromY, s.toX, s.toY))
      f.write("\n")

def read_solution(pizza, solution_file_path):
  slices = []
  num_slices = 0
  
  with open(solution_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()  # those nasty line endings get on my nerves
      if line_count == 0:
        num_slices = int(l)
      elif len(l)>0:
        ls = [int(a) for a in l.split(' ')]
        s = Slice(ls[0], ls[2], ls[1], ls[3], pizza)
        slices.append(s)
      line_count += 1
  if num_slices != len(slices):
    logging.error("inconsistent solution file {}".format(solution_file_path))
  
  solution = Solution(pizza)
  [solution.add_slice(s) for s in slices]  # do not assign slices directly, this computes cached values
  return solution

def is_legal_solution(solution):
  """ put outside of class "solution" on purpose": this is a checking function,
  so it does not use any caching variable """
  
  taken = np.zeros((solution.pizza.R, solution.pizza.C), dtype=bool)  # for each cell, whether it is in a slice
  for s in solution.get_slices():
    if not s.legal:
      logging.error("illegal slice")
      return False
  
    if np.count_nonzero(taken[s.fromX: s.toX + 1, s.fromY: s.toY + 1]) > 0:
      logging.error("pizza cell already taken")
      return False
    
    taken[s.fromX: s.toX + 1, s.fromY: s.toY + 1] = np.ones((s.height, s.width), dtype=bool)
    
  return True

class Pizza:

  def __init__(self, R, C, L, H):
    self.R = R # num rows
    self.C = C # num columns
    self.L = L # minimum number of each ingredient in each slice
    self.H = H # maximum size of each slice
    self.grid = np.zeros((R, C), dtype=int) # on each cell, 0 if tomato and 1 if mushroom

  def __repr__(self):
    s = "R={}, C={}, L={}, H={}".format(self.R, self.C, self.L, self.H);
    s += "\n" + str(self.grid)
    return s
  
  def random_legal_slice(self):
    # first decide the size, then the position, then check if it is a "legal" slice
    while True:
      height = random.randint(1, min(self.R-1, self.H))
      width = random.randint(1, min(self.C-1, int(self.H/height)))
      r = random.randrange(0, self.R - height)
      c = random.randrange(0, self.C - width)
      
      s = Slice(r, r+height-1, c, c+width-1, self)
      if s.legal:
        return s
  
  def describe(self):
    print("total cells: {}, tomato proportion: {}".format(self.R*self.C, np.sum(self.grid)*1.0/(self.R*self.C)))
    
class Slice:
  def __init__(self, fromX, toX, fromY, toY, pizza):
    self.pizza = pizza
    self.fromX = fromX
    self.toX = toX
    self.fromY = fromY
    self.toY = toY
    
    # we do some computation here instead of creating methods for fast access (cache-like)
    # and because there are not too many slices (low memory cost)
    self.height = self.toX + 1 - self.fromX
    self.width = self.toY + 1 - self.fromY
    self.area = self.height * self.width
    self.mushrooms = np.count_nonzero(pizza.grid[self.fromX:self.toX + 1, self.fromY: self.toY + 1])
    self.tomatoes = self.area - self.mushrooms
    self.is_inside_boundaries = self._is_inside_boundaries()
    self.legal = self._is_legal_slice()
    
    
  def __hash__(self):
    """ we use slice hashes to better we exploit numpy functions """
    h = self.fromX
    h = h * self.pizza.C + self.fromY
    h = h * self.pizza.R + self.toX
    h = h * self.pizza.C + self.toY
    return h
  
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.fromX == other.fromX and self.toX == other.toX and self.fromY == other.fromY and self.toY == other.toY
    else:
      return False
  
  def _is_inside_boundaries(self):
    if self.fromX < 0 or self.fromX >= self.pizza.R:
      return False
  
    if self.toX < 0 or self.toX < self.fromX or self.toX >= self.pizza.R:
      return False
  
    if self.fromY < 0 or self.fromY >= self.pizza.C:
      return False
  
    if self.toY < 0 or self.toY < self.fromY or self.toY >= self.pizza.C:
      return False
    
    return True
  
  def _is_legal_slice(self):
    
    if not self.is_inside_boundaries:
      return False
    
    if self.area > self.pizza.H:
      return False
    
    if self.tomatoes < self.pizza.L or self.mushrooms < self.pizza.L:
      return False
      
    return True
 
class Solution:
  
  def __init__(self, pizza):
    self.pizza = pizza # the problem statement
    
    # cache variables
    self._slices = []  # each slice we want to cut
    self._hash_to_slice = dict() # map from slice "hash" to slice
    self._cell_slice = np.zeros((self.pizza.R, self.pizza.C), dtype=int) # the slice (hash) each cell belongs to (0 if none)
    self._score = 0 # current number of cells in slices, "solution" stays always legal
  
  def get_slices(self):
    return self._slices
  
  def get_score(self):
    return self._score
  
  def get_existing_slices_on_slice_position(self, s):
    return [self._hash_to_slice[h] for h in np.unique(self._cell_slice[s.fromX: s.toX + 1, s.fromY: s.toY + 1]) if h != 0]
  
  def score_gain_if_add_slice(self, s):
    loss = sum([existing_slice.area for existing_slice in self.get_existing_slices_on_slice_position(s)])
    return s.area - loss

  def add_slice(self, s):
    """ removes present slices if needed first """
    for existing_slice in self.get_existing_slices_on_slice_position(s):
      self.remove_slice(existing_slice)

    self._slices.append(s)
    self._cell_slice[s.fromX: s.toX + 1, s.fromY: s.toY + 1] = hash(s) * np.ones((s.height, s.width), dtype=int)
    self._hash_to_slice[hash(s)] = s
    self._score += s.area
  
  def remove_slice(self, s):
    self._cell_slice[s.fromX: s.toX + 1, s.fromY: s.toY + 1] = np.zeros((s.height, s.width), dtype=int)
    del self._hash_to_slice[hash(s)]
    self._score -= s.area
    self._slices.remove(s)

def solve(input_file_path, output_file_path, start_from_output = False):
  """ We randomly select a rectangle that could be a pizza slice (enough mushrooms, tomatoes)
  Then we check if we gain score by putting it. It requires removing slices which are on the same spot.
  """
  
  pizza = read_pizza(input_file_path)
  solution = Solution(pizza)
  if start_from_output and os.path.exists(output_file_path):
    solution = read_solution(pizza, output_file_path)
    if not is_legal_solution(solution):
      print("ERROR: starting solution from file {} is not legal".format(output_file_path))
      return
  
  num_iter = pizza.R * pizza.C # * 5
  
  start_time = time.time() # in seconds
  for i in range(num_iter):
    if i % (num_iter/10) == 0:
      print("iter {}/{}, score {}".format(i, num_iter, solution.get_score()))
    
    s = solution.pizza.random_legal_slice()
    g = solution.score_gain_if_add_slice(s)
    if g >= 0:
      solution.add_slice(s)
    
  if not is_legal_solution(solution):
    print("ERROR: wrong Solution, not saving.")
  else:
    print("legal solution with score: {}".format(solution.get_score()))
    write_solution(solution, output_file_path)
    print("saved in: {}".format(output_file_path))

  print("time taken: {} seconds".format(int(time.time() - start_time)))


def compute_solutions():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["example", "small", "medium", "big"]
  
  for sample in samples:
    print("#####\n")
    random.seed(17)  # set the seed for reproducibility
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    solve(input_file_path, output_file_path, start_from_output=True)
  
def read_solutions():
  """ read all solutions and compute total score """
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["example", "small", "medium", "big"]
  scores = []
  
  for sample in samples:
    print("#####\n")
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    pizza = read_pizza(input_file_path)
    pizza.describe()
    solution = read_solution(pizza, output_file_path)
    scores.append(solution.get_score())
    print("read sample '{}', score {}".format(sample, solution.get_score()))
    print("solution is legal: {}".format(is_legal_solution(solution)))
    print("num slices: {}".format(len(solution.get_slices())))
    total_slices_area = sum([s.area for s in solution.get_slices()])
    print("area covered by slices: {}/{}".format(total_slices_area, pizza.R*pizza.C))

  print("#####\n")
  print("total score = {} + {} + {} + {} = {}".format(scores[0], scores[1], scores[2], scores[3], sum(scores)))


def main():
  #compute_solutions()
  print("####################\n####################\n")
  read_solutions()
 
if __name__ == "__main__":
  main()