"""
Created on 7 Fev. 2018

Python 3.6
@author: emmanuelcharon
"""

import os
import numpy as np
import logging
import random

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
    
  @staticmethod
  def read_file(file_path):
    pizza = None
    grid = []
    # efficient for large files
    with open(file_path, 'r') as f:
      line_count = 0
      for line in f.readlines():
        l = line.rstrip()
        if line_count == 0:
          ls = [int(a) for a in l.split(' ')]
          pizza = Pizza(ls[0], ls[1], ls[2], ls[3])
        else:
          ls = [0 if a=='T' else 1 for a in l]
          grid.append(ls)
        line_count += 1
    pizza.grid = np.array(grid)
    return pizza
  
  def random_legal_slice(self):
    # first decide the size
    
    while True:
      height = random.randint(1, min(self.R, self.H))
      width = random.randint(1, min(self.C, int(self.H/height)))
 
      r = random.randrange(0, self.R - height)
      c = random.randrange(0, self.C - width)
      p_slice = Slice(r, r+height-1, c, c+width-1)
      if p_slice.is_legal(self):
        return p_slice
      
class Slice:
  def __init__(self, r1, r2, c1, c2):
    if r1 > r2 or c1 > c2:
      logging.error("wrong slice init")
    self.r1 = r1
    self.r2 = r2
    self.c1 = c1
    self.c2 = c2
  
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.r1 == other.r1 and self.r2 == other.r2 and self.c1 == other.c1 and self.c2 == other.c2
    else:
      return False
    
  def __str__(self):
    return "{}, {}, {}, {}".format(self.r1, self.c1, self.r2, self.c2)
  
  def area(self):
    return (self.r2 + 1  - self.r1) * (self.c2 + 1  - self.c1)
  
  def counts(self, pizza):
    mushrooms = np.count_nonzero(pizza.grid[self.r1:self.r2 + 1, self.c1: self.c2 + 1])
    tomatoes = self.area() - mushrooms
    return [tomatoes, mushrooms]
  
  def is_legal(self, pizza):
    if self.r1 < 0 or self.r1 >= pizza.R:
      return False
    
    if self.r2 < 0 or self.r2 >= pizza.R:
      return False
    
    if self.c1 < 0 or self.c1 >= pizza.C:
      return False
    
    if self.c2 < 0 or self.c2 >= pizza.C:
      return False
    
    if self.area() > pizza.H:
      return False
    
    [tomatoes, mushrooms] = self.counts(pizza)
    if tomatoes < pizza.L or mushrooms < pizza.L:
      return False
      
    return True

class Solution:
  
  def __init__(self, pizza):
    self.pizza = pizza # the problem statement
    self.slices = [] # each slice we want to cut
    self.used_cells = np.zeros((self.pizza.R, self.pizza.C), dtype=bool)
  
  def print_output(self, output_file_path):
    with open(output_file_path, 'w') as f:
      f.write(str(len(self.slices)))
      f.write("\n")
      
      for p_slice in self.slices:
        f.write(str(p_slice))
        f.write("\n")
  
  def can_add(self, p_slice):
    return not np.any(self.used_cells[p_slice.r1:p_slice.r2 + 1, p_slice.c1: p_slice.c2 + 1])
   
  def add(self, p_slice):
    """ assumes this is permitted """
    self.slices.append(p_slice)
    self.used_cells[p_slice.r1:p_slice.r2 + 1, p_slice.c1: p_slice.c2 + 1] = np.ones((p_slice.r2 + 1 - p_slice.r1, p_slice.c2 + 1 - p_slice.c1), dtype=bool)
   
  def remove(self, p_slice):
    self.used_cells[p_slice.r1:p_slice.r2 + 1, p_slice.c1: p_slice.c2 + 1] = np.zeros((p_slice.r2 + 1 - p_slice.r1, p_slice.c2 + 1 - p_slice.c1), dtype=bool)
    self.slices.remove(p_slice)
  
  def compute_score(self):
    """ returns -1 if illegal """
    taken = np.zeros((self.pizza.R, self.pizza.C), dtype=bool)
    for p_slice in self.slices:
      
      if p_slice.area() > self.pizza.H:
        logging.error("slice is too big")
        return -1
      
      [tomatoes, mushrooms] = p_slice.counts(self.pizza)
      if tomatoes < self.pizza.L:
        logging.error("not enough tomatoes on slice")
        return -1
      
      if mushrooms < self.pizza.L:
        logging.error("not enough mushrooms on slice")
        return -1
      
      if np.count_nonzero(taken[p_slice.r1:p_slice.r2+1, p_slice.c1:p_slice.c2+1]) > 0:
        logging.error("pizza cell already taken")
        return -1
      taken[p_slice.r1:p_slice.r2+1, p_slice.c1:p_slice.c2+1] = np.ones((p_slice.r2+1-p_slice.r1, p_slice.c2+1-p_slice.c1), dtype=bool)

    return taken.sum()

  def solve(self):
    
    # make many iterations
    # at each step, consider a random "legal" slice
    # compute the score if we removed slices on its way and used this slice instead
    # perform modification if the score would be greater
    for i in range(100000):
      p_slice = self.pizza.random_legal_slice()
      
      if self.can_add(p_slice):
        self.add(p_slice)
        print("added slice " + str(p_slice) + " with area " + str(p_slice.area()))
        

def main():
  sample = "small" # example small medium big
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  pizza = Pizza.read_file(os.path.join(dir_path, "input", sample+".in"))
  print(pizza)
  
  random.seed(17)
  solution = Solution(pizza)
  solution.solve()
  score = solution.compute_score()
  print("score: {}".format(score))
  if score >= 0:
    solution.print_output(os.path.join(dir_path, "output", sample+".out"))
  else:
    print("not saving wrong solution")

if __name__ == "__main__":
  main()