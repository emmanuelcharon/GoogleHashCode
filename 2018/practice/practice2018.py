"""
Created on 24 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import os
import numpy as np
import random
import math

def read_pizza(input_file_path):
  """ read input file and return a pizza (which is a problem statement instance) """
  pizza = None
  # efficient for large files
  with open(input_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        ls = [int(a) for a in l.split(' ')]
        pizza = Pizza(ls[0], ls[1], ls[2], ls[3])
      else:
        ls = [True if a == 'T' else False for a in l]
        r = len(pizza.grid)
        pizza.grid.append([Cell(ls[c], r, c) for c in range(len(ls))])
      line_count += 1
  return pizza


def write_solution(pizza, output_file_path):
  with open(output_file_path, 'w') as f:
    f.write(str(len(pizza.slices)))
    f.write("\n")
    for s in pizza.slices:
      f.write("{} {} {} {}".format(s.fromX, s.fromY, s.toX, s.toY))
      f.write("\n")


def read_slices(solution_file_path):
  slices = []
  num_slices = 0
  
  with open(solution_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()  # those nasty line endings get on my nerves
      if line_count == 0:
        num_slices = int(l)
      elif len(l) > 0:
        ls = [int(a) for a in l.split(' ')]
        s = Slice(fromX=ls[0], toX=ls[2], fromY=ls[1], toY=ls[3])
        slices.append(s)
      line_count += 1
  if num_slices != len(slices):
    print("error: inconsistent solution file {}".format(solution_file_path))
  
  return slices
  

class Cell:
  
  def __init__(self, isTomato, r, c):
    self.isTomato = isTomato
    self.r = r
    self.c = c
    self.currentSlice = None

    # cache
    # we call "potential slice" a legal slice we could put (respects all conditions of size and availability)
    self.bestPotentialSlice = None # a potential slice starting from this cell with maximal score
                                   # compared to other potential slices starting from this cell
    self.potentialSlices = set() # set of potential slices covering this cell
                                 # (slices that are a the bestPotentialSlice of a other cell or of this cell)
                                 # this must be empty when the cell has a currentSlice

class Slice:
  
  def __init__(self, fromX, toX, fromY, toY):
    self.fromX = fromX
    self.toX = toX
    self.fromY = fromY
    self.toY = toY

    self.height = self.toX + 1 - self.fromX
    self.width = self.toY + 1 - self.fromY
    self.area = self.height * self.width
    
  def __repr__(self):
    return "fromX:{}, toX:{}, fromY:{}, toY:{}, area: {}".format(self.fromX, self.toX, self.fromY, self.toY, self.area)

class Pizza:

  def __init__(self, R, C, L, H):
    self.R = R # num rows
    self.C = C # num columns
    self.L = L # minimum number of each ingredient in each slice
    self.H = H # maximum size of each slice
    self.grid = [] # grid of cells or size (R,C)
    
    self.slices = [] # the solution
    self.score = 0
    self.score_for_unavailable = -1
  
  def list_cells_in_slice(self, s):
    result = list()
    for r in range(s.fromX, s.toX + 1):
      for c in range(s.fromY, s.toY + 1):
        result.append(self.grid[r][c])
    return result
  
  def count_mushrooms_tomatoes_in_slice(self, s):
    [num_mushrooms, num_tomatoes] = [0, 0]
    for cell in self.list_cells_in_slice(s):
      if cell.isTomato:
        num_tomatoes += 1
      else:
        num_mushrooms += 1
    return [num_mushrooms, num_tomatoes]
   
  def find_max_unused_x(self, fromX, y):
    """
    increments x as long as the cell is not "used" and we stay in bounds
    also stops at maximum fromX + self.H (a max area slice of a single column)
    """

    if self.grid[fromX][y].currentSlice:
      print("error: must call this function on a free cell")
      return None
    
    result = fromX
    
    while True:
      # test if result + 1 works fine
      r = result + 1
      if r >= self.R:
        break
      if r >= fromX + self.H:
        break
      if self.grid[r][y].currentSlice:
        break
      result = r
    return result
  
  def has_used_cell(self, fromX, toX, fromY, toY):
    """ toX and toY are inclusive in this problem """
    for r in range(fromX, toX + 1):
      for c in range(fromY, toY + 1):
        if self.grid[r][c].currentSlice:
          return True
    return False

  def find_max_unused_columns_y(self, fromX, toX, fromY):
    """ finds the max toY such that the slice is completely unused
    and stays within bounds, given fromX, toX and fromY
    also stops at maximum fromY + self.H,
    also stops before area of rectangle becomes larger than self.H
    """
    
    if self.has_used_cell(fromX, toX, fromY, fromY):
      print("error: must call this function on a free column")
      return None

    result = fromY
    
    while True:
      # test if result + 1 works fine
      c = result + 1
      if c >= self.C:
        break
      if c >= fromY + self.H:
        break
      area = (toX + 1 - fromX) * (c + 1 - fromY)
      if area > self.H:
        break
      if self.has_used_cell(fromX, toX, c, c): # need only compute for the new column
        break
      result = c
    return result
  
  def best_slice_for_cell(self, cell):
    """ finds a maximum size slice starting from this cell
    This cell is a the top left of the slice (we go right and down only)
    
    For each possible number of rows (1 up to H), finds the widest legal slice.
    Then choose one maximum area slice among these.
    
    It can be slow to compute this for all cells at the beginning, so we stop
    early if a legal slice of size R is found.
    
    returns None if: the cell is already in a slice, or if no legal slice can be found
    """
    
    if cell.currentSlice:
      print("error: cell is already in a slice")
      return None
    
    best_legal_slice = None
    best_area = 0
    
    [fromX, fromY] = [cell.r, cell.c]

    # because of early termination and our way of selecting the max, the order in which we seek this slice matters a lot

    list_of_toX = list(range(fromX, self.find_max_unused_x(fromX, fromY)+1))
    
    # order A: keeping it like that prioritizes very horizontal slices
    # order B: prioritize vertically
    #  list_of_toX = reversed(list_of_toX)
    # order C: simply random
    #  random.shuffle(list_of_toX)
    # order D: prioritize "squares" : orders toX so that toX with resulting height closest to math.sqrt(self.H) come first
    #  list_of_toX = sorted(list_of_toX, key=lambda x: abs(math.sqrt(self.H) - (x + 1 - fromX)))
    
    for toX in list_of_toX:
      toY = self.find_max_unused_columns_y(fromX, toX, fromY)
      s = Slice(fromX=cell.r, toX=toX, fromY=cell.c, toY=toY)

      #print("cell ({},{}), slice: {}".format(fromX, fromY, s))

      # s is now the largest "free" slice of height toX-fromX
      # check area again, to be sure
      if s.area > self.H:
        print("error: slice too big")
        return None

      # check that it contains enough mushrooms and tomatoes
      [num_mushrooms, num_tomatoes] = self.count_mushrooms_tomatoes_in_slice(s)
      if num_mushrooms >= self.L and num_tomatoes >= self.L:
        # slice s is legal
        if s.area == self.H:  # early termination
          return s

        if s.area > best_area:
          best_area = s.area
          best_legal_slice = s
   
    return best_legal_slice

  def remove_potential_slice(self, s):
    start_cell = self.grid[s.fromX][s.fromY]
    if s == start_cell.bestPotentialSlice:
      start_cell.bestPotentialSlice = None
    else:
      print("weird: start cell does not have s as best potential slice")
    
    for cell in self.list_cells_in_slice(s):
      if s not in cell.potentialSlices:
        print("weird potential cell's potentialSlices set does not contain s")
      else:
        cell.potentialSlices.remove(s)

  def add_potential_slice(self, s):
    
    start_cell = self.grid[s.fromX][s.fromY]
    if start_cell.bestPotentialSlice:
      print("error: must remove start_cell's potential slice first")
      return

    # check that all cells in this slice are available
    if self.has_used_cell(s.fromX, s.toX, s.fromY, s.toY):
      print("error: when adding a potential slice, it must be on empty cells")
      return
    
    start_cell.bestPotentialSlice = s
  
    for cell in self.list_cells_in_slice(s):
      cell.potentialSlices.add(s)
    
  def add_actual_slice(self, s):
    # check that all cells in this slice are available
    if self.has_used_cell(s.fromX, s.toX, s.fromY, s.toY):
      print("error: a cell is already in a slice")
      return

    for cell in self.list_cells_in_slice(s):
      cell.currentSlice = s
    self.slices.append(s)
    self.score += s.area

  def remove_actual_slice(self, s):
    # check that all cells in this slice were using it as current slice
    for cell in self.list_cells_in_slice(s):
      if cell.currentSlice != s:
        print("error: a cell was not using this as currentSlice")
        return
    if s not in self.slices:
      print("error: slice is not in the pizza slices list")
      return
  
    for cell in self.list_cells_in_slice(s):
      cell.currentSlice = None
    self.slices.remove(s)
    self.score -= s.area

  def solve_greedy(self, optimize_steps = 0):
    """ if optimize_steps>0, when the greedy part converged, we destroy part of the allocation and re-build a greedy one,
    we repeat optimize_steps times. """
    
    optimize_steps_done = 0
    
    # find a maximum area slice starting from each cell
    gains = np.zeros((self.R, self.C), dtype=int) # we keep this as a separate np.array to use the np.argmax function
    for r in range(self.R):
      if r%(self.R/10) == 0:
        print("computing initial gains, row {}/{} ".format(r, self.R))
      for c in range(self.C):
        cell = self.grid[r][c]
        if cell.currentSlice: # the cell is occupied
          gains[cell.r, cell.c] = self.score_for_unavailable
        else:
          s = self.best_slice_for_cell(cell)
          if s:
            gains[r, c] = s.area
            self.add_potential_slice(s)
          else:
            gains[r, c] = self.score_for_unavailable
    print("done computing initial gains")
    
    while True:
      # among all bestPotentialSlices, choose the slice with maximum area, add it,
      # then update what the best potential slice is for affected cells (local)
      [best_r, best_c] = np.unravel_index(gains.argmax(), gains.shape)
      best_gain = gains[best_r, best_c]

      if best_gain <= self.score_for_unavailable:
        if optimize_steps_done >= optimize_steps:
          print("stopping because no potential slice adds score")
          break
        else:
          print("optimize step {}/{}, slices: {}, score: {}".format(
            optimize_steps_done, optimize_steps, len(self.slices), self.score))
  
          optimize_steps_done += 1
          self.optimize_step(gains)
          continue
          
      best_slice = self.grid[best_r][best_c].bestPotentialSlice
      self.add_actual_slice(best_slice)
      # print("best gain= {}, best cell ({},{}), added best slice: {}".format(best_gain, best_r, best_c, best_slice))

      # now we remove all the potentialSlices that are now illegal
      # and we update the score for each affected cell
      #  - cells that are starting points for the potential slices now illegal
      #  - cells part of the newly added slice
      
      potential_slices_now_illegal = set([best_slice])
      for cell in self.list_cells_in_slice(best_slice):
        for s in cell.potentialSlices:
          potential_slices_now_illegal.add(s)
         
      cells_affected = set(self.list_cells_in_slice(best_slice))
      for s in potential_slices_now_illegal:
        cells_affected.add(self.grid[s.fromX][s.fromY])
        
      for s in potential_slices_now_illegal:
        self.remove_potential_slice(s)
  
      for cell in cells_affected:
        if cell.currentSlice: # the cell is occupied
          gains[cell.r, cell.c] = self.score_for_unavailable
        else:
          s = self.best_slice_for_cell(cell)
          if s:
            gains[cell.r, cell.c] = s.area
            self.add_potential_slice(s)
          else:
            gains[cell.r, cell.c] = self.score_for_unavailable
            
      if len(self.slices)%self.R==0:
        print("slices: {}, score: {}".format(len(self.slices), self.score))

  def remove_actual_slices_in_area(self, fromX, toX, fromY, toY):
    slices_to_remove = set()
    for cell in self.list_cells_in_slice(Slice(fromX=fromX, toX=toX, fromY=fromY, toY=toY)):
      if cell.currentSlice:
        slices_to_remove.add(cell.currentSlice)
    for s in slices_to_remove:
      self.remove_actual_slice(s)

  def optimize_step(self, gains):
    """
    try to discover better greedy solutions by removing a lot of slices
    after removing slices, we must re-compute the best potential slice for potentially affected cells.
    (maximum) affected cells:
      - cells on a random rectangle 1/4th the size of the pizza
      - all cells that are up to self.H cells before fromX and self.H cells before fromY
      => in total we can re-compute potential slice for cells in (fromX-self.H, toX, fromY-self.H, toY)
    """
    
    # choose area and size: take a quarter of the grid
    fromX = random.randrange(0, self.R)
    fromY = random.randrange(0, self.C)
    toX = min(self.R - 1, fromX + int(self.R / 4))
    toY = min(self.C - 1, fromY + int(self.C / 4))

    self.remove_actual_slices_in_area(fromX=fromX, toX=toX, fromY=fromY, toY=toY)
    
    recomputeFromX = max(0, fromX - self.H)
    recomputeFromY = max(0, fromY - self.H)
    
    for cell in self.list_cells_in_slice(Slice(fromX=recomputeFromX, fromY=recomputeFromY, toX=toX, toY=toY)):
      
      if cell.bestPotentialSlice:
        self.remove_potential_slice(cell.bestPotentialSlice)
      
      if cell.currentSlice:  # the cell is occupied
        gains[cell.r, cell.c] = self.score_for_unavailable
      else:
        s = self.best_slice_for_cell(cell)
        if s:
          gains[cell.r, cell.c] = s.area
          self.add_potential_slice(s)
        else:
          gains[cell.r, cell.c] = self.score_for_unavailable


def main(solve = True, start_from_existing_solution = True, optimize_steps = 0):
  """
  if solve is False and start_from_existing_solution is True, we just read the existing solutions
  if compute_solutions is false, it will read existing solutions from the output folder
  
  
  """
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["example", "small", "medium", "big"]
  scores = []
  
  for sample in samples:
    print("\n##### " + sample)
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    
    pizza = read_pizza(input_file_path)
    if start_from_existing_solution:
      [pizza.add_actual_slice(s) for s in read_slices(output_file_path)]
      print("read solution with {} slices and score is {}".format(len(pizza.slices), pizza.score))
  
    if solve:
    pizza.solve_greedy(optimize_steps=optimize_steps)
    
    print("{}: solution has {} slices and score is {}/{}".format(
      sample, len(pizza.slices), pizza.score, pizza.R*pizza.C))
    [total_mushrooms, total_tomatoes] = pizza.count_mushrooms_tomatoes_in_slice(Slice(0, pizza.R-1, 0, pizza.C-1))
    
    print ("tomato proportion in pizza: {}".format(total_tomatoes/(total_mushrooms+total_tomatoes)))

    slice_widths = [s.width for s in pizza.slices]
    slice_heights = [s.height for s in pizza.slices]
    print("average slice width:{}, height:{}".format(sum(slice_widths)/len(slice_widths), sum(slice_heights)/len(slice_heights)))

    write_solution(pizza, output_file_path)

    if sample != "example":
      scores.append(pizza.score)
    
  print("\n#####\nscores: {} => total score (without 'example') = {}".format(scores, sum(scores)))
  
if __name__ == "__main__":
  random.seed(17) # reproducibility
  main(start_from_existing_solution=False, optimize_steps=0)
  