"""
Created on 24 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import os
import numpy as np
import random

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

def write_visual_solution(pizza, visual_output_file_path):
  with open(visual_output_file_path, 'w') as f:
    for r in range(pizza.R):
      for c in range(pizza.C):
        if pizza.grid[r][c].currentSlice:
          f.write(".")
        else:
          f.write("T" if pizza.grid[r][c].isTomato else "M")
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
    raise ValueError("inconsistent solution file {}".format(solution_file_path))
  
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
      raise ValueError("must call this function on a free cell")
    
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
      raise ValueError("must call this function on a free column")
      
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
      raise ValueError("cell is already in a slice")
    
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
        raise ValueError("slice too big")

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
      raise ValueError("must remove start_cell's potential slice first")
      
    # check that all cells in this slice are available
    if self.has_used_cell(s.fromX, s.toX, s.fromY, s.toY):
      raise ValueError("when adding a potential slice, it must be on empty cells")
    
    start_cell.bestPotentialSlice = s
  
    for cell in self.list_cells_in_slice(s):
      cell.potentialSlices.add(s)
    
  def add_actual_slice(self, s):
    # check that all cells in this slice are available
    if self.has_used_cell(s.fromX, s.toX, s.fromY, s.toY):
      raise ValueError("a cell is already in a slice")

    for cell in self.list_cells_in_slice(s):
      cell.currentSlice = s
    self.slices.append(s)
    self.score += s.area

  def remove_actual_slice(self, s):
    # check that all cells in this slice were using it as current slice
    for cell in self.list_cells_in_slice(s):
      if cell.currentSlice != s:
        raise ValueError("a cell was not using this as currentSlice")
    if s not in self.slices:
      raise ValueError("slice is not in the pizza slices list")
  
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

  def enumerate_shapes(self):
    """ enumerates all the possible slice shapes (rectangles, no rotation): [height, width]
        it is all the rectangle shapes with area between 2L and H inclusive
    """
    result = list()
    for num_rows in range(1, self.H + 1):
      for num_columns in range(1, self.H + 1):
        if 2 * self.L <= num_rows * num_columns <= self.H:
          result.append([num_rows, num_columns])
    return result
  
  
  def potential_slice_spot(self, cell, cells_set, height, width):
    """ given a cell, tells if a slice of (height,width) respects:
    * stays within pizza bounds
    * all cells in the slice are in the set
    * the resulting slice has enough mushrooms and tomatoes and is not too big
    
    if True, returns the list of cells in this shape,
    else returns None
    
    """
    if height*width > self.H:
      return False
    num_tomatoes, num_mushrooms = 0, 0
    for x in range(cell.r, cell.r + height):
      for y in range(cell.c, cell.c + width):
        if x < 0 or x >= self.R or y < 0 or y >= self.C:
          return False
        other_cell = self.grid[x][y]
        if other_cell not in cells_set:
          return False
        if other_cell.isTomato:
          num_tomatoes += 1
        else:
          num_mushrooms += 1
    return num_tomatoes >= self.L and num_mushrooms >= self.L

  def get_cells_set_in_slice_spot(self, cell, height, width):
    result = set()
    for x in range(cell.r, cell.r + height):
      for y in range(cell.c, cell.c + width):
        result.add(self.grid[x][y])
    return result

  def optimal_score(self, cells_set, shapes, depth=0):
    """
    given a set of cells, computes all the possible ways to fill it up with with shapes,
    and returns the best score along with the corresponding slices
    * shapes must stay inside cells_set
    * recursive (long) but not memory consuming: stack of maximum [size(cells_set)/max_shape_area] calls
    * does not look at current pizza state (works any time)
    """
    
    if len(cells_set) > 4*self.H:
      print("WARNING: calling optimal_score with a big set. We recommend less than 4*H")
    
    # for each cell in the set, try to position each shape there, and recursively compute score
    best_score = 0
    best_slices = list()
    
    for cell in cells_set:
      for shape in shapes:
        if self.potential_slice_spot(cell, cells_set, shape[0], shape[1]):
          slice_set = self.get_cells_set_in_slice_spot(cell, shape[0], shape[1])
          cells_left = set([c for c in cells_set if c not in slice_set])
          
          score, slices = self.optimal_score(cells_left, shapes, depth=depth+1)
          
          score += shape[0]*shape[1]
          slices.append(Slice(cell.r, cell.r+shape[0]-1, cell.c, cell.c+shape[1]-1))

          
          if score > best_score:
            best_score = score
            best_slices = slices

            #if depth <= 0:
            #  print("(depth= {}) score {} with {} slices: {}".format(depth, score, len(slices), slices))
            
            if best_score == len(cells_set): # terminate early because we found a solution covering the full set
              return best_score, best_slices
            
    return best_score, best_slices
  
  
  def get_neighbors(self, cell):
    result = list()
    if cell.c + 1 < self.C:
      result.append(self.grid[cell.r][cell.c + 1])
    if cell.r - 1 >= 0:
      result.append(self.grid[cell.r - 1][cell.c])
    if cell.c - 1 >= 0:
      result.append(self.grid[cell.r][cell.c - 1])
    if cell.r + 1 < self.R:
      result.append(self.grid[cell.r + 1][cell.c])
    return result

  def find_available_connected_set(self, available_cell, size_limit=-1):
    # a simple BFS on available_cell, 4 neighbors per cell
    if available_cell.currentSlice:
      raise ValueError("use only on an available cell")
    
    cells_stack = [available_cell]
    visited = set()
    while len(cells_stack) > 0:
      cell = cells_stack.pop(0)
      visited.add(cell)
      if 0 <= size_limit <= len(visited):
        break
      for neighbor in self.get_neighbors(cell):
        if neighbor not in visited and not neighbor.currentSlice:
          cells_stack.append(neighbor)
      
    return visited
    
  def local_optimal_moves(self):
    if self.score == 0:
      raise ValueError("do not do this on empty solution")
    
    shapes = self.enumerate_shapes()
    
    for x in range(self.R):
      print("row {}/{}".format(x, self.R))
      for y in range(self.C):
        
        if not self.grid[x][y].currentSlice:
          
          # this cell has no slice, find the connected set of available cells around it (connected = ortho neighbors)
          # then for each neighbor slice of this set, compute the optimal score on
          # the union of slice area + connected available set
          # (limit the union to size 3*H for num_shapes = 19 or 4*H for num_shapes = 12)
          # perform the best move (if better than doing nothing)
          # in a second pass, we can try removing pairs of adjacent neighbors

          connected_set = self.find_available_connected_set(self.grid[x][y], size_limit=3*self.H)
          
          neighbor_slices = set()
          for cell in connected_set:
            for neighbor in self.get_neighbors(cell):
              if neighbor.currentSlice:
                neighbor_slices.add(neighbor.currentSlice)
          
          best_score, best_replacing_slices = self.optimal_score(connected_set, shapes) # probably 0
          best_ns = None
          if best_score > 0:
            print("gain without removal: {}".format(best_score))
          
          for ns in neighbor_slices:
            ns_set = set(self.list_cells_in_slice(ns))
            ns_set.update(connected_set)
            ns_score, ns_slices = self.optimal_score(ns_set, shapes)
            
            if ns_score > best_score:
              best_score = ns_score
              best_ns = ns
              best_replacing_slices = ns_slices
          
          if best_score > 0:
            if not best_ns:
              for s in best_replacing_slices:
                self.add_actual_slice(s)
            elif best_score > best_ns.area: # better to do the move than to do nothing
              self.remove_actual_slice(best_ns)
              for s in best_replacing_slices:
                self.add_actual_slice(s)
            
              print("score: {}".format(self.score))

    print("final score: {}".format(self.score))

  def incremental_optimal_local_moves(self):
    shapes = self.enumerate_shapes()
    
    h = self.H
    w = 3
    
    for x in range(0, self.R, h):
      print("row {}/{}, score: {}".format(x, self.R, self.score))
      for y in range(0, self.C, w):
        
          cells = set([self.grid[i][j] for i in range(x, x+h) for j in range(y, y+w) if 0 <= j < self.C if 0 <= i < self.R])
          score, slices = self.optimal_score(cells, shapes)
          for s in slices:
            self.add_actual_slice(s)
    print("final score: {}".format(self.score))


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
    visual_output_file_path = os.path.join(dir_path, "visual_output", sample + ".out")

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
    write_visual_solution(pizza, visual_output_file_path)

    if sample != "example":
      scores.append(pizza.score)
    
  print("\n#####\nscores: {} => total score (without 'example') = {}".format(scores, sum(scores)))


def test():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["big"] #["example", "small", "medium", "big"]
  
  for sample in samples:
    print("\n##### " + sample)
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output2", sample + ".out")
    visual_output_file_path = os.path.join(dir_path, "visual_output2", sample + ".out")

    pizza = read_pizza(input_file_path)
   # [pizza.add_actual_slice(s) for s in read_slices(output_file_path)]
   # print("read solution with {} slices and score is {}".format(len(pizza.slices), pizza.score))

    #shapes = pizza.enumerate_shapes()
    #print("L={}, H={}, shapes: {}".format(pizza.L, pizza.H, len(shapes)))
    #print(shapes)
    
    #cells_set = set()
    #for x in range(0, min(pizza.R, 14)):
    #  for y in range(0, min(pizza.C, 5)):
    #    cells_set.add(pizza.grid[x][y])
    #o = pizza.optimal_score(cells_set, shapes)
    #print(o)

    pizza.incremental_optimal_local_moves()
    write_solution(pizza, output_file_path)
    write_visual_solution(pizza, visual_output_file_path)

if __name__ == "__main__":
  random.seed(17) # reproducibility
  # main(solve=True, start_from_existing_solution=False, optimize_steps=0)  # compute solutions
  #main(solve=False, start_from_existing_solution=True, optimize_steps=0) # read solutions
  test()
  
  