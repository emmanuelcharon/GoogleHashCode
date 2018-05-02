"""
Created on 30 Apr. 2018

Python 3.6.4
@author: emmanuelcharon

  N = number of routers desired
  (H,W) = building dimensions
  R = router range (but there are walls too)
  
  I- we will implement a greedy approach:
  - must compute the score gain for a router placed in each position, taking existing routers into account
  - then we will iteratively take the best move and update scores
  - initial scores are just the number of cells covered by each position
  - to update scores: only need to update scores in a radius 2*R around new router position
  
  II- then find a local optimum using swaps (maximisation):
  - compute for each router the gain when swapping its position somewhere else
  - then we will iteratively take the best move and update scores
  - there are N*H*W moves possible moves in each iteration (N=numRouters) )
  
  - faster option taken:
    repeat until no router has moved
      for each router, remove it and place it in the best position
  
  III- we will want to remove random routers from time to time (exploration)
  - need to remove router and update necessary gains & swaps
  
"""
import os
import time
from routers_basics import Building, Utils
import numpy as np
import random

class MCCell:
  
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.S = None # the set of targets covered if a router is placed on this cell
    
    # dynamic fields:
    self.hasRouter = False
    self.numCoveringRouters = 0 # the number of routers currently covering cell

  def __repr__(self):
    return "({},{})".format(self.x, self.y);

  def __str__(self):
    return str(self.__repr__())

class MCBuilding:
 
  def __init__(self, building):
    self.H = building.H
    self.W = building.W
    self.R = building.R
    self.grid = list()
    for x in range(self.H):
      row = list()
      for y in range(self.W):
        row.append(MCCell(x, y))
      self.grid.append(row)

    for x in range(self.H):
      for y in range(self.W):
        self.grid[x][y].S = set([self.grid[c.r][c.c] for c in building.grid[x][y].get_targets_covered_if_router()])

    self.routers = set()
    self.numTargetsCovered = 0
  
  def print_solution_info(self):
    print("routers: {}, targets covered: {}".format(len(self.routers), self.numTargetsCovered))
  
  def write_routers(self, routers_file_path):
    with open(routers_file_path, 'w') as f:
      # 1 line per router
      for router in self.routers:
        f.write("{} {}\n".format(router.x, router.y))
  
  def read_routers(self, routers_file_path, clear_first=False):
    
    if clear_first:
      for router in list(self.routers):
        self.remove_router(router)
    else:
      if len(self.routers) > 0:
        raise ValueError("trying to read solution file but given building has routers")
    
    with open(routers_file_path, 'r') as f:
      for line in f.readlines():
        l = line.rstrip()
        [x, y] = [int(a) for a in l.split(' ')]
        self.add_router(self.grid[x][y])
    print("read routers: {}, targets covered: {}".format(len(self.routers), self.numTargetsCovered))
  
  def add_router(self, cell):
    if cell.hasRouter:
      raise ValueError(str(cell) + " cell already has router")

    cell.hasRouter = True
    self.routers.add(cell)

    for target in cell.S:
      if target.numCoveringRouters == 0:
        self.numTargetsCovered += 1
      target.numCoveringRouters += 1
  
  def remove_router(self, cell):
    if not cell.hasRouter:
      raise ValueError(str(cell) + " cell has no router")

    cell.hasRouter = False
    self.routers.remove(cell)

    for target in cell.S:
      if target.numCoveringRouters == 0:
        raise ValueError(str(cell) + " cell was not counted as covered")
      if target.numCoveringRouters == 1:
        self.numTargetsCovered -= 1
      target.numCoveringRouters -= 1
  
  def get_cells_within_distance(self, cell, distance):
    result = list()
    for x in range(cell.x-distance, cell.x+distance+1):
      for y in range(cell.y - distance, cell.y + distance + 1):
        if 0 <= x < self.H and 0 <= y < self.W:
          result.append(self.grid[x][y])
    return result
  
  @staticmethod
  def compute_gain_if_add_router(cell):
    if cell.hasRouter:
      return -1
    num_new_targets_covered = 0
    for target in cell.S:
      if target.numCoveringRouters == 0:
        num_new_targets_covered += 1
    return num_new_targets_covered
  
  def greedy_add_router_step(self, add_router_gains, max_routers, verbose):
    
    [best_x, best_y] = np.unravel_index(add_router_gains.argmax(), add_router_gains.shape)
    best_cell_for_router = self.grid[best_x][best_y]
    best_gain = add_router_gains[best_x, best_y]
  
    if best_gain <= 0:
      if verbose:
        print("no router addition has a positive gain\n")
      return None
    else:
      self.add_router(best_cell_for_router)
      
      if verbose:
        print("router added at {}, gain: {}, routers {}/{}, score: {}".format(
          best_cell_for_router, best_gain, len(self.routers), max_routers, self.numTargetsCovered))
    
      for c in self.get_cells_within_distance(best_cell_for_router, 2 * self.R):
        add_router_gains[c.x, c.y] = self.compute_gain_if_add_router(c)
      return best_cell_for_router
  
  def greedy_solve(self, max_routers, verbose):
    """
    applies greedy step until no budget is left or no router addition improves score
    this works whether the building is empty or already has routers placed.

    if gain_per_budget_point is True: the gain is computed per budget point.
    """
    
    if len(self.routers) > max_routers:
      raise ValueError("already {} routers, where max_routers = {}".format(len(self.routers), max_routers))
  
    start_time = time.time()
    
    if verbose:
      print("greedy_solve")

    add_router_gains = np.full((self.H, self.W), -1, dtype=int)
    for x in range(0, self.H):
      for y in range(0, self.W):
        if len(self.routers) == 0:
          add_router_gains[x, y] = len(self.grid[x][y].S) # shortcut when no router exists yet
        else:
          add_router_gains[x, y] =  self.compute_gain_if_add_router(self.grid[x][y])
      if verbose and Utils.do_log(x, self.H, 5):
        print("initial gains computation: row {}/{}".format(x, self.H))
    if verbose:
      print(
        "initial gains computation ({} cells) took {:.2f}s".format(self.H * self.W, time.time() - start_time))
  
    if verbose and len(self.routers) == max_routers:
      print("already {} routers (max routers: {})".format(len(self.routers), max_routers))
      return add_router_gains
  
    while len(self.routers) < max_routers:
      new_router = self.greedy_add_router_step(add_router_gains, max_routers, verbose)
      if new_router is None:
        break
  
    if verbose:
      self.print_solution_info()
      print("greedy_solve took {:.2f}s".format(time.time() - start_time))
    return add_router_gains

  def swap_until_local_maximum(self, add_router_gains, verbose):
    """
    keeping the same number of routers, swaps router positions until no router swap brings any improvement

    while any router moves:
      for each router:
        remove router from position
        add router on the best spot of the grid
    """
    
    n_iter = 0
    routers_moved = 1
    while routers_moved > 0:
      n_iter += 1
      routers_moved = 0
      routers_to_move = list(self.routers)
    
      for router in routers_to_move:
        self.remove_router(router)
        for c in self.get_cells_within_distance(router, 2 * self.R):
          add_router_gains[c.x, c.y] = self.compute_gain_if_add_router(c)

        [best_x, best_y] = np.unravel_index(add_router_gains.argmax(), add_router_gains.shape)
        best_cell_for_router = self.grid[best_x][best_y]
        best_gain = add_router_gains[best_x, best_y]
      
        if best_gain > 0:
          self.add_router(best_cell_for_router)
          for c in self.get_cells_within_distance(best_cell_for_router, 2 * self.R):
            add_router_gains[c.x, c.y] = self.compute_gain_if_add_router(c)

          if best_cell_for_router is not router:
            routers_moved += 1
        else:
          routers_moved += 1
          if verbose:
            print("router not replaced")
    
      if verbose:
        print("swaps iteration {}: {}/{} routers have moved, score: {}".format(
          n_iter, routers_moved, len(routers_to_move), self.numTargetsCovered))

  def greedy_solve_with_random_improvements(self, max_routers, routers_file_path, num_iterations,  verbose):
    """ goes forever if num_iterations <= 0"""
    
    if verbose:
      print("greedy_solve_with_random_improvements")
  
    best_score = self.numTargetsCovered
  
    add_router_gains = self.greedy_solve(max_routers, verbose)
    
    n_iter = 0
    while True:
    
      # remove 10% random routers
      routers_to_remove = list(self.routers)
      random.shuffle(routers_to_remove)
      routers_to_remove = routers_to_remove[:int(len(routers_to_remove) / 10)]
    
      cells_to_update = set()
      for router in routers_to_remove:
        self.remove_router(router)
        cells_to_update.update(self.get_cells_within_distance(router, 2 * self.R))
      for cell in cells_to_update:
        add_router_gains[cell.x, cell.y] = self.compute_gain_if_add_router(cell)
      
      # fill greedily
      new_router = self.grid[0][0]  # just need a not none value
      while new_router is not None and len(self.routers) < max_routers:
        new_router = self.greedy_add_router_step(add_router_gains, max_routers, verbose=False)
      
      if verbose:
        print("after removing {} random routers and greedy, num routers = {} and score = {}".format(
          len(routers_to_remove), len(self.routers), self.numTargetsCovered))
      
      # save result if improved
      if self.numTargetsCovered > best_score:
        best_score = self.numTargetsCovered
        self.print_solution_info()
        if routers_file_path is not None:
          self.write_routers(routers_file_path)
        
      n_iter += 1
      if verbose:
        print("iteration {}/{} done, score: {}".format(n_iter, num_iterations, self.numTargetsCovered))
      if 0 < num_iterations <= n_iter:
        break
    
    return add_router_gains

def grand_loop(sample):
  # perform random iterations, and from time to time, swap to local max and see if this beats the best score
  # note: swap to local max forces a rigid local max, so we do not follow the random iterations after local max but
  # from where we left random improvements.
  
  max_routers_dict = {"example":2, "opera":854, "rue_de_londres":189}
  max_routers = max_routers_dict[sample]

  dir_path = os.path.dirname(os.path.realpath(__file__))
  input_file_path = os.path.join(dir_path, "input", sample + ".in")
  covered_targets_file_path = os.path.join(dir_path, "input", "covered_targets", "ct_" + sample + ".txt")
  routers_file_path = os.path.join(dir_path, "max_coverage", sample + str(max_routers) + ".txt")

  general_problem_building = Utils.read_problem_statement(input_file_path)
  Utils.read_targets_covered(general_problem_building, covered_targets_file_path)
  mc_building = MCBuilding(general_problem_building)
  del general_problem_building

  best_score = mc_building.numTargetsCovered

  while True:
    # random exploration (but can also bring improvements), start where we left off
    mc_building.read_routers(routers_file_path, clear_first=True)
    add_router_gains = mc_building.greedy_solve_with_random_improvements(max_routers, routers_file_path, 100, verbose=True)
    mc_building.write_routers(routers_file_path)
    
    # swaps for maximization, but save on a different file, if better
    mc_building.swap_until_local_maximum(add_router_gains, verbose=True)
    if mc_building.numTargetsCovered > best_score:
      best_score = mc_building.numTargetsCovered
      specific_routers_file_path = os.path.join(dir_path, "max_coverage", "{}{}_{}.txt".format(
        sample,len(mc_building.routers), mc_building.numTargetsCovered))
      mc_building.write_routers(specific_routers_file_path)
      
    

def main(sample):
  random.seed(17)
  
  if sample not in ["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]:
    raise ValueError("unknown sample: {}".format(sample))
  
  print("\n####### {}\n".format(sample))
  grand_loop(sample)
  

if __name__ == "__main__":
  # main("example")
  # main("opera")
  main("rue_de_londres")


