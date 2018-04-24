"""
Created on 20 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import random
import os
import logging
import numpy as np
import time

def read_building(input_file_path):
  """ read input file and return a building (which is a problem statement instance)"""
  [H, W, R, Pb, Pr, B, br, bc] = [0, 0, 0, 0, 0, 0, 0, 0]
  grid = []
  # efficient for large files
  with open(input_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        ls = [int(a) for a in l.split(' ')]
        H = ls[0]
        W = ls[1]
        R = ls[2]
      elif line_count == 1:
        ls = [int(a) for a in l.split(' ')]
        Pb = ls[0]
        Pr = ls[1]
        B = ls[2]
      elif line_count == 2:
        ls = [int(a) for a in l.split(' ')]
        br = ls[0]
        bc = ls[1]
      else:
        grid.append(l) # '-' = void, '.' = target, '#' = wall
      line_count += 1
  
  building =  Building(H, W, R, Pb, Pr, B, br, bc)
  building.grid = []
  for r in range(0, building.H):
    row = []
    for c in range(0, building.W):
      cell = Cell(r, c, isTarget=grid[r][c]=='.', isWall=grid[r][c]=='#', isInitialBackboneCell= r==br and c==bc)
      row.append(cell)
    building.grid.append(row)

  return building


def write_solution(building, output_file_path):
  with open(output_file_path, 'w') as f:
    
    f.write("{}\n".format(building.numBackBoneCells))
    backbone_cells_to_write = [building.get_initial_backbone_cell()]
    backbone_cells_written = set()

    while len(backbone_cells_to_write)>0:
      bb = backbone_cells_to_write.pop()
      if bb in backbone_cells_written:
        continue
      if bb != building.get_initial_backbone_cell():
        f.write("{} {}\n".format(bb.r, bb.c))
      backbone_cells_written.add(bb)
      
      # add its neighbors backbone cells that have not been added yet
      neighbors = building.get_cells_at_distance(bb, 1)
      new_bb_neighbors = [cell for cell in neighbors if cell.isBackbone and cell not in backbone_cells_written]
      backbone_cells_to_write.extend(new_bb_neighbors)
      
      

    f.write("{}\n".format(len(building.routers)))
    
    for router_cell in building.routers:
      f.write("{} {}\n".format(router_cell.r, router_cell.c))
    
def write_visual_solution(building, visual_output_file_path):
  with open(visual_output_file_path, 'w') as f:
    for x in range(0, building.H):
      for y in range(0, building.W):
        cell = building.grid[x][y]
        if cell.hasRouter:
          f.write('R')
        elif cell.isInitialBackboneCell:
          f.write('B')
        elif cell.isBackbone:
          f.write('b')
        elif cell.isTarget and len(cell.coveringRouters) > 1:
          f.write('w') # wasted coverage
        elif cell.isTarget and len(cell.coveringRouters) > 0:
          f.write('.')
        elif cell.isTarget:
          f.write('t') # uncovered target
        elif cell.isWall:
          f.write('#')
        else:
          f.write('-')
      f.write('\n');
     

def read_solution(building, solution_file_path):
  """ transforms 'building' parameter """
  num_backbone_cells = 0
  num_routers = 0

  with open(solution_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        num_backbone_cells = int(l)
      elif line_count <= num_backbone_cells:
        [x, y] = [int(a) for a in l.split(' ')]
        building.add_backbone_to_cell(building.grid[x][y])
      elif line_count == 1 + num_backbone_cells:
        num_routers = int(l)
      else:
        [x, y] = [int(a) for a in l.split(' ')]
        building.add_router_to_cell(building.grid[x][y])
      line_count += 1
  
  if num_backbone_cells != building.numBackBoneCells:
    logging.error("expected {} backbone cells but added {}".format(num_backbone_cells, building.numBackBoneCells))

  if num_routers != len(building.routers):
    logging.error("expected {} router cells but added {}".format(num_routers, len(building.routers)))


class Cell:
  
  def __init__(self, r, c, isTarget=False, isWall=False, isInitialBackboneCell=False):
    self.r = r
    self.c = c
    self.isTarget = isTarget
    self.isWall = isWall
    self.isInitialBackboneCell = isInitialBackboneCell
    
    self.hasRouter = False # whether there is a router on this cell
    self.isBackbone = self.isInitialBackboneCell # boolean, whether this cell is connected to the backbone
    
    # cache variables
    self.distance_to_backbone = -1
    self.neighbor_to_backbone_cell = None
    self.closest_backbone_cell = None
    
    self.coveringRouters = set() # cache the set of routers covering cell
    
    # cache the set of target cells that would be covered by a router on this cell
    # (stays empty if this is a wall)
    self._targetsCoveredIfRouter = None
    
    # a convenience integer for temporary operations
    self.mark = 0

  def get_targets_covered_if_router(self, building):
    """
    naive approach is R^4: for each cell in range, checkout the rectangle from potential router (this) to cell
    
    our linear programming approach is R^2: within range, a cell is covered if its two predecessors are covered
    - the two predecessors are the two neighbors out of the 4 direct neighbors (right, top, left, down)
    which are closest to the potential router cell (this)
    - if a cell in range is in the same row or column as this, then it has only 1 predecessor
    - we must be careful in the order in which we go though the cells in range,
    so that predecessors are computed before each cell
    """
    
    if not self._targetsCoveredIfRouter:
      self._targetsCoveredIfRouter = set()
      if not self.isWall:
        
        # we mark cells: 0 = unseens, 1 = covered, -1 = not covered
        
        # mark all cells in range as unseen
        for dx in range(-building.R - 1, building.R + 1):
          for dy in range(-building.R - 1, building.R + 1):
            [x, y] = [self.r + dx, self.c + dy]
            if 0 <= x < building.H and 0 <= y < building.W:
              cell_in_range = building.grid[x][y]
              cell_in_range.mark = 0
  
        # go through each quarter of square: bottom right, top right, bottom left, top left
        successor_pairs = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
        for successor_pair in successor_pairs:
          [px, py] = successor_pair
          [x, y] = [self.r, self.c]

          while 0 <= x < building.H and abs(x - self.r) <= building.R:
            while 0 <= y < building.W and abs(y - self.c) <= building.R:
              cell_in_range = building.grid[x][y]
              if cell_in_range.isWall:
                cell_in_range.mark = -1
              else:  # this is not a wall
                if x == self.r and y == self.c: # no predecessor
                  cell_in_range.mark = 1
                elif x == self.r: # only 1 predecessor
                  if building.grid[x][y - py].mark == 0:
                    logging.error("predecessor has not been marked")
                  cell_in_range.mark = building.grid[x][y - py].mark
                elif y == self.c: # only 1 predecessor
                  if building.grid[x - px][y].mark == 0:
                    logging.error("predecessor has not been marked")
                  cell_in_range.mark = building.grid[x - px][y].mark
                else:  # the regular case: 2 predecessors
                  if building.grid[x - px][y].mark == 0 or building.grid[x][y - py].mark == 0:
                    logging.error("1 or 2 predecessor(s) have not been marked")
                  if building.grid[x - px][y].mark == -1 or building.grid[x][y - py].mark == -1:
                    cell_in_range.mark = -1
                  else:
                    cell_in_range.mark = 1
  
              if cell_in_range.mark == 1 and cell_in_range.isTarget:
                self._targetsCoveredIfRouter.add(cell_in_range)
  
              y = y + py
            [x, y] = [x + px, self.c]
        
    return self._targetsCoveredIfRouter
    
  def __repr__(self):
    return "({},{})".format(self.r, self.c);
  
  def __str__(self):
    return str(self.__repr__())
  
  def compute_backbone_distance(self, other):
    """ the backbone distance between two cells is the number of steps on the grid where steps can be diagonal """
    return max(abs(self.r-other.r), abs(self.c-other.c))
  
class Building:
  
  def __init__(self, H, W, R, Pb, Pr, B, br, bc):
    self.H = H  # num rows
    self.W = W  # num columns
    self.R = R  # router radius
    self.Pb = Pb  # price of connecting 1 cell to backbone
    self.Pr = Pr  # price of placing a router on a backbone cell
    self.B = B  # budget
    self.br = br
    self.bc = bc
    self.grid = None # array of arrays of cells, initialised later
    
    # cache
    self.routers = set()
    self.numBackBoneCells = 0
    self.numTargetCellsCovered = 0

    # for each cell, gain if router is added to it
    self.add_router_gains = np.full((self.H, self.W), -1000000000, dtype=int)

  def __repr__(self):
    return "H={}, W={}, R={}, Pb={}, Pr={}, B={}, br={}, bc={}".format(
      self.H, self.W, self.R, self.Pb,
      self.Pr, self.B, self.br, self.bc)

  def cache_targets_covered_if_router(self):
    """
    compute & save, for each cell, all the target cells that would be covered if it had a router
    complexity is H * W * R^2
    """
    
    for x in range(0, self.H):
      if x==0 or (int(self.H/5)>0 and x % int(self.H / 5) == 0):
        print("computation of covered targets by each possible router cell: row {}/{}".format(x, self.H))
  
      for y in range(0, self.W):
        cell = self.grid[x][y]
        cell.get_targets_covered_if_router(self)

  def get_initial_backbone_cell(self):
    return self.grid[self.br][self.bc]

  def get_neighbors(self, cell):
    """ returns the (up to) 8 neighbor cells in the grid"""
    return self.get_cells_at_distance(cell, 1)
  
  def get_cells_at_distance(self, cell, d):
    """ returns cells exactly at distance d from cell """
    if d<0:
      logging.error("d<0")
      return None
    if d == 0:
      return [cell]
    
    coords = []
    for y in range(cell.c - d, cell.c + d + 1):
      coords.append([cell.r - d, y])
      coords.append([cell.r + d, y])
    for x in range(cell.r - d + 1, cell.r + d):
      coords.append([x, cell.c - d])
      coords.append([x, cell.c + d])
    
    result = []
    for [x, y] in coords:
      if 0 <= x < self.H and 0 <= y < self.W:
        result.append(self.grid[x][y])
    return result

  def get_cells_within_distance(self, cell, d):
    """ returns cells at distance <= d from cell """
    result = []
    for dd in range(0, d+1):
      result.extend(self.get_cells_at_distance(cell, dd))
    return result

  # def get_closest_backbone_cell(self, cell):
  #   if self.initialState:
  #     return self.get_initial_backbone_cell()
  #
  #   if cell.isBackbone:
  #     return cell
  #   for d in range(1, cell.compute_backbone_distance(self.get_initial_backbone_cell())):
  #     for other_cell in self.get_cells_at_distance(cell, d):
  #       if other_cell.isBackbone:
  #         return other_cell
  #   return self.get_initial_backbone_cell()
  
  def has_neighbor_backbone(self, cell):
    for neighbor in self.get_neighbors(cell):
      if neighbor.isBackbone:
        return True
    return False
  
  def add_backbone_to_cell(self, cell):
    if cell.isBackbone:
      logging.warning(str(cell) + " already backbone")
    elif not self.has_neighbor_backbone(cell):
      logging.warning(str(cell) + " no neighbor backbone")
    else:
      cell.isBackbone = True
      self.numBackBoneCells += 1
      
  def remove_backbone_from_cell(self, cell):
    if not cell.isBackbone:
      logging.warning(str(cell) + " is not backbone, cannot remove backbone")
    else:
      cell.isBackbone = False
      self.numBackBoneCells -= 1
  
  def add_router_to_cell(self, cell):
    if cell.hasRouter:
      logging.error(str(cell) + " cell already has router")
    elif cell.isWall:
      logging.warning(str(cell) + " cell is a wall")
    elif not cell.isBackbone:
      logging.warning(str(cell) + " cell is not backbone")
    else:
      cell.hasRouter = True
      self.routers.add(cell)
      
      num_new_covered_targets = 0
      for target in cell.get_targets_covered_if_router(self):
        if len(target.coveringRouters) == 0:
          num_new_covered_targets += 1
        target.coveringRouters.add(cell)
      self.numTargetCellsCovered += num_new_covered_targets

  def remove_router_from_cell(self, cell):
    if not cell.hasRouter:
      logging.error("cell has no router")
    elif cell not in self.routers:
      logging.error("cell is not in routers")
    elif cell.isWall:
      logging.error("wall cell cannot have a router")
    else:
      cell.hasRouter = False
      self.routers.remove(cell)

      num_targets_uncovered = 0
      for target in cell.get_targets_covered_if_router(self):
        if cell not in target.coveringRouters:
          logging.error("a target was not considered cov ered by this router")
        else:
          target.coveringRouters.remove(cell)
          if len(target.coveringRouters) == 0:
            num_targets_uncovered += 1
      self.numTargetCellsCovered -= num_targets_uncovered
  
  def remove_router_and_its_backbones(self, cell):
    self.remove_router_from_cell(cell)
    
    path = self.get_backbone_branch_path(cell)
    for backbone_cell in path:
      self.remove_backbone_from_cell(backbone_cell)
   
  def gain_if_add_router_to_cell(self, cell):
    if cell.hasRouter:
      return -1
    elif cell.isWall:
      return -1
    else:
      if cell.distance_to_backbone == -1:
        logging.error("distance to backbone was not computed")
      
      cost = cell.distance_to_backbone * self.Pb + self.Pr

      if self.get_remaining_budget() < cost:
        return -1

      num_new_covered_targets = 0

      #exp_num_free_targets_further = 0 # experimental

      for target in cell.get_targets_covered_if_router(self):
        if len(target.coveringRouters) == 0:
          num_new_covered_targets += 1
          
          #for neighbor in self.get_neighbors(target):
            #if neighbor.isTarget and neighbor not in cell.get_targets_covered_if_router(self) and len(neighbor.coveringRouters) == 0:
              #exp_num_free_targets_further += 1
        
      #exp_cost = 1 * exp_num_free_targets_further
      
      return 1000*num_new_covered_targets - cost #- exp_cost

  def loss_if_remove_router_from_cell(self, cell, building):
    if not cell.hasRouter:
      logging.error("cell has no router")
      return -1000000000
    elif cell.isWall:
      logging.error("wall cell cannot have a router")
      return -1000000000
    else:
      # compute the number of targets that would stop being covered
      num_targets_uncovered = 0
      for target in cell.get_targets_covered_if_router(building):
        if len(target.coveringRouters) == 1:
          num_targets_uncovered += 1
      
      # compute the number of backbone cells we would be able to remove
      num_removable_backbones = len(self.get_backbone_branch_path(cell))
      
      budget_saved = self.Pr + num_removable_backbones * self.Pb
      loss = budget_saved - 1000 * num_targets_uncovered
      return loss
  
  def get_backbone_branch_path(self, cell):
    """ path stops at either: next router, next intersection or initial backbone cell"""
    
    path = list()
    current_cell = cell
    
    while True:
      next_cell = None
      
      if not current_cell.isBackbone:
        logging.error("next_cell must always be on backbone")
      
      if current_cell.isInitialBackboneCell:
        break
      if current_cell != cell and current_cell.hasRouter:
        break
      
      num_backbone_neighbors = 0
      for neighbor in self.get_neighbors(current_cell):
        if neighbor.isBackbone:
          num_backbone_neighbors += 1
          if len(path) == 0 or neighbor != path[-1]:
            next_cell = neighbor
      
      if num_backbone_neighbors >= 3:  # means this is an intersection
        break
      
      if current_cell != cell and num_backbone_neighbors <= 1:  # means this is an end, but there should not be one
        logging.error("looks like an end, num_backbone_neighbors: "+str(num_backbone_neighbors))
        break
        
      if next_cell is None:
        logging.error("next cell is none, num_backbone_neighbors: "+str(num_backbone_neighbors))
        break
      
      path.append(current_cell)
      current_cell = next_cell
    return path
    
    
    
  def is_branch_end(self, cell):
    if cell.isInitialBackboneCell:
      return True
    if cell.hasRouter:
      return True
    num_backbone_neighbors = 0
    for neighbor in self.get_neighbors(cell):
      if neighbor.isBackbone:
        num_backbone_neighbors += 1
    if num_backbone_neighbors >= 3: # means this is an intersection
      return True
    return False

  def get_backbone_path_basic(self, backbone_cell, target_cell):
    # go in diagonal then straight
    result = []
    
    next_cell = backbone_cell
    while next_cell.r != target_cell.r or next_cell.c != target_cell.c:
      direction_r = 1 if target_cell.r > next_cell.r else 0 if target_cell.r == next_cell.r else -1
      direction_c = 1 if target_cell.c > next_cell.c else 0 if target_cell.c == next_cell.c else -1
      next_cell = self.grid[next_cell.r+direction_r][next_cell.c+direction_c]
      result.append(next_cell)
    return result

  def get_backbone_path(self, target_cell):
    result = []
    previous_cell = target_cell
    while not previous_cell.isBackbone:
      result.append(previous_cell)
      previous_cell = previous_cell.neighbor_to_backbone_cell
    result.reverse()
    return result
  
  def get_num_total_targets(self):
    result = 0
    for x in range(0, self.H):
      for y in range(0, self.W):
        if self.grid[x][y].isTarget:
          result+=1
    return result
  
  def count_wasted_coverage(self):
    result = 0
    for x in range(0, self.H):
      for y in range(0, self.W):
        cell = self.grid[x][y]
        if cell.isTarget and len(cell.coveringRouters) > 1:
          result += len(cell.coveringRouters) - 1
    return result
  
  def get_remaining_budget(self):
    return self.B - self.numBackBoneCells * self.Pb - len(self.routers) * self.Pr
  
  def get_score(self):
    return 1000 * self.numTargetCellsCovered + self.get_remaining_budget()

  def compute_cell_distances_to_backbone(self, start_backbone_cells = None, update_backbone_gains = False):
    """
    computes the actual distance of each cell to the back bone, and one of the corresponding closest backbone cells
    time is linear in the number of cells: self.H * self.W
    we compute all cells at distance 0, then all cells at distance 1, etc...
    
    if start_cells = None, we erase any information saved and we re-compute the distance for the whole grid
    
    """
    all_updated_cells = set()
    cells_at_distance_n = list()
    
    if start_backbone_cells is not None:
      for cell in start_backbone_cells:
        if not cell.isBackbone:
          logging.error("start cells must be backbone cells")
        cell.distance_to_backbone = 0
        cell.closest_backbone_cell = cell
        cell.neighbor_to_backbone_cell = None
        cells_at_distance_n.append(cell)
    else:
      for x in range(0, self.H):
        for y in range(0, self.W):
          cell = self.grid[x][y]
          if cell.isBackbone:
            cell.distance_to_backbone = 0
            cell.closest_backbone_cell = cell
            cell.neighbor_to_backbone_cell = None
            cells_at_distance_n.append(cell)
          else:
            cell.distance_to_backbone = -1
            cell.closest_backbone_cell = None
            cell.neighbor_to_backbone_cell = None

    while len(cells_at_distance_n) > 0:
      all_updated_cells.update(cells_at_distance_n)
      
      cells_at_distance_n_plus_1 = list()
      
      for cell_at_distance_n in cells_at_distance_n:
        for neighbor in self.get_neighbors(cell_at_distance_n):
          if neighbor.distance_to_backbone == -1 or neighbor.distance_to_backbone > 1 + cell_at_distance_n.distance_to_backbone:
            
            if neighbor.distance_to_backbone > 1 + cell_at_distance_n.distance_to_backbone:
              if update_backbone_gains and not neighbor.hasRouter and not neighbor.isWall:
                distance_gained = neighbor.distance_to_backbone - 1 - cell_at_distance_n.distance_to_backbone
                self.add_router_gains[neighbor.r, neighbor.c] += distance_gained
            
            neighbor.distance_to_backbone = 1 + cell_at_distance_n.distance_to_backbone
            neighbor.closest_backbone_cell = cell_at_distance_n.closest_backbone_cell
            neighbor.neighbor_to_backbone_cell = cell_at_distance_n
        
            # it could have more than one way to reach backbone at this length, but we just save 1 way
            cells_at_distance_n_plus_1.append(neighbor)

      #print("found {} cells at distance {} from backbone".format(
      #  len(cells_at_distance_n_plus_1), 1 + cells_at_distance_n[0].distance_to_backbone))

      cells_at_distance_n = cells_at_distance_n_plus_1

    #print("'distance to backbone' updated cells: {}/{}".format(len(all_updated_cells), self.H * self.W))
    return all_updated_cells

  def recompute_backbone_tree(self):
    """
    create a spanning tree over the initial backbone and the routers, and replace backbone with it
    we use an algorithm close to Prim's MST algo,
    we find a tree close to a minimal spanning tree over theses cells
    """
    
    routers_to_add = set(self.routers)
    
    for router in routers_to_add:
      self.remove_router_from_cell(router)
    for x in range(0, self.H):
      for y in range(0, self.W):
        cell = self.grid[x][y]
        if cell.isBackbone and not cell.isInitialBackboneCell:
          self.remove_backbone_from_cell(cell)

    self.compute_cell_distances_to_backbone()
    
    while len(routers_to_add)>0:
      """ select the router closest to backbone, add it """
      
      min_distance = self.H + self.W + 1
      closest_router = None
      
      for router in routers_to_add:
        if router.distance_to_backbone < min_distance:
          min_distance = router.distance_to_backbone
          closest_router = router
          
      if closest_router is None:
        logging.error("closest_router is None")
        break
      routers_to_add.remove(closest_router)
      path = self.get_backbone_path(closest_router)

      for cell in path:
        self.add_backbone_to_cell(cell)
      self.add_router_to_cell(closest_router)

      self.compute_cell_distances_to_backbone(start_backbone_cells=path, update_backbone_gains=False)
      
    # recompute all gains
    #for x in range(0, self.H):
    #  if x == 0 or (int(self.H / 5) > 0 and x % int(self.H / 5) == 0):
    #    print("computation of initial backbone gains: row {}/{}".format(x, self.H))
    #  for y in range(0, self.W):
    #    self.add_router_gains[x, y] = self.gain_if_add_router_to_cell(self.grid[x][y])
    #print("gains re-computed")
    

  def greedy_step(self):
     """
     assumes that all cached values are up-to-date
     
     selects the cell with the best gain if a router is added to it, then adds backbones and router to it
     updates cached values, including "add_router_gains" BUT NOT "remove_router_losses"
     returns the new router cell
     
     if no cell has a positive gain, does not select any cell and returns None
     """

     if self.get_remaining_budget() < self.Pr:
       print("remaining budget is lower than Pr (the price of 1 router)\n")
       return None

     [best_x, best_y] = np.unravel_index(self.add_router_gains.argmax(), self.add_router_gains.shape)
     best_cell_for_router = self.grid[best_x][best_y]
     best_gain = self.add_router_gains[best_x, best_y]

     if best_gain <= 0:
       print("no router addition has a positive gain\n")
       return None
     else:
       # path = building.get_backbone_path_basic(best_cell_for_router.closest_backbone_cell, best_cell_for_router)
       path = self.get_backbone_path(best_cell_for_router)
  
       for cell in path:
         self.add_backbone_to_cell(cell)
       self.add_router_to_cell(best_cell_for_router)
  
       print("router added at {}, with {} new backbone cells, gain: {}, remaining budget {},  new score: {}".format(
         best_cell_for_router, len(path), best_gain, self.get_remaining_budget(), self.get_score()))
  
       self.compute_cell_distances_to_backbone(start_backbone_cells=path, update_backbone_gains=True)
       
       cells_to_update = self.get_cells_within_distance(best_cell_for_router, 2 * self.R)
       for cell in cells_to_update:
         self.add_router_gains[cell.r, cell.c] = self.gain_if_add_router_to_cell(cell)
       return best_cell_for_router

  def optimize_step(self):
    
    # select a random router, remove it, and place it in the optimal stop in a neighborhood of where it was
    router = random.choice(list(self.routers))
    self.remove_router_and_its_backbones(router)
    self.compute_cell_distances_to_backbone()

    max_gain_cells = list()
    max_gain = 0

    for cell in self.get_cells_within_distance(router, 2):
      gain = self.gain_if_add_router_to_cell(cell)
      if gain > max_gain:
        max_gain_cells = [cell]
        max_gain = gain
      elif gain == max_gain:
        max_gain_cells.append(cell)
    
    if len(max_gain_cells) == 0:
      logging.error("max_gain_cells of length 0")
    
    new_router = random.choice(max_gain_cells)
    
    

    path = self.get_backbone_path(new_router)

    for cell in path:
      self.add_backbone_to_cell(cell)
    self.add_router_to_cell(new_router)
    
    self.compute_cell_distances_to_backbone(start_backbone_cells=path, update_backbone_gains=False)
    
    return new_router != router
    
def optimize(building, output_file_path, visual_output_file_path):
  """ should follow solve """
  
  budget_threshold =  4*building.R*building.Pb + building.Pr
  
  n=0
  while True:
    n += 1
    
    
    while building.get_remaining_budget() > budget_threshold:
      while building.greedy_step() is not None:
        pass
     
      print(" backbone cells: {},\n routers: {},\n targets covered: {}/{},\n "
            "wasted coverage: {},\n remaining budget {}/{},\n score: {}\n".format(
        building.numBackBoneCells, len(building.routers),
        building.numTargetCellsCovered, building.get_num_total_targets(),
        building.count_wasted_coverage(),
        building.get_remaining_budget(), building.B,
        building.get_score()))

      write_solution(building, output_file_path)
      write_visual_solution(building, visual_output_file_path)
      
    if building.optimize_step():
      print("n = {}, remaining budget = {}, score = {}".format(n, building.get_remaining_budget(), building.get_score()))

def solve(building):
  
  start_time = time.time()
  
  building.cache_targets_covered_if_router() # pre-compute these so we know which initial steps are slow
  building.compute_cell_distances_to_backbone()
  
  # compute initial gains
  for x in range(0, building.H):
    if x == 0 or (int(building.H / 5) > 0 and x % int(building.H / 5) == 0):
      print("computation of initial backbone gains: row {}/{}".format(x, building.H))
    for y in range(0, building.W):
      building.add_router_gains[x, y] = building.gain_if_add_router_to_cell(building.grid[x][y])
  print("initial gains computed")

  n_rm = 0

  while True:
    new_router = building.greedy_step()
    if new_router is None:
      if n_rm >= 0:
        break
      else:
        n_rm += 1
      
      print("n_rm = {}".format(n_rm))
      
      all_routers = list(building.routers)

      # remove 10% worst routers
      #for router_cell in all_routers:
      #  router_cell.mark = building.loss_if_remove_router_from_cell(router_cell, building)
      
      #all_routers.sort(key=lambda _: _.mark)
      #routers_to_remove = all_routers[:max(1,int(len(all_routers)/10))]
      
      # remove 10% random routers
      
      # remove routers in an area
      routers_to_remove = list()
      center_cell = building.grid[random.randrange(building.H)][random.randrange(building.W)]
      for router_cell in all_routers:
        if router_cell.compute_backbone_distance(center_cell) < building.H/4:
          routers_to_remove.append(router_cell)
      
      print("removing {}/{} routers".format(len(routers_to_remove), len(all_routers)))
      
      cells_to_update = set()
      for router_cell in routers_to_remove:
        building.remove_router_and_its_backbones(router_cell)
        cells_to_update.update(building.get_cells_within_distance(router_cell, 2 * building.R))
      
      building.compute_cell_distances_to_backbone(update_backbone_gains=True)
      for cell in cells_to_update:
        building.add_router_gains[cell.r, cell.c] = building.gain_if_add_router_to_cell(cell)

  print("solve time: {}s".format(time.time()-start_time))
  # 0.2s, 75s, 349s, 414s, 1087s

def main():
  random.seed(17) # reproducibility
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  #samples = ["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]
  
  samples = ["opera"]
  scores = []
  
  for sample in samples:
    print("\n####### {}\n".format(sample))
    random.seed(17)  # set the seed for reproducibility
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    visual_output_file_path = os.path.join(dir_path, "visual_outputs", "visual_" + sample + ".txt")

    building = read_building(input_file_path)
    print(building)
    read_solution(building, output_file_path)
    
    #solve(building)
    #building.recompute_backbone_tree()
    #optimize(building, output_file_path, visual_output_file_path)

    print(" backbone cells: {},\n routers: {},\n targets covered: {}/{},\n "
          "wasted coverage: {},\n remaining budget {}/{},\n score: {}\n".format(
      building.numBackBoneCells, len(building.routers),
      building.numTargetCellsCovered, building.get_num_total_targets(),
      building.count_wasted_coverage(),
      building.get_remaining_budget(), building.B,
      building.get_score()))

    write_solution(building, output_file_path)
    write_visual_solution(building, visual_output_file_path)

    if sample != "example":
      scores.append(building.get_score())
  
  print("scores: {} => {} i.e {}M".format(scores, sum(scores), int(sum(scores)/1000000)))

if __name__ == "__main__":
  main()
  

  
  
  