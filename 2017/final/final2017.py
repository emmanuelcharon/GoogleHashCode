"""
Created on 7 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import random
import os
import logging

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
      
      

    f.write("{}\n".format(building.numRouters))
    for x in range(0, building.H):
      for y in range(0, building.W):
        if building.grid[x][y].hasRouter:
          f.write("{} {}\n".format(x, y))
    

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

  if num_routers != building.numRouters:
    logging.error("expected {} router cells but added {}".format(num_routers, building.numRouters))


class Cell:
  
  def __init__(self, r, c, isTarget=False, isWall=False, isInitialBackboneCell=False):
    self.r = r
    self.c = c
    self.isTarget = isTarget
    self.isWall = isWall
    self.isInitialBackboneCell = isInitialBackboneCell
    
    self.hasRouter = False # whether there is a router on this cell
    self.isBackbone = self.isInitialBackboneCell # whether this cell is connected to the backbone
    self.coveringRouters = set() # list of routers covering cell
    
    # cache
    self.gain = -1

  def __repr__(self):
    return "({},{})".format(self.r, self.c);
  
  def __str__(self):
    return str(self.__repr__())
  
  def distance(self, other):
    return abs(self.r-other.r) + abs(self.c-other.c)
  
  
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
    self.grid = None # array of arrays of cells
    
    # cache
    self.initialState = True
    self.numRouters = 0
    self.numBackBoneCells = 0
    self.targetCellsCovered = 0

  def __repr__(self):
    return "H={}, W={}, R={}, Pb={}, Pr={}, B={}, br={}, bc={}".format(
      self.H, self.W, self.R, self.Pb,
      self.Pr, self.B, self.br, self.bc);

  def get_initial_backbone_cell(self):
    return self.grid[self.br][self.bc]

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

  def get_closest_backbone_cell(self, cell):
    if self.initialState:
      return self.get_initial_backbone_cell()
    
    if cell.isBackbone:
      return cell
    for d in range(1, cell.distance(self.get_initial_backbone_cell())):
      for other_cell in self.get_cells_at_distance(cell, d):
        if other_cell.isBackbone:
          return other_cell
    return self.get_initial_backbone_cell()
  
  def has_neighbor_backbone(self, cell):
    for neighbor in self.get_cells_at_distance(cell, 1):
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
      self.initialState = False
  
  def is_covered(self, router_cell, cell_in_range):
    for x in range( min(router_cell.r, cell_in_range.r), min(router_cell.r, cell_in_range.r)):
      for y in range(min(router_cell.c, cell_in_range.c), min(router_cell.c, cell_in_range.c)):
        if self.grid[x][y].isWall:
          return False
    return True
    
  def add_router_to_cell(self, cell):
    if cell.hasRouter:
      logging.warning(str(cell) + " cell already has router")
    elif cell.isWall:
      logging.warning(str(cell) + " cell is a wall")
    elif not cell.isBackbone:
      logging.warning(str(cell) + " cell is not backbone")
    else:
      cell.hasRouter = True
      self.numRouters += 1
      self.initialState = False
      
      new_covered_cells = 0
      for cell_in_range in self.get_cells_within_distance(cell, self.R):
        if cell_in_range.isTarget and self.is_covered(cell, cell_in_range):
          if len(cell_in_range.coveringRouters) == 0:
            new_covered_cells += 1
          cell_in_range.coveringRouters.add(cell)
      self.targetCellsCovered += new_covered_cells

  def gain_if_add_router_to_cell(self, cell):
    if cell.hasRouter:
      logging.warning(str(cell) + " cell already has router")
    elif cell.isWall:
      logging.warning(str(cell) + " cell is a wall")
    else:
      new_covered_cells = 0
      for cell_in_range in self.get_cells_within_distance(cell, self.R):
        if cell_in_range.isTarget and self.is_covered(cell, cell_in_range):
          if len(cell_in_range.coveringRouters) == 0:
            new_covered_cells += 1

      closest = self.get_closest_backbone_cell(cell)
      cost = cell.distance(closest) * self.Pb + self.Pr
      
      if self.get_remaining_budget() < cost:
        #logging.warning(str(cell) + " not enough budget")
        return -1
      return 1000*new_covered_cells - cost

  def get_backbone_path(self, backbone_cell, target_cell):
    # go in diagonal then straight
    result = []
    
    next_cell = backbone_cell
    while next_cell.r != target_cell.r or next_cell.c != target_cell.c:
      direction_r = 1 if target_cell.r > next_cell.r else 0 if target_cell.r == next_cell.r else -1
      direction_c = 1 if target_cell.c > next_cell.c else 0 if target_cell.c == next_cell.c else -1
      next_cell = self.grid[next_cell.r+direction_r][next_cell.c+direction_c]
      result.append(next_cell)
    return result
    
  def get_num_total_targets(self):
    result = 0
    for x in range(0, self.H):
      for y in range(0, self.W):
        if self.grid[x][y].isTarget:
          result+=1
    return result
  
  def get_remaining_budget(self):
    return self.B - self.numBackBoneCells * self.Pb - self.numRouters * self.Pr
  
  def get_score(self):
    return 1000 * self.targetCellsCovered + self.get_remaining_budget()


def solve(building):
  
  initial_gains_computed = 0
  
  for x in range(0, building.H):
    if x%(building.H/10)==0:
      print("row {}/{}".format(x, building.H))
    
    for y in range(0, building.W):
      cell = building.grid[x][y]
  
      keep = x%building.R == 0 and y%building.R==0
      keep = keep or cell.distance(building.get_initial_backbone_cell()) < building.R
      
      if keep and not cell.hasRouter and not cell.isWall:
        cell.gain = building.gain_if_add_router_to_cell(cell)
        initial_gains_computed += 1
      else:
        cell.gain = -1

  print("initial gains computed for sparse grid of cells: {}/{} cells.".format(initial_gains_computed, building.H*building.W))

  for n_iter in range(0, building.H*building.W):
    if building.get_remaining_budget() < building.Pr:
      print("stopping because remaining budget is lower than Pr")
      break
    
    best_cell_for_router = None
    best_gain = 0
    
    for x in range(0, building.H):
      for y in range(0, building.W):
        cell = building.grid[x][y]
        if cell.gain > best_gain:
            best_gain = cell.gain
            best_cell_for_router = cell
    
    print(best_gain)
    
    if best_cell_for_router:
      closest = building.get_closest_backbone_cell(best_cell_for_router)
      path = building.get_backbone_path(closest, best_cell_for_router)
      
      for cell in path:
        building.add_backbone_to_cell(cell)
      building.add_router_to_cell(best_cell_for_router)
      
      # now update gains cells in this area: all cells in a range 2r of this new router
      for cell in building.get_cells_within_distance(best_cell_for_router, 2*building.R):
        if not cell.hasRouter and not cell.isWall:
          cell.gain = building.gain_if_add_router_to_cell(cell)
        else:
          cell.gain = -1
      
      
      print("router added at {}, with {} new backbone cells, new score: {}".format(
        best_cell_for_router, len(path), building.get_score()))
      
    else:
      print("stopping because no router addition has a positive gain")
      break

def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]
  scores = []
  
  for sample in samples:
    print("\n####### {}\n".format(sample))
    random.seed(17)  # set the seed for reproducibility
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    
    building = read_building(input_file_path)
    print(building)
    read_solution(building, output_file_path)
    #solve(building)
    
    print("backbone cells: {}, routers: {}, targets covered: {}/{}, remaining budget {}/{}, score: {}".format(
      building.numBackBoneCells, building.numRouters,
      building.targetCellsCovered, building.get_num_total_targets(),
      building.get_remaining_budget(), building.B,
      building.get_score()))
    
    write_solution(building, output_file_path)
    
    if sample != "example":
      scores.append(building.get_score())
  
  print("scores: {} => {} i.e {}M".format(scores, sum(scores), sum(scores)/1000000))

if __name__ == "__main__":
  main()
  # next: write code to be able to remove a router and corresponding backbones, and update local scores
  # then: when optimisation step gives 0 improvement, destroy 10% of routers randomly and simultaneously
  # then keep optimising with the same strategy
  # then: write solution to file if best score
  
  
  