import numpy as np
import random
import time
import logging

class Cell:
  
  def __init__(self, r, c, isTarget, isWall, isInitialBackboneCell, building):
    # static variables
    self.r = r
    self.c = c
    self.isTarget = isTarget
    self.isWall = isWall
    self.isInitialBackboneCell = isInitialBackboneCell
    self.targetsCoveredIfRouter = None  # the set of target cells that would be covered by a router on this cell
                                        # computed once then cached (empty set if this is a wall)
    self.building = building # a reference to the building, sometimes we need it.
    
    # dynamic variables
    self.hasRouter = False  # whether there is a router on this cell
    self.isBackbone = self.isInitialBackboneCell  # boolean, whether this cell is currently connected to the backbone
    self.coveringRouters = set()  # the set of routers currently covering this cell
    self.mark = 0  # a convenience integer for temporary operations
  
  def get_targets_covered_if_router(self):
    if self.targetsCoveredIfRouter is None:
      self.targetsCoveredIfRouter = self.building.compute_targets_covered_if_router_lp(cell=self)
    return self.targetsCoveredIfRouter
  
  def __repr__(self):
    return "({},{})".format(self.r, self.c);
  
  def __str__(self):
    return str(self.__repr__())
  
  def chebyshev_distance(self, other):
    """ the backbone distance between two cells is the number of steps on the grid where steps can be diagonal """
    return max(abs(self.r - other.r), abs(self.c - other.c))

class Utils:
  """ regroups Input/Output and other useful methods. """
  
  @staticmethod
  def read_problem_statement(input_file_path):
    """ read input file and return a building """
    [H, W, R, Pb, Pr, B, br, bc] = [0, 0, 0, 0, 0, 0, 0, 0]
    grid = []
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
          grid.append(l)  # '-' = void, '.' = target, '#' = wall
        line_count += 1
    
    building = Building(H, W, R, Pb, Pr, B, br, bc)
    building.grid = []
    for r in range(0, building.H):
      row = []
      for c in range(0, building.W):
        cell = Cell(r, c, isTarget=grid[r][c] == '.', isWall=grid[r][c] == '#',
                    isInitialBackboneCell=r == br and c == bc, building=building)
        row.append(cell)
      building.grid.append(row)
    
    return building
  
  @staticmethod
  def write_solution(building, output_file_path):
    with open(output_file_path, 'w') as f:
      
      # 1. backbones (we do a BFS starting at the initial backbone cell)
      f.write("{}\n".format(building.numBackBoneCells))
      backbone_cells_to_write = [building.get_initial_backbone_cell()]
      backbone_cells_written = set()
      while len(backbone_cells_to_write) > 0:
        bb = backbone_cells_to_write.pop()
        if bb in backbone_cells_written:
          continue
        if bb != building.get_initial_backbone_cell():
          f.write("{} {}\n".format(bb.r, bb.c))
        backbone_cells_written.add(bb)
        
        # add its neighbors backbone cells that have not been added yet
        neighbors = building.get_neighbors(bb)
        new_bb_neighbors = [cell for cell in neighbors if cell.isBackbone and cell not in backbone_cells_written]
        backbone_cells_to_write.extend(new_bb_neighbors)
      
      # 2. routers
      f.write("{}\n".format(len(building.routers)))
      for router_cell in building.routers:
        f.write("{} {}\n".format(router_cell.r, router_cell.c))
  
  @staticmethod
  def write_visual_solution(building, visual_output_file_path):
    """
    This writes a file that looks like the input file,
    but where characters on cells show what we put on each cell.
    """
    
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
            f.write('w')  # wasted coverage (target covered twice or more)
          elif cell.isTarget and len(cell.coveringRouters) > 0:
            f.write('.')
          elif cell.isTarget:
            f.write('t')  # uncovered target
          elif cell.isWall:
            f.write('#')
          else:
            f.write('-')
        f.write('\n')
  
  @staticmethod
  def read_solution(building, solution_file_path):
    """ transforms 'building' parameter, adding the backbones and routers found in the solution file """
    
    if building.numTargetCellsCovered > 0 or building.numBackBoneCells > 0 or len(building.routers) > 0:
      raise ValueError("building must be empty when reading a solution file.")
    
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
    
    print("read solution")
    building.print_solution_info()
  
  @staticmethod
  def do_log(index, max_index, num_logs, log_beginning_until_index = 0):
    
    if index <= log_beginning_until_index:
      return True
    log_interval = int(max_index / num_logs)
    if log_interval > 0 and index % log_interval == 0:
      return True
    return False

class Building:
  """ represents a problem instance, as well as a solution """
  
  def __init__(self, H, W, R, Pb, Pr, B, br, bc):
    
    # static variables
    self.H = H  # num rows
    self.W = W  # num columns
    self.R = R  # router radius
    self.Pb = Pb  # price of connecting 1 cell to backbone
    self.Pr = Pr  # price of placing a router on a backbone cell
    self.B = B  # budget
    self.br = br
    self.bc = bc
    self.grid = None  # array of arrays of cells, initialised later
    
    # dynamic & cache variables
    self.routers = set()  # we could also go through all cells to get this
    self.numBackBoneCells = 0
    self.numTargetCellsCovered = 0
  
  def __repr__(self):
    return "H={}, W={}, R={}, Pb={}, Pr={}, B={}, br={}, bc={}".format(
      self.H, self.W, self.R, self.Pb,
      self.Pr, self.B, self.br, self.bc)
  
  def print_solution_info(self):
    print(" backbone cells: {},\n routers: {},\n targets covered: {}/{},\n "
          "wasted coverage: {},\n remaining budget {}/{},\n score: {}\n".format(
      self.numBackBoneCells, len(self.routers),
      self.numTargetCellsCovered, self.compute_num_total_targets(),
      self.count_wasted_coverage(),
      self.get_remaining_budget(), self.B,
      self.get_score()))

  def compute_num_total_targets(self):
    result = 0
    for x in range(0, self.H):
      for y in range(0, self.W):
        if self.grid[x][y].isTarget:
          result += 1
    return result

  def get_remaining_budget(self):
    return self.B - self.numBackBoneCells * self.Pb - len(self.routers) * self.Pr

  def get_score(self):
    return 1000 * self.numTargetCellsCovered + self.get_remaining_budget()

  def count_wasted_coverage(self):
    result = 0
    for x in range(0, self.H):
      for y in range(0, self.W):
        cell = self.grid[x][y]
        if cell.isTarget and len(cell.coveringRouters) > 1:
          result += len(cell.coveringRouters) - 1
    return result
  
  def get_all_cells_as_list(self):
    result = list()
    for row in self.grid:
      result.extend(row)
    return result
  
  def compute_targets_covered_if_router_naive(self, cell):
    """
    Returns the set of targets covered when this cell has a router.
    
    This naive approach takes R^4 time.
    
    For each cell in range (there are R^2),
    check if there is a wall inside the rectangle from potential router (this) to cell (check R^2 cells).
    """
    
    result = set()
    if not cell.isWall:
      for target in self.get_cells_within_distance(cell, self.R):
        if target.isTarget:
          reaches = True
          for x in range(min(cell.r, target.r), max(cell.r, target.r) + 1):
            for y in range(min(cell.c, target.c), max(cell.c, target.c) + 1):
              if self.grid[x][y].isWall:
                reaches = False
          if reaches:
            result.add(target)
    return result
  
  def compute_targets_covered_if_router_lp(self, cell):
    """
    Returns the set of targets covered when this cell has a router.
    
    This linear programming approach takes R^2 time.
    
    Within range, a cell is covered if its two predecessors are covered
     - the two predecessors are the two neighbors out of the 4 direct neighbors (right, top, left, down)
     which are closest to the potential router cell
     - if a cell in range is in the same row or column as this, then it has only 1 predecessor
     - we must be careful in the order in which we go though the cells in range,
     so that predecessors are computed before each cell.
    """
    
    if cell.isWall:
      return set()
    
    result = set()
    
    # we mark cells: 0 = unseen, 1 = covered (void or target cell), -1 = not covered
    for cell_in_range in self.get_cells_within_distance(cell, self.R):
      cell_in_range.mark = 0
    
    # go through each quarter of square: bottom right, top right, bottom left, top left
    # we must start from central cell and cover the square quarter in order
    successor_pairs = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    for successor_pair in successor_pairs:
      [px, py] = successor_pair
      [x, y] = [cell.r, cell.c]
      
      while 0 <= x < self.H and abs(x - cell.r) <= self.R:
        while 0 <= y < self.W and abs(y - cell.c) <= self.R:
          cell_in_range = self.grid[x][y]
          
          if self.grid[x][y].isWall:
            cell_in_range.mark = -1
          else:  # this is not a wall
            if x == cell.r and y == cell.c:  # no predecessor
              cell_in_range.mark = 1
            elif x == cell.r:  # only 1 predecessor
              if self.grid[x, y - py].mark == 0:
                raise ValueError("predecessor has not been marked, (x,y) = ({},{})".format(x, y))
              cell_in_range.mark = self.grid[x, y - py].mark
            elif y == cell.c:  # only 1 predecessor
              if self.grid[x - px][y].mark == 0:
                raise ValueError("predecessor has not been marked, (x,y) = ({},{})".format(x, y))
              cell_in_range.mark = self.grid[x - px, y].mark
            else:  # the regular case: 2 predecessors
              if self.grid[x - px][y].mark == 0 or self.grid[x][y - py].mark == 0:
                raise ValueError("1 or 2 predecessor(s) have not been marked, (x,y) = ({},{})".format(x, y))
              elif self.grid[x - px][y].mark == -1 or self.grid[x][y - py].mark == -1:
                cell_in_range.mark = -1
              else:
                cell_in_range.mark = 1
          if cell_in_range.mark == 1 and cell_in_range.isTarget:
            result.add(cell_in_range)
          y = y + py
        [x, y] = [x + px, cell.c]
    
    return result
  
  def cache_targets_covered_if_router(self, verbose = True):
    """
    compute & save, for each cell, all the target cells that would be covered if it had a router
    complexity is H * W * R^2
    """
    
    start_time = time.time()
    for x in range(0, self.H):
      for y in range(0, self.W):
        self.grid[x][y].get_targets_covered_if_router()
      if verbose and Utils.do_log(x, self.H, 5):
        print("cached covered targets: row {}/{}".format(x, self.H))
    if verbose:
      print("targets covered cached for {} cells, took {:.2f}s".format(self.H * self.W, time.time() - start_time))
  
  def get_initial_backbone_cell(self):
    return self.grid[self.br][self.bc]
  
  def get_neighbors(self, cell):
    """ returns the (up to) 8 neighbor cells in the grid"""
    return self.get_cells_at_distance(cell, 1)
  
  def get_cells_at_distance(self, cell, d):
    """ returns cells exactly at distance d from cell """
    if d < 0:
      logging.error("d<0")
      return None
    if d == 0:
      return [cell]
    
    coords = []
    for dy in range(- d, d + 1):
      coords.append([cell.r - d, cell.c + dy])
      coords.append([cell.r + d, cell.c + dy])
    for dx in range(- d + 1, d):
      coords.append([cell.r + dx, cell.c - d])
      coords.append([cell.r + dx, cell.c + d])
    
    result = []
    for [x, y] in coords:
      if 0 <= x < self.H and 0 <= y < self.W:
        result.append(self.grid[x][y])
    return result
  
  def get_cells_within_distance(self, cell, d):
    """ returns cells at distance <= d from cell """
    result = []
    for dd in range(0, d + 1):
      result.extend(self.get_cells_at_distance(cell, dd))
    return result
  
  def get_neighbor_backbones(self, cell):
    result = []
    for neighbor in self.get_neighbors(cell):
      if neighbor.isBackbone:
        result.append(neighbor)
    return result
  
  def has_neighbor_backbone(self, cell):
    return len(self.get_neighbor_backbones(cell)) > 0
  
  def add_backbone_to_cell(self, cell, check=True):
    if cell.isBackbone:
      logging.warning(str(cell) + " already backbone")
      return
    if check and not self.has_neighbor_backbone(cell):
      logging.warning(str(cell) + " no neighbor backbone")
      return
    cell.isBackbone = True
    self.numBackBoneCells += 1
  
  def add_router_to_cell(self, cell):
    if cell.hasRouter:
      raise ValueError(str(cell) + " cell already has router")
    elif cell.isWall:
      raise ValueError(str(cell) + " cell is a wall")
    elif not cell.isBackbone:
      raise ValueError(str(cell) + " cell is not backbone")
    else:
      cell.hasRouter = True
      self.routers.add(cell)
      
      num_new_covered_targets = 0
      for target in cell.get_targets_covered_if_router():
        if len(target.coveringRouters) == 0:
          num_new_covered_targets += 1
        target.coveringRouters.add(cell)
      self.numTargetCellsCovered += num_new_covered_targets

  def add_router_and_backbone_path(self, new_router_cell):
    """ returns the path of the new backbones created """
    if new_router_cell.hasRouter:
      raise ValueError("cell already has router")
    else:
      backbone_cell = self.get_closest_backbone_cell(new_router_cell)
      path = self.get_backbone_path_basic(backbone_cell, new_router_cell)
      for cell in path:
        self.add_backbone_to_cell(cell)
      self.add_router_to_cell(new_router_cell)
      return path

  def get_closest_backbone_cell(self, cell):
    """ this goes in increasing squares around cell until we find a backbone cell """
  
    if self.numBackBoneCells == 0:  # shortcut for initial computation
      return self.get_initial_backbone_cell()
    if cell.isBackbone:
      return cell
    for d in range(1, 1 + cell.chebyshev_distance(self.get_initial_backbone_cell())):
      for other_cell in self.get_cells_at_distance(cell, d):
        if other_cell.isBackbone:
          return other_cell
    logging.warning("did not find closest backbone cell, default to initial backbone cell")
    return self.get_initial_backbone_cell()

  def get_backbone_path_basic(self, backbone_cell, target_cell):
    """ goes in diagonal from backbone_cell to target_cell then straight """
    result = []
  
    next_cell = backbone_cell
    while next_cell.r != target_cell.r or next_cell.c != target_cell.c:
      direction_r = 1 if target_cell.r > next_cell.r else 0 if target_cell.r == next_cell.r else -1
      direction_c = 1 if target_cell.c > next_cell.c else 0 if target_cell.c == next_cell.c else -1
      next_cell = self.grid[next_cell.r + direction_r][next_cell.c + direction_c]
      result.append(next_cell)
  
    if len(result) != backbone_cell.chebyshev_distance(target_cell):
      print("ERROR: backbone distance from {} to {} =  {}, basic path length: {}".format(
        backbone_cell, target_cell, backbone_cell.chebyshev_distance(target_cell), len(result)))
  
    return result

  def compute_gain_if_add_router_to_cell(self, cell, gain_per_budget_point):
    """
    this function does no modification at all on the input,
    except if some cached values are computed on the fly, like cell.targetsCoveredIfRouter
    
    if gain_per_budget_point==true: we divide the gain by the budget required to put it.
    """
    if cell.hasRouter:
      return -1
    elif cell.isWall or not cell.isTarget:
      return -1
    elif len(cell.get_targets_covered_if_router()) == 0:
      return -1
    else:
      closest_backbone_cell = self.get_closest_backbone_cell(cell)
      distance_to_backbone = cell.chebyshev_distance(closest_backbone_cell)
    
      cost = distance_to_backbone * self.Pb + self.Pr
    
      if self.get_remaining_budget() < cost:
        return -1
    
      num_new_covered_targets = 0
      for target in cell.get_targets_covered_if_router():
        if len(target.coveringRouters) == 0:
          num_new_covered_targets += 1
    
      gain = 1000 * num_new_covered_targets - cost
    
      if gain_per_budget_point:
        return gain / cost
      else:
        return gain

  def remove_router_from_cell(self, cell):
    if not cell.hasRouter:
      raise ValueError("cell has no router")
    elif cell not in self.routers:
      raise ValueError("cell is not in routers")
    elif cell.isWall:
      raise ValueError("wall cell cannot have a router")
    else:
      cell.hasRouter = False
      self.routers.remove(cell)
      
      num_targets_uncovered = 0
      for target in cell.get_targets_covered_if_router():
        if cell not in target.coveringRouters:
          logging.error("a target was not considered covered by this router")
        else:
          target.coveringRouters.remove(cell)
          if len(target.coveringRouters) == 0:
            num_targets_uncovered += 1
      self.numTargetCellsCovered -= num_targets_uncovered

  def remove_backbone_from_cell(self, cell, check=True):
    if not cell.isBackbone:
      raise ValueError(str(cell) + " is not backbone, cannot remove backbone")
    elif cell.isInitialBackboneCell:
      raise ValueError(str(cell) + " is initial backbone: cannot remove backbone from it")
    elif cell.hasRouter:
      raise ValueError(str(cell) + " has a router, cannot remove backbone")
    else:
      if check:
        if len(self.get_neighbor_backbones(cell)) >= 2:
          logging.warning("removing backbone from cell {} with {} neighbors".format(
            str(cell), len(self.get_neighbor_backbones(cell))))
    
      cell.isBackbone = False
      self.numBackBoneCells -= 1
  
  def remove_router_and_backbone_path(self, cell):
    self.remove_router_from_cell(cell)
    
    path = self.backtrack_backbone_branch_path(cell)
    for backbone_cell in path:
      self.remove_backbone_from_cell(backbone_cell)
  
  def backtrack_backbone_branch_path(self, cell, stop_if_cell_has_router=True):
    """
    backtracks the backbone until we find: next router, next intersection or initial backbone cell
    then returns the path found. These are the backbone cells we can remove if we remove a router on cell.
    """
    
    path = list()
    current_cell = cell
    
    while True:
      
      if not current_cell.isBackbone:
        logging.error("next_cell must always be on backbone")
      
      if current_cell.isInitialBackboneCell:
        break
      if current_cell != cell or stop_if_cell_has_router:
        if current_cell.hasRouter:
          break
      
      backbone_neighbors = self.get_neighbor_backbones(current_cell)
      
      if len(backbone_neighbors) >= 3:  # means this is an intersection
        break
      
      if len(backbone_neighbors) == 0:
        raise ValueError("isolated backbone, num_backbone_neighbors: " + str(len(backbone_neighbors)))
        
      next_cell = None
      
      if len(backbone_neighbors) == 1:
        if len(path) > 0 or current_cell != cell:
          raise ValueError("should have 2 backbone neighbors, num_backbone_neighbors: " + str(len(backbone_neighbors)))
        else:
          next_cell = backbone_neighbors[0]
      
      else:  # 2 backbone neighbors
        if len(path) == 0:
          break  # the starting cell/router is on a route
        
        elif backbone_neighbors[0] == path[-1]:
          next_cell = backbone_neighbors[1]
        else:
          next_cell = backbone_neighbors[0]
      
      if next_cell is None:
        raise ValueError("next cell is none, num_backbone_neighbors: " + str(len(backbone_neighbors)))
        
      path.append(current_cell)
      current_cell = next_cell
    return path

class Greedy:
  
  @staticmethod
  def greedy_add_router_step(building, add_router_gains, gain_per_budget_point, verbose):
    """
    1. selects the cell with the best gain if a router is added to it,
       add_router_gains is a numpy array
    2. then adds required backbones, then the router
    3. updates the gains in "add_router_gains": these become approximate gains since
       the distance to backbone is not recomputed for all the cells, but only locally
    4. returns the new router cell
    
    if no cell has a positive gain, does not select any cell and returns None
    """
    
    if building.get_remaining_budget() < building.Pr:
      if verbose:
        print("remaining budget is lower than Pr (the price of 1 router)\n")
      return None
    
    [best_x, best_y] = np.unravel_index(add_router_gains.argmax(), add_router_gains.shape)
    best_cell_for_router = building.grid[best_x][best_y]
    best_approx_gain = add_router_gains[best_x, best_y]
    
    if best_approx_gain <= 0:
      if verbose:
        print("no router addition has a positive gain\n")
      return None
    else:
      score_before = building.get_score()
      path = building.add_router_and_backbone_path(best_cell_for_router)
      score_after = building.get_score()
      
      if verbose:
        print("router added at {}, new backbone cells: {}, best approx gain: {:2f}, actual gain = {},"
              " remaining budget {},  new score: {}".format(
          best_cell_for_router, len(path), best_approx_gain, score_after-score_before,
          building.get_remaining_budget(), score_after))
      
      cells_to_update = building.get_cells_within_distance(best_cell_for_router, 2 * building.R + 1)
      for cell in cells_to_update:
        add_router_gains[cell.r, cell.c] = building.compute_gain_if_add_router_to_cell(cell, gain_per_budget_point)
      return best_cell_for_router

  @staticmethod
  def greedy_solve(building, gain_per_budget_point, verbose):
    """
    applies greedy step until no budget is left or no router addition improves score
    this works whether the building is empty or already has routers placed.
    
    if gain_per_budget_point is True: the gain is computed per budget point.
    """

    start_time = time.time()
    
    if verbose:
      print("greedy_solve")
    
    if verbose and building.get_remaining_budget() < building.Pr:
      print("not enough budget for even 1 router")
      return

    add_router_gains = np.full((building.H, building.W), -1, dtype=float) # for each cell, gain if router is added to it
    for x in range(0, building.H):
      for y in range(0, building.W):
        add_router_gains[x, y] = building.compute_gain_if_add_router_to_cell(building.grid[x][y], gain_per_budget_point)
      if verbose and Utils.do_log(x, building.H, 8):
        print("initial gains computation: row {}/{}".format(x, building.H))
    if verbose:
      print("initial gains computation ({} cells) took {:.2f}s".format(building.H*building.W, time.time()-start_time))

    while True:
      new_router = Greedy.greedy_add_router_step(building, add_router_gains, gain_per_budget_point, verbose)
      if new_router is None:
        break
    
    if verbose:
      building.print_solution_info()
      print("greedy_solve took {:.2f}s".format(time.time() - start_time))

  @staticmethod
  def greedy_solve_with_random_improvements(building, output_file_path, visual_output_file_path,
                                            num_iterations, gain_per_budget_point, verbose):
    """
    when the budget is depleted, removes 10% of the routers randomly, then greedily adds routers again
    saves the solution when the score improved (does not save if the file paths are "None")
    
    if num_iterations <= 0, this will run forever and must be interrupted by hand.
    """
    
    if verbose:
      print("greedy_solve_with_random_improvements")
    
    start_time = time.time()
    best_score = building.get_score()

    add_router_gains = np.full((building.H, building.W),-1,dtype=float)  # for each cell, gain if router is added to it
    for x in range(0, building.H):
      for y in range(0, building.W):
        add_router_gains[x, y] = building.compute_gain_if_add_router_to_cell(building.grid[x][y], gain_per_budget_point)
      if verbose and Utils.do_log(x, building.H, 8):
        print("initial gains computation: row {}/{}".format(x, building.H))
    if verbose:
      print("initial gains computation ({} cells) took {:.2f}s".format(building.H * building.W, time.time() - start_time))

    iteration = 0
    while True:
      
      # remove 10% random routers
      routers_to_remove = list(building.routers)
      random.shuffle(routers_to_remove)
      routers_to_remove = routers_to_remove[:int(len(routers_to_remove) / 10)]
      
      cells_to_update = set()
      for router in routers_to_remove:
        building.remove_router_and_backbone_path(router)
        cells_to_update.update(building.get_cells_within_distance(router, 2 * building.R + 1))
      for cell in cells_to_update:
        add_router_gains[cell.r, cell.c] = building.compute_gain_if_add_router_to_cell(cell)
  
      # fill greedily
      new_router = building.grid[0][0]  # just need a not none value
      while new_router is not None:
        new_router = Greedy.greedy_add_router_step(building, add_router_gains, gain_per_budget_point, verbose)
        
      # save result if improved
      if building.get_score() > best_score:
        best_score = building.get_score()
        building.print_solution_info()
        if output_file_path is not None:
          Utils.write_solution(building, output_file_path)
        if visual_output_file_path is not None:
          Utils.write_visual_solution(building, visual_output_file_path)
      
      iteration += 1
      if verbose:
        print("iteration {}/{} done, score: {}".format(iteration, num_iterations, building.get_score()))
      if 0 < num_iterations < iteration:
        break