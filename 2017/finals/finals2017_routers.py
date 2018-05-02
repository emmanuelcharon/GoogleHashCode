"""
Created on 20 Fev. 2018

Python 3.6.4
@author: emmanuelcharon
"""

import random
import os
import logging
import time
from routers_basics import Cell, Building, Utils, Greedy
from steiner_tree import SteinerSolverItf, Point, HighwayMST, B1S_DMSTM, Local_B1S_DMSTM, CitiesMST
from maximum_coverage import MCCell, MCBuilding

def compute_and_save_covered_targets():
  samples = ["rue_de_londres", "lets_go_higher"] #["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]
  for sample in samples:
    print("\n####### {}: computing targets\n".format(sample))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    covered_targets_file_path = os.path.join(dir_path, "input", "covered_targets", "ct_" +sample + ".txt")
    
    building = Utils.read_problem_statement(input_file_path)
    building.cache_targets_covered_if_router(verbose=True)
    Utils.write_targets_covered(building, covered_targets_file_path)
  
def compute_greedy_solution_with_random_improvements(sample = "example", start_from_existing=True):
  if sample not in ["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]:
    raise ValueError("unknown sample: {}".format(sample))
  
  print("\n####### {}\n".format(sample))
  random.seed(17)  # set the seed for reproducibility

  dir_path = os.path.dirname(os.path.realpath(__file__))
  input_file_path = os.path.join(dir_path, "input", sample + ".in")
  covered_targets_file_path = os.path.join(dir_path, "input", "covered_targets", "ct_" + sample + ".txt")

  output_file_path = os.path.join(dir_path, "greedy_output", sample + ".out")
  visual_output_file_path = os.path.join(dir_path, "greedy_output", "visual_outputs", "visual_" + sample + ".txt")

  building = Utils.read_problem_statement(input_file_path)
  #Utils.read_targets_covered(building, covered_targets_file_path)
  building.cache_targets_covered_if_router(verbose=True)
  
  if start_from_existing:
    Utils.read_solution(building, output_file_path)
  else:
    Greedy.greedy_solve(building, gain_per_budget_point=True, verbose=True)
    Utils.write_solution(building, output_file_path)
    Utils.write_visual_solution(building, visual_output_file_path)
    
  #Greedy.greedy_solve_with_random_improvements(building, output_file_path, visual_output_file_path,
  #                                             num_iterations= 10, verbose = True)
  add_router_gains = Greedy.greedy_solve(building, gain_per_budget_point=False, verbose=True)
  building.print_solution_info()
  Utils.write_solution(building, output_file_path)
  Utils.write_visual_solution(building, visual_output_file_path)
  
  #Greedy.swap_until_local_maximum(building, add_router_gains, gain_per_budget_point=False, verbose=True)
  #building.print_solution_info()

def read_solutions(option):
  if option not in ["two_step_outputs", "greedy_output"]:
    raise ValueError("mow")
  
  samples = ["example", "charleston_road", "opera", "rue_de_londres", "lets_go_higher"]
  
  scores = []
  
  for sample in samples:
    print("\n####### {}\n".format(sample))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, option, sample + ".out")

    building = Utils.read_problem_statement(input_file_path)
    Utils.read_solution(building, output_file_path)
    
    if sample != "example":
      scores.append(building.get_score())

  print("scores: {} => {} i.e {:.2f}M".format(scores, sum(scores), sum(scores) / 1000000))

def optimize_local_max(building, output_file_path, visual_output_file_path, distance = 1):
  """
  repeat until this provides no score improvement:
    for each router:
      repeat until the new router is placed on the same spot twice in a row
        remove router
        compute gain for adding a new router around and including this position
        place a new router on one of the best positions, take a random position if several best choices
        (this can remove a router without replacement if, after removing it, all local positions provide no improvement)
  """
  
  best_score = building.get_score()
  n_iterations = 0
  
  touched_router_cells_set = set()
  
  while True:
    n_iterations += 1
    print("optimize_local_max iteration {}".format(n_iterations))
    
    routers_this_iteration = list(building.routers)
    for router in routers_this_iteration:
      
      moved = True
      while moved:
        building.remove_router_and_backbone_path(router)
        
        #neighbors = list(building.get_neighbors(router))
        neighbors = list(building.get_cells_within_distance(router, distance))

        neighbors.append(router)
        gains = []
        for neighbor in neighbors:
          gains.append(building.compute_gain_if_add_router_to_cell(neighbor))
        

        max_gain_cells = list()
        max_gain = 0
        
        for i in range(len(neighbors)):
          if gains[i] > max_gain:
            max_gain_cells = [neighbors[i]]
            max_gain = gains[i]
          elif gains[i] == max_gain:
            max_gain_cells.append(neighbors[i])
          
        if len(max_gain_cells) == 0:
          logging.warning("max_gain_cells of length 0, not replacing router")
          moved = False
        else:
          new_router = random.choice(max_gain_cells)
          path = building.add_router_and_backbone_path(new_router)
          
          moved = router != new_router
          if moved:
            touched_router_cells_set.add(router)
            touched_router_cells_set.add(new_router)
          router = new_router
        
    
    if building.get_score() >= best_score:
      best_score = building.get_score()
      
      if output_file_path is not None and visual_output_file_path is not None:
        building.print_solution_info()
        Utils.write_solution(building, output_file_path)
        Utils.write_visual_solution(building, visual_output_file_path)
    
    if building.get_score() <= best_score:
      break
      
  return touched_router_cells_set

def optimize_annealing(building, output_file_path, visual_output_file_path):
  """
    repeat n times:
      decrease T (the temperature)
      take a random router and a random neighbor cell
      compute the score gained if router is moved to neighbor position
      if the move improves score, do it
      of score loss is less than R*1000, do it with probability T
      else do not do it
  """
  
  N = 5*5*100*len(building.routers)
  for n in range(N):
    if n==0 or (N/10 > 0 and n % (N/10) == 0):
      print("optimize_annealing, n = {}/{}".format(n, N))
      building.print_solution_info()
    
    T = 0.1*(1-n/N) # T will decrease to 0 over time
    
    router = random.choice(list(building.routers))
    #neighbor = random.choice(building.get_neighbors(router))
    neighbor = random.choice(list(building.get_cells_within_distance(router, 2*building.R)))
    
    if neighbor.hasRouter or not neighbor.isTarget:
      continue
    
    building.remove_router_and_backbone_path(router)
    gain_router = building.compute_gain_if_add_router_to_cell(router)
    gain_neighbor = building.compute_gain_if_add_router_to_cell(neighbor)
    
    if gain_neighbor >= 0:
      if gain_neighbor >= gain_router or (gain_neighbor >= gain_router - building.R*1000 and random.random() < T):
        building.add_router_and_backbone_path(neighbor)
        continue
    building.add_router_and_backbone_path(router)
    
    
def compute_tree(sample, num_routers):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  input_file_path = os.path.join(dir_path, "input", sample + ".in")
  output_file_path = os.path.join(dir_path, "two_step_outputs", sample + ".out")
  visual_output_file_path = os.path.join(dir_path, "two_step_outputs", "visual_outputs", "visual_" + sample + ".txt")

  routers_file_path = os.path.join(dir_path, "max_coverage", sample + str(num_routers) + ".txt")
  
  h_dict = {"example": 8, "opera": 667, "rue_de_londres": 559, "lets_go_higher":872, "charleston_road": 240}
  w_dict = {"example": 22, "opera": 540, "rue_de_londres": 404, "lets_go_higher": 975, "charleston_road": 180}
  original_bb_dict = {"example": [2,7], "opera": [333,270], "rue_de_londres": [279,202],
              "lets_go_higher": [436,487], "charleston_road": [120,90]}
  routers_coords = SteinerSolverItf.read_cities_coords(routers_file_path)

  # find backbone tree:
  start_time = time.time()
  cities_coords = list(routers_coords)
  cities_coords.append(original_bb_dict[sample]) # MUST NOT FORGET THIS ONE :)
  highwayMST = CitiesMST(h_dict[sample], w_dict[sample], cities_coords).find_steiner_tree()

  #highwayMST = HighwayMST(h_dict[sample], w_dict[sample], cities_coords).find_steiner_tree(variation=True, verbose=False)
  #highwayMST = Local_B1S_DMSTM(h_dict[sample], w_dict[sample], cities_coords)\
  #  .find_steiner_tree(cluster_iterations=len(cities_coords), max_cities_per_cluster=10, verbose=False)
  SteinerSolverItf.check_steiner_tree(highwayMST, cities_coords)
  print("highwayMST cost: {}, time taken: {:.2f}s\n".format(SteinerSolverItf.cost(highwayMST), time.time() - start_time))
  
  # now form actual building
  # building = Utils.read_problem_statement(input_file_path)
  # for x in range(building.H):
  #   for y in range(building.W):
  #     if highwayMST[x, y] > 0 and not (building.br==x and building.bc==y):
  #       building.add_backbone_to_cell(building.grid[x][y], check_neighbors=False)
  # for coords in routers_coords:
  #   building.add_router_to_cell(building.grid[coords[0]][coords[1]])
  # building.print_solution_info()
  #
  # add_router_gains = Greedy.greedy_solve(building, gain_per_budget_point=False, verbose=True)
  # Greedy.swap_until_local_maximum(building, add_router_gains, gain_per_budget_point=False, verbose=True)
  # building.print_solution_info()
  #
  # Utils.write_solution(building, output_file_path)
  # Utils.write_visual_solution(building, visual_output_file_path)

def replace_backbone(building):
  routers_coords = [[router.r, router.c] for router in building.routers]
  
  # find backbone tree:
  start_time = time.time()
  cities_coords = list(routers_coords)
  cities_coords.append([building.br, building.bc])  # MUST NOT FORGET TO CONNECT THE ORIGINAL BACKBONE
  mst = Local_B1S_DMSTM(building.H, building.W, cities_coords).find_steiner_tree(
    cluster_iterations=len(cities_coords), max_cities_per_cluster=5, verbose=True)
  SteinerSolverItf.check_steiner_tree(mst, cities_coords)
  mst_cost = SteinerSolverItf.cost(mst)
  print("mst cost: {}, time taken: {:.2f}s\n".format(mst_cost, time.time() - start_time))
  print("had {} backbone cells and computed a steiner tree with {}".format(building.numBackBoneCells, mst_cost-1))
  if mst_cost - 1 < building.numBackBoneCells:
    # then replace
    building.clear_all_routers_and_backbones()
    for x in range(building.H):
      for y in range(building.W):
        if mst[x, y] > 0 and not (building.br == x and building.bc == y):
          building.add_backbone_to_cell(building.grid[x][y], check_neighbors=False)
    for coords in routers_coords:
      building.add_router_to_cell(building.grid[coords[0]][coords[1]])

def hand_managed(sample):
  # perform random iterations, and from time to time, swap to local max and see if this beats the best score
  # note: swap to local max forces a rigid local max, so we do not follow the random iterations after local max but
  # from where we left random improvements.
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  input_file_path = os.path.join(dir_path, "input", sample + ".in")
  output_file_path = os.path.join(dir_path, "two_step_outputs", sample + ".out")
  visual_output_file_path = os.path.join(dir_path, "two_step_outputs", "visual_outputs", "visual_" + sample + ".txt")
  covered_targets_file_path = os.path.join(dir_path, "input", "covered_targets", "ct_" + sample + ".txt")
  
  
  building = Utils.read_problem_statement(input_file_path)
  #Utils.read_targets_covered(building, covered_targets_file_path)
  building.cache_targets_covered_if_router(verbose=True)

  Utils.read_solution(building, output_file_path)
  
  
  #replace_backbone(building)
  #add_router_gains = Greedy.greedy_solve_with_random_improvements(building, output_file_path,
  #  visual_output_file_path, num_iterations= 100, verbose=True)
  add_router_gains = Greedy.greedy_solve(building, gain_per_budget_point=False, verbose=True)
  Greedy.swap_until_local_maximum(building, add_router_gains, gain_per_budget_point=False, verbose=True)
  
  Utils.write_solution(building, output_file_path)
  Utils.write_visual_solution(building, visual_output_file_path)

def main():
  # compute_and_save_covered_targets()
  # compute_greedy_solution_with_random_improvements(sample="opera")
  read_solutions(option="two_step_outputs")
  # compute_tree("rue_de_londres", 191)
  # hand_managed("opera")

if __name__ == "__main__":
 main()
 

  
  
  