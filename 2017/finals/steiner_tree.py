"""
Created on 25 Apr. 2018

Python 3.6.4
@author: emmanuelcharon

In this file, we provide different heuristics to solve the Steiner tree problem.
We use "highways" to connect "cities" (instead of backbone to connect routers).

"""

import numpy as np
import random
import math
import time
from routers_basics import Utils
from multiprocessing import Pool
import os

class SteinerSolverItf:
  
  def find_steiner_tree(self):
    """
    returns a numpy array of ints of shape (H,W),
    where each cell has value:
      2 if there is a city (and a highway)
      1 if there is just a highway
      0 else.
    """
    
    raise NotImplementedError("must implement this method")
  
  @staticmethod
  def check_steiner_tree(steiner_tree, cities_coords):
    """ checks that all cells are 0 or 1 or 2? and that cities are 2"""
    for v in np.nditer(steiner_tree):
      if v != 0 and v != 1 and v != 2:
        raise ValueError("steiner_tree has a value: {}".format(v))
    for coords in cities_coords:
      if steiner_tree[coords[0], coords[1]] != 2:
        raise ValueError("steiner_tree is not passing in city: {}".format(coords))
  
  @staticmethod
  def cost(steiner_tree):
    result = 0
    for v in np.nditer(steiner_tree):
      if v > 0:
        result += 1
    return result
    
  @staticmethod
  def read_cities_coords(routers_file_path):
    cities_coords = list()
    with open(routers_file_path, 'r') as f:
      for line in f.readlines():
        l = line.rstrip()
        [x, y] = [int(a) for a in l.split(' ')]
        cities_coords.append([x, y])
    return cities_coords

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return "{},{}".format(self.x, self.y)
  def __str__(self):
    return str(self.__repr__())
  
  @staticmethod
  def chebyshev_distance(point_a, point_b):
    """ one vertical, horizontal, or diagonal step costs 1 """
    return max(abs(point_a.x - point_b.x), abs(point_a.y - point_b.y))

  def get_basic_path(self, destination):
    """
    returns a list of pairs of coordinates
    goes in diagonal from self to destination then straight
    includes destination coordinates, but not self coordinates.
    """
    
    result = []
  
    next_point = [self.x, self.y]
    while next_point[0] != destination.x or next_point[1] != destination.y:
      direction_x = 1 if destination.x > next_point[0] else 0 if destination.x == next_point[0] else -1
      direction_y = 1 if destination.y > next_point[1] else 0 if destination.y == next_point[1] else -1
      next_point = [next_point[0] + direction_x, next_point[1] + direction_y]
      result.append(next_point)
  
    if len(result) != Point.chebyshev_distance(self, destination):
      raise ValueError("basic path length unexpected from {} to {}".format(self, destination))
      
    return result
  
  @staticmethod
  def other_2_parallelogram_summits(a, b):
    """
    return the other 2 summits of the smallest parallelogram formed by points a and b,
    when using only vertical, horizontal and 1x1 diagonal lines
    
    case 1:
    a..........X
     .          .
      X..........b
    
    case 2:
    a
    ..
    . X
    . .
    . .
    X .
     ..
      b
    
    case 3:
      X..........a
     .          .
    b..........X
    
    case 4:
      a
     ..
    X .
    . .
    . .
    . X
    ..
    b
    
    """
    
    if a.x > b.x:
      return Point.other_2_parallelogram_summits(b, a)
    
    dist_x = abs(b.x - a.x)
    dist_y = abs(b.y - a.y)
    
    if a.y <= b.y:
      if dist_x <= dist_y: # case 1
        return [[b.x, a.y+dist_x], [a.x, b.y-dist_x]]
      else: # case 2
        return [[b.x-dist_y, a.y], [a.x+dist_y, b.y]]
    else:
      if dist_x <= dist_y: # case 3
        return [[a.x, b.y+dist_x], [b.x, a.y - dist_x]]
      else: # case 4
        return [[a.x+dist_y, b.y], [b.x-dist_y, a.y]]
    
class CitiesMST(SteinerSolverItf):
  """ simply find a Minimum Spanning Tree over the cities """

  def __init__(self, H, W, cities_coordinates):
    self.H = H
    self.W = W
    self.cities = [Point(coords[0], coords[1]) for coords in cities_coordinates]
  
  def find_steiner_tree(self):
    """ this will put the first city in the list in the tree first """
    
    tree_edges = list()  # each edge is a pair of cities that we connect with the highway
    connection_costs = dict()  # cost of connecting each remaining city to the tree, and associated closest city
    for city in self.cities:
      connection_costs[city] = [self.H + self.W + 1, None]

    while len(connection_costs) > 0:
      closest_city = None # city to connect that is closest to a connected city
      min_distance = self.H + self.W + 2
      closest_dest = None # closest connected destination
  
      for city, [distance, destination] in connection_costs.items():
        if distance < min_distance:
          min_distance = distance
          closest_city = city
          closest_dest = destination

      if closest_city is None:
        raise ValueError("closest_city is none")
  
      del connection_costs[closest_city]
      if closest_dest is not None:
        tree_edges.append([closest_dest, closest_city])
      
      # for each city not yet connected, check if the new connected city makes them closer to highway
      for city, [distance, destination] in connection_costs.items():
        new_distance = Point.chebyshev_distance(city, closest_city)
        if new_distance < distance:
          connection_costs[city] = [new_distance, closest_city]
    
    # now tree_edges contains the highways we want to construct.
    # there are many ways to connect 2 cities with the same cost: we choose to go diagonal first, then straight
    result = np.full((self.H, self.W), 0, dtype=int)
    if len(tree_edges) > 0:
      first_city = tree_edges[0][0]
      result[first_city.x, first_city.y] += 1
      for edge in tree_edges:
        for coords in edge[0].get_basic_path(edge[1]):
          result[coords[0], coords[1]] += 1
    for city in self.cities:
      result[city.x, city.y] += 1
    return result

class HighwayMST(SteinerSolverItf):
  """
  we connect cities 1 by 1 greedily (a bit like in the CitiesMST) but we count the distance from each remaining city
  to the highway (instead of their distances to other cities)
  """

  def __init__(self, H, W, cities_coordinates, grid = None):
    self.H = H
    self.W = W
    
    if grid is not None:
      if len(grid) != self.H or len(grid[0]) != self.W:
        raise ValueError("wrong dimensions of grid")
      self.grid = grid
    else:
      self.grid = [[Point(x, y) for y in range(self.W)] for x in range(self.H)]

    
    self.cities = [self.grid[coords[0]][coords[1]] for coords in cities_coordinates]
  
  def get_basic_points_path(self, start_point, end_point):
    return [self.grid[coords[0]][coords[1]] for coords in start_point.get_basic_path(end_point)]
  
  def get_neighbors(self, point):
    result = []
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == dy == 0 or point.x + dx < 0 or point.x + dx >= self.H or point.y + dy < 0 or point.y + dy >= self.W:
          continue
        else:
          result.append(self.grid[point.x + dx][point.y + dy])
    return result
  
  def get_better_points_path(self, start_point, end_point, remaining_cities):
    """
    this path chooses, at each step, between all neighbors closest to the end_point,
    the one closest to the closest remaining_city
    """

    result = []

    next_point = start_point
    
    while next_point.x != end_point.x or next_point.y != end_point.y:
      neighbors = self.get_neighbors(next_point)
      distance_to_end = min([Point.chebyshev_distance(end_point, n) for n in neighbors])
      neighbors = [n for n in neighbors if Point.chebyshev_distance(end_point, n) == distance_to_end]
      
      if len(neighbors) == 1:
        next_point = neighbors[0]
        result.append(next_point)
      
      else:
        best_neighbor = None
        best_dist_to_city = self.H + self.W + 2
        for neighbor in neighbors:
          neighbor_dist = self.H + self.W + 1
          for city in remaining_cities:
            d = Point.chebyshev_distance(neighbor, city)
            if d < neighbor_dist:
              neighbor_dist = d
          if neighbor_dist < best_dist_to_city:
            best_dist_to_city = neighbor_dist
            best_neighbor = neighbor

        next_point = best_neighbor
        result.append(next_point)
        

    if len(result) != Point.chebyshev_distance(start_point, end_point):
      raise ValueError("better path length unexpected from {} to {}".format(start_point, end_point))
    
    return result

  def find_steiner_tree(self, variation=True, verbose=True):
    """
    this will put the first city in the list in the tree first
    
    if variation == True:
      uses a better way to choose the path from the highway to the next closest city
      the run time is 2x longer but the score is much better
    """

    result = np.full((self.H, self.W), 0, dtype=int)
    for city in self.cities:
      result[city.x, city.y] += 1
    connection_costs = dict()  # cost of connecting each remaining city to the highway,
                               # and associated closest HIGHWAY point
    for city in self.cities:
      connection_costs[city] = [self.H + self.W + 1, None]

    while len(connection_costs) > 0:
      if verbose and Utils.do_log(len(self.cities)-len(connection_costs), len(self.cities), 10):
        print("optimising highway, cities remaining: {}/{}".format(len(connection_costs), len(self.cities)))
      
      closest_city = None  # city to connect that is closest to existing highway
      min_distance = self.H + self.W + 2
      closest_highway = None  # closest highway point
  
      for city, [distance, highway] in connection_costs.items():
        if distance < min_distance:
          min_distance = distance
          closest_city = city
          closest_highway = highway
  
      if closest_city is None:
        raise ValueError("closest_city is none")
  
      del connection_costs[closest_city]
      if closest_highway is not None:
        if variation:
          new_highways = self.get_better_points_path(closest_highway, closest_city, connection_costs.keys())
        else:
          new_highways = self.get_basic_points_path(closest_highway, closest_city)
      else:
        # we build the first highway on the first city
        new_highways = [closest_city]
       
      for new_highway in new_highways:
        result[new_highway.x, new_highway.y] += 1
        # for each city not yet connected, check if each new point connected to the highway is closer
        for city, [distance, highway] in connection_costs.items():
          new_distance = Point.chebyshev_distance(city, new_highway)
          if new_distance < distance:
            connection_costs[city] = [new_distance, new_highway]
    
    return result

class Candidate:
  def __init__(self, point):
    self.point = point
    self.score = -1
  
class B1S_DMSTM(SteinerSolverItf):
  """
  A different approach: we greedily add "Steiner points" which are highway intersections. The result will be an MST
  on the set of cities and highway intersections. We stop when no additional intersection improves score.
  
  See article 'Closing the Gap: Near-Optimal Steiner Trees in Polynomial Time'.
  
  We perform the Batched 1-Steiner greedy algorithm with Dynamic MST Maintenance, so the whole thing
  runs in O(n^3) where n is the number of cities.
  
  Our situation is different from the article since we have the Chebyshev distance instead of the Rectilinear distance.
  To adjust for that, we used 8 regions (cut with horizontal, vertical and 2 diagonal lines) instead of 4, and we
  have a lot more Hannan candidates.
  
  """
  
  def __init__(self, H, W, cities_coordinates, grid = None):
    self.H = H
    self.W = W
    if grid is not None:
      if len(grid) != self.H or len(grid[0]) != self.W:
        raise ValueError("wrong dimensions of grid")
      self.grid = grid
    else:
      self.grid = [[Point(x, y) for y in range(self.W)] for x in range(self.H)]

    self.cities = [self.grid[coords[0]][coords[1]] for coords in cities_coordinates]
  
  def mst(self, P):
    """
    computes a mst of the list of points P,
    returns a list of edges (an edge is a pair of cities) and the mst cost
    this is simply an implementation Prim's algorithm, and takes O(n^2) where n = len(P)
    """
    
    tree_edges = list()
    connection_costs = dict()
    for p in P:
      connection_costs[p] = [self.H + self.W + 1, None]

    while len(connection_costs) > 0:
      closest_p = None  # point to connect that is closest to a connected point
      min_distance = self.H + self.W + 2
      closest_dest = None  # corresponding closest point already connected
  
      for p, [distance, destination] in connection_costs.items():
        if distance < min_distance:
          min_distance = distance
          closest_p = p
          closest_dest = destination
  
      if closest_p is None:
        raise ValueError("closest_p is none")
  
      del connection_costs[closest_p]
      if closest_dest is not None: # it is None during the first iteration, when putting the first point
        tree_edges.append([closest_dest, closest_p])
  
      # for each point not yet connected, check if the new connected city makes them closer to highway
      for p, [distance, destination] in connection_costs.items():
        new_distance = Point.chebyshev_distance(p, closest_p)
        if new_distance < distance:
          connection_costs[p] = [new_distance, closest_p]
    
    tree_cost = 1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in tree_edges])
    return tree_edges, tree_cost

  def find_hannan_candidates_naive(self, P):
    """ this just takes the all points in the smallest rectangle containing P """
    min_x = min([point.x for point in P])
    max_x = max([point.x for point in P])
    min_y = min([point.y for point in P])
    max_y = max([point.y for point in P])
    forbid = set(P)
    result = list()
    for x in range(min_x, max_x+1):
      for y in range(min_y, max_y+1):
        if self.grid[x][y] not in forbid:
          result.append(Candidate(self.grid[x][y]))
    return result

  def find_hannan_candidates_old(self, P):
    
    if not isinstance(P, list):
      raise TypeError("P must be a list")
    forbid = set(P)
    result = list()
    for i in range(len(P)):
      for j in range(i+1, len(P)):
        # take the 2 other summits of the parallelogram linking these 2 points
        for summit_coords in Point.other_2_parallelogram_summits(P[i], P[j]):
          summit = self.grid[summit_coords[0]][summit_coords[1]]
          if summit not in forbid:
            result.append(Candidate(summit))
            forbid.add(summit)
    return result

  def find_hannan_candidates(self, P):
    """ in the smallest rectangle containing P, takes points that are on
    lines (vertical, horizontal or diagonal) going through any point in P
    """
  
    min_x = min([point.x for point in P])
    max_x = max([point.x for point in P])
    min_y = min([point.y for point in P])
    max_y = max([point.y for point in P])
    forbid = set(P)
    result = list()
    for x in range(min_x, max_x + 1):
      for y in range(min_y, max_y + 1):
        if self.grid[x][y] not in forbid:
          add_xy = False
          for p in P:
            if x==p.x or y==p.y or abs(x-p.x) == abs(y-p.y):
              # x,y is on a vertical, horizontal or diagonal line going through p
              add_xy = True
          if add_xy:
            result.append(Candidate(self.grid[x][y]))
    return result
  
  def b1s(self):
    """
    return the list of steiner points, the set of edges of the tree, and the score
    note n the number of cities to connect: this runs in about O(n^4) time per outer loop: O(n^2) per hannan candidate,
    and we expect less than 5 of these outer loops.
    """
    P = list(self.cities)
    mst_P, mst_P_cost = self.mst(P)
    cities_set = set(self.cities)

    while True:
      candidates = self.find_hannan_candidates(P)
      for candidate in candidates:
        can_P = P[:]
        can_P.append(candidate.point)
        mst_can_P, mst_can_P_cost = self.mst(can_P)
        candidate.score = mst_P_cost - mst_can_P_cost
    
      helpful_candidates = [can for can in candidates if can.score > 0]
      helpful_candidates.sort(key=lambda _: _.score, reverse=True)  # best scores first
    
      print("outer loop: P size: {}, helpful candidates: {}/{}".format(len(P), len(helpful_candidates), len(candidates)))
      
      if len(helpful_candidates) == 0:
        break
    
      for candidate in helpful_candidates:
        can_P = P[:]
        can_P.append(candidate.point)
        mst_can_P, mst_can_P_cost = self.mst(can_P)
      
        if mst_can_P_cost - mst_P_cost >= candidate.score:
          P, mst_P, mst_P_cost = can_P, mst_can_P, mst_can_P_cost
          #print("inner loop, candidate added: P size: {}, helpful candidates: {}".format(len(P), len(candidates)))
      
      # remove steiner points which became unnecessary
      degrees_in_mst_P = dict()
      for p in P:
        degrees_in_mst_P[p] = 0
      for edge in mst_P:
        degrees_in_mst_P[edge[0]] += 1
        degrees_in_mst_P[edge[1]] += 1

      steiner_points = [p for p in P if p not in cities_set]
      unnecessary_steiner_points = set([sp for sp in steiner_points if degrees_in_mst_P[sp] <= 2])
      if len(unnecessary_steiner_points) > 0:
        P = [p for p in P if p not in unnecessary_steiner_points]
        mst_P, mst_P_cost = self.mst(P)
        print("removed {}  unnecessary steiner points".format(len(unnecessary_steiner_points)))

      print("outer loop iteration ended: P size: {}, edges: {}, score: {}".format(len(P), len(mst_P), mst_P_cost))
    
    steiner_points = [p for p in P if p not in cities_set]
    return steiner_points, mst_P, mst_P_cost
  
  def remove_cycles(self, start_point, edges, expected_cycles):
    has_cycle = True
    result = list(edges)
    
    while has_cycle:
  
      cycle = self.find_cycle(start_point, result)
      if cycle is None:
        has_cycle = False
      else:
        # find longest edge on cycle and remove it from edges
        longest_length = -1
        longest_edge = None
        for edge in cycle:
          d = Point.chebyshev_distance(edge[0], edge[1])
          if d > longest_length:
            longest_length = d
            longest_edge = edge
         
        if longest_edge is None:
          raise ValueError("longest edge is None")
        else:
          result = [edge for edge in result if edge is not longest_edge]
        
    if len(edges)-len(result) != expected_cycles:
      raise ValueError("should have removed exactly 1 edge, but removed {}".format(len(edges)-len(result)))
    return result
  
  @staticmethod
  def find_cycle(start_point, edges):
    """
    consider the graph G formed by the edges (and the points in the edges)
    performs a DFS on G starting at start_point, and returns the first cycle found
    returns None if no cycle was found
    """
    
    # perform DFS on graph starting at start_point
    
    points_set = set()
    
    for edge in edges:
      points_set.add(edge[0]) # mark each point as not visited
      points_set.add(edge[1])

    point_visited = dict()  # convenience booleans to mark points as visited or not
    point_edges = dict()  # for each point, the list of edges this point touches

    for point in points_set:
      point_visited[point] = False
      point_edges[point] = list()
    
    for edge in edges:
      point_edges[edge[0]].append(edge)
      point_edges[edge[1]].append(edge)

    # using python list as stack: we add and remove elements at end of list
    edges_stack = list(point_edges[start_point])
    point_visited[start_point] = True
    edges_path = []
    
    while len(edges_stack) > 0:
      edge = edges_stack.pop()
      edges_path.append(edge)
    
      if not point_visited[edge[0]] and not point_visited[edge[1]]:
        raise ValueError("expected at least one of the two points already visited")
    
      elif point_visited[edge[0]] and point_visited[edge[1]]:
        # we found a cycle, and "edges_path" are the edges in this cycle
        if edge[0] is not start_point and edge[1] is not start_point:
          print("Warning: expected loop to be found on start_point and not somewhere else: start_point: {}, edges_path: {}".format(start_point, edges_path))
        return edges_path
        
      else:
        current_point = edge[0] if point_visited[edge[1]] else edge[1]
        point_visited[current_point] = True
        next_edges = [e for e in point_edges[current_point] if e is not edge]
      
        if len(next_edges) > 0:
          edges_stack.extend(next_edges)
        else:
          # we found a tree leaf, backtrack the edges_path
          if len(edges_stack) > 0:
            next_edge = edges_stack[-1]
            
            if point_visited[next_edge[0]] != point_visited[next_edge[1]]:  # else we will find a cycle there next step
              path_point = next_edge[1] if point_visited[next_edge[1]] else next_edge[0] # all points on path have been seen
              while len(edges_path) > 0 and path_point not in edges_path[-1]:
                edges_path.pop()
              if len(edges_path) > 0:
                edges_path.pop()
    return None

  def mst_dmstm(self, P, mst_P, candidate_point):
    """
    does NOT modify P or mst_P
    
    for explanations, see DMSTM in article "Closing the Gap: Near-Optimal Steiner Trees in Polynomial Time"
    since our geometry is different (Chebyshev distance instead of Rectilinear distance),
    we use 8 regions instead of 4, using vertical/horizontal and diagonal lines to split space around candidate_point.
    
    note Q the list P with the candidate_point
    finds the mst_Q (hopefully exact: unproven) in linear time (in the size of P) using geometric properties and mst_P
    
    returns Q, mst_Q, mst_Q_cost
    """
    
    neighbors = [None for _ in range(8)] # P-neighbor of candidate_point in each region
    for p in P:
      region = B1S_DMSTM.quadrant(p, candidate_point, num_quadrants=8)
      if neighbors[region] is None:
        neighbors[region] = p
      else:
        # noinspection PyTypeChecker
        current_min_dist = Point.chebyshev_distance(candidate_point, neighbors[region])
        # noinspection PyTypeChecker
        dist = Point.chebyshev_distance(p, neighbors[region])
        if dist < current_min_dist:
          neighbors[region] = p
    neighbors = [neighbor for neighbor in neighbors if neighbor is not None]

    # Follow DMSTM:
    #   form a graph formed of points in Q and edges in mst_P
    #   then the neighbor in each region:
    #     add the edge neighbor<->candidate
    #     while there are cycles, remove longest edge from cycle
    
    Q = list(P)
    Q.append(candidate_point)
    edges = list(mst_P)

    for neighbor in neighbors:
      # form a graph formed of points in Q and edges in mst_P + the edge neighbor<->candidate
      edges.append([neighbor, candidate_point])
      # then while there are cycles, remove longest edge from cycle
      edges = self.remove_cycles(candidate_point, edges, expected_cycles = 0 if neighbor is neighbors[0] else 1)
      
    # edges is now mst_Q
    tree_cost = 1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in edges])
    return Q, edges, tree_cost
  
  def b1s_dmstm(self, verbose, max_outer_loops):
    """
    same as B1S, except that we compute the mst P with each candidate in linear time instead of quadratic,
    by using geometric properties and the existing mst on P
    """

    P = list(self.cities)
    mst_P, mst_P_cost = self.mst(P)
    cities_set = set(self.cities)
    
    outer_loop = 0
    while True:
  
      outer_loop += 1
      if 0 < max_outer_loops < outer_loop:
        break
      
      candidates = self.find_hannan_candidates(P)
      if verbose:
        print("P of size {} and score {}, and {} hannan candidates found".format(len(P), mst_P_cost, len(candidates)))
        
      for i in range(len(candidates)):
        if verbose and  len(candidates) > 1000 and Utils.do_log(i, len(candidates), 100):
           print("treating candidate {}/{}".format(i, len(candidates)))
        candidate = candidates[i]
        Q, mst_Q, mst_Q_cost = self.mst_dmstm(P, mst_P, candidate.point)
        candidate.score = mst_P_cost - mst_Q_cost
      
      helpful_candidates = [can for can in candidates if can.score > 0]
      helpful_candidates.sort(key=lambda _: _.score, reverse=True)  # best scores first
      
      if verbose:
        print("outer loop: P size: {}, helpful candidates: {}/{}".format(len(P),len(helpful_candidates),len(candidates)))
  
      if len(helpful_candidates) == 0:
        break
      
      max_gain_this_iteration = sum([can.score for can in helpful_candidates])
      cost_before_iteration = mst_P_cost
      
      for i in range(len(helpful_candidates)):
        if verbose and len(helpful_candidates) > 1000 and Utils.do_log(i, len(helpful_candidates), 10):
           print("treating candidate {}/{}".format(i, len(candidates)))
        candidate = helpful_candidates[i]
        Q, mst_Q, mst_Q_cost = self.mst_dmstm(P, mst_P, candidate.point)
    
        if mst_P_cost - mst_Q_cost >= candidate.score:
          # then add the candidate as a new steiner point
          P, mst_P, mst_P_cost = Q, mst_Q, mst_Q_cost
      
      # now that all "independent" candidates have been added, remove steiner points which became unnecessary
      degrees_in_mst_P = dict()
      for p in P:
        degrees_in_mst_P[p] = 0
      for edge in mst_P:
        degrees_in_mst_P[edge[0]] += 1
        degrees_in_mst_P[edge[1]] += 1
     
      steiner_points = [p for p in P if p not in cities_set]
      unnecessary_steiner_points = set([sp for sp in steiner_points if degrees_in_mst_P[sp] <= 2])
      if len(unnecessary_steiner_points) > 0:
        P = [p for p in P if p not in unnecessary_steiner_points]
        mst_P, mst_P_cost = self.mst(P)

      gain_this_iteration = cost_before_iteration - mst_P_cost
      max_future_gain = max_gain_this_iteration - gain_this_iteration

      if verbose:
        print("removed {}  unnecessary steiner points".format(len(unnecessary_steiner_points)))
        print("outer loop iteration ended: P size: {}, edges: {}, score: {}, estimated max future gain: {}"
              .format(len(P), len(mst_P), mst_P_cost, max_future_gain))
      
      
      
    steiner_points = [p for p in P if p not in cities_set]
    return steiner_points, mst_P, mst_P_cost
  
  def create_np_array_from_edges(self, edges):
    result = np.full((self.H, self.W), 0, dtype=int)
    if len(edges) > 0:
      for edge in edges:
        highways = edge[0].get_basic_path(edge[1])
        highways.append([edge[0].x, edge[0].y])
        for coords in highways:
          result[coords[0], coords[1]] = 1
    for city in self.cities:
      result[city.x, city.y] += 1
    return result
  
  def find_steiner_tree(self, method="B1S_DMSTM", verbose = True, max_outer_loops = -1):
    
    if method == "B1S":
      steiner_points, mst_P, mst_P_cost = self.b1s()
      print("B1S: {} cities, {} steiner points, {} edges, cost {}".format(
        len(self.cities), len(steiner_points), len(mst_P), mst_P_cost))
      
    elif method == "B1S_DMSTM":
      steiner_points, mst_P, mst_P_cost = self.b1s_dmstm(verbose=verbose, max_outer_loops=max_outer_loops)
      print("B1S DMSTM: {} cities, {} steiner points, {} edges, cost {}".format(
        len(self.cities), len(steiner_points), len(mst_P), mst_P_cost))
    else:
      raise ValueError("method '{}' does not exist".format(method))

    return self.create_np_array_from_edges(mst_P)

  @staticmethod
  def angle_360(x, y):
    """ given a point (x,y), returns the angle in [0 360[ from (1,0) to (x,y)"""
    norm_xy = math.sqrt(x * x + y * y)
    if norm_xy == 0:
      raise ValueError("norm 0")
    dot_product = x * 1 + y * 0
    cos_xy = dot_product / norm_xy
    angle_rad = math.acos(cos_xy)
    angle_degrees = math.degrees(angle_rad)
    if y < 0:
      angle_degrees = 360.0 - angle_degrees
    return angle_degrees

  @staticmethod
  def quadrant(center, target, num_quadrants=8):
    degrees = B1S_DMSTM.angle_360(center.x - target.x, center.y - target.y)
    return int(math.floor(degrees * num_quadrants / 360))

class Local_B1S_DMSTM(SteinerSolverItf):
  
  def __init__(self, H, W, cities_coordinates, grid = None):
    self.H = H
    self.W = W
    if grid is not None:
      if len(grid)!=self.H or len(grid[0])!=self.W:
        raise ValueError("wrong dimensions of grid")
      self.grid = grid
    else:
      self.grid = [[Point(x, y) for y in range(self.W)] for x in range(self.H)]
    self.cities = [self.grid[coords[0]][coords[1]] for coords in cities_coordinates]
    self.cities_set = set(self.cities)
    
  # def find_rectangle_clusters(self, max_cities_per_cluster):
  #   """
  #   divide the grid in rectangle (clusters), such that each rectangle has less than max_cities_per_square cities
  #   each cluster is returned as a list of cities (concatenating them should make the full list of cities)
  #   """
  #
  #   if max_cities_per_cluster < 10:
  #     raise ValueError("try with bigger clusters")
  #
  #   clusters = self._find_rectangle_clusters(max_cities_per_cluster, 0, 0, self.H, self.W, list(self.cities))
  #
  #   # check result
  #   concat_clusters = [c for cluster in clusters for c in cluster]
  #   if len(concat_clusters) != len(self.cities):
  #     raise ValueError("wrong")
  #   if len(set(concat_clusters)) != len(concat_clusters):
  #     raise ValueError("wrong")
  #   if set(concat_clusters) != set(self.cities):
  #     raise ValueError("wrong")
  #
  #   return clusters
  #
  # def _find_rectangle_clusters(self, max_cities_per_cluster, min_x, min_y, height, width, all_cities_in_area):
  #   """
  #   returns the clusters found in the rectangle delimited by (min_x, min_y, max_x, max_y)
  #   min_x, min_y are inclusive and max_x, max_y are exclusive
  #   """
  #
  #   if len(all_cities_in_area) <= max_cities_per_cluster:
  #     return [all_cities_in_area] # a list of clusters
  #   else:
  #     h, w = int(height/2), int(width/2)
  #
  #     areas = [
  #       [min_x, min_y, h, w],
  #       [min_x + h, min_y, h, w],
  #       [min_x, min_y + w, h, w],
  #       [min_x + h, min_y + w, h, w]
  #     ]
  #
  #     areas_cities = [list(), list(), list(), list()]
  #
  #     for city in all_cities_in_area:
  #       if city.x < min_x + h:
  #         if city.y < min_y + h:
  #           areas_cities[0].append(city)
  #         else:
  #           areas_cities[2].append(city)
  #       else:
  #         if city.y < min_y + h:
  #           areas_cities[1].append(city)
  #         else:
  #           areas_cities[3].append(city)
  #
  #     clusters = list()
  #     for i in range(4):
  #       _x, _y, _h, _w = areas[i]
  #       clusters.extend(self._find_rectangle_clusters(max_cities_per_cluster, _x, _y, _h, _w, areas_cities[i]))
  #
  #     return clusters
  #
  # def test_rectangle_clusters(self, max_cities_per_cluster):
  #
  #   clusters = self.find_rectangle_clusters(max_cities_per_cluster)
  #   cost = 0
  #   for cluster in clusters:
  #     print("solving cluster of size {}: {}".format(len(cluster), cluster))
  #     cluster_cities_coords = [[city.x, city.y] for city in cluster]
  #     result = HighwayMST(self.H, self.W, cluster_cities_coords).find_steiner_tree(verbose=False)
  #     # steiner_points, mst_cluster, mst_cluster_cost = B1S_DMSTM(self.H, self.W, cluster_cities_coords).b1s_dmstm(verbose=False)
  #     cost += result.sum()
  #   print(cost)
  #
  #   # we noticed that in general, this does not provide a good steiner tree on the full problem.
  
  def get_neighbors(self, point):
    result = []
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == dy == 0 or point.x + dx < 0 or point.x + dx >= self.H or point.y + dy < 0 or point.y + dy >= self.W:
          continue
        else:
          result.append(self.grid[point.x + dx][point.y + dy])
    return result
  
  def mst(self, P):
    """ same as mst in B1S_DMSTM (actually using it)"""

    b1s = B1S_DMSTM(self.H, self.W, [[city.x, city.y] for city in self.cities])
    b1s_P = [b1s.grid[p.x][p.y] for p in P]
    b1s_mst_P, cost = b1s.mst(b1s_P)
    
    mst_P = [[self.grid[b1s_p.x][b1s_p.y] for b1s_p in b1s_edge] for b1s_edge in b1s_mst_P]
    if cost !=  1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in mst_P]):
      raise ValueError("costs are different")
    return mst_P, cost

  def b1s_dmstm(self, cluster_cities, verbose = False, max_outer_loops=-1):
    
    b1s = B1S_DMSTM(self.H, self.W, [[city.x, city.y] for city in cluster_cities], grid = self.grid)
    b1s_steiner_points, b1s_mst_P, b1s_mst_P_cost = b1s.b1s_dmstm(
      verbose=verbose, max_outer_loops=max_outer_loops)
    
    # must convert to local points
    cluster_steiner_points = [self.grid[p.x][p.y] for p in b1s_steiner_points]
    cluster_mst = [[self.grid[b1s_p.x][b1s_p.y] for b1s_p in b1s_edge] for b1s_edge in b1s_mst_P]
    
    if b1s_mst_P_cost !=  1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in cluster_mst]):
      raise ValueError("costs are different")
    
    return cluster_steiner_points, cluster_mst, b1s_mst_P_cost
    
  def create_np_array_from_edges(self, edges):
    result = np.full((self.H, self.W), 0, dtype=int)
    if len(edges) > 0:
      for edge in edges:
        highways = edge[0].get_basic_path(edge[1])
        highways.append([edge[0].x, edge[0].y])
        for coords in highways:
          result[coords[0], coords[1]] = 1
    for city in self.cities:
      result[city.x, city.y] += 1
    return result
  
  def convert_highway_to_steiner_tree(self, highway_array):
    """
    returns (steiner_points, mst_P, mst_P_cost)
    note that there are can be several ways to choose steiner points
    
    example:
         1
          1  11
           11
           1
          1
    In this example, we could choose 1, 2 or 3 of the central points as steiner points.
    Our implementation chooses all of them.
    Note that there could be 2 legit adjacent steiner points): 1  1
                                                                11
                                                               1  1
    """

    steiner_points = list()
    for x in range(self.H):
      for y in range(self.W):
        if highway_array[x, y] == 1: # this is a highway and not a city
          neighbor_highways_and_cities = [n for n in self.get_neighbors(self.grid[x][y]) if highway_array[n.x, n.y] > 0]
          if len(neighbor_highways_and_cities) > 2: # the degree of point [x, y]
            steiner_points.append(self.grid[x][y])
  
    P = list(self.cities)
    P.extend(steiner_points)
    mst_P, mst_P_cost = self.mst(P)
    return steiner_points, mst_P, mst_P_cost

  def _find_neighbor_cities_and_steiner_points(self, start_city, steiner_edges):
    """
    Given a start_city, returns the neighbor cities in the graph defined by steiner_edges
    (walking through steiner points)
    also returns the steiner points between the city and its neighbors

    This is a DFS (but we could have used a BFS) around start_city, considering cities found as leaves.
    
    :param steiner_edges: a dict() of point -> set of connected points,
                          where for each key is a city or a steiner point
    
    Remark:
    To go further we could divide them into "sections"

    example:
          21    12
            2111
                12
    in this example, consider that the central city is the start city.
    there are 2 "sections":
      one goes to the left and reaches 1 city,
      the other goes to the right and reaches 2 cities

    
    """
    neighbor_cities = list()
    neighbor_steiner_points = list()
    points_visited = set()
    points_to_visit = [start_city]
  
    while len(points_to_visit) > 0:
      point = points_to_visit.pop()  # to transform that into a BFS, just do: point = points_stack.pop(0)
      points_visited.add(point)
    
      if point in self.cities_set and point is not start_city:
        neighbor_cities.append(point)
      else:  # this is a steiner point (or the start_city)
        if point is not start_city:
          neighbor_steiner_points.append(point)
        for p in steiner_edges[point]:
          if p not in points_visited:
            points_to_visit.append(p)
  
    return neighbor_cities, neighbor_steiner_points

  def find_cluster_cities_and_steiner_points(self, start_city, max_cities, steiner_edges):
    """
     walks on the steiner tree in a "BFS per highway section" manner around start_city
     returns a set of cities that is the cluster
  
     steiner_edges: a dict() of point -> set of other connected points, for each city or steiner point
    
     Each highway "section" is an edge that can lead to several child cities,
     adds cities found on the tree until the cluster reaches a size of max_cities
     
     
     IMPORTANT: once a highway section is started
     (potentially connecting more than 2 cities if there is an intersection on this section)
     then all the routers reached by the section must be added to the cluster.
     This is because, outside of this function, we will remove all the highways connecting cities in this cluster.
     
    """

    cluster_cities = set() # the cities visited
    cluster_steiner_points = set()
    sections_to_visit = [([start_city], list())] # a list of tuples
    # each section is a list of neighbors cities that we will add together, and the steiner points of the section

    while len(sections_to_visit) > 0 and len(cluster_cities) < max_cities:
      section_cities, section_sp = sections_to_visit.pop(0) # BFS is what we want
      
      for steiner_point in section_sp:
        cluster_steiner_points.add(steiner_point)
      
      for city in section_cities:
        cluster_cities.add(city)
        
        # instead of actually separating sections, here we simply take all the next neighbors together in a section:
        next_section_cities, next_section_sp = self._find_neighbor_cities_and_steiner_points(city, steiner_edges)
        next_section_cities = [nc for nc in next_section_cities if nc not in cluster_cities]
        next_section_sp = [nsp for nsp in next_section_sp if nsp not in cluster_steiner_points]
        sections_to_visit.append((next_section_cities, next_section_sp))
        
    return cluster_cities, cluster_steiner_points
  
  def select_cluster(self, start_city, max_cities_per_cluster, steiner_points, mst_P):
    """ select a cluster of cities and return the associated steiner subtree"""
  
    P = list(self.cities)
    P.extend(steiner_points)
    P_edges = dict()
    for p in P:
      P_edges[p] = set()
    for edge in mst_P:
      P_edges[edge[0]].add(edge[1])
      P_edges[edge[1]].add(edge[0])

    # take a random city and improve its local graph
    cluster_cities, cluster_steiner_points_set = self.find_cluster_cities_and_steiner_points(
      start_city, max_cities_per_cluster, P_edges)

    cluster_P = set(cluster_cities)
    cluster_P.update(cluster_steiner_points_set)
    cluster_tree = list()
    for edge in mst_P:
      if edge[0] in cluster_P and edge[1] in cluster_P:
        cluster_tree.append(edge)
    cluster_tree_cost = 1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in cluster_tree])
    
    return cluster_cities, list(cluster_steiner_points_set), cluster_tree, cluster_tree_cost
  
  @staticmethod
  def recreate_steiner_tree(cluster_cities,
                            steiner_points, mst_P, mst_P_cost,
                            pc_steiner_points, pc_tree, pc_tree_cost,
                            b1s_steiner_points, b1s_mst_P, b1s_mst_P_cost):
    """
    given the steiner tree on the full grid,
    the current sub-steiner tree on the cluster_cities,
    and a new sub-steiner tree on the cluster_cities,
    
    creates a new steiner tree on the full grid, the same as the previous one but replacing
    the previous sub-tree with the new one
    """
    
    new_steiner_points = set(steiner_points)
    for p in pc_steiner_points:
      new_steiner_points.remove(p)
    for p in b1s_steiner_points:
      new_steiner_points.add(p)
    new_steiner_points = list(new_steiner_points)
  
    pc_P = set(cluster_cities)
    pc_P.update(pc_steiner_points)
  
    new_mst_P = list()
    for edge in mst_P:
      if not (edge[0] in pc_P and edge[1] in pc_P): # all the edges outside of cluster are kept
        new_mst_P.append(edge)
    for edge in b1s_mst_P: # add the new edges
      new_mst_P.append(edge)
    
    new_mst_P_cost = mst_P_cost - pc_tree_cost + b1s_mst_P_cost

    expected_cost = 1 + sum([Point.chebyshev_distance(edge[0], edge[1]) for edge in new_mst_P])
    if new_mst_P_cost != expected_cost:
      raise ValueError("cost {} is not the expected cost: {}".format(new_mst_P_cost, expected_cost))

    return new_steiner_points, new_mst_P, new_mst_P_cost
    
  def find_steiner_tree(self, cluster_iterations=100, max_cities_per_cluster=10, verbose=True):
    
    if max_cities_per_cluster > 12 or cluster_iterations < 100:
      print("WARNING: this function is better used with a lot of small iterations than with few big iterations.")
    if cluster_iterations < len(self.cities):
      print("WARNING: recommendation: cluster_iterations should be at least the number of cities")

    highway_array = HighwayMST(self.H, self.W, [[city.x, city.y] for city in self.cities], grid=self.grid
                              ).find_steiner_tree(variation=True, verbose=verbose)
    #highway_array =  CitiesMST(self.H, self.W, [[city.x, city.y] for city in self.cities]
    #                          ).find_steiner_tree()
    
    if verbose:
      print("HighwayMST cost: {}".format(SteinerSolverItf.cost(highway_array)))

    steiner_points, mst_P, mst_P_cost = self.convert_highway_to_steiner_tree(highway_array)

    if verbose:
      print("Converted. steiner points: {}, edges: {}, cost: {}".format(len(steiner_points), len(mst_P), mst_P_cost))
    
    for n_iter in range(cluster_iterations):
      #print(self.create_np_array_from_edges(mst_P))
      
      # select cities, and save the existing steiner sub-tree of this cluster
      if n_iter < len(self.cities): # go through each city once
        start_city = self.cities[n_iter]
      else:
        start_city = random.choice(self.cities)
      
      cluster_cities, pc_steiner_points, pc_tree, pc_tree_cost = \
        self.select_cluster(start_city, max_cities_per_cluster, steiner_points, mst_P)
      
      if verbose:
        print("previous cluster - steiner points: {}, edges: {}, cost: {}".format(
          len(pc_steiner_points), len(pc_tree), pc_tree_cost))

      # find a (hopefully) better steiner tree on this small cluster. Complexity O(n^3) in size of cluster
      b1s_steiner_points, b1s_mst_P, b1s_mst_P_cost = self.b1s_dmstm(cluster_cities, verbose = False)
      
      if verbose:
        print("new cluster - steiner points: {}, edges: {}, cost: {}".format(
          len(b1s_steiner_points), len(b1s_mst_P), b1s_mst_P_cost))

      if b1s_mst_P_cost < pc_tree_cost:
        steiner_points, mst_P, mst_P_cost = Local_B1S_DMSTM.recreate_steiner_tree(
          cluster_cities = cluster_cities,
          steiner_points = steiner_points, mst_P=mst_P, mst_P_cost=mst_P_cost,
          pc_steiner_points=pc_steiner_points, pc_tree=pc_tree, pc_tree_cost=pc_tree_cost,
          b1s_steiner_points=b1s_steiner_points, b1s_mst_P=b1s_mst_P, b1s_mst_P_cost=b1s_mst_P_cost)
          
      if True or verbose:
        print("iter {}/{} done after selecting city {} and a cluster of {} cities. "
              "Current full solution: steiner points {}, edges {}, cost {}".format(
          n_iter, cluster_iterations, start_city, len(cluster_cities), len(steiner_points), len(mst_P), mst_P_cost))

    return self.create_np_array_from_edges(mst_P)

def test_1():
  for [H, W] in [[8, 16], [16, 32], [32, 64], [64, 128]]:
    # with size [128, 256] b1s_dmstm finds a tree with cost 1373 or less, where highwayMST finds 1389 but takes ~500s
    
    random.seed(17) # for reproducibility
    all_cities_coords = [[x, y] for y in range(W) for x in range(H)]
    random.shuffle(all_cities_coords)
    cities_coords = all_cities_coords[:int((H+W)/2)]
    print_steiner_trees = H <= 27 and W <= 36
    
    print("\n#######")
    print("# computing steiner trees with different methods on a grid of shape ({},{}) with {} cities"
          .format(H, W, len(cities_coords)))
    print("#######\n")
  
    start_time = time.time()
    citiesMST = CitiesMST(H, W, cities_coords).find_steiner_tree()
    SteinerSolverItf.check_steiner_tree(citiesMST, cities_coords)
    if print_steiner_trees:
      print(citiesMST)
    print("citiesMST cost: {}, time taken: {:.2f}s\n".format(SteinerSolverItf.cost(citiesMST), time.time()-start_time))

    start_time = time.time()
    highwayMST = HighwayMST(H, W, cities_coords).find_steiner_tree(variation=True, verbose=False)
    SteinerSolverItf.check_steiner_tree(highwayMST, cities_coords)
    if print_steiner_trees:
      print(highwayMST)
    print("highwayMST (v) cost: {}, time taken: {:.2f}s\n".format(SteinerSolverItf.cost(highwayMST), time.time()-start_time))

    start_time = time.time()
    b1s_dmstm = B1S_DMSTM(H, W, cities_coords).find_steiner_tree(method="B1S_DMSTM", verbose= False)
    SteinerSolverItf.check_steiner_tree(b1s_dmstm, cities_coords)
    if print_steiner_trees:
      print(b1s_dmstm)
    print("b1s_dmstm cost: {}, time taken: {:.2f}s\n".format(SteinerSolverItf.cost(b1s_dmstm), time.time() - start_time))

    start_time = time.time()
    mx = 50 if len(cities_coords) > 50 else int(len(cities_coords)/2)
    local_b1s_dmstm = Local_B1S_DMSTM(H, W, cities_coords).find_steiner_tree(max_cities_per_cluster=mx, verbose=False)
    SteinerSolverItf.check_steiner_tree(local_b1s_dmstm, cities_coords)
    if print_steiner_trees:
      print(local_b1s_dmstm)
    print("local_b1s_dmstm  with max_cities_per_cluster={}, cost: {}, time taken: {:.2f}s\n".format(
      mx, SteinerSolverItf.cost(local_b1s_dmstm), time.time() - start_time))


def test_2():
  # with size [128, 256] b1s_dmstm finds a tree with cost 1373 or less, where highwayMST finds 1389 but takes ~500s
  
  start_time = time.time()
  random.seed(17)  # for reproducibility
  H, W = [128, 256] #[667, 540]
  all_cities_coords = [[x, y] for y in range(W) for x in range(H)]
  random.shuffle(all_cities_coords)
  #cities_coords = all_cities_coords[:857]
  cities_coords = all_cities_coords[:int((H+W)/2)]

  #b1s_dmstm = B1S_DMSTM(H, W, cities_coords).find_steiner_tree(method="B1S_DMSTM", verbose=True, max_outer_loops=3)
  #print(b1s_dmstm)
  #SteinerSolverItf.check_steiner_tree(b1s_dmstm, cities_coords)
  #print("cities: {}, cost: {}".format(len(cities_coords), SteinerSolverItf.cost(b1s_dmstm)))
  #print("time {:.2f}s".format(time.time()-start_time))

  local_b1s_dmstm = Local_B1S_DMSTM(H, W, cities_coords).find_steiner_tree(cluster_iterations=200, max_cities_per_cluster=20, verbose = True)
  print(local_b1s_dmstm)
  SteinerSolverItf.check_steiner_tree(local_b1s_dmstm, cities_coords)
  print("cities: {}, cost: {}".format(len(cities_coords), SteinerSolverItf.cost(local_b1s_dmstm)))

  
if __name__ == "__main__":
  test_2()