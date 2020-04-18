import os
import numpy as np
import scipy
import scipy.sparse
import time

def read_problem(input_file_path):
  problem = None
  
  with open(input_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        ls = [int(a) for a in l.split(' ')]
        problem = Problem(ls[0],ls[1],ls[2],ls[3],ls[4])
      elif line_count <= problem.N:
        ls = [float(a) for a in l.split(' ')]
        junction = Junction(len(problem.junctions),ls[0], ls[1])
        problem.junctions.append(junction)
      else:
        ls = [int(a) for a in l.split(' ')]
        street = Street(len(problem.streets), ls[0], ls[1], True if ls[2]==2 else False, ls[3], ls[4])
        # signal to junctions jA and jB that they touch this street
        problem.junctions[street.jA].streets.append(street) # we can use this street if we are in jA
        if street.bi_directional:
          problem.junctions[street.jB].streets.append(street) # we can use this street if we are in jB
        
        problem.streets.append(street)

      line_count += 1
  
  return problem


def write_solution(problem, output_file_path):
  with open(output_file_path, 'w') as f:
    f.write("{}\n".format(len(problem.cars)))
    for car in problem.cars:
      f.write("{}\n".format(car.ID))
      for junction in car.get_junctions_list(problem.S):
        f.write("{}\n".format(junction))

class Junction:
  def __init__(self, ID, latitude, longitude):
    self.ID = ID
    self.lat = latitude
    self.lon = longitude
    self.streets = list() # all the streets where we can go FROM this junction
  
class Street:
  def __init__(self, ID, jA, jB, bi_directional, cost, length):
    self.ID = ID
    self.jA = jA # the ID of junctionA, not the object itself
    self.jB = jB
    self.bi_directional = bi_directional # a boolean
    self.length = length
    self.cost = cost
    
    # dynamic
    self.num_visits = 0 # the number of times a street view car goes through this street

  def __repr__(self):
    return "street {}: jA={}, jB={}, two-way: {}".format(self.ID, self.jA, self.jB, self.bi_directional)


class Car:
  def __init__(self, ID, T, initial_junction):
    self.ID = ID
    self.itinerary = list() # the list of streets this car will go through
    self.last_junction = initial_junction # the last junction on the itinerary
    self.time_left = T
  
  def add_street(self, street):
    if self.time_left < street.cost:
      raise ValueError("car {}: not enough time to add street {} (time left: {}, street cost: {}".format(
        self.ID, street.ID, self.time_left, street.cost))
    
    if self.last_junction != street.jA and self.last_junction != street.jB:
      raise ValueError("car {}'s last junction is not in the street {}".format(self.ID, street.ID))

    if not street.bi_directional and self.last_junction != street.jA:
      raise ValueError("car {}'s last junction is {}, and is not is not the uni-directional street {}'s start: jA={} (where jB={})".format(
        self.ID, self.last_junction, street.ID, street.jA, street.jB))
    
    self.itinerary.append(street)
    self.last_junction = street.jB if self.last_junction  == street.jA else street.jA
    self.time_left -= street.cost
    street.num_visits += 1
  
  def get_junctions_list(self, initial_junction):
    """ creates the list of junctions based on the initial junction and the streets itinerary """
    junctions = [initial_junction]
    for street in self.itinerary:
      junctions.append(street.jB if junctions[-1] == street.jA else street.jA)
    return junctions

  
class Problem:
  
  def __init__(self, N, M, T, C, S):
    self.N = N # number of junctions
    self.M = M # number of streets
    self.T = T # total time
    self.C = C # num cars
    self.S = S # cars initial junction
    
    self.junctions = list()
    self.streets = list()
    self.cars = [Car(i, self.T, self.S) for i in range(0, self.C)]
   
    self.paths_dist_matrix = None
    self.paths_predecessors = None
   
  def describe(self):
    print("junctions: {}, streets: {} , time: {}s, cars: {}, initial junction: {}".format(
      len(self.junctions), len(self.streets), self.T, len(self.cars), self.S))
    
    all_lengths = [street.length for street in self.streets]
    all_costs = [street.cost for street in self.streets]
    all_ratios = [street.length/street.cost for street in self.streets]
    all_degrees = [len(j.streets) for j in self.junctions]

    print("total_length (max possible score): {}, max_length: {}, min_length: {}, avg_length: {}".format(
      sum(all_lengths), max(all_lengths), min(all_lengths), sum(all_lengths)/len(all_lengths)))

    print("total_cost per car: {}, max_cost: {}, min_cost: {}, avg_cost: {}".format(
      sum(all_costs)/self.C, max(all_costs), min(all_costs), sum(all_costs) / len(all_costs)))

    print("max_ratio: {}, min_ratio: {}, avg_ratio (points per second of travel): {}".format(
      max(all_ratios), min(all_ratios), sum(all_ratios) / len(all_ratios)))

    print("max_degree: {}, min_degree: {}, avg_degree: {}".format(
      max(all_degrees), min(all_degrees), sum(all_degrees) / len(all_degrees)))
  
  
  def distance(self, junctionA, junctionB):
    if self.paths_predecessors[junctionA, junctionB] >= 0:
      return self.paths_dist_matrix[junctionA, junctionB]
    else:
      raise ValueError("no path found from {} to {} => infinite distance".format(junctionA, junctionB))
  
  def shortest_path(self, junctionA, junctionB):
    path = []
    predecessor = junctionB
    while predecessor != junctionA and predecessor >= 0:
      next_predecessor = self.paths_predecessors[junctionA, predecessor]
      # now find the street going from next_predecessor to predecessor
      for s in self.junctions[next_predecessor].streets:
        if (s.jA == next_predecessor and s.jB == predecessor) or (s.bi_directional and s.jA==predecessor and s.jB == next_predecessor):
          path.append(s)
      predecessor = next_predecessor
    return path
      
  def scipy_compute_shortest_paths(self):
    edges_dense = np.full(fill_value=0, shape=(len(self.junctions), len(self.junctions)), dtype=np.int32)
    
    for s in self.streets:
      edges_dense[s.jA, s.jB] = s.cost
      if s.bi_directional:
        edges_dense[s.jB, s.jA] = s.cost
    edges_sparse = scipy.sparse.csgraph.csgraph_from_dense(edges_dense, null_value=0)
    
    print(np.max(edges_sparse))
    print(np.min(edges_sparse))

    print("starting shortest paths computation")
    start_time = time.time()
    self.paths_dist_matrix, self.paths_predecessors = scipy.sparse.csgraph.dijkstra(edges_sparse, return_predecessors=True)
    print("shortest paths computation finished in {}s".format(time.time()-start_time))
 
  def score(self):
    result = 0
    for street in self.streets:
      if street.num_visits > 0:
        result += street.length
    return result
  
  def solve_least_visits(self):
    
    while True:
      
      # select the next possible "move" with the least visits
      min_visits = self.T  # like infinity
      best_car = None
      best_street = None

      for car in self.cars:
        for street in self.junctions[car.last_junction].streets:
          if car.time_left >= street.cost and street.num_visits < min_visits:
            min_visits = street.num_visits
            best_car = car
            best_street = street
      
      if best_car is None:
        break # we return here, because no car could be moved
      else:
        best_car.add_street(best_street)
        #print("car {}: added street {}".format(best_car.ID, best_street.ID))

  def solve_greedy(self):
    
    # at each step, we will list our possible moves and select one
    # we select the one with the best length/cost ratio (points per second)
    step = 0
    while True:
      step += 1
      # a move is: a car, a street, and "move score"
      possible_moves = list()
      for car in self.cars:
        for street in self.junctions[car.last_junction].streets:
          if car.time_left >= street.cost:
            # move_score = 0 if street.num_visits > 0 else street.length/street.cost
            move_score = 0 if street.num_visits > 0 else street.length
            possible_moves.append([car, street, move_score])

      best_move_score = 0
      best_move = None
      for move in possible_moves:
        if move[2] > best_move_score:
          best_move = move
          best_move_score = move[2]
      
      if step % 10 == 0:
        print("step {}; possible moves: {}, best move score: {}".format(step, len(possible_moves), best_move_score))
      
      if best_move_score > 0:
        # perform this move
        best_move[0].add_street(best_move[1])
      elif len(possible_moves) == 0:
        # no time to move any car
        break
      else:
        # all the moves are worth 0 points, but we can move a car closer to unvisited streets
        # I decided to move a car to the least visited street
        best_move_visits = self.T # like infinity
        best_move = None
        for move in possible_moves:
          if move[1].num_visits < best_move_visits:
            best_move_visits = move[1].num_visits
            best_move = move
        if best_move is None:
          raise ValueError("bestMove is still none")
        else:
          best_move[0].add_street(best_move[1])

  def yield_all_trips_of_depth(self, start_junction, depth):
    """
    returns sequences of streets of length depth or less, which are feasible trips (with possible loops)
    starting from the current junction at which the car is
    """
    if depth < 0:
      raise ValueError("depth is {}".format(depth))
    elif depth == 0:
      yield []
    else:
      for s in self.junctions[start_junction].streets:
        end_junction = s.jB if start_junction==s.jA else s.jA
        for trip in self.yield_all_trips_of_depth(end_junction, depth-1):
          yield [s] + trip
  
  def yield_all_trips_up_to_depth(self, start_junction, depth):
    for d in range(1, depth+1):
      for trip in self.yield_all_trips_of_depth(start_junction, d):
        if len(trip) > 0:
          yield trip
    
  def solve_greedy_depth(self, depth):
    """
    in the basic greedy approach, we looked for the next "move" each car can make
    we now consider a "move" is a trip of several streets, for each car (a max of depth streets)
    we compare all possible "moves" at the given max_depth at each step
    """

    step = 0
    while True:
      step += 1
      
      num_trips_considered = 0
      best_trip = None
      best_trip_car = None
      best_trip_gain = 0
      best_trip_cost = 1000000000000
      
      percent_time_left = sum([car.time_left for car in self.cars])/len(self.cars)/self.T
      
      for car in self.cars:
        for trip in self.yield_all_trips_up_to_depth(car.last_junction, depth):
          num_trips_considered += 1
          trip_cost = sum([s.cost for s in trip])
          if trip_cost <= car.time_left:
            # points in the trip: only the new streets
            new_streets_visited = set([s for s in trip if s.num_visits == 0])
            trip_gain = sum([s.length  for s in new_streets_visited])
            #if percent_time_left > 0.1:
              # at the beginning optimize gain/cost, at the end just look for the best gain
              #trip_gain /= trip_cost
            
            if trip_gain > best_trip_gain or (trip_gain>0 and trip_gain==best_trip_gain and trip_cost<best_trip_cost):
              best_trip = trip
              best_trip_car = car
              best_trip_gain = trip_gain
              best_trip_cost = trip_cost
      
      if best_trip is None:
        
        # take the immediate move with least visits
        least_visits = 1000000
        least_visits_car = None
        least_visits_street = None
        
        for car in self.cars:
          for street in self.junctions[car.last_junction].streets:
            if car.time_left >= street.cost and street.num_visits < least_visits:
              least_visits = street.num_visits
              least_visits_car = car
              least_visits_street = street
        if least_visits_car is None:
          break # no more possible move
        else:
          least_visits_car.add_street(least_visits_street)
            
      else:
        # do the best trip
        for s in best_trip:
          best_trip_car.add_street(s)

        if step % 1000 == 0:
          print("step {}; % time left: {}, trips considered: {}; best trip num steps {}, cost {}, gain {}".format(
            step, percent_time_left, num_trips_considered, len(best_trip), best_trip_cost, best_trip_gain))
     
  
  def describe_solution(self):
    print("number of streets for each car: {}".format([len(car.itinerary) for car in self.cars]))
    print("number of streets visited: {}/{}".format(sum([1 if street.num_visits>0 else 0 for street in self.streets]),len(self.streets)))
    print("number of streets visited more than once {}".format(sum([1 if street.num_visits > 1 else 0 for street in self.streets])))

    print("avg travel time per car: {}s".format(sum([street.cost for car in self.cars for street in car.itinerary ]) / len(self.cars)))

    print("avg wasted travel time per car: {}s".format(sum([street.cost*(street.num_visits - 1) if street.num_visits > 1 else 0 for street in self.streets])/len(self.cars)))

def test_shortest_path(problem):
  for i in range(0, 1000, 10):
    jA, jB = i, i + 3
    print(problem.distance(jA, jB))
    print(sum([s.cost for s in problem.shortest_path(jA, jB)]))


def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  input_file_path = os.path.join(dir_path, "input", "paris_54000.txt")
  output_file_path = os.path.join(dir_path, "output", "greedy_depth_1.out")

  distances_file_path = os.path.join(dir_path, "output", "distances.npy")
  predecessors_file_path = os.path.join(dir_path, "output", "predecessors.npy")

  problem = read_problem(input_file_path)
  
  #problem.scipy_compute_shortest_paths()
  #np.save(distances_file_path, problem.paths_dist_matrix)
  #np.save(predecessors_file_path, problem.paths_predecessors)
  problem.paths_dist_matrix = np.load(distances_file_path)
  problem.paths_predecessors = np.load(predecessors_file_path)

  problem.describe()
  #problem.solve_greedy_depth(3)
  
  # for each car, find sub-trips composed of consecutive streets visited at least twice
  # replace this subtrip with a shortest path between start and end of trip
  
  #print("\n\nscore for greedy: {}".format(problem.score()))
  #problem.describe_solution()
  #write_solution(problem, output_file_path)

if __name__ == "__main__":
  main()





   
    
