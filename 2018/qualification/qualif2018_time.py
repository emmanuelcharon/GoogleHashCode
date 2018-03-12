import os
import numpy as np
import time

def read_problem(input_file_path):
  problem = None
  
  with open(input_file_path, 'r') as f:
    line_count = 0
    for line in f.readlines():
      l = line.rstrip()
      if line_count == 0:
        ls = [int(a) for a in l.split(' ')]
        problem = Problem(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5])
      else:
        ls = [int(a) for a in l.split(' ')]
        ride = Ride(len(problem.rides),ls[0],ls[1],ls[2],ls[3],ls[4],ls[5])
        problem.rides.append(ride)
      line_count += 1
  
  if len(problem.rides) != problem.N:
    print("error")
  
  return problem

def write_solution(problem, output_file_path):
  with open(output_file_path, 'w') as f:
    for car in problem.cars:
      f.write("{} {}\n".format(len(car.rides), " ".join([str(ride.ID) for ride in car.rides])))

def distance(point_a, point_b):
  return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

class Problem:
  
  def __init__(self, R, C, F, N, B ,T):
    self.R = R
    self.C = C
    self.F = F
    self.N = N
    self.B = B
    self.T = T

    self.cars = [Car(i) for i in range(0, F)]
    self.rides = list()

    self.score = 0
    self.assignments = list()
    # self.assignments[car.ID][ride.ID] = assignment object if we were to add ride as the next ride of car

    for car_id in range(self.F):
      l = list()
      for ride_id in range(self.N):
        l.append(0)
      self.assignments.append(l)
    self.ranking_scores = np.zeros((self.F, self.N), dtype=float)
    
  def __str__(self):
    return "cars: {} , rides : {}, bonus {}, city size: ({}, {}), total time: {};\nscore {}".format(
      len(self.cars), len(self.rides), self.B, self.R, self.C, self.T, self.score)
  
  def compute_assignment(self, car, ride):
    
    if ride.assigned:
      print("error: ride is already assigned")
    
    [t, pos] = car.time_and_pos_next_available()
  
    initial_distance = distance(pos, ride.start)
    start_time = max(t + initial_distance, ride.earliest_start)
    end_time = start_time + ride.length
  
    gain = 0
    if end_time <= ride.latest_finish:
      gain += ride.length
      if start_time == ride.earliest_start:
        gain += self.B
    
    # the bigger the ranking score, the better, this is how we choose the next assignment
    ranking_score = -1
    if gain > 0:
      ranking_score = self.T - start_time +  gain/(self.B*(self.R+self.C)) - initial_distance/(self.R+self.C)
    
    return Assignment(ride, car, start_time, end_time, initial_distance, gain, ranking_score)

  def solve(self):
    
    
    # compute initial assignments
    for car in self.cars:
      for ride in self.rides:
        a = self.compute_assignment(car, ride)
        self.assignments[car.ID][ride.ID] = a
        self.ranking_scores[car.ID, ride.ID] = a.ranking_score

    remaining_rides = set(self.rides)

    while len(remaining_rides) > 0:
      if len(remaining_rides) % max(1, int(self.N/50)) == 0:
        print("remaining rides: {}/{}, score: {}".format(len(remaining_rides), self.N, self.score))

      # select the best assignment
      [best_car_id, best_ride_id] = np.unravel_index(self.ranking_scores.argmax(), self.ranking_scores.shape)

      if self.ranking_scores[best_car_id, best_ride_id] <= 0:
        print("stopping because no best assignment")
        break

      best_assignment = self.assignments[best_car_id][best_ride_id]
      
      # do the assignment
      best_assignment.car.add_ride(best_assignment.ride, best_assignment.end_time)
      remaining_rides.remove(best_assignment.ride)
      self.score += best_assignment.gain
      
      # update affected assignments & ranking_scores
      for car in self.cars:
        self.ranking_scores[car.ID][best_assignment.ride.ID] = -1
        self.assignments[car.ID][best_assignment.ride.ID] = None
      for ride in remaining_rides:
        a = self.compute_assignment(best_assignment.car, ride)
        self.assignments[best_assignment.car.ID][ride.ID] = a
        self.ranking_scores[best_assignment.car.ID][ride.ID] = a.ranking_score
    print("stopped with {} remaining rides".format(len(remaining_rides)))
    
class Assignment:
  def __init__(self, ride, car, start_time, end_time, initial_distance, gain, ranking_score):
    self.ride = ride
    self.car = car
    
    self.start_time = start_time
    self.end_time = end_time
    self.initial_distance = initial_distance
    self.gain = gain
    
    self.ranking_score = ranking_score
    
class Ride:
  def __init__(self, ID, a, b, x, y, s, f):
    self.ID = ID
    self.start = [a, b]
    self.end = [x, y]
    self.earliest_start = s
    self.latest_finish = f
    self.length = distance(self.end, self.start)
    self.assigned = False

class Car:
  def __init__(self, ID):
    self.ID = ID
    self.rides = list()
  
    self.time_next_available = 0
    self.position_next_available = [0, 0]
    
    
  def add_ride(self, ride, end_time):
    self.rides.append(ride)
    ride.assigned = True
    self.time_next_available = end_time
    self.position_next_available = ride.end
  
  def time_and_pos_next_available(self):
    return [self.time_next_available, self.position_next_available]
  
  #def is_available(self, t):
  #  return t >= self.time_next_available
    
def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["a_example", "b_should_be_easy", "c_no_hurry", "d_metropolis", "e_high_bonus"]
  # ["a_example"]
  scores = []
  
  for sample in samples:
    print("\n####### {}\n".format(sample))
    start_time = time.time()
    
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output_time", sample + ".out")

    problem = read_problem(input_file_path)
    
    problem.solve()
    
    print(problem)
    print("computation time: {}".format(time.time()-start_time))
    
    write_solution(problem, output_file_path)
    scores.append(problem.score)
  print("\n#####\nscores: {} => total score = {}".format(scores, sum(scores)))


if __name__ == "__main__":
  main()





   
    
