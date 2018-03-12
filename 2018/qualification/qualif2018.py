import os


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
  
  def __str__(self):
    return "cars: {} , rides : {}, bonus {}, city size: ({}, {}), total time: {};\nscore {}".format(
      len(self.cars), len(self.rides), self.B, self.R, self.C, self.T, self.score)
  
  def solve(self):
    
    # sort rides
    remaining_rides = [ride for ride in self.rides]
    
    for ride in remaining_rides:
      ride.potential_car = self.cars[0]
      self.cars[0].potential_rides.add(ride)
    
    remaining_rides = sorted(remaining_rides, key=lambda ride: ride.earliest_start)
    #remaining_rides = sorted(remaining_rides, key=lambda ride: max(ride.earliest_start, distance([0, 0], ride.start)))

    while len(remaining_rides) > 0:
      if len(remaining_rides) % max(1, int(self.N/100)) == 0:
        print("remaining rides: {}/{}, score: {}".format(len(remaining_rides), self.N, self.score))

      # re-sort remaining rides
      earliest_start_ride = None
      earliest_start_time = self.T
      
      for ride in remaining_rides:
        if ride.earliest_start_possible < earliest_start_time:
          earliest_start_ride = ride
          earliest_start_time = ride.earliest_start_possible
        
      if not earliest_start_ride:
        break
      remaining_rides.remove(earliest_start_ride)
      ride = earliest_start_ride
      
      # find best gain cars for this ride
      best_assignments = []
      best_gain = 0
      
      for car in self.cars:
        [t, pos] = car.time_and_pos_next_available()
        
        initial_distance = distance(pos, ride.start)
        start_time = max(t + initial_distance, ride.earliest_start)
        end_time = start_time + ride.length
        
        gain = 0
        if end_time <= ride.latest_finish:
          gain += ride.length
          if start_time == ride.earliest_start:
            gain += self.B
        
        ranking_score = (self.R+self.C)*start_time+initial_distance
        
        a = Assignment(ride, car, start_time, end_time, initial_distance, gain, ranking_score)
        
        if a.gain > 0:
          if a.gain > best_gain:
            best_gain = a.gain
            best_assignments = [a]
          elif a.gain == best_gain:
            best_assignments.append(a)
      
      if best_gain > 0 and len(best_assignments) > 0:
        best_assignments = sorted(best_assignments, key=lambda assignment: assignment.ranking_score)
        a = best_assignments[0]
        a.car.add_ride(a.ride, a.end_time)
        self.score += a.gain
        
        # update
        rides_to_update = list(a.car.potential_rides)
        if ride in rides_to_update:
          rides_to_update.remove(ride)
        a.car.potential_rides = set()
        for potential_ride in rides_to_update:
          [best_time, best_car] = self.find_earliest_start_time(potential_ride)
          if best_car:
            potential_ride.potential_car = best_car
            potential_ride.earliest_start_possible = best_time
            best_car.potential_rides.add(potential_ride)
      
  def find_earliest_start_time(self, ride):
    """ finds the earliest time we could start this ride (with gain > 0)"""
    
    best_time = self.T
    best_car = None
    
    for car in self.cars:
      [t, pos] = car.time_and_pos_next_available()
      initial_distance = distance(pos, ride.start)
      start_time = max(t + initial_distance, ride.earliest_start) # if this car was to take this ride
      end_time = start_time + ride.length

      if start_time < best_time and end_time <= ride.latest_finish:
        best_time = start_time
        best_car = car

    #best_time = best_time - self.T - ride.length
    
    #if best_time == ride.earliest_start:
      #best_time -= self.T

    return [best_time, best_car]

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
  
  
    self.earliest_start_possible = max(self.earliest_start, distance([0, 0], self.start))
    self.potential_car = None # should be the first car

class Car:
  def __init__(self, ID):
    self.ID = ID
    self.rides = list()
  
    self.time_next_available = 0
    self.position_next_available = [0, 0]
    
    self.potential_rides = set()
    
  def add_ride(self, ride, end_time):
    self.rides.append(ride)
    
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

    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")

    problem = read_problem(input_file_path)
    
    problem.solve()
    
    print(problem)
    
    write_solution(problem, output_file_path)
    scores.append(problem.score)
  print("\n#####\nscores: {} => total score = {}".format(scores, sum(scores)))


if __name__ == "__main__":
  main()





   
    
