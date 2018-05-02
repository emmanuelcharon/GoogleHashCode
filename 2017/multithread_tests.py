import time
import math
import threading
import os
from multiprocessing import Pool
import random

def factorize_naive(n):
  """ A naive factorization method. Take integer 'n', return list of
      factors.
  """
  if n < 2:
    return []
  factors = []
  p = 2
  
  while True:
    #print(p, factors)
    if n == 1:
      return factors
    
    r = n % p
    if r == 0:
      factors.append(p)
      n = n // p # integer division in python3
    elif p * p >= n:
      factors.append(n)
      return factors
    elif p > 2:
      # Advance in steps of 2 over odd numbers
      p += 2
    else:
      # If p == 2, get to 3
      p += 1

def serial_factorizer(nums):
  return {n: factorize_naive(n) for n in nums}

def threaded_factorizer(nums, nthreads):
  def worker(nums, outdict):
    """ The worker function, invoked in a thread. 'nums' is a
        list of numbers to factor. The results are placed in
        outdict.
    """
    for n in nums:
      outdict[n] = factorize_naive(n)

  # Each thread will get 'chunksize' nums and its own output dict
  chunksize = int(math.ceil(len(nums) / float(nthreads)))
  threads = []
  outs = [{} for i in range(nthreads)]

  for i in range(nthreads):
    # Create each thread, passing it its chunk of numbers to factor
    # and output dict.
    t = threading.Thread(
      target=worker,
      args=(nums[chunksize * i:chunksize * (i + 1)],
            outs[i]))
    threads.append(t)
    t.start()

  # Wait for all threads to finish
  for t in threads:
    t.join()

  # Merge all partial output dicts into a single dict and return it
  return {k: v for out_d in outs for k, v in out_d.items()}

def mp_factorizer(nums, nprocs):
  with Pool(nprocs) as p:
    return p.map(factorize_naive, nums)

def test():
  numbers = [1234567898765432103] * 20
  numbers.extend([1234567898765432101] * 20)
  num_processes = os.cpu_count()  # multiprocessing.cpu_count()
  print(numbers)
  print(num_processes)
  
  for i in range(3):
    start_time = time.time()
    result = None
    if i == 0:
      result = serial_factorizer(numbers)
      print("serial time_taken: {:.2f}s".format(time.time() - start_time))
    elif i == 1:
      result = threaded_factorizer(numbers, num_processes)
      print("threaded time_taken: {:.2f}s".format(time.time() - start_time))
    elif i == 2:
      result = mp_factorizer(numbers, num_processes)
      print("multiprocess time_taken: {:.2f}s".format(time.time() - start_time))

if __name__ == "__main__":
  test()
  