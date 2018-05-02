import numpy as np
import os
import time

def read_streaming_problem_statement(input_file_path):
  
  with open(input_file_path, 'r') as f:
    [V, E, R, C, X] = [int(a) for a in next(f).strip().split(' ')]
    streaming = Streaming(V, E, R, C, X)
    
    streaming.servers = [Server(s, X) for s in range(C)]
    
    video_sizes = [int(a) for a in next(f).strip().split(' ')]
    
    for v in range(V):
       streaming.videos.append(Video(v, video_sizes[v]))
    
    for e in range(E):
      [Ld, K] = [int(a) for a in next(f).strip().split(' ')]
      endpoint = Endpoint(e, Ld)
      for i in range(K):
        [c, Lc] = [int(a) for a in next(f).strip().split(' ')]
        endpoint.latencies[c] = Lc
        #streaming.servers[c].connectedEndpoints.add(endpoint)
      streaming.endpoints.append(endpoint)
    
    for r in range(R):
      [Rv, Re, Rn] = [int(a) for a in next(f).strip().split(' ')]
      request = Request(r, streaming.videos[Rv], streaming.endpoints[Re], Rn)
      streaming.requests.append(request)
      #streaming.endpoints[Re].requests.add(request)
  
  return streaming

def write_streaming_solution(streaming, output_file_path):
  servers_used = [server for server in streaming.servers if len(server.videos) > 0]

  with open(output_file_path, 'w') as f:
    f.write(str(len(servers_used))+"\n")
    for server in servers_used:
      line = str(server.id) + " " + " ".join([str(video.id) for video in server.videos])
      f.write(line+"\n")

def read_streaming_solution(streaming, output_file_path):
  """ acts as if streaming starts "empty" """
  pass


class Video:
  def __init__(self, v, size):
    self.id = v
    self.size = size

class Server:
  def __init__(self, s, X):
    self.id = s
    self.space = X # remaining capacity, starts at X
    #self.connectedEndpoints = set() # endpoints connected to this server, fixed
    self.videos = set() # videos in this server, dynamic
  
class Endpoint:
  def __init__(self, e, Ld):
    self.id = e
    self.Ld = Ld
    self.latencies = dict() # key = connected server id, value = latency
    #self.requests = set() # requests that happen on this endpoint
    
class Request:
  def __init__(self, r, video, endpoint, n):
    self.id = r
    self.video = video
    self.endpoint = endpoint
    self.n = n
    self.server = None # the current server used by this request, dynamic
    
class Streaming:
  def __init__(self, V, E, R, C, X):
    self.videos = list()
    self.servers = list()
    self.endpoints = list()
    self.requests = list()
    self.X = X
    
    # cache
    self.videos_added = 0
    self.summed_score = 0
    self.affected_requests = None # will be a a list of lists of sets: affected requests by video v being in server s
  
  def get_total_score(self):
    #nope : must sum all request, and also the 1000x
    total_requests = sum([request.n for request in self.requests])
    return int((self.summed_score * 1000)/total_requests)
  
  def cache_affected_requests(self):
    self.affected_requests = [[set() for s in range(len(self.servers))] for v in range(len(self.videos))]
    
    for request in self.requests:
      for s in request.endpoint.latencies.keys():
        self.affected_requests[request.video.id][s].add(request)
   
  def __repr__(self):
    return "videos: {}, endpoints: {}, requests: {}, servers: {}, server capacity: {}".format(
      len(self.videos), len(self.endpoints), len(self.requests), len(self.servers), self.X)
  
  def compute_gain(self, video, server):
    """ total score gain if we put video v in server s,
     returns -1 if the server has not enough space
     returns -2 if the video already is in the server """
    
    if video in server.videos:
      return -2
    
    if server.space < video.size:
      return -1
    
    score_gain = 0

    for request in self.affected_requests[video.id][server.id]:
      if request.video == video:
        current_latency = request.endpoint.latencies[request.server.id] if request.server else request.endpoint.Ld
        new_latency = request.endpoint.latencies[server.id]
        if new_latency < current_latency:
          # current_request_score = request.n * (endpoint.Ld - current_latency)
          # new_request_score = request.n * (endpoint.Ld - new_latency)
          # score_gain += new_request_score - current_request_score
          score_gain += request.n * (current_latency - new_latency) # faster

    return score_gain
    
  def add_video_to_server(self, video, server):
    if video in server.videos:
      print("video already in server")
      return
  
    if server.space < video.size:
      print("not enough space in server")
    
    server.videos.add(video)
    server.space -= video.size
    
    # also modify the server used by each request potentially affected
    for request in self.affected_requests[video.id][server.id]:
      if request.video == video:
        current_latency = request.endpoint.latencies[request.server.id] if request.server else request.endpoint.Ld
        new_latency = request.endpoint.latencies[server.id]
        if new_latency < current_latency:
          request.server = server
  
  def solve(self):
    self.cache_affected_requests()
    # compute initial gains
    gains = np.zeros((len(self.videos), len(self.servers)), dtype=int)
    for v in range(len(self.videos)):
      if v%(len(self.videos)/10) == 0:
        print("initial gains {}/{}".format(v, len(self.videos)))
  
      for s in range(len(self.servers)):
        gains[v, s] = self.compute_gain(self.videos[v], self.servers[s])
    print("done computing initial gains")
    
    while True:
      [best_v, best_s] = np.unravel_index(gains.argmax(), gains.shape)
      best_gain = gains[best_v, best_s]
      if best_gain <= 0:
        print("stopping because no video addition is possible or brings new score")
        break

      [best_video, best_server] = [self.videos[best_v], self.servers[best_s]]
      self.add_video_to_server(best_video, best_server)
      
      # compute gains that have changed:
      # 1. for each server, gain if put best_v in it
      for s in range(len(self.servers)):
        gains[best_v, s] = self.compute_gain(best_video, self.servers[s])
      # 2. for each video, whether we can still put it in our best_server
      for v in range(len(self.videos)):
        if self.videos[v].size > best_server.space:
          gains[v, best_s] = -1

      # print("added video {} to server {}".format(best_video.id, best_server.id))

      self.videos_added += 1
      self.summed_score += best_gain
      if self.videos_added % max(1, int(len(self.videos)/10)) == 0:
        print("videos added: {}, total score: {}".format(self.videos_added, self.get_total_score()))


def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  samples = ["example", "me_at_the_zoo", "trending_today", "videos_worth_spreading", "kittens"]
  scores = []
  
  for sample in samples:
    start_time = time.time()
    print("\n##### {}\n".format(sample))
    input_file_path = os.path.join(dir_path, "input", sample + ".in")
    output_file_path = os.path.join(dir_path, "output", sample + ".out")
    
    streaming = read_streaming_problem_statement(input_file_path)
    print(streaming)
    streaming.solve()
    print("Total videos added: {}, Total score: {}".format(streaming.videos_added, streaming.get_total_score()))
    
    write_streaming_solution(streaming, output_file_path)

    print("{} solve time: {:.2}s".format(sample, time.time() - start_time))
    if sample != "example":
      scores.append(streaming.get_total_score())

  print("scores: {} => {}".format(scores, sum(scores)))
  
if __name__=="__main__":
  main()