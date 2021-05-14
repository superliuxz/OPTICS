import math
import numpy as np
from sklearn.cluster import OPTICS as OPTICS_SKL
import sys
import matplotlib.pyplot as plt


np.random.seed(0)


def euclidean_distance(p1, p2):
  return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def get_dist_mat(data):
  N = len(data)
  dist_mat = [[0.0]*N for _ in range(N)]
  for i in range(N):
    for j in range(i+1, N):
      dist_mat[i][j] = dist_mat[j][i] = euclidean_distance(data[i], data[j])
  return dist_mat


class OPTICS:
  def __init__(self, data, min_pts):
    self.min_pts = min_pts
    self.data = data
    self.N = len(data)
    self.order = list()
    self.reachability_dist = [float("inf")] * self.N
    self.core_dist = [float("inf")] * self.N
    self.processed = [False] * self.N
    self.dist_mat = get_dist_mat(self.data)

  def get_core_dist(self, p):
    return sorted(self.dist_mat[p])[self.min_pts]

  def get_reachability_dist(self, o, p):
    return max(self.core_dist[p], euclidean_distance(self.data[p], self.data[o]))

  def update(self, p, seeds):
    self.core_dist[p] = self.get_core_dist(p)
    for o in range(self.N):
      if self.processed[o]:
        continue
      new_reach_dist = self.get_reachability_dist(o, p)
      if self.reachability_dist[o] == float("inf"):
        self.reachability_dist[o] = new_reach_dist
        seeds.append((o, new_reach_dist))
        seeds.sort(key=lambda x: x[1], reverse=True)
      elif new_reach_dist < self.reachability_dist[o]:
        seeds.remove((o, self.reachability_dist[o]))
        self.reachability_dist[o] = new_reach_dist
        seeds.append((o, new_reach_dist))
        seeds.sort(key=lambda x: x[1], reverse=True)

  def run(self):
    for p in range(self.N):
      if self.processed[p]:
        continue
      self.processed[p] = True
      self.order.append(p)
      if self.core_dist[p] == float("inf"):
        self.core_dist[p] = self.get_core_dist(p)
      seeds = list()
      self.update(p, seeds)
      while seeds:
        q, _ = seeds[-1]
        seeds.pop()
        self.processed[q] = True
        self.order.append(q)
        if self.core_dist[q] == float("inf"):
          self.core_dist[q] = self.get_core_dist(q)
        self.update(q, seeds)

    # TODO
    def extract_cluster_xi(xi):
      pass


if __name__ == "__main__":
  if len(sys.argv) == 2:
    n_points_per_cluster = 150

    C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
    C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    data = np.vstack((C1, C2, C3, C4, C5, C6))
    min_samples = int(sys.argv[1])
  else:
    raise Exception("'python3 optics k' or 'python3 optics.py <input> k'")

  algo = OPTICS(data, min_samples)
  algo.run()
  # print(algo.order)

  algo2 = OPTICS_SKL(min_samples=min_samples)
  algo2.fit(np.array(data))
  # print(list(algo2.ordering_))

  plt.figure(figsize=(10, 7))
  ax1 = plt.subplot()
  xx = np.arange(len(data))
  reachability = np.array(algo.reachability_dist)[np.array(algo.order)]
  ax1.plot(xx, reachability, '-')

  ax2 = plt.subplot()
  reachability = algo2.reachability_[algo2.ordering_]
  ax2.plot(xx, reachability, '-.')

  plt.show()
