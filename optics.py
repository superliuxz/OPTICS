import math
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from bintrees import FastRBTree
from scipy.spatial import distance_matrix
from sklearn.cluster import OPTICS as OPTICS_SKL

np.random.seed(0)


def _euclidean_distance(p1, p2):
  return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


class OPTICS:
  def __init__(self, points, min_pts):
    self.min_pts = min_pts
    self.points = points
    self.N = len(points)
    self.order = list()

    self.reachability_dist = np.empty(self.N)
    self.reachability_dist.fill(np.inf)

    self.core_dist = np.empty(self.N)
    self.core_dist.fill(np.inf)
    dist_mat = distance_matrix(points, points)
    for p in range(self.N):
      self.core_dist[p] = np.partition(dist_mat[p], self.min_pts)[self.min_pts]

    self.processed = np.zeros(self.N, dtype=bool)

  def _get_reachability_dist(self, o, p):
    return max(self.core_dist[p], _euclidean_distance(self.points[p], self.points[o]))

  def _update(self, p, seeds: FastRBTree):
    for o in range(self.N):  # since eps is INF, the neighbours are all the points in the dataset.
      if self.processed[o]:
        continue
      new_reach_dist = self._get_reachability_dist(o, p)
      if self.reachability_dist[o] == np.inf:
        self.reachability_dist[o] = new_reach_dist
        seeds.insert((new_reach_dist, o), o)
      elif new_reach_dist < self.reachability_dist[o]:
        seeds.remove((self.reachability_dist[o], o))
        self.reachability_dist[o] = new_reach_dist
        seeds.insert((new_reach_dist, o), o)

  def run(self):
    for p in range(self.N):
      if self.processed[p]:
        continue
      self.processed[p] = True
      self.order.append(p)
      seeds = FastRBTree()
      self._update(p, seeds)
      while seeds:
        # pop_min returns ( (reachability, q), q ).
        _, q = seeds.pop_min()
        self.processed[q] = True
        self.order.append(q)
        self._update(q, seeds)

  # TODO
  def extract_cluster_xi(self, xi):
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

  start = timer()
  algo = OPTICS(data, min_samples)
  algo.run()
  print("my impl:", timer() - start)
  # print(algo.order)

  start = timer()
  algo2 = OPTICS_SKL(min_samples=min_samples)
  algo2.fit(np.array(data))
  print("sklearn impl:", timer() - start)
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
