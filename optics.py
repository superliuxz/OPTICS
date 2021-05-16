import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from bintrees import FastRBTree
from scipy.spatial import distance_matrix
from sklearn.cluster import OPTICS as OPTICS_SKL

np.random.seed(0)


class OPTICS:
  def __init__(self, points, min_pts, xi=0.05):
    self.xi = xi
    self.min_pts = min_pts
    self.points = points
    self.dist_mat = distance_matrix(self.points, self.points)
    self.N = len(points)
    self.order = list()

    self.reachability_dist = np.full(self.N, np.inf, dtype=float)

    self.core_dist = np.full(self.N, np.inf, dtype=float)
    # heapq yields different reachability, order and predecessor, but does not
    # alter the quality of cluster.
    for p in range(self.N):
      self.core_dist[p] = \
        np.partition(self.dist_mat[p], self.min_pts)[self.min_pts]

    self.processed = np.zeros(self.N, dtype=bool)
    self.predecessor = np.full(self.N, -1, dtype=int)
    self.clusters = list()
    self.labels = np.full(self.N, -1, dtype=int)

  def _update(self, p, seeds: FastRBTree):
    # since eps is INF, the neighbours are all the points in the dataset.
    for o in range(self.N):
      if self.processed[o]:
        continue
      new_reach_dist = max(self.core_dist[p], self.dist_mat[o][p])
      if np.isinf(self.reachability_dist[o]):
        self.reachability_dist[o] = new_reach_dist
        seeds.insert((new_reach_dist, o), o)
        self.predecessor[o] = p
      elif new_reach_dist < self.reachability_dist[o]:
        seeds.remove((self.reachability_dist[o], o))
        self.reachability_dist[o] = new_reach_dist
        seeds.insert((new_reach_dist, o), o)
        self.predecessor[o] = p

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

    self._extract_cluster_xi()

  def _extract_cluster_xi(self):
    reachability_plot = self.reachability_dist[self.order]
    predecessor_plot = self.predecessor[self.order]
    reachability_plot = np.hstack((reachability_plot, np.inf))

    xi_complement = 1 - self.xi
    # Steep Down Areas
    sdas = list()
    index = 0
    # max in-between
    global_mib = 0.0

    def _update_sdas():
      if np.isinf(global_mib):
        return []
      res = [sda for sda in sdas
             if
             global_mib <= reachability_plot[sda['start']] * xi_complement]
      for sda in res:
        sda['mib'] = max(sda['mib'], global_mib)
      return res

    def _move_end_pointer(is_steep_xward, is_xward, start):
      """
      To extend end pointer for steep upward region, pass |steep_upward| and
      |downward|;
      To extend end pointer for steep downward region, pass |steep_downward| and
      |upwnward|;
      """
      N = len(is_steep_xward)
      consecutive_non_xward_pts = 0
      i = start
      end = start
      while i < N:
        if is_steep_xward[i]:
          # if current point is steep up/down-ward, reset the number of
          # consecutive non up/down-ward points, and move end pointer.
          consecutive_non_xward_pts = 0
          end = i
        # For example, with |steep_upward| and |downward|, if a point is not
        # steep upward, then it can be: upward, downward and steep downward.
        # By checking if it's not downward, then it must be an upward point.
        # Upward is not steep upward, so increment the consecutive counter.
        elif not is_xward[i]:
          consecutive_non_xward_pts += 1
          if consecutive_non_xward_pts > self.min_pts:
            break
        else:
          return end
        i += 1
      return end

    def _correct_predecessor(s, e):
      while s < e:
        if reachability_plot[s] > reachability_plot[e]:
          return s, e
        p_e = self.order[predecessor_plot[e]]
        for i in range(s, e):
          if p_e == self.order[i]:
            return s, e
        e -= 1
      return None, None

    with np.errstate(invalid='ignore'):
      ratio = reachability_plot[:-1] / reachability_plot[1:]
      steep_upward = ratio <= xi_complement
      steep_downward = ratio >= 1 / xi_complement
      downward = ratio > 1
      upward = ratio < 1

    for steep_index in range(self.N):
      # only care about the area that is either steep up or steep down
      if not steep_upward[steep_index] and not steep_downward[steep_index]:
        continue
      # skip if this index has been visited.
      if steep_index < index:
        continue

      global_mib = max(global_mib,
                       np.max(reachability_plot[index:steep_index + 1]))

      if steep_downward[steep_index]:
        sdas = _update_sdas()
        D_start = steep_index
        D_end = _move_end_pointer(steep_downward, upward, D_start)
        sdas.append({"start": D_start, "end": D_end, "mib": 0.0})
        index = D_end + 1
        global_mib = reachability_plot[index]
      else:
        sdas = _update_sdas()
        U_start = steep_index
        U_end = _move_end_pointer(steep_upward, downward, U_start)
        index = U_end + 1
        global_mib = reachability_plot[index]

        U_clusters = list()
        for D in sdas:
          c_start = D['start']
          c_end = U_end

          # line (**), sc2*
          if reachability_plot[c_end + 1] * xi_complement < D['mib']:
            continue

          # Definition 11: criterion 4

          # 4b, if ReachStart is more than Xi% higher than ReachEnd
          if (reachability_plot[c_start] * xi_complement >=
              reachability_plot[c_end + 1]):
            # Find the first index from the left side which is almost
            # at the same level as the end of the detected cluster.
            while (reachability_plot[c_start + 1] >
                   reachability_plot[c_end + 1] and c_start < D['end']):
              c_start += 1
          # 4c, if ReachEnd is more than Xi% higher than ReachStart
          elif (reachability_plot[c_end + 1] * xi_complement >=
                reachability_plot[c_start]):
            # Find the first index from the right side which is almost
            # at the same level as the beginning of the detected cluster.
            while (reachability_plot[c_end - 1] > reachability_plot[c_start] and
                   c_end > U_start):
              c_end -= 1

          # Applies Algorithm 2 of Schubert, Erich, Michael Gertz.
          # "Improving the Cluster Structure Extracted from OPTICS Plots."
          # Proc. of the Conference "Lernen, Wissen, Daten, Analysen" (LWDA)
          # (2018): 318-329.
          c_start, c_end = _correct_predecessor(c_start, c_end)
          if c_start is None:
            continue
          # Definition 11: criterion 3.a
          if c_end - c_start + 1 < self.min_pts:
            continue
          # Definition 11: criterion 1
          if c_start > D['end']:
            continue
          # Definition 11: criterion 2
          if c_end < U_start:
            continue

          U_clusters.append((c_start, c_end))

        # add smaller clusters first.
        U_clusters.reverse()
        self.clusters.extend(U_clusters)

    label = 0
    for c in self.clusters:
      if not np.any(self.labels[c[0]:(c[1] + 1)] != -1):
        self.labels[c[0]:(c[1] + 1)] = label
        label += 1
    self.labels[self.order] = self.labels.copy()


if __name__ == "__main__":
  if len(sys.argv) == 2:
    n_points_per_cluster = 500

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
  optics = OPTICS(data, min_samples)
  optics.run()
  print("my impl:", timer() - start)

  start = timer()
  optics_skl = OPTICS_SKL(min_samples=min_samples)
  optics_skl.fit(np.array(data))
  print("sklearn impl:", timer() - start)

  try:
    # artifacts from reachability plot
    np.testing.assert_array_almost_equal(optics.core_dist,
                                         optics_skl.core_distances_)
    np.testing.assert_array_almost_equal(optics.reachability_dist,
                                         optics_skl.reachability_)
    np.testing.assert_array_equal(optics.order, optics_skl.ordering_)
    np.testing.assert_array_equal(optics.predecessor, optics_skl.predecessor_)
    # artifacts from extract xi method
    np.testing.assert_array_equal(optics.clusters,
                                  optics_skl.cluster_hierarchy_)
    np.testing.assert_array_equal(optics.labels,
                                  optics_skl.labels_)
  except AssertionError as e:
    print(e)

  fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 7))

  xx = np.arange(len(data))
  reachability = optics.reachability_dist[optics.order]
  ax1.plot(xx, reachability, '-')

  reachability = optics_skl.reachability_[optics_skl.ordering_]
  ax1.plot(xx, reachability, '-.')

  colors = ['g.', 'r.', 'b.', 'y.', 'c.']
  for klass, color in zip(range(0, 5), colors):
    X1 = data[optics.labels == klass]
    X2 = data[optics_skl.labels_ == klass]
    ax2.plot(X1[:, 0], X1[:, 1], color, alpha=0.3)
    ax3.plot(X2[:, 0], X2[:, 1], color, alpha=0.3)
  ax2.plot(data[optics.labels == -1, 0], data[optics.labels == -1, 1], 'k+',
           alpha=0.1)
  ax3.plot(data[optics_skl.labels_ == -1, 0], data[optics_skl.labels_ == -1, 1],
           'k+', alpha=0.1)

  plt.show()
