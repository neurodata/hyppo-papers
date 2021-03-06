from math import ceil

import numpy as np
from hyppo.ksample._utils import k_sample_transform
from hyppo.sims import gaussian_3samp
from scipy._lib._util import MapWrapper, check_random_state
from sklearn.metrics import pairwise_distances


class _ParallelP3Samp(object):
    """
    Helper function to calculate parallel power.
    """

    def __init__(
        self, test, n, epsilon=1, weight=0, case=1, rngs=[], d=2, c=0.3, multiway=False
    ):
        if multiway:
            self.test = test(compute_distance=False)
        else:
            self.test = test()

        self.n = n
        self.epsilon = epsilon
        self.weight = weight
        self.case = case
        self.rngs = rngs
        self.d = d
        self.c = c
        self.multiway = multiway

    def __call__(self, index):
        if self.case not in [4, 5]:
            x, y, z = gaussian_3samp(
                self.n, epsilon=self.epsilon, case=self.case, d=self.d, c=self.c
            )
        else:
            x, y, z = gaussian_3samp(
                self.n, weight=self.weight, case=self.case, d=self.d, c=self.c
            )

        if self.multiway:
            ways = [[0, 0], [0, 1], [1, 0]]
            u, v = k_sample_transform([x, y, z], ways=ways)
            u = pairwise_distances(u, metric="euclidean")
            v = pairwise_distances(v, metric="sqeuclidean")
        else:
            u, v = k_sample_transform([x, y, z])

        obs_stat = self.test.statistic(u, v)

        if self.multiway:
            idx = self.rngs[index].permutation(np.arange(len(v)))
            permv = v[idx][:, idx]
        else:
            idx = self.rngs[index].permutation(np.arange(len(v)))
            permv = v[idx]

        # calculate permuted stats, store in null distribution
        perm_stat = self.test.statistic(u, permv)

        return obs_stat, perm_stat


def _perm_test_3samp(
    test,
    n=100,
    epsilon=1,
    weight=0,
    case=1,
    reps=1000,
    workers=1,
    random_state=None,
    d=2,
    c=0.3,
    multiway=False,
):
    r"""
    Helper function that calculates the statistical.

    Parameters
    ----------
    test : callable()
        The independence test class requested.
    sim : callable()
        The simulation used to generate the input data.
    reps : int, optional (default: 1000)
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    workers : int, optional (default: -1)
        The number of cores to parallelize the p-value computation over.
        Supply -1 to use all cores available to the Process.
    d : int, optional (default 2)
        The number of ds in the simulation. The first two are signal,
        the rest are noise.
    c : int, optional (default 0.2)
        The one-way epsilon in case 6.
    multiway : boolean, optional (default False)
        If True, label distance matrix is computed in a multiway-aware fashion

    Returns
    -------
    null_dist : list
        The approximated null distribution.
    """
    # set seeds
    random_state = check_random_state(random_state)
    rngs = [
        np.random.RandomState(random_state.randint(1 << 32, size=4, dtype=np.uint32))
        for _ in range(reps)
    ]

    # use all cores to create function that parallelizes over number of reps
    mapwrapper = MapWrapper(workers)
    parallelp = _ParallelP3Samp(test, n, epsilon, weight, case, rngs, d, c, multiway)
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power_3samp_epsweight(
    test,
    n=100,
    epsilon=0.5,
    weight=0,
    case=1,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
    d=2,
    c=0.3,
    multiway=False,
):
    alt_dist, null_dist = _perm_test_3samp(
        test,
        n=n,
        epsilon=epsilon,
        weight=weight,
        case=case,
        reps=reps,
        workers=workers,
        random_state=random_state,
        d=d,
        c=c,
        multiway=multiway,
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
