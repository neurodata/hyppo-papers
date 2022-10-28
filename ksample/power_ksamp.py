from math import ceil

import numpy as np
from hyppo.ksample._utils import k_sample_transform
from scipy._lib._util import MapWrapper, check_random_state


class _ParallelP(object):
    """
    Helper function to calculate parallel power.
    """

    def __init__(self, test, ksim, sim, n, p, noise, rngs, angle, pow_type):
        self.test = test()
        self.ksim = ksim
        self.sim = sim
        self.angle = angle

        self.n = n
        self.p = p
        self.noise = noise
        self.rngs = rngs
        self.pow_type = pow_type

    def __call__(self, index):
        if self.sim == "multimodal_independence":
            x, y, z = self.ksim(
                self.sim, self.n, self.p, k=3, degree=[0, 0], pow_type=self.pow_type
            )
        elif self.sim == "multiplicative_noise":
            x, y, z = self.ksim(
                self.sim, self.n, self.p, k=3, degree=[self.angle, -self.angle], pow_type=self.pow_type
            )
        else:
            x, y, z = self.ksim(
                self.sim, self.n, self.p, k=3, degree=[self.angle, -self.angle], noise=self.noise, pow_type=self.pow_type
            )

        u, v = k_sample_transform([x, y, z])
        obs_stat = self.test.statistic(u, v)

        permv = self.rngs[index].permutation(v)

        # calculate permuted stats, store in null distribution
        perm_stat = self.test.statistic(u, permv)

        return obs_stat, perm_stat


def _perm_test(
    test,
    ksim,
    sim,
    n=100,
    p=1,
    noise=False,
    reps=1000,
    workers=1,
    random_state=None,
    angle=90,
    pow_type="samp"
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
    parallelp = _ParallelP(
        test=test,
        ksim=ksim,
        sim=sim,
        n=n,
        p=p,
        noise=noise,
        rngs=rngs,
        angle=angle,
        pow_type=pow_type
    )
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power(
    test,
    ksim,
    sim,
    n=100,
    p=1,
    angle=90,
    noise=True,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
    pow_type="samp"
):
    """
    [summary]

    Parameters
    ----------
    test : [type]
        [description]
    sim : [type]
        [description]
    n : int, optional
        [description], by default 100
    p : int, optional
        [description], by default 1
    noise : int, optional
        [description], by default 0
    reps : int, optional
        [description], by default 1000
    alpha : float, optional
        [description], by default 0.05
    """

    alt_dist, null_dist = _perm_test(
        test,
        ksim,
        sim,
        n=n,
        p=p,
        angle=angle,
        noise=noise,
        reps=reps,
        workers=workers,
        random_state=random_state,
        pow_type=pow_type
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power


def power_ksamp_sample(
    test,
    ksim,
    sim,
    n=100,
    p=1,
    angle=90,
    noise=True,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
):
    """
    [summary]

    Parameters
    ----------
    test : [type]
        [description]
    sim : [type]
        [description]
    n : int, optional
        [description], by default 100
    p : int, optional
        [description], by default 1
    noise : int, optional
        [description], by default 0
    reps : int, optional
        [description], by default 1000
    alpha : float, optional
        [description], by default 0.05
    """

    return power(
        test,
        ksim,
        sim,
        n=n,
        p=p,
        angle=angle,
        noise=noise,
        alpha=alpha,
        reps=reps,
        workers=workers,
        random_state=random_state,
        pow_type="samp"
    )


def power_ksamp_dimension(
    test,
    ksim,
    sim,
    n=100,
    p=1,
    angle=90,
    noise=False,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
):
    """
    hello

    Parameters
    ----------
    test : [type]
        [description]
    sim : [type]
        [description]
    n : int, optional
        [description], by default 100
    p : int, optional
        [description], by default 1
    noise : int, optional
        [description], by default 0
    reps : int, optional
        [description], by default 1000
    alpha : float, optional
        [description], by default 0.05
    """
    return power(
        test,
        ksim,
        sim,
        n=n,
        p=p,
        angle=angle,
        noise=noise,
        alpha=alpha,
        reps=reps,
        workers=workers,
        random_state=random_state,
        pow_type="dim"
    )


def power_ksamp_angle(
    test,
    ksim,
    sim,
    n=100,
    p=1,
    angle=90,
    noise=True,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
):
    """
    hello

    Parameters
    ----------
    test : [type]
        [description]
    sim : [type]
        [description]
    n : int, optional
        [description], by default 100
    p : int, optional
        [description], by default 1
    noise : int, optional
        [description], by default 0
    reps : int, optional
        [description], by default 1000
    alpha : float, optional
        [description], by default 0.05
    """

    return power(
        test,
        ksim,
        sim,
        n=n,
        p=p,
        angle=angle,
        noise=noise,
        alpha=alpha,
        reps=reps,
        workers=workers,
        random_state=random_state,
        pow_type="samp"
    )
