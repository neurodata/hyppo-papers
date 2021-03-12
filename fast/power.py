import numpy as np
from math import ceil

from hyppo.tools import linear


def _sim_gen(sim, n, p, noise=True):
    """
    Generate x, y from each sim
    """
    if sim in ["multiplicative_noise", "multimodal_independence"]:
#         x, y = indep_sim(sim, n, p)
        x, y = multimodal_independence(n, p)
    else:
#         x, y = indep_sim(sim, n, p, noise=noise)
        x, y = linear(n, p, noise=noise)


    return x, y


def _perm_stat(test, sim, n=100, p=1, noise=True):
    """
    Calculated permuted and observed test statistics
    """
    x, y = _sim_gen(sim, n, p, noise=noise)
    obs_stat = test()._statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test()._statistic(x, permy)

    return obs_stat, perm_stat


def _fast_perm_stat(test, sim, n=100, p=1, noise=True):
    """
    Generates fast  permutation pvalues
    """
    x, y = _sim_gen(sim, n, p, noise=noise)
    pvalue = test().test(x, y, auto=True)[1]

    return pvalue


def power(test, sim, n=100, p=1, noise=True, alpha=0.05, reps=1000, auto=False):
    """
    Calculates empirical power
    """
    if test.__name__ in ["Dcorr", "Hsic"] and auto:
        if n < 20:
            empirical_power = np.nan
        else:
            pvals = np.array(
                [_fast_perm_stat(test, sim, n, p, noise=noise) for _ in range(reps)]
            )
            empirical_power = (pvals <= alpha).sum() / reps
    else:
        alt_dist, null_dist = map(
            np.float64,
            zip(*[_perm_stat(test, sim, n, p, noise=noise) for _ in range(reps)]),
        )
        cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
        empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
