import numpy as np
from math import ceil

from hyppo.tools import rot_ksamp
from hyppo.ksample import k_sample_transform


def _indep_sim_gen(sim, n, p, noise=True):
    """
    Generate x, y from each sim
    """
    if sim in ["multiplicative_noise", "multimodal_independence"]:
        x, y = indep_sim(sim, n, p)
    else:
        x, y = indep_sim(sim, n, p, noise=noise)

    return x, y


def _perm_stat(test, test_name, sim, n=100, p=1, noise=True):
    """
    Generates null and alternate distributions
    """
    u, v = rot_ksamp(sim, n, p, noise=noise, pow_type="dim")
    if test_name == "KMERF":
        x, y = k_sample_transform([u, v], test_type="rf")
    else:
        x, y = k_sample_transform([u, v])
    obs_stat = test().statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test().statistic(x, permy)

    return obs_stat, perm_stat


def _nonperm_pval(test, test_name, sim, n=100, p=1, noise=True):
    """
    Generates fast  permutation pvalues
    """
    u, v = rot_ksamp(sim, n, p, noise=noise, pow_type="dim")
    if test_name == "KMERF":
        x, y = k_sample_transform([u, v], test_type="rf")
    else:
        x, y = k_sample_transform([u, v])
    pvalue = test().test(x, y)[1]

    return pvalue


def power_ksamp(test, sim, n=100, p=1, noise=True, alpha=0.05, reps=1000):
    """
    Calculates empirical power
    """
    test_name = test.__name__
    if test_name == "KMERF":
        pvals = np.array([
            _nonperm_pval(test, test_name, sim, n, p, noise=noise) for _ in range(reps)
        ])
        empirical_power = (1 + (pvals <= alpha).sum()) / (1 + reps)
    else:
        alt_dist, null_dist = map(
            np.float64,
            zip(*[_perm_stat(test, test_name, sim, n, p, noise=noise) for _ in range(reps)]),
        )
        cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
        empirical_power = (1 + (alt_dist >= cutoff).sum()) / (1 + reps)

    return empirical_power
