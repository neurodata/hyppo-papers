import numpy as np
from math import ceil

from hyppo.tools import indep_sim


def _indep_sim_gen(sim, n, p, noise=True):
    """
    Generate x, y from each sim
    """
    if sim in ["multiplicative_noise", "multimodal_independence"]:
        x, y = indep_sim(sim, n, p)
    else:
        x, y = indep_sim(sim, n, p, noise=noise)

    return x, y


def _perm_stat(test, sim, n=100, p=1, noise=True, **test_kwargs):
    """
    Generates null and alternate distributions
    """
    x, y = _indep_sim_gen(sim, n, p, noise=noise)
    obs_stat = test(**test_kwargs).statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test(**test_kwargs).statistic(x, permy)

    return obs_stat, perm_stat


def _nonperm_pval(test, sim, n=100, p=1, noise=True, **test_kwargs):
    """
    Generates fast  permutation pvalues
    """
    x, y = _indep_sim_gen(sim, n, p, noise=noise)
    pvalue = test(**test_kwargs).test(x, y)[1]

    return pvalue

def _casual_pval(test, n=100, p=1, noise=True, **test_kwargs):
    """
    Generates 6 uniform samples, 3 independent, 3 dependent
    """
    noise = np.random.normal(0, 1, size=(n, 1))
    x = [np.array(np.random.uniform(-1, 1, size=(n, p))) for _ in range(6)]
    y = x[0] + x[1] ** 2 + x[2] + noise * 0.5 * np.random.normal(0, 1, size=(n, 1))
    pvalues = [test(**test_kwargs).test(x[i], y)[1] for i in range(6)]
    
    return pvalues


def power_indep(test, sim, n=100, p=1, noise=True, alpha=0.05, reps=1000, **test_kwargs):
    """
    Calculates empirical power
    """
    test_name = test.__name__
    if test_name == "KMERF":
        if sim == "causal":
            pvals = list(map(
                np.float64,
                zip(*[_casual_pval(test, n, p, noise=noise, **test_kwargs) for _ in range(reps)]),
            ))
            empirical_tpr = np.mean([(1 + (pvals[i] <= (alpha/3)).sum()) / (1 + reps) for i in [0, 1, 2]])
            empirical_fpr = np.mean([(1 + (pvals[i] <= (alpha/3)).sum()) / (1 + reps) for i in [3, 4, 5]])
            return empirical_tpr, empirical_fpr
        else:
            pvals = np.array([
                _nonperm_pval(test, sim, n, p, noise=noise, **test_kwargs) for _ in range(reps)
            ])
            empirical_power = (1 + (pvals <= alpha).sum()) / (1 + reps)
    else:
        alt_dist, null_dist = map(
            np.float64,
            zip(*[_perm_stat(test, sim, n, p, noise=noise, **test_kwargs) for _ in range(reps)]),
        )
        cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
        empirical_power = (1 + (alt_dist >= cutoff).sum()) / (1 + reps)

    return empirical_power
