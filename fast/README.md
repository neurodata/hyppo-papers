# The Chi-Square Test of Distance Correlation

Cencheng Shen, Joshua T. Vogelstein

**Abstract**: Distance correlation has gained much recent attention in the data science community: the sample statistic is straightforward to compute and asymptotically equals zero if and only if independence, making it an ideal choice to test any type of dependency structure given sufficient sample size. One major bottleneck is the testing process: because the null distribution of distance correlation depends on the underlying random variables and metric choice, it typically requires a permutation test to estimate the null and compute the p-value, which is very costly for large amount of data. To overcome the difficulty, we propose a centered chi-square distribution, demonstrate it well-approximates the limiting null distribution of unbiased distance correlation, and prove upper tail dominance and distribution bound. The resulting distance correlation chi-square test is a nonparametric test for independence, is valid and universally consistent using any strong negative type metric or characteristic kernel, enjoys a similar finite-sample testing power as the standard permutation test, is provably most powerful among all valid tests of distance correlation using known distributions, and is also applicable to K-sample and partial testing.

arXiv: https://arxiv.org/abs/1912.12150
