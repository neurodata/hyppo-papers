# Learning Interpretable Characteristic Kernels via Decision Forests

Cencheng Shen, Sambit Panda, Joshua T. Vogelstein

**Abstract**: Decision forests are popular tools for classification and regression. These forests naturally produce proximity matrices  measuring how often each pair of observations lies in the same leaf node. It has been demonstrated that these proximity matrices can be thought of as kernels, connecting the decision forest literature to the extensive kernel machine literature. While other kernels are known to have strong theoretical properties such as being characteristic, no similar result is available for any decision forest based kernel. In this manuscript, we prove that the decision forest induced proximity can be made characteristic. Therefore any theoretical results that apply to characteristic kernel machines immediately also apply to kernel machines using these characteristic decision tree-based kernels, such as universal consistency for hypothesis testing. We demonstrate the performance of the induced kernel on a suite of 20 high-dimensional independence and two-sample test settings. The decision forest induced kernel typically achieves substantially higher testing power than existing methods. Finally, we demonstrate that the learned kernel is interpretable, in that the most important features are readily apparent.  This work therefore demonstrate the existence of a test that is both more powerful and more interpretable than existing methods, flying in the face of conventional wisdom of the trade-off between the two.

Note, the real data figure code was created by modifying this MATLAB script and running our test: https://github.com/neurodata/MGC-paper/blob/master/Code/Experiments/run_realData.m

arXiv: https://arxiv.org/abs/1812.00029
