# Kurkova-Kolmogorov-Arnold-Newtorks-KKANs

Please check our official implementation with more examples here: https://github.com/jdtoscano94/Kurkova_Kolmogorov_Arnold_Networks_KKANs.git


Inspired by the Kolmogorov-Arnold representation theorem and Kurkova's principle of using approximate representations, we propose the Kurkova-Kolmogorov-Arnold Network (KKAN), a new two-block architecture that combines robust multi-layer perceptron (MLP) based inner functions with flexible linear combinations of basis functions as outer functions. We first prove that  KKAN is a universal approximator, and then we demonstrate its versatility across scientific machine-learning applications, including function regression, physics-informed machine learning (PIML), and operator-learning frameworks. The benchmark results show that KKANs outperform MLPs and the original Kolmogorov-Arnold Networks (KANs) in function approximation and operator learning tasks and achieve performance comparable to fully optimized MLPs for PIML. To better understand the behavior of the new representation models, we analyze their geometric complexity and learning dynamics using information bottleneck theory, identifying three universal learning stages, fitting, transition, and diffusion, across all types of architectures. We find a strong correlation between geometric complexity and signal-to-noise ratio (SNR), with optimal generalization achieved during the diffusion stage. Additionally, we propose self-scaled residual-based attention weights to maintain high SNR dynamically, ensuring uniform convergence and prolonged learning. 


## Papers

Currently, this repository contains the code for the following paper:

1. **Juan Diego Toscano, Li-Lian Wang, George Em Karniadakis** , KKANs:
Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics, Neural Networks (2025), doi:
https://doi.org/10.1016/j.neunet.2025.107831.

If you find this content useful please consider citing our work as follows:

```bibtex
@article{Toscano_kkans_2025,
title = {KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics},
journal = {Neural Networks},
pages = {107831},
year = {2025},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2025.107831},
url = {https://www.sciencedirect.com/science/article/pii/S0893608025007117},
author = {Juan Diego Toscano and Li-Lian Wang and George Em Karniadakis},
keywords = {Kolmogorov-Arnold representation theorem, physics-informed neural networks, Kolmogorov-Arnold networks, optimization algorithms, self-adaptive weights, information bottleneck theory}}
```
