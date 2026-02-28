# A Variational Framework for Residual-Based Adaptivity (vRBA)

A principled variational framework for adaptive sampling and weighting in Physics-Informed Neural Networks (PINNs) and Neural Operators, published at **npj Artificial Intelligence**.

> Residual-based adaptive strategies are widely used in scientific machine learning yet remain largely heuristic. We introduce a variational framework that formalizes these methods through convex transformations of the residual, where different transformations correspond to distinct objective functionals. For instance, exponential weights target uniform error minimization, while linear weights recover quadratic error minimization. This perspective reveals adaptive weighting as a means of selecting sampling distributions that optimize a primal objective, directly linking discretization choices to error metrics. This principled approach yields three key benefits: it enables systematic design of adaptive schemes, reduces discretization error by lowering estimator variance, and enhances learning dynamics by improving gradient signal-to-noise ratio. Extending the framework to operator learning, we demonstrate substantial performance gains across diverse optimizers and architectures.



(DOI coming soon) Our paper has beed accepted at npj Artificial Intelligence. In the meantime you can read our preprint at arXiv: [arXiv:2509.14198](https://arxiv.org/abs/2509.14198).


## About the Code
This repository contains the official JAX implementation for the paper: **A Variational Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning**.

This code provides a JAX-native implementation of **vRBA**, a principled adaptive sampling and weighting method for PINNs and Neural Operators. It also includes a custom, high-performance **SSBroyden optimizer** for second-order training to accelerate convergence and achieve state-of-the-art accuracy.

---

## About the Framework

Residual-based adaptive methods are powerful but often heuristic. Our work introduces **vRBA**, a unifying variational framework that provides a formal justification and a principled design strategy for these techniques.

The core idea is to connect adaptive schemes to a primal optimization objective using convex transformations of the PDE residual. This establishes a direct link between the choice of sampling distribution and the error metric being minimized. For example:
* **Exponential weights** correspond to minimizing the **$L^\infty$ (uniform) error**.
* **Linear/quadratic weights** correspond to Variance Minimization.
* **Other potentials** Our framework allows for a wide range of potentials (e.g., $(r+1)log(r+1)$, $\cosh(r)$, etc), enabling the design of adaptive schemes tailored to specific error metrics or optimization objectives.


**Table: Summary of the seven generated adaptive schemes.** The choice of potential $\Phi(r)$ determines the update rule for the sampling distribution $q$ and the strategy for the regularizer $\epsilon$. Note that the Baseline is simply the linear potential case, where the induced distribution is uniform.

| **Name** | **Potential** $\Phi(r)$ | **Optimal Dist.** $q(x) \propto$ | **Regularizer** $\epsilon$ |
|---|---|---|---|
| Baseline (Uniform) | $r$ | $1$ | N/A |
| Quadratic (RAD/RBA) | $r^2 + 1$ | $r$ | Analytic |
| Exponential (Attention) | $e^r$ | $e^r$ | Any |
| Polynomial ($L^p$) | $r^p$ | $r^{p-1}$ | Analytic |
| Dual-KL | $(1+r)\log(1+r) - r$ | $\log(1+r)$ | Newton |
| Hybrid Robust | $\cosh(r)$ | $\sinh(r)$ | Newton |
| Super-Exponential | $e^{r^2}$ | $r e^{r^2}$ | Newton |




This approach transforms adaptive sampling from a heuristic into a principled optimization strategy.

---

## Key Contributions & Features

### 1. The vRBA Framework
* **Principled Design:** A unified method for creating adaptive sampling and weighting schemes based on formal optimization theory.
* **Reduced Discretization Error:** vRBA provably reduces the variance of the loss estimator, leading to more stable training and a smaller discretization error.
* **Improved Training Dynamics:** The framework enhances the gradient's signal-to-noise ratio (SNR), allowing models to enter the productive "diffusion" phase of training more rapidly and accelerating convergence.
* **Extension to Operator Learning:** We introduce a novel **hybrid strategy** that combines importance weighting for spatial/temporal domains with importance sampling over function instances. This makes vRBA highly effective for architectures like DeepONets, FNOs, and TC-UNets.

### 2. High-Performance Second-Order Optimizer
* This repository includes a custom JAX implementation of the **Self-Scaled Broyden (SSBroyden)** optimizer from Urbán et al..
* Our implementation is designed for **GPU acceleration**, overcoming the performance bottlenecks of the original CPU-bound SciPy version.
* It features a robust **three-stage fallback line search**, making it stable and effective for the challenging, ill-conditioned loss landscapes found in PINNs.
---

## Getting Started

### Data Availability

All datasets for the Operator Learning examplesare available on Google Drive:

[**Download Data Here (Google Drive)**](https://drive.google.com/drive/folders/1IiSt-g_Vs0wKcSbvzgnwCFTBP_v75Tc-?usp=drive_link)

#### Instructions

1.  Download the data from the link above.
2.  The downloaded folder contains several subfolders 
3.  Unzip the files.
4.  Move the contents of *each* of these subfolders into the corresponding, identically-named folder within this project repository.

For example, the 'data' folder from the downloaded `wave_tcunet` folder should be placed into the `wave_tcunet` folder in this repository.

### Prerequisites

This project has two sets of requirements. The core vRBA framework and PINN experiments are built in JAX, while the operator learning benchmarks (FNO, TC-UNet, etc.) are built in PyTorch.


### 1. JAX Setup Instructions (Required for vRBA)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jdtoscano94/NABLA-SciML.git](https://github.com/jdtoscano94/NABLA-SciML.git)
    cd NABLA-SciML
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    conda create -n nabla_env python=3.10
    conda activate nabla_env
    ```

3.  **Install JAX with GPU support:**
    *(This is required for all modules. We recommend installing it explicitly first to ensure CUDA support)*
    ```bash
    pip install -U "jax[cuda12]"
    ```

4.  **Install the NABLA package:**
    This installs the library in editable mode, which is necessary to import the vRBA models and the SSBroyden optimizer.
    ```bash
    pip install -e .
    ```

### 2. PyTorch Environment (Operator Learning)

For running the operator learning tasks, you will need a separate environment with PyTorch and related libraries.

Key requirements include:

```
torch==2.7.0
tensorly-torch==0.5.0
torch_harmonics==0.7.3
torchaudio==2.7.0
torchinfo==1.8.0
torchvision==0.22.0
```

### 3. General Dependencies

Both environments will also require:
* NumPy
* Matplotlib (for visualizations)

## 📚 References

### [4] Variational Residual-Based Adaptivity (vRBA)
```bibtex
@article{toscano2025variational,
  title={A Variational Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning},
  author={Toscano, Juan Diego and Chen, Daniel T and Oommen, Vivek and Darbon, J{'e}r{\^o}me and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2509.14198},
  year={2025}
}
```