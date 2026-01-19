# $\nabla$ NABLA-SciML
**N**eural **A**lgorithms & **B**asis **L**earning **A**pproximations for **S**cientific **M**achine **L**earning

Welcome to **$\nabla$ NABLA-SciML**!

I am **Juan Diego Toscano**, a PhD Candidate working under the mentorship of **Prof. George Karniadakis** since 2022. My research delves into the realm of Scientific Machine Learning (SciML), with a specific focus on developing reliable and stable machine learning methods to study and understand complex physical systems that cannot be analyzed using traditional techniques (such as cerebrospinal fluid flow or turbulent flows).

**NABLA** is a collection of my ongoing work and serves as a unified framework for efficient and reproducible implementations of Physics-Informed Neural Networks (PINNs), DeepONets, and novel architectures like KANs.

### Repository Structure
The codebase includes separate modules for:

1.  **Tutorials**: Basic introductions to PINNs and DeepONets using both PyTorch and JAX.
2.  **RBA**: Code for Residual-Based Attention mechanisms [1]. Our official repository is available at: [rba-pinns](https://github.com/soanagno/rba-pinns).
3.  **cKANs**: Implementations for the comprehensive comparison between MLP and KAN representations [2].
4.  **KKANs**: Source code for Kurkova-Kolmogorov-Arnold Networks [3]. Our official repository is available at: [KKANs](https://github.com/jdtoscano94/Kurkova_Kolmogorov_Arnold_Networks_KKANs.git).
5.  **vRBA**: A Variational Framework for Residual-Based Adaptivity [4]. **Note:** This module includes our custom, highly accurate implementation of the **Self-Scaling Broyden (SSBroyden)** optimizer.
6.  **AIVT**: The code for Turbulent Thermal Convection [5] is hosted in its own dedicated repository: [Instant-AIVT](https://github.com/jdtoscano94/Instant-AIVT).
7.  **MR-AIV**: Code for Brain-wide Fluid Flow [6] will be released soon.
## 🛠️ Installation

**Note on Repository Structure:**
* **RBA and cKANs:** These directories are **self-contained**. They can be run directly as standalone scripts, provided you have a valid JAX installation.
* **KKANS and vRBA (and SSBroyden):** To use the Variational Residual-Based Adaptivity framework and our highly accurate **SSBroyden** optimizer, you must install the `Crunch` library using the steps below.

### Setup Instructions (Required for vRBA)

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


## 📚 References

If you use code from this repository in your research, please consider citing the relevant papers below:

### [1] Residual-Based Attention (RBA)
```bibtex
@article{anagnostopoulos2024residual,
  title={Residual-based attention in physics-informed neural networks},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={421},
  pages={116805},
  year={2024},
  publisher={Elsevier}
}
```

### [2] Comprehensive KANs (cKANs)
```bibtex
@article{shukla2024comprehensive,
  title={A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks},
  author={Shukla, Khemraj and Toscano, Juan Diego and Wang, Zhicheng and Zou, Zongren and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117290},
  year={2024},
  publisher={Elsevier}
}
```

### [3] Kurkova-Kolmogorov-Arnold Networks (KKANs)
```bibtex
@article{toscano2025kkans,
  title={KKANs: Kurkova-Kolmogorov-Arnold networks and their learning dynamics},
  author={Toscano, Juan Diego and Wang, Li-Lian and Karniadakis, George Em},
  journal={Neural Networks},
  volume={191},
  pages={107831},
  year={2025},
  publisher={Elsevier}
}
```

### [4] Variational Residual-Based Adaptivity (vRBA)
```bibtex
@article{toscano2025variational,
  title={A Variational Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning},
  author={Toscano, Juan Diego and Chen, Daniel T and Oommen, Vivek and Darbon, J{'e}r{\^o}me and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2509.14198},
  year={2025}
}
```

### [5] AIVT (Turbulence Inference)
```bibtex
@article{toscano2025aivt,
  title={AIVT: Inference of turbulent thermal convection from measured 3D velocity data by physics-informed Kolmogorov-Arnold networks},
  author={Toscano, Juan Diego and K{\"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={Science advances},
  volume={11},
  number={19},
  pages={eads5236},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```

### [6] MR-AIV (Brain Fluid Flow)
```bibtex
@article{toscano2025mr,
  title={MR-AIV reveals in vivo brain-wide fluid flow with physics-informed AI},
  author={Toscano, Juan Diego and Guo, Yisen and Wang, Zhibo and Vaezi, Mohammad and Mori, Yuki and Karniadakis, George Em and Boster, Kimberly AS and Kelley, Douglas H},
  journal={bioRxiv},
  year={2025}
}
```