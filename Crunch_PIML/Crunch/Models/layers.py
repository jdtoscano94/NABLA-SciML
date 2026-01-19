# %% [markdown]
# # Layers
# This is the documentation for the Layers module imported as  
# 
# ```python
# from Crunch.Models.layers import *
# ```

# %%
import os
import sys
import os
import time
import jax
from jax import vmap
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import trange
from jax import jvp, vjp, value_and_grad
from flax import linen as nn
from typing import Sequence, Callable
from functools import partial
import scipy
from pyDOE import lhs
import scipy.io as sio
from Crunch.Models.polynomials import  *

# %% [markdown]
# ### Polynomial_Embedding (Feature Expansion)
# 
# It is a **feature expansion** module that implements the "Chebyshev Layer" (or "Legendre Layer") blocks used within the **Inner Block (ebMLP)** of the KKAN architecture.
# 
# 
# Instead of summing terms into a single scalar, this module expands an input $ X $ into a **high-dimensional feature vector $ H(X) $** by horizontally concatenating the outputs of scaled, trainable polynomial basis functions.
# 
# The mathematical formula for the output vector $ H(X) $ is (assuming `step=1`):
# 
# $$
# 
# H(X) = \left[ \frac{C_0}{1} P_0(X), \frac{C_1}{2} P_1(X), \frac{C_2}{3} P_2(X), \dots, \frac{C_D}{D+1} P_{D}(X) \right]
# 
# $$
# 
# Where:
# * $ P_n $ is the $ n $-th polynomial (Chebyshev $ T_n $ or Legendre $ L_n $) selected by `polynomial_type`.
# * $ D $ is the `degree`.
# * $ C_n $ are the trainable coefficients (`c_i`), which are initialized to ones.
# 
# The `step` parameter allows for using only a subset of polynomials (e.g., $ P_0, P_2, P_4, \dots $ if `step=2`). The resulting feature vector $ H(X) $ is then passed as input to a subsequent network, such as the MLP in the core of the ebMLP.
# """

# %%
class Polynomial_Embedding(nn.Module):   
    degree: int
    step: int = 1 
    polynomial_type: str ='T'
    def setup(self):
        self.T_funcs = [globals()[f"{self.polynomial_type}{i}"] for i in range(0, self.degree+1, self.step)]
    @nn.compact
    def __call__(self, X):
        C_n = self.param('c_i', nn.initializers.ones, (len(self.T_funcs)))  # Adjust size based on the step
        C_n_T_n = jnp.hstack([C_n[i] / (i + 1) * self.T_funcs[i](X) for i in range(len(self.T_funcs))])
        return C_n_T_n


# %% [markdown]
# ### Random_Fourier_Embedding (Feature Expansion)
# 
# This module implements **Random Fourier Features (RFF)**, a technique used to map low-dimensional inputs into a higher-dimensional feature space. This mapping helps neural networks learn high-frequency functions more effectively, a common requirement in scientific machine learning.
# 
# 
# The module follows these steps:
# 
# 1.  **Random Projection Matrix:** A non-trainable (static) projection matrix $ B $ is initialized with shape `(input_features, degree)`. The elements of $ B $ are drawn from a normal distribution scaled by $ s $:
# 
#     $$
# 
#     B \sim \mathcal{N}(0, s^2)
# 
#     $$
# 
# 2.  **Linear Projection:** The input vector $ X $ is projected onto this random matrix:
# 
#     $$
# 
#     X_{\text{proj}} = X \cdot B
# 
#     $$
# 
# 3.  **Feature Concatenation:** The final output feature vector is created by concatenating the original input $ X $ with the sine and cosine of the projected features:
# 
#     $$
# 
#     H(X) = \text{concatenate}[X, \sin(X_{\text{proj}}), \cos(X_{\text{proj}})]
# 
#     $$
# 
# The resulting vector $ H(X) $ has a shape of `(batch_size, input_features + 2 * degree)` and serves as the embedded input for subsequent network layers.

# %%
class Random_Fourier_Embedding(nn.Module):
    """Random Fourier Feature embedding layer."""
    degree: int
    s: float = 10.0
    @nn.compact
    def __call__(self, X: Array) -> Array:
        # Infer input dimension dynamically from the input tensor X.
        input_features = X.shape[-1]
        def b_init(key, shape, dtype=jnp.float32):
            return self.s * jax.random.normal(key, shape, dtype)
        
        B = self.param('B', b_init, (input_features, self.degree))
        X_proj = jnp.dot(X, B)
        # Concatenate original features with their sinusoidal projections.
        return jnp.concatenate([X, jnp.sin(X_proj), jnp.cos(X_proj)], axis=-1)

# %% [markdown]
# ### Periodic_Fourier_Features (Feature Expansion)
# 
# This module is a **feature expansion** layer that maps a low-dimensional input $ X $ to a high-dimensional feature vector $ H(X) $ using a deterministic **Periodic Fourier Series expansion**.
# 
# This technique is a cornerstone of scientific machine learning for encoding known periodicities in a domain (e.g., in periodic boundary conditions).
# 
# 
# The module computes a feature vector $ H(X) $ by horizontally concatenating a basis of $ D $ harmonic sine and cosine functions, corresponding to the wave numbers $ k = \{1, \dots, D\} $.
# 
# The mathematical formula for the output vector $ H(X) $ is:
# 
# $$
# 
# H(X) = \text{concatenate}\left[
#     \cos(\omega_1 X), \sin(\omega_1 X),
#     \cos(\omega_2 X), \sin(\omega_2 X),
#     \dots,
#     \cos(\omega_D X), \sin(\omega_D X)
# \right]
# 
# $$
# 
# Where:
# * $ D $ is the `kmax` (maximum wave number).
# * $ \omega_k = \frac{2\pi k}{L_x} $ is the $ k $-th angular frequency.
# * $ L_x $ is the period of the domain.
# 
# This implementation is fully deterministic and not trainable. It differs from the `Fourier_Embedding` by focusing exclusively on the periodic components and omitting the $ k=0 $ (constant) term.

# %%
class Periodic_Fourier_Features(nn.Module):
    kmax: int = 1
    Lx: float = 2.0 
    @nn.compact
    def __call__(self, X: Array) -> Array:
        ks = jnp.arange(1, self.kmax + 1)
        Xper = 2 * jnp.pi * jnp.matmul(X, ks[None, :]) / self.Lx
        xcos = jnp.cos(Xper)
        xsin = jnp.sin(Xper)
        Xper = jnp.concatenate([xcos, xsin], axis=1)
        return Xper

# %% [markdown]
# # Weight normalization (Architecture Enhancement)
# 
# This is a JAX/Flax module WN_layer that implements Weight Normalization. Its goal is to accelerate training by decoupling the weight vector's magnitude (a learnable scalar $g$) from its direction (a unit vector $V$).The core operation is:$$y = g \frac{\mathbf{W}}{\|\mathbf{W}\|} \cdot \mathbf{H} + b$$Where $\mathbf{W}$ is the raw weight matrix, $V = \mathbf{W} / \|\mathbf{W}\|$ is the normalized direction, $g$ is the learnable scale, and $b$ is the learnable bias.

# %%
class WN_layer(nn.Module):
    out_features: int  # Number of output features
    kernel_init: nn.initializers.Initializer  # Custom initializer for W
    def setup(self):
        # Define bias and scale parameters; W will be initialized later
        self.b = self.param('b', nn.initializers.zeros, (self.out_features,))
        self.g = self.param('g', nn.initializers.ones, (self.out_features,))
    @nn.compact
    def __call__(self, H):
        # Determine input size from H dynamically
        in_features = H.shape[-1]
        # Initialize W with the specified kernel initializer and dynamic shape
        W = self.param('W', self.kernel_init, (in_features, self.out_features))
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        return self.g * jnp.dot(H, V) + self.b

# %% [markdown]
# ## Adaptive Residual Connections  (Architecture Enhancement)
# AdaptiveResNetThis module implements an adaptive residual block.Instead of the standard ResNet connection $ \mathbf{H}{\text{out}} = \mathbf{H}{\text{in}} + \mathbf{F}(\mathbf{H}{\text{in}}) $, this block uses a single learnable scalar parameter $\alpha$ to create a weighted average between the identity path ($\mathbf{H}{\text{in}}$) and the transformation path ($\mathbf{G}$).The core operation is:$$\mathbf{H}_{\text{out}} = \tanh(\alpha \cdot \mathbf{G} + (1 - \alpha) \cdot \mathbf{H}_{\text{in}})$$Where $\mathbf{G}$ is the output of the transformation block (which, according to your code, uses two WN_layers):$$\mathbf{G} = \text{WN\_layer}_2(\tanh(\text{WN\_layer}_1(\mathbf{H}_{\text{in}})))$$This $\alpha$ parameter allows the network to learn how much of the original signal $\mathbf{H}_{\text{in}}$ to preserve, effectively learning the "skip" connection's strength.As you noted, this architecture is especially useful for deep networks. Like standard residual connections, it helps to prevent vanishing gradients by ensuring a clear path for gradient flow back through the identity component ($(1 - \alpha) \cdot \mathbf{H}_{\text{in}}$).

# %%
class AdaptiveResNet(nn.Module):
    out_features: int
    def setup(self):
        # Initialize alpha as a trainable parameter with a single value
        self.alpha = self.param('alpha', nn.initializers.ones, ())
    @nn.compact
    def __call__(self, H):
        init = nn.initializers.glorot_normal()
        F = nn.activation.tanh(WN_layer(self.out_features, kernel_init=init)(H))
        G = WN_layer(self.out_features, kernel_init=init)(F)
        H = nn.activation.tanh(self.alpha * G + (1 - self.alpha) * H)
        return H

# %% [markdown]
# ### eMLP (Enhanced-basis MLP) (Architecture Enhancement)
# 
# This module implements the **Enhanced-basis Multi-Layer Perceptron (ebMLP)**, the advanced architecture used as the **Inner Block** of the KKAN model (see `KKANs_Journal_Version.pdf`, Section 2.3.1, Figure 2).
# 
# 
# The design is inspired by the **"adaptive basis viewpoint"**: the network learns a set of adaptive features $ \beta $, and the final output is a linear combination of a basis $ T(\beta) $ applied to those features.
# 
# ---
# 
# ### Core Idea
# 
# The key idea is to **impose structure** on the network's representations by enclosing a standard MLP core (built from `WN_layer` and `AdaptiveResNet` blocks) between two `Polynomial_Embedding` layers.
# 
# 1.  **Input Embedding ($ T(x_p) $):** The input $ x_p $ is first expanded into a feature space $ T(x_p) $ by a `Polynomial_Embedding` layer.
# 2.  **MLP Core ($ \beta = \text{MLP}(T(x_p)) $):** A deep MLP core processes these features to learn an internal representation $ \beta $.
# 3.  **Pre-Output Embedding ($ T(\beta) $):** This internal representation $ \beta $ is expanded *again* by a second `Polynomial_Embedding` layer, $ T(\beta) $.
# 4.  **Final Linear Combination ($ \Psi(x_p) = \text{Linear}(T(\beta)) $):** The final output is a simple linear combination (a `WN_layer`) of this $ T(\beta) $ basis.
# 
# By using an orthogonal basis (e.g., Chebyshev) for $ T $, this architecture endows the **"last layer"** (the final `WN_layer`) with beneficial properties, such as promoting **sparse or orthogonal representations**, which can improve stability and generalization.
# 
# ---
# 
# ### Mathematical Flow
# 
# $$
# 
# \Psi(x_p) = \text{Linear} \Big( T \big( \text{MLP}( T(x_p) ) \big) \Big)
# 
# $$
# 
# Where:
# * $ T(\cdot) $ is the `Polynomial_Embedding` layer.
# * $ \text{MLP}(\cdot) $ is the core stack of `AdaptiveResNet` blocks.
# * $ \text{Linear}(\cdot) $ is the final `WN_layer`.
# """

# %%
class eMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable = nn.relu
    degree: int=5
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        # 1. Input Embedding
        X = Polynomial_Embedding_Layer(degree=self.degree)(x)
        
        # 2. Main MLP Body (e.g., with ResNets)
        H = nn.activation.tanh(WN_layer(self.layers[0], kernel_init=init)(X))
        for feat in self.layers[1:-1]:
            H = AdaptiveResNet(out_features=feat)(H) 
        # 3. Pre-Output Embedding
        H = Polynomial_Embedding_Layer(degree=self.degree)(H)
        
        # 4. Final Layer
        H = WN_layer(self.layers[-1], kernel_init=init)(H)
        return H

# %% [markdown]
# # KAN (Kolmogorov-Arnold Network)
# 
# Based on the appendix (B.2), your `KAN` module defines the overall network architecture, which is inspired by the **Kolmogorov-Arnold representation theorem**.
# 
# The mathematical formula for the full network, which your `KAN` class implements by stacking layers, is a nested composition of univariate functions as defined in (B.1):
# [Image of KAN architecture diagram]
# 
# $$
# 
# f(x) \approx \sum_{i_{L-1}=1}^{n_{L-1}} \phi_{L-1,i_{L},i_{L-1}}(\cdot\cdot\cdot\phi_{1,i_{2},i_{1}}(\sum_{i_{0}=1}^{n_{0}} \phi_{0,i_{1},i_{0}}(x_{i_{0}}))\cdot\cdot\cdot)
# 
# $$
# 
# Where:
# * $ L $ is the number of layers (the length of your `layers` sequence).
# * $ n_j $ is the number of neurons in layer $ j $ (the values `feat` in your `layers` sequence).
# * $ \phi_{i,j,k} $ are the learnable **univariate activation functions**.
# 

# %%
class KAN(nn.Module):
    layers: Sequence[int]
    degree: int = 5
    @nn.compact
    def __call__(self, x):
        for feat in self.layers:
            x = Cheby_KAN_layer(out_dim=feat, degree=self.degree)(x)
        return x

# %% [markdown]
# The possibilities are extensive. The basis functions could also be:
# * **Polynomial basis such as Chebyshev**
# * **Radial Basis Functions (RBFs)**
# * **Fourier Series**
# * **Wavelets**
# * In other words any other universal approximator in 1D.
# 
# ### Cheby_KAN_layer (cKAN) (KAN Layer)
# 
# This specific implementation makes it a **Recursive Chebyshev KAN (cKAN)**.
# 
# The `Cheby_KAN_layer` module itself provides the mathematical definition for each individual $ \phi $ function, which in this case is a linear combination of Chebyshev polynomials as defined in (B.4):
# 
# $$
# 
# \phi(x) = \sum_{n=1}^{D} C_n T_n(\tanh(x))
# 
# $$
# 
# Where:
# * $ T_n $ is the $ n $-th Chebyshev polynomial.
# * $ C_n $ are the trainable coefficients.
# * $ D $ is the `degree`.
# * $ \tanh(x) $ is used to embed the input $ x $ and ensure normalization.
# 

# %%
class Cheby_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    polynomial_type: str = 'T' # 'T' for Chebyshev or 'L' for Legendre
    normalization: callable = jnp.tanh

    def setup(self):
        # Dynamically get the Chebyshev T(0), T(1)... functions
        self.T_funcs = [globals()[f"{self.polynomial_type}{i}"] for i in range(self.degree + 1)]
        self.normalization_fn = self.normalization

    @nn.compact
    def __call__(self, X):
        X = self.normalization_fn(X) # Normalize input
        in_dim = X.shape[1]
        
        # Trainable coefficients C_n
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
            (in_dim, self.out_dim, self.degree + 1)
        )
        
        # Stack the polynomial evaluations T_n(X)
        T_n = jnp.stack([func(X) for func in self.T_funcs], axis=1)
        
        # Compute the final output via tensor contraction
        # This is a batched sum over the input dim and degree
        C_n_Tn = jnp.einsum('bdi,iod->bo', T_n, C_n)

        return C_n_Tn

# %% [markdown]
# ### RBF_KAN_layer (KAN Layer)
# 
# This module implements a KAN layer where each univariate activation function $ \phi(x) $ is a linear combination of **Radial Basis Functions (RBFs)**.
# 
# This corresponds to the "Radial Basis Functions (RBF)" case in the appendix (B.3.2). The mathematical formula for a single univariate function $ \phi(x) $ is:
# 
# 
# $$
# 
# \phi(x) = \sum_{n=1}^{D} C_n e^{-\frac{(x-p_n)^2}{2\sigma^2}}
# 
# $$
# 
# Where:
# * $ D $ is the number of centers, corresponding to the `degree` hyperparameter.
# * $ p_n $ are the centers of the RBFs, which the code initializes as a uniform grid (`jnp.linspace`) from `grid_min` to `grid_max`. These are the `centers` in the code.
# * $ C_n $ are the trainable coefficients, corresponding to the `C_n` parameter.
# * $ \sigma $ is the hyperparameter controlling the spread (width) of the basis functions. The code sets this as `(self.grid_max - self.grid_min) / (self.num_centers - 1)`.

# %%
class RBF_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    grid_min: float = -2.0
    grid_max: float = 2.0
    normalization: callable = jnp.tanh
    def setup(self):
        self.num_centers = self.degree
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)
        self.normalization_fn=self.normalization
    @nn.compact
    def __call__(self, X):
        X = X # It is possible to add normalization self.normalization_fn(X) but is generally not needed
        batch_size, in_dim = X.shape
        centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )

        X_expanded = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        centers_expanded = centers[None, :, None]  # Shape: (1, num_centers, 1)

        diff = X_expanded - centers_expanded   # Shape: (batch_size, num_centers, in_dim)

        RBF_n = jnp.exp(- (diff ** 2) / (2 * self.sigma ** 2))

        RBF_n = jnp.transpose(RBF_n, (0, 2, 1))

        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', RBF_n, C_n)

        return output

# %% [markdown]
# ### AcNet_KAN_layer (Sine Series) (KAN Layer)
# 
# This module implements a KAN layer where each univariate activation function $ \phi(x) $ is a linear combination of normalized **Sine Series** basis functions. This layer is referred to as "Sine Series" in the appendix (B.3.2).
# 
# The mathematical formula for the full activation $ \phi(x) $ is a sum of basis functions $ b_i $:
# 
# $$
# 
# \phi(x) = \sum_{i=1}^{D} C_i b_i(x)
# 
# $$
# 
# Where $ D $ is the `degree`, $ C_i $ are the trainable coefficients (`C_n`), and each basis function $ b_i(x) $ is a normalized sine function:
# 
# $$
# 
# b_i(x) = \frac{\sin(w_i x + p_i) - \mu(w_i, p_i)}{\sigma(w_i, p_i)}
# 
# $$
# 
# The $ w_i $ (`W_i`) are trainable frequencies and $ p_i $ (`centers`) are trainable phases. The basis is normalized using its statistical mean $ \mu $ and standard deviation $ \sigma $:
# 
# $$
# 
# \mu(w_i, p_i) = e^{-w_i^2/2} \sin(p_i)
# 
# $$
# 
# $$
# 
# \sigma(w_i, p_i) = \sqrt{\frac{1}{2} - e^{-w_i^2} \cos(p_i) - \mu(w_i, p_i)^2}
# 
# $$
# 

# %%
class AcNet_KAN_layer(nn.Module):
    out_dim: int
    degree: int = 5
    grid_min: float = -0.1
    grid_max: float = 0.1
    normalization: callable = jnp.sin ## This should not be changed!!! otherwise we need other normalization
    def setup(self):
        self.num_centers=self.degree
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )
        self.normalization_fn=self.normalization
    @nn.compact
    def __call__(self, X):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        p_i = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_i = self.param(
            'W_i',
            nn.initializers.normal(1),
            (p_i.shape)
        )
        mu=jnp.exp(-(W_i**2)/2)*self.normalization_fn(p_i)
        sigma=jnp.sqrt(0.5-0.5*jnp.exp(-(2*W_i**2)*jnp.cos(2*p_i)-mu**2))+1e-12
        b_i=(self.normalization_fn(W_i*X+p_i)-mu)/sigma
        b_i = jnp.transpose(b_i, (0, 2, 1))
        # Initialize trainable parameters C_n
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', b_i, C_n)
        return output

# %% [markdown]
# ### Polynomial_KAN_layer (cKAN / Legendre) (KAN Layer)
# 
# This module implements a KAN layer where each univariate activation function $ \phi(x) $ is a linear combination of **orthogonal polynomials**, specifically **Chebyshev ($ T_n $)** or **Legendre ($ L_n $)**.
# 
# This corresponds to the "Chebyshev" and "Legendre" bases described in the appendix (B.3.2). The mathematical formula for a single univariate function $ \phi(x) $ is:
# 
# 
# $$
# 
# \phi(x) = \sum_{n=0}^{D} C_n P_n(\tanh(x))
# 
# $$
# 
# Where:
# * $ P_n $ is the $ n $-th polynomial (Chebyshev or Legendre), selected by `polynomial_type`.
# * $ D $ is the `degree` hyperparameter.
# * $ C_n $ are the trainable coefficients.
# * $ \tanh(x) $ is the `normalization_fn` used to map the input to the polynomial's valid domain (typically [-1, 1]).
# 
# This implementation is highly efficient. It first computes the polynomial features $ [P_0(x_i), ..., P_D(x_i)] $ for all input dimensions $ i $ in parallel (the $ T\_n $ tensor). The final output is then computed with a single matrix multiplication (`T_n @ C_n`), which acts as a large **linear layer** whose inputs are the polynomial features and whose weights are the trainable coefficients $ C\_n $.
# """

# %%
class Polynomial_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    polynomial_type: str='T' # L is for Legendre
    normalization: callable = jnp.tanh
    def setup(self):
        self.T_funcs = [globals()[f"{self.polynomial_type}{i}"] for i in range(self.degree + 1)]
        self.normalization_fn=self.normalization
    @nn.compact
    def __call__(self, X):
        X = self.normalization_fn(X)
        in_dim = X.shape[1]
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                        (in_dim, self.out_dim, (self.degree + 1)))  # Shape: (In, Out, Degree+1)
        
        # Compute polynomial functions T_n
        T_n = jnp.stack([func(X) for func in self.T_funcs], axis=1)  # Shape: (Batch, Degree+1, In)
        
        # Reshape and transpose tensors for matrix multiplication
        T_n = T_n.transpose(0, 2, 1).reshape(X.shape[0], -1)      # Shape: (Batch, In * (Degree+1))
        C_n = C_n.transpose(0, 2, 1).reshape(-1, self.out_dim)    # Shape: (In * (Degree+1), Out)
        
        # Perform matrix multiplication
        C_n_Tn = T_n @ C_n  # Shape: (Batch, Out)
        return C_n_Tn

# %% [markdown]
# # Polynomial_grid_KAN_layer (Chebyshev/Legendre Grid) (KAN Layer)
# 
# This module implements a KAN layer using a "Chebyshev Grid" or "Legendre Grid" basis, as described in the appendix (B.3.2).
# 
# This architecture enhances the standard `Polynomial_KAN_layer` by first processing the input $ x $ through a small, trainable sub-expansion based on a grid of centers. The output of this sub-expansion is then fed into the polynomial basis.
# 
# The mathematical formula for a single univariate function $ \phi(x) $ is:
# 
# 
# $$
# 
# \phi(x) = \sum_{n=0}^{D} C_n P_n \left( \sum_{i=1}^{c} \tanh(W_i x + b_i) \right)
# 
# $$
# 
# Where:
# 1.  **Grid Sub-expansion (Inner step):** The input $ x $ is first transformed by a sum of $ c $ (`num_centers`) hyperbolic tangent functions, each with its own trainable weight $ W_i $ (`W_p`) and bias/center $ b_i $ (`centers`). This is the $ \sum \tanh(\dots) $ term.
# 2.  **Polynomial Basis (Outer step):** The result of this sum is then fed as the input to the standard polynomial basis.
# 3.  **Final Summation:** The output is a linear combination of these polynomial functions, $ P_n $ (Chebyshev or Legendre), weighted by the trainable coefficients $ C_n $.
# 
# This combination allows the layer to learn a more complex, displaced, and scaled input representation before applying the polynomial expansion, increasing its expressiveness.

# %%
class Polynomial_grid_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    num_centers: int = 5
    grid_min: float = -0.5
    grid_max: float = 0.5
    polynomial_type: str='T'
    normalization: callable = jnp.tanh
    def setup(self):
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )
        self.T_funcs = [globals()[f"{self.polynomial_type}{i}"] for i in range(self.degree + 1)]
        self.normalization_fn=self.normalization
    @nn.compact
    def __call__(self, X):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        b = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_p = self.param(
            'W_p',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (b.shape)
        )
        X=self.normalization_fn(W_p*X+b)
        X=jnp.sum(X,axis=1)
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_n= jnp.stack([self.T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn

# %% [markdown]
# 
# ### KKAN (Kürková-Kolmogorov-Arnold Network)
# 
# This module implements the full **Kürková-Kolmogorov-Arnold Network (KKAN)** architecture.
# 
# 
# The KKAN is a two-block model designed to approximate a multivariate function $ f(x) $ by separating it into **inner** and **outer** functions, adhering to the Kolmogorov-Arnold representation:
# 
# $$
# 
# f(x_1, \dots, x_d) = \sum_{q=0}^{m} g_q \left( \sum_{p=1}^{d} \Psi_{p,q}(x_p) \right)
# 
# $$
# 
# This class implements this architecture by composing two main sub-modules:
# 
# ---
# 
# ### 1. Inner Block + Combination Layer (`get_Psi`)
# 
# The `get_Psi` submodule (which is not shown but inferred) is responsible for the entire inner part of the formula. It performs two steps:
# 
# * **Inner Block:** It computes the inner functions $ \Psi(x_p) $ for each input dimension $ x_p $ (where $ p = 1, \dots, d $). Each of these inner blocks is an `eMLP` (Enhanced-basis MLP) module.
#     $$
#     \Psi(x_p) = \text{eMLP}(x_p)
#     $$
# * **Combination Layer:** It sums the outputs of all `eMLP` blocks to produce the intermediate vector $ \xi $ (named `sum_psi` in the code).
#     $$
#     \xi_q = \sum_{p=1}^{d} \Psi_{p,q}(x_p)
#     $$
# 
# The output of `get_Psi` is the vector $ \xi = [\xi_0, \dots, \xi_m] $.
# 
# ---
# 
# ### 2. Outer Block + Final Sum (KAN Layer)
# 
# The second part of the model is implemented by a **swappable KAN layer** (e.g., `RBF_KAN_layer`, `Polynomial_KAN_layer`, etc.). This single layer performs both the outer function evaluation and the final summation.
# 
# It takes the vector $ \xi $ (`sum_psi`) as its input and applies its univariate functions $ g_q $ to each component $ \xi_q $, summing the results to produce the final output $ f(x) $:
# 
# $$
# 
# f(x) = \sum_{q=0}^{m} g_q(\xi_q)
# 
# $$
# 
# In this implementation, the $ g_q $ functions are the univariate functions $ \phi_{q,o} $ defined by the chosen KAN layer (e.g., RBFs, Chebyshev polynomials, etc.).

# %%
class KKAN(nn.Module):
    features: Sequence[int]
    degree: int=5
    degree_T: int=5
    M: int = 10
    output_dim: int =1
    activation: Callable = nn.tanh
    def setup(self):
        # Initialize the GetPhi submodule
        self.get_Psi = get_Psi(degree=self.degree_T, features=self.features, M=self.M)
    @nn.compact
    def __call__(self, X):
        # Process inputs through the GetPhi function
        inputs = [X[:,0:1],X[:,1:2]]
        sum_psi = self.get_Psi(inputs)
        sum_Phi = RBF_KAN_layer(out_dim=self.output_dim, degree=self.degree)(sum_psi)
        return sum_Phi


# %% [markdown]
# ## KKAN Component
# 
# This module implements the core of the **KKAN** architecture: the **Inner Block** and the first **Combination Layer**, as shown in Figure 1 of the `KKANs_Journal_Version.pdf`.
# 
# 
# It is responsible for executing the first half of the Kolmogorov-Arnold representation:
# 
# $$
# 
# \xi_q = \sum_{p=1}^{d} \Psi_{p,q}(x_p)
# 
# $$
# 
# This is achieved in two distinct steps within the module's `__call__` method.
# 
# ---
# 
# ### 1. Inner Block (The Loop)
# 
# The `for` loop iterates over each input dimension $ x_p $ (where $ p = 1, \dots, d $). Inside the loop, each $ x_p $ is processed by its own independent **`eMLP` (Enhanced-basis MLP)** module.
# 
# 
# Is the explicit, unrolled implementation of the `eMLP` architecture. The output $ H $ of this process is the inner function vector $ \Psi_p(x_p) $ for that specific dimension $ p $.
# 
# $$
# 
# H = \Psi_p(x_p) = \text{eMLP}(x_p)
# 
# $$
# 
# ---
# 
# ### 2. Combination Layer (The Sum)
# 
# The line `sum_psi += H` implements the first **Combination Layer** of the KART formula (the $ \sum_{p=1}^{d} $). It performs an element-wise sum, accumulating the inner function vectors $ \Psi_p(x_p) $ from all input dimensions.
# 
# ---
# 
# ### Final Output
# 
# The returned value `sum_psi` is the intermediate vector $ \xi = [\xi_0, \dots, \xi_m] $, which is the direct result of this summation.
# 
# $$
# 
# \text{sum\_psi} = \xi = \sum_{p=1}^{d} \Psi_p(x_p)
# 
# $$
# 
# This vector $ \xi $ is then passed as the input to the **KKAN Outer Block** (e.g., `RBF_KAN_layer`, `Polynomial_KAN_layer`, etc.) to complete the computation.

# %%
class get_Psi(nn.Module):
    degree: int
    features: Sequence[int]
    M: int = 10
    def setup(self):
        self.T_funcs = [globals()[f"T{i}"] for i in range(self.degree + 1)]
    @nn.compact
    def __call__(self, inputs):
        init = nn.initializers.glorot_normal()
        sum_psi = 0
        
        # Loop over each input dimension (e.g., x_1, x_2, ...)
        for i, X in enumerate(inputs):
            # Each input is processed by its OWN eMLP
            
            # 1. Input Embedding
            X = Polynomial_Embedding_Layer(degree=self.degree)(X)
            # 2. Main MLP Body
            H = nn.activation.tanh(WN_layer(self.features[0], kernel_init=init)(X))
            for fs in self.features[1:-1]:
                H = AdaptiveResNet(out_features=fs)(H) 
            # 3. Pre-Output Embedding
            H = Polynomial_Embedding_Layer(degree=self.degree)(H)
            # 4. Final Layer
            H = WN_layer(self.features[-1], kernel_init=init)(H)
            # 5. Accumulate results (the \sum_p part)
            sum_psi += H
        return sum_psi

# %% [markdown]
# ## Automatic Differentiation schemes 
# 
# Adapted from https://github.com/stnamjef/SPINN.

# %%
# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out

# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out

# reverse over reverse
def hvp_revrev(f, primals, tangents, return_primals=False):
    g = lambda primals: vjp(f, primals)[1](tangents)
    primals_out, vjp_fn = vjp(g, primals)
    tangents_out = vjp_fn((tangents,))[0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# forward over reverse
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    g = lambda primals: vjp(f, primals)[1](tangents[0])[0]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over forward
def hvp_revfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, primals, tangents)[1]
    primals_out, vjp_fn = vjp(g, primals)
    tangents_out = vjp_fn(tangents[0])[0][0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


