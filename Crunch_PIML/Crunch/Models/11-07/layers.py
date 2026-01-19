import os
import sys
import os
import time
import jax
from jax import vmap
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from jax import jvp, vjp, value_and_grad
from flax import linen as nn
from typing import Sequence
from functools import partial
import scipy
from pyDOE import lhs
import scipy.io as sio
from Crunch.Models.polynomials import  *



class Polynomial_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    @nn.compact
    def __call__(self, X, normalization_fn=jnp.tanh, polynomial_type='T'):
        X = normalization_fn(X)
        in_dim = X.shape[1]
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                        (in_dim, self.out_dim, (self.degree + 1)))  # Shape: (In, Out, Degree+1)
        
        # Compute polynomial functions T_n
        T_funcs = [globals()[f"{polynomial_type}{i}"] for i in range(self.degree + 1)]
        T_n = jnp.stack([func(X) for func in T_funcs], axis=1)  # Shape: (Batch, Degree+1, In)
        
        # Reshape and transpose tensors for matrix multiplication
        T_n = T_n.transpose(0, 2, 1).reshape(X.shape[0], -1)      # Shape: (Batch, In * (Degree+1))
        C_n = C_n.transpose(0, 2, 1).reshape(-1, self.out_dim)    # Shape: (In * (Degree+1), Out)
        
        # Perform matrix multiplication
        C_n_Tn = T_n @ C_n  # Shape: (Batch, Out)
        return C_n_Tn




class Cheby_KAN_layer_norm(nn.Module):
    out_dim: int
    degree: int
    @nn.compact
    def __call__(self, X):
        X=jnp.tanh(X)
        in_dim = X.shape[1]
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_funcs = [globals()[f"T{i}"] for i in range(self.degree+1)]
        T_n= jnp.stack([1 / (i + 1) *T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn


class Legendre_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    @nn.compact
    def __call__(self, X):
        X=jnp.tanh(X)
        in_dim = X.shape[1]
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_funcs = [globals()[f"L{i}"] for i in range(self.degree+1)]
        T_n= jnp.stack([T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn

class Legendre_KAN_layer_norm(nn.Module):
    out_dim: int
    degree: int
    @nn.compact
    def __call__(self, X):
        X=jnp.tanh(X)
        in_dim = X.shape[1]
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_funcs = [globals()[f"L{i}"] for i in range(self.degree+1)]
        T_n= jnp.stack([1 / (i + 1) *T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn



class Sigmoid_KAN_Layer(nn.Module):
    F_l1: int 
    F_l2: int
    
    @nn.compact
    def __call__(self, X,sigma=jnp.tanh):
        dim = X.shape[1]
        # Initialize trainable parameters a_i, b_i, c_i
        a_i = self.param('a_i', nn.initializers.normal(1.0), (dim, self.F_l1, self.F_l2))  # dim,Features,output
        b_i = self.param('b_i', nn.initializers.normal(1.0), (dim, self.F_l1, self.F_l2))  # dim,Features,output
        c_i = self.param('c_i', nn.initializers.zeros, (dim, self.F_l1, self.F_l2))  # dim,Features,output
        def single_batch_forward(x):
            b_iX = b_i * x[:, None, None] 
            psi = jnp.sum(a_i * sigma(b_iX + c_i), axis=1)
            return psi
        # Use vmap to vectorize over the batch dimension
        psi = vmap(single_batch_forward)(X)
        sum_psi=psi.sum(axis=1)
        return sum_psi
    
class RBF_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    grid_min: float = -2.0
    grid_max: float = 2.0
    def setup(self):
        self.num_centers = self.degree
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.tanh):
        X = normalization_fn(X)
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

class RBF_KAN_single_layer(nn.Module):
    out_dim: int
    degree: int
    grid_min: float = -2.0
    grid_max: float = 2.0
    def setup(self):
        self.num_centers = self.degree
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.tanh):
        X = normalization_fn(X)
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
            (self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,no->bo', RBF_n, C_n)

        return output

class Polynomial_grid_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    num_centers: int = 5
    grid_min: float = -0.5
    grid_max: float = 0.5
    def setup(self):
        self.sigma = (self.grid_max - self.grid_min) / (self.num_centers - 1)
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.tanh,polynomial_type='T'):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        b = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_p = self.param(
            'W_p',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (b.shape)
        )
        X=normalization_fn(W_p*X+b)
        X=jnp.sum(X,axis=1)
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_funcs = [globals()[f"{polynomial_type}{i}"] for i in range(self.degree+1)]
        T_n= jnp.stack([T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn

class AcNet_T_KAN_layer(nn.Module):
    out_dim: int
    degree: int
    num_centers: int = 5
    grid_min: float = -0.1
    grid_max: float = 0.1
    def setup(self):
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.sin,polynomial_type='T'):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        p_i = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_i = self.param(
            'W_i',
            nn.initializers.normal(1),
            (p_i.shape)
        )
        mu=jnp.exp(-(W_i**2)/2)*jnp.sin(p_i)
        sigma=jnp.sqrt(0.5-0.5*jnp.exp(-(2*W_i**2)*jnp.cos(2*p_i)-mu**2))+1e-12
        b_i=(jnp.sin(W_i*X+p_i)-mu)/sigma
        X=jnp.sum(X,axis=1)
        # Initialize trainable parameters C_n
        C_n = self.param('C_n', nn.initializers.normal(1 / (in_dim * (self.degree + 1))),
                         (in_dim, self.out_dim, (self.degree + 1)))  # In,Out,Degree+1
        T_funcs = [globals()[f"{polynomial_type}{i}"] for i in range(self.degree+1)]
        T_n= jnp.stack([T_funcs[i](X) for i in range(self.degree + 1)],axis=1)
        C_n_Tn = jnp.einsum("bdi,iod->bo", T_n, C_n)
        return C_n_Tn

class AcNet_Rand_KAN_layer(nn.Module):
    out_dim: int
    degree: int = 5
    grid_min: float = -0.5
    grid_max: float = 0.5

    def setup(self):
        # Generate random key
        self.num_centers=self.degree
        self.rng = jax.random.PRNGKey(42)  # You can set your seed here
        # Fixed centers and W_i, initialized but not trainable
        self.p_i = jnp.linspace(self.grid_min, self.grid_max, self.num_centers)
        self.p_i = self.p_i[None, :, None]  # Shape: (1, num_centers, 1)
        #  W_i initialization
        self.W_i = nn.initializers.normal(1)(self.rng, (1, self.num_centers, 1))  # Use the valid rng key
        self.mu = jnp.exp(-(self.W_i**2)/2) * jnp.sin(self.p_i)
        self.sigma = jnp.sqrt(0.5 - 0.5 * jnp.exp(-(2 * self.W_i**2) * jnp.cos(2 * self.p_i) - self.mu**2)) + 1e-12


    @nn.compact
    def __call__(self, X, normalization_fn=jnp.sin, polynomial_type='T'):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        b_i = (normalization_fn(self.W_i * X + self.p_i ) - self.mu) / self.sigma
        b_i = jnp.transpose(b_i, (0, 2, 1))

        # Initialize trainable parameters C_n
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', b_i, C_n)
        return output
class AcNet_KAN_layer(nn.Module):
    out_dim: int
    degree: int = 5
    grid_min: float = -0.1
    grid_max: float = 0.1
    def setup(self):
        self.num_centers=self.degree
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.sin,polynomial_type='T'):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        p_i = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_i = self.param(
            'W_i',
            nn.initializers.normal(1),
            (p_i.shape)
        )
        mu=jnp.exp(-(W_i**2)/2)*jnp.sin(p_i)
        sigma=jnp.sqrt(0.5-0.5*jnp.exp(-(2*W_i**2)*jnp.cos(2*p_i)-mu**2))+1e-12
        b_i=(jnp.sin(W_i*X+p_i)-mu)/sigma
        b_i = jnp.transpose(b_i, (0, 2, 1))
        # Initialize trainable parameters C_n
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', b_i, C_n)
        return output



class Exapand_Tanh_layer(nn.Module):
    out_dim: int
    degree: int = 5
    grid_min: float = -0.1
    grid_max: float = 0.1
    def setup(self):
        self.num_centers=self.degree
        self.centers = self.param(
            'centers',
            lambda rng, shape: jnp.linspace(self.grid_min, self.grid_max, self.num_centers),
            (self.num_centers,)
        )

    @nn.compact
    def __call__(self, X, normalization_fn=jnp.tanh,polynomial_type='T'):
        batch_size, in_dim = X.shape
        X = X[:, None, :]             # Shape: (batch_size, 1, in_dim)
        p_i = self.centers[None, :, None]  # Shape: (1, num_centers, 1)
        W_i = self.param(
            'W_i',
            nn.initializers.normal(1),
            (p_i.shape)
        )
        b_i=(jnp.normalization_fn(W_i*X+p_i))
        b_i = jnp.transpose(b_i, (0, 2, 1))
        # Initialize trainable parameters C_n
        C_n = self.param(
            'C_n',
            nn.initializers.normal(1 / (in_dim * self.num_centers)),
            (in_dim, self.num_centers, self.out_dim)
        )
        output = jnp.einsum('bin,ino->bo', b_i, C_n)
        return output





# WN

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


# Ress Nets
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

### Derivatives
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
