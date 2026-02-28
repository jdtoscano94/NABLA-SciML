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
from typing import Sequence
from functools import partial
import scipy
from pyDOE import lhs
import scipy.io as sio
import numpy as np
from jax import random
from jax import Array



# Chebyshev's Polynomials
def T0(x):
    return x*0+1
def T1(x):
    return x
def T2(x):
    return 2*x**2-1
def T3(x):
    return 4*x**3-3*x
def T4(x):
    return 8*x**4-8*x**2+1
def T5(x):
    return 16*x**5-20*x**3+5*x
def T6(x):
    return 32*x**6-48*x**4+18*x**2-1
def T7(x):
    return 64*x**7-112*x**5+56*x**3-7*x
def T8(x):
    return 128*x**8-256*x**6+160*x**4-32*x**2+1
def T9(x):
    return 256*x**9-576*x**7+432*x**5-120*x**3+9*x
def T10(x):
    return 512*x**10-1280*x**8+1120*x**6-400*x**4+50*x**2-1
def T11(x):
    return 1024*x**11-2816*x**9+2816*x**7-1232*x**5+220*x**3-11*x
def T12(x):
    return 2048*x**12-6144*x**10+6912*x**8-3584*x**6+840*x**4-72*x**2+1
def T13(x):
    return x * (4096 * x**12 - 13312 * x**10 + 16640 * x**8 - 9984 * x**6 + 2912 * x**4 - 364 * x**2 + 13)

def T14(x):
    return 8192 * x**14 - 28672 * x**12 + 39424 * x**10 - 26880 * x**8 + 9408 * x**6 - 1568 * x**4 + 98 * x**2 - 1

def T15(x):
    return x * (16384 * x**14 - 61440 * x**12 + 92160 * x**10 - 70400 * x**8 + 28800 * x**6 - 6048 * x**4 + 560 * x**2 - 15)

def T16(x):
    return 32768 * x**16 - 131072 * x**14 + 212992 * x**12 - 180224 * x**10 + 84480 * x**8 - 21504 * x**6 + 2688 * x**4 - 128 * x**2 + 1

def T17(x):
    return x * (65536 * x**16 - 278528 * x**14 + 487424 * x**12 - 452608 * x**10 + 239360 * x**8 - 71808 * x**6 + 11424 * x**4 - 816 * x**2 + 17)

def T18(x):
    return 131072 * x**18 - 589824 * x**16 + 1105920 * x**14 - 1118208 * x**12 + 658944 * x**10 - 228096 * x**8 + 44352 * x**6 - 4320 * x**4 + 162 * x**2 - 1

def T19(x):
    return x * (262144 * x**18 - 1245184 * x**16 + 2490368 * x**14 - 2723840 * x**12 + 1770496 * x**10 - 695552 * x**8 + 160512 * x**6 - 20064 * x**4 + 1140 * x**2 - 19)

def T20(x):
    return 524288 * x**20 - 2621440 * x**18 + 5570560 * x**16 - 6553600 * x**14 + 4659200 * x**12 - 2050048 * x**10 + 549120 * x**8 - 84480 * x**6 + 6600 * x**4 - 200 * x**2 + 1

# --- Analytical Derivatives of Chebyshev Polynomials, dT_0 to dT_10 ---

def dT0(x):
    return x*0

def dT1(x):
    return x*0+1


def dT2(x):
    return 4 * x

def dT3(x):
    return 12 * x**2 - 3

def dT4(x):
    return 32 * x**3 - 16 * x

def dT5(x):
    return 80 * x**4 - 60 * x**2 + 5

def dT6(x):
    return 192 * x**5 - 192 * x**3 + 36 * x

def dT7(x):
    return 448 * x**6 - 560 * x**4 + 168 * x**2 - 7

def dT8(x):
    return 1024 * x**7 - 1536 * x**5 + 640 * x**3 - 64 * x

def dT9(x):
    return 2304 * x**8 - 4032 * x**6 + 2160 * x**4 - 360 * x**2 + 9

def dT10(x):
    return 5120 * x**9 - 10240 * x**7 + 6720 * x**5 - 1600 * x**3 + 100 * x
# Legendre Polynomials from L0 to L20

def L0(x):
    return x*0+1

def L1(x):
    return x

def L2(x):
    return 0.5 * (3 * x**2 - 1)

def L3(x):
    return 0.5 * (5 * x**3 - 3 * x)

def L4(x):
    return (1 / 8) * (35 * x**4 - 30 * x**2 + 3)

def L5(x):
    return (1 / 8) * (63 * x**5 - 70 * x**3 + 15 * x)

def L6(x):
    return (1 / 16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)

def L7(x):
    return (1 / 16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)

def L8(x):
    return (1 / 128) * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)

def L9(x):
    return (1 / 128) * (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x)

def L10(x):
    return (1 / 256) * (46189 * x**10 - 109395 * x**8 + 90090 * x**6 - 30030 * x**4 + 3465 * x**2 - 63)

def L11(x):
    return x * (344.44921875 * x**10 - 902.12890625 * x**8 + 854.6484375 * x**6 - 351.9140625 * x**4 + 58.65234375 * x**2 - 2.70703125)

def L12(x):
    return (660.1943359375 * x**12 - 1894.470703125 * x**10 + 2029.7900390625 * x**8 - 997.08984375 * x**6 + 219.9462890625 * x**4 - 17.595703125 * x**2 + 0.2255859375)

def L13(x):
    return x * (1269.6044921875 * x**12 - 3961.166015625 * x**10 + 4736.1767578125 * x**8 - 2706.38671875 * x**6 + 747.8173828125 * x**4 - 87.978515625 * x**2 + 2.9326171875)

def L14(x):
    return (2448.52294921875 * x**14 - 8252.42919921875 * x**12 + 10893.2065429688 * x**10 - 7104.26513671875 * x**8 + 2368.08837890625 * x**6 - 373.90869140625 * x**4 + 21.99462890625 * x**2 - 0.20947265625)

def L15(x):
    return x * (4733.81103515625 * x**14 - 17139.6606445313 * x**12 + 24757.2875976563 * x**10 - 18155.3442382813 * x**8 + 7104.26513671875 * x**6 - 1420.85302734375 * x**4 + 124.63623046875 * x**2 - 3.14208984375)

def L16(x):
    return (9171.75888061523 * x**16 - 35503.5827636719 * x**14 + 55703.8970947266 * x**12 - 45388.3605957031 * x**10 + 20424.7622680664 * x**8 - 4972.98559570313 * x**6 + 592.022094726563 * x**4 - 26.707763671875 * x**2 + 0.196380615234375)

def L17(x):
    return x * (17804.002532959 * x**16 - 73374.0710449219 * x**14 + 124262.539672852 * x**12 - 111407.794189453 * x**10 + 56735.4507446289 * x**8 - 16339.8098144531 * x**6 + 2486.49279785156 * x**4 - 169.149169921875 * x**2 + 3.33847045898438)

def L18(x):
    return (34618.8938140869 * x**18 - 151334.021530151 * x**16 + 275152.766418457 * x**14 - 269235.502624512 * x**12 + 153185.717010498 * x**10 - 51061.905670166 * x**8 + 9531.55572509766 * x**6 - 888.033142089844 * x**4 + 31.7154693603516 * x**2 - 0.185470581054688)

def L19(x):
    return x * (67415.7405853271 * x**18 - 311570.044326782 * x**16 + 605336.086120605 * x**14 - 642023.121643066 * x**12 + 403853.253936768 * x**10 - 153185.717010498 * x**8 + 34041.2704467773 * x**6 - 4084.95245361328 * x**4 + 222.008285522461 * x**2 - 3.52394104003906)

def L20(x):
    return (131460.694141388 * x**20 - 640449.535560608 * x**18 + 1324172.68838882 * x**16 - 1513340.21530151 * x**14 + 1043287.57266998 * x**12 - 444238.579330444 * x**10 + 114889.287757874 * x**8 - 17020.6352233887 * x**6 + 1276.54764175415 * x**4 - 37.0013809204102 * x**2 + 0.176197052001953)


# Embedding

class Legendre_Embedding_Layer(nn.Module):   
    degree: int
    @nn.compact
    def __call__(self, X):
        T_funcs = [globals()[f"L{i}"] for i in range(self.degree+1)]
        C_n = self.param('c_i', nn.initializers.ones, (1, self.degree+1))  # 1,degree
        T_n= jnp.hstack([1 / (i + 1) * T_funcs[i](X) for i in range(self.degree + 1)])
        return C_n*T_n

class Fourier_Embedding(nn.Module):   
    degree: int
    def setup(self):
        self.k = jnp.arange(1,self.degree+1)
        self.L =2.0
        self.w = 2.0*jnp.pi/self.L
    @nn.compact
    def __call__(self, X):
        T_n=jnp.hstack([jnp.cos(self.k*self.w*X),jnp.sin(self.k*self.w*X),jnp.ones_like(X)])
        return T_n
class Fourier_Embedding_no_ones(nn.Module):   
    degree: int
    def setup(self):
        self.k = jnp.arange(1,self.degree+1)
        self.L =2.0
        self.w = 2.0*jnp.pi/self.L
    @nn.compact
    def __call__(self, X):
        T_n=jnp.hstack([jnp.cos(self.k*self.w*X),jnp.sin(self.k*self.w*X)])
        return T_n
class Random_Fourier_Embedding_univariate(nn.Module):
    degree: int 
    sigma: float = 10.0 

    @nn.compact
    def __call__(self, X):
        freqs = self.param(
            'frequencies',
            lambda key, shape, dtype: random.normal(key, shape, dtype),
            (1, self.degree), 
            jnp.float64
        )
        
        X_freqs = X @ (self.sigma * freqs)
        
        # Return the sine and cosine components
        T_n = jnp.concatenate([jnp.cos(X_freqs), jnp.sin(X_freqs)], axis=-1)
        return T_n


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


class Polynomial_Embedding_Layer(nn.Module):   
    degree: int
    step: int = 1  # Set default step to 1
    
    @nn.compact
    def __call__(self, X, polynomial_type='T'):
        # Create the list of functions based on the step size
        T_funcs = [globals()[f"{polynomial_type}{i}"] for i in range(0, self.degree+1, self.step)]
        C_n = self.param('c_i', nn.initializers.ones, (len(T_funcs)))  # Adjust size based on the step
        C_n_T_n = jnp.hstack([C_n[i] / (i + 1) * T_funcs[i](X) for i in range(len(T_funcs))])
        return C_n_T_n


class Fourier_Embedding_back(nn.Module):   
    degree: int

    def setup(self):
        self.k = jnp.arange(1, self.degree + 1)
        self.L = 2.0
        self.w = jnp.array(2.0 * jnp.pi / self.L)  # Explicitly initialize as a JAX array

    @nn.compact
    def __call__(self, X):
        X = X.reshape(-1, 1)  # Ensure compatibility for broadcasting
        cos_terms = jnp.cos(self.k * self.w * X)  # Now self.w has a shape and can broadcast
        sin_terms = jnp.sin(self.k * self.w * X)
        T_n = jnp.hstack([cos_terms, sin_terms, jnp.ones_like(X)])
        return T_n


# Derivatives:

def dL0(x):
    return x * 0

def dL1(x):
    return np.ones_like(x)

def dL2(x):
    return 3 * x

def dL3(x):
    return 0.5 * (15 * x**2 - 3)

def dL4(x):
    return (1 / 8) * (140 * x**3 - 60 * x)

def dL5(x):
    return (1 / 8) * (315 * x**4 - 210 * x**2 + 15)

def dL6(x):
    return (1 / 16) * (1386 * x**5 - 1260 * x**3 + 210 * x)

def dL7(x):
    return (1 / 16) * (3003 * x**6 - 3465 * x**4 + 945 * x**2 - 35)

def dL8(x):
    return (1 / 128) * (51480 * x**7 - 72072 * x**5 + 27720 * x**3 - 2520 * x)

def dL9(x):
    return (1 / 128) * (109395 * x**8 - 180180 * x**6 + 90090 * x**4 - 13860 * x**2 + 315)

def dL10(x):
    return (1 / 256) * (461890 * x**9 - 875160 * x**7 + 540540 * x**5 - 120120 * x**3 + 6930 * x)
