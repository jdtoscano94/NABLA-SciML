
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu, tanh
#from jax.config import config
from jax.numpy import index_exp as index
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange, tqdm
import numpy as np0
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import scipy.io as sio
import tqdm as tqdm
import sys
import os
# Define the neural net
class MLP(nn.Module):
    layers: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.layers[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.layers[-1])(x)
        return x

def init_A(rng_key, N,K):
    k1, k2 = random.split(rng_key)
    glorot_stddev = 1. / np.sqrt((N + K) / 2.)
    A= glorot_stddev * random.normal(k1, (N, K))
    return A
