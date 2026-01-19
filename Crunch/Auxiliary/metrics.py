import time
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np

import optax

def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def relative_l2m(u, u_gt):
    return jnp.linalg.norm(u-u_gt,2) / jnp.linalg.norm(u_gt,2)

def relative_error2(pred,exact):
    return np.linalg.norm(exact-pred,2)/np.linalg.norm(exact,2)
    
def relative_l2_2(u, u_gt):
    return jnp.linalg.norm(u.flatten()-u_gt.flatten(),2) / jnp.linalg.norm(u_gt.flatten(),2)

def MAE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.abs(pred - exact))

def MSE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact))

def relative_error(pred,exact):
    return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
