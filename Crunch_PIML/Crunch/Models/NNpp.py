 # Libraries

import numpy as np
from jax import jit, vmap, grad
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import Tuple

from typing import Tuple, List, Dict, Sequence
import h5py
# from Instant_AIV.models.metrics import *
# from Instant_AIV.manage.dataloader import *
import optax

#Initialization
from typing import Tuple
def select_model(WN=False, Mod_MLP=False, Adaptive_AF=False,Use_ResNet=False,Adaptive=False,Light=False,Network_type='mlp',degree=5):
    if Network_type.lower()=='mlp':
        model_map = {
            (True, True, True,False, False, False): FCN_WN_MMLP_Adaptive,
            (True, True, False,False, False, False): FCN_WN_MMLP,
            (True, False, True,False, False, False): FCN_WN_Adaptive,
            (True, False, False,False, False, False): FCN_WN,
            (False, True, True,False, False, False): FCN_MMLP_Adaptive,
            (False, True, False,False, False, False): FCN_MMLP,
            (False, False, True,False, False, False): FCN_Adaptive,
            (False, False, False,False, False, False): FCN,
            (False, False, False,True, False, False): ResNet,
            (False, False, False,True, False, True): ResNet_light,
            (True, False, False,True, False, False): WN_ResNet,
            (True, False, False,True, False, True): WN_ResNet_light,
            (True, False, False,True, True, False): WN_ResNet_adaptive,
        }
        return model_map[(WN, Mod_MLP, Adaptive_AF,Use_ResNet,Adaptive,Light)]
    elif Network_type.lower()=='kan':
        if degree==3:
            return KAN_Net3
        elif degree==5:
            if Use_ResNet:
                return KAN5_ResNet
            else:
                return KAN_Net5
        elif degree==7:
            return KAN_Net7
        elif degree==9:
            return KAN_Net9
        elif degree==11:
            return KAN_Net11
        else:
            return KAN_Net
    elif Network_type.lower()=='kan_theta':
            return KAN_Net_theta

def initialize_optimizer(lr0, decay_rate, lrf, decay_step, T_e,optimizer_type='Adam',weight_decay=1e-5):
    print('Optimizer',optimizer_type.lower())
    if optimizer_type.lower()=='adam':
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adam(lr0), decay_step
        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            return optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate,)),decay_step
    elif optimizer_type.lower()=='adamw':
        print('Weight decay:',weight_decay)
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adamw(learning_rate=lr0, weight_decay=weight_decay), decay_step

        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.adamw(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step
    elif optimizer_type.lower()=='lion':
        if decay_rate == 0 or lrf == lr0:
            weight_decay=weight_decay*3
            print('No decay')
            return optax.lion(learning_rate=lr0, weight_decay=weight_decay), decay_step
        else:
            if decay_step == 0:
                weight_decay=weight_decay*3
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.lion(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step

def transfer_params(params_s,params_t,levels=[0,1,2]):
    for level in levels:
        params_t['params'][level]=params_s['params'][level]
    return params_t
