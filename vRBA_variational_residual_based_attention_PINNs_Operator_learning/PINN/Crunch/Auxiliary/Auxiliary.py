import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu
from jax import config
from jax.numpy import index_exp as index
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange, tqdm
import numpy as np0
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.tri as tri

import scipy.io as sio
import sys
import os

# Use double precision to generate data (due to GP sampling)
#https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/PI_DeepONet_DR.ipynb

def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

# A diffusion-reaction numerical solver
#https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/PI_DeepONet_DR.ipynb
def solve_ADR(key, Nx, Nt, P, length_scale,m):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01*np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: 0.01*u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function  
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = np.zeros((Nx, Nt))
    u = u.at[:,0].set(u0(x))
    #u = index_update(u, index[:,0], u0(x))
    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = u.at[1:-1, i + 1].set(np.linalg.solve(A, b1 + b2))
        #u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
        return u
    # Run loop
    UU = lax.fori_loop(0, Nt-1, body_fn, u)

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkeys[1], (P,2), 0, max(Nx,Nt))
    y = np.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = UU[idx[:,0], idx[:,1]]
    # x, t: sampled points on grid
    return (x, t, UU), (u, y, s)
# A diffusion-reaction numerical solver
#https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/PI_DeepONet_DR.ipynb
def solve_ADR_QR(key, Nx, Nt, P, length_scale,m,same_points=False):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01*np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: 0.01*u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function  
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = np.zeros((Nx, Nt))
    u = u.at[:,0].set(u0(x))
    #u = index_update(u, index[:,0], u0(x))
    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = u.at[1:-1, i + 1].set(np.linalg.solve(A, b1 + b2))
        #u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
        return u
    # Run loop
    UU = lax.fori_loop(0, Nt-1, body_fn, u)

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Generate subkeys
    if same_points:
        key=random.PRNGKey(1234)
    subkeys = random.split(key, 2)
    idx = random.randint(subkeys[1], (P,2), 0, max(Nx,Nt))
    XX, TT = np.meshgrid(x, t)
    y = np.concatenate([XX.flatten()[:,None], TT.flatten()[:,None]], axis = 1)
    s = UU.flatten()
    # x, t: sampled points on grid
    return (x, t, UU), (u, y, s)
    # Plots
def plot(X,T,f):
  fig = plt.figure(figsize=(7,5))
  plt.pcolor(X,T,f, cmap='rainbow')
  plt.colorbar()
  plt.xlabel('$x$')
  plt.ylabel('$t$')
  plt.title('$f(x,t)$')
  plt.tight_layout()

# Geneate training data corresponding to one input sample
def generate_one_training_data_DN(key, P, Q,Nx,Nt,length_scale,m):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx , Nt, P, length_scale,m)

    # Geneate subkeys
    subkeys = random.split(key, 4)

    # Training data for BC and IC
    u_train = np.tile(u, (P,1))
    y_train = y
    s_train = s

    # Sample collocation points
    x_r_idx= random.choice(subkeys[2], np.arange(Nx), shape = (Q,1))
    x_r = x[x_r_idx]
    t_r = random.uniform(subkeys[3], minval = 0, maxval = 1, shape = (Q,1))

    # Training data for the PDE residual
    '''For the operator'''
    u_r_train = np.tile(u, (Q,1))
    y_r_train = np.hstack([x_r, t_r])
    '''For the function'''
    f_r_train = u[x_r_idx]
    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train
# Geneate training data corresponding to one input sample
def generate_one_training_data_QR(key, P, Q,Nx,Nt,length_scale,m):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR_QR(key, Nx , Nt, P, length_scale,m,same_points=True)

    # Geneate subkeys
    subkeys = random.split(key, 4)

    # Training data for BC and IC
    u_train = u
    y_train = y
    s_train = s

    # Sample collocation points
    x_r_idx= random.choice(subkeys[2], np.arange(Nx), shape = (Q,1))
    x_r = x[x_r_idx]
    t_r = random.uniform(subkeys[3], minval = 0, maxval = 1, shape = (Q,1))

    # Training data for the PDE residual
    '''For the operator'''
    u_r_train = np.tile(u, (Q,1))
    y_r_train = np.hstack([x_r, t_r])
    '''For the function'''
    f_r_train = u[x_r_idx]
    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train



  # Geneate training data corresponding to one input sample (PHYSICS INFORMED)
def generate_one_training_data(key, P, Q,Nx,Nt,length_scale,m):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx , Nt, P, length_scale,m)

    # Geneate subkeys
    subkeys = random.split(key, 4)

    # Sample points from the boundary and the inital conditions
    # Here we regard the initial condition as a special type of boundary conditions
    x_bc1 = np.zeros((P // 3, 1))
    x_bc2 = np.ones((P // 3, 1))
    x_bc3 = random.uniform(key = subkeys[0], shape = (P // 3, 1))
    x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))

    t_bc1 = random.uniform(key = subkeys[1], shape = (P//3 * 2, 1))
    t_bc2 = np.zeros((P//3, 1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    # Training data for BC and IC
    u_train = np.tile(u, (P,1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P, 1))

    # Sample collocation points
    x_r_idx= random.choice(subkeys[2], np.arange(Nx), shape = (Q,1))
    x_r = x[x_r_idx]
    t_r = random.uniform(subkeys[3], minval = 0, maxval = 1, shape = (Q,1))

    # Training data for the PDE residual
    '''For the operator'''
    u_r_train = np.tile(u, (Q,1))
    y_r_train = np.hstack([x_r, t_r])
    '''For the function'''
    f_r_train = u[x_r_idx]
    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train

    # Geneate test data corresponding to one input sample
def generate_one_test_data(key, P,length_scale,m):
    Nx = P
    Nt = P
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx , Nt, P, length_scale,m)

    XX, TT = np.meshgrid(x, t)

    u_test = np.tile(u, (P**2,1))
    y_test = np.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test

# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.y = y # location
        self.s = s # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), )#replace=False)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

class DataGenerator_SNR(data.Dataset):
    def __init__(self, u, y, s, num_batch=100):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size_u = u.shape[0]//num_batch
        self.batch_size_y = y.shape[0]//num_batch
        self.batch_size_s = s.shape[0]//num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Calculate start and end indices for the batch
        start_idx_u = index * self.batch_size_u
        end_idx_u = (index+1) * self.batch_size_u
        start_idx_y = index * self.batch_size_y
        end_idx_y = (index+1) * self.batch_size_y
        start_idx_s = index * self.batch_size_s
        end_idx_s = (index+1) * self.batch_size_s

        # Slice the data arrays to get the batch
        s = self.s[start_idx_s:end_idx_s, :]
        y = self.y[start_idx_y:end_idx_y, :]
        u = self.u[start_idx_u:end_idx_u, :]

        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

    def __len__(self):
        'Denotes the total number of batches'
        return (self.N + self.batch_size - 1) // self.batch_size

def flatten_parameters(params,QR=False):
    """
    Flatten the branch and trunk network parameters into a single list of NumPy arrays.

    :param branch_params: List of lists of NumPy arrays for the branch network
    :param trunk_params: List of lists of NumPy arrays for the trunk network
    :return: List of NumPy arrays containing all parameters
    """
    if QR:
        branch_params, trunk_params,A=params
    else:
        branch_params, trunk_params=params
    flattened = []
    for layer in branch_params:
        flattened.extend(layer)
    for layer in trunk_params:
        flattened.extend(layer)
    if QR:
        flattened.extend(A)
    return flattened
def reconstruct_parameters(flattened, N_layers_branch, N_layers_trunk, QR=False):
    """
    Reconstruct the original structure of branch and trunk network parameters from a flattened list.
    Also reconstructs the matrix A if QR is True.

    :param flattened: Flattened list of NumPy arrays containing all parameters
    :param N_layers_branch: Number of layers in the branch network
    :param N_layers_trunk: Number of layers in the trunk network
    :param QR: Flag indicating whether the additional matrix A is included
    :return: Tuple of lists (branch_params, trunk_params) and optionally the matrix A
    """
    branch_params = [flattened[i:i + 2] for i in range(0, 2 * N_layers_branch, 2)]
    trunk_start = 2 * N_layers_branch
    trunk_params = [flattened[i:i + 2] for i in range(trunk_start, trunk_start + 2 * N_layers_trunk, 2)]

    if QR:
        A = flattened[-1]  # Assuming A is the last element in the flattened list
        params = (branch_params, trunk_params, A)
    else:
        params = (branch_params, trunk_params)

    return params

def save_flattened_params(flattened_params, filename):
    """
    Save the flattened list of NumPy arrays to an .npz file.

    :param flattened_params: The flattened list of NumPy arrays
    :param filename: The filename for the .npz file
    """
    # Create a dictionary where each array is assigned a unique key
    arrays_dict = {f'arr_{i}': arr for i, arr in enumerate(flattened_params)}
    
    # Save the arrays to an .npz file
    np.savez(filename, **arrays_dict)

def load_flattened_params(filename):
    """
    Load the flattened list of NumPy arrays from an .npz file.

    :param filename: The filename of the .npz file to load
    :return: List of NumPy arrays
    """
    with np.load(filename) as data:
        # Extract all arrays from the .npz file and store them in a list
        arrays = [data[f'arr_{i}'] for i in range(len(data))]
    return arrays

def vtu_to_npy(data="",id_data=0):
    #Choose the vtu file
    # Read the source file.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(data)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    num_of_points = reader.GetNumberOfPoints()
    print(f"Number of Points: {num_of_points}")
    num_of_cells = reader.GetNumberOfCells()
    print(f"Number of Cells: {num_of_cells}")
    points = output.GetPoints()
    npts = points.GetNumberOfPoints()
    ## Each elemnts of x is list of 3 float [xp, yp, zp]
    x = vtk_to_numpy(points.GetData())
    print(f"Shape of point data:{x.shape}")

    ## Field value Name:
    n_arrays = reader.GetNumberOfPointArrays()
    num_of_field = 0 
    field = []
    for i in range(n_arrays):
        f = reader.GetPointArrayName(i)
        field.append(f)
        print(f"Id of Field: {i} and name:{f}")
        num_of_field += 1 
    print(f"Total Number of Field: {num_of_field}")
    u = vtk_to_numpy(output.GetPointData().GetArray(id_data))
    print(f"Shape of field: {np.shape(u)}")
    print('u: ', u.shape)
    print('x: ', x.shape)
    print(np.min(u), np.max(u))
    return x,u

def process_uneven_data(X,Y,V):
    n_x=np.unique(X).shape[0]
    n_y=np.unique(Y).shape[0]
    xi = np.linspace(np.min(X), np.max(X), n_x)
    yi = np.linspace(np.min(Y), np.max(Y), n_y)
    triang = tri.Triangulation(X, Y)
    interpolator = tri.LinearTriInterpolator(triang, V)
    x, y = np.meshgrid(xi, yi)
    Vi = interpolator(x, y)
    return x,y,Vi

def plot_losses_grid(log_loss,num_cols=3,fig_h=16,fig_v=12):
    
    titles = list(log_loss[0].keys())
    
    # Make sure the subplot grid dimensions match the number of titles
    num_rows = (len(titles) + 2) // 3  # Add 2 to round up
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_h, fig_v))

    # Ensure axs is a 2D array
    if num_rows == 1:
        axs = np.array([axs])

    for i, title in enumerate(titles):
        row = i // 3
        col = i % 3
        ax = axs[row][col]

        # Extract values for a specific loss from all dictionary entries
        loss_values = [entry[title] for entry in log_loss]

        ax.plot(loss_values, label=title, color='k')
        ax.set_title(title)
        ax.set_yscale('log')  # set y-axis to log scale
        ax.grid(True, which="both", ls="--", c='0.65')

    for ax in axs[-1, :]:
        ax.set_xlabel('Iterations (10e2)')

    plt.tight_layout()