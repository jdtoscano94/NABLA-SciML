import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from don import DeepONet
from scipy.interpolate import interp1d
from torchinfo import summary
from scipy.stats import wasserstein_distance
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse


# Initialize Argument Parser
parser = argparse.ArgumentParser(description="BGD")

# vRBA Potential Methods
parser.add_argument('--weighting_potential', type=str, default='exponential', 
                    choices=['exponential', 'cosh', 'lp', 'quadratic', 'linear', 'logarithmic', 'sublinear', 'superexp'],
                    help='Potential function for spatial/latent weighting')
args = parser.parse_args()
args.sampling_potential=args.weighting_potential
# Set Device based on Argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Weighting Method: {args.weighting_potential}")


import pickle
torch.manual_seed(23)

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def get_exact_epsilon_torch(r_i, reduction_dims, potential_type, n_newton=20, eps_floor=1e-8):
    """
    Helper: Solves for the exact epsilon satisfying the variational constraint.
    - Handles arbitrary reduction dimensions (spatial/temporal).
    - Uses 20 Newton-Raphson iterations for high precision.
    - Safe for Float32/Mixed Precision.
    """
    device = r_i.device
    
    # Calculate N (number of points involved in the mean reduction)
    N = 1
    for d in reduction_dims:
        N *= r_i.shape[d]
    
    # Safety initialization: Max value for scaling
    r_max = torch.amax(r_i, dim=reduction_dims, keepdim=True)
    r_max_safe = torch.maximum(r_max, torch.tensor(eps_floor, device=device))
    
    # 1. Initialization (Asymptotic Approximations)
    if potential_type == 'cosh':
        # Approx: eps ~ r_max / ln(2N)
        denom = torch.log(torch.tensor(2.0 * N, device=device)) + eps_floor
        eps = r_max_safe / denom
        
    elif potential_type == 'superexp':
        # Approx: eps ~ r_max / sqrt(ln N)
        denom = torch.sqrt(torch.log(torch.tensor(float(N), device=device)) + eps_floor)
        eps = r_max_safe / denom

    elif potential_type == 'logarithmic':
        # Approx: eps ~ mean(r) / (e - 1)
        # We use the mean of the residuals as a robust starting point
        r_mean = torch.mean(r_i, dim=reduction_dims, keepdim=True)
        eps = r_mean / (2.718281828 - 1.0)
        
    else:
        return torch.ones_like(r_max)

    # 2. Newton-Raphson Loop
    for _ in range(n_newton):
        # Clamp eps to prevent division by zero
        eps_safe = torch.maximum(eps, torch.tensor(eps_floor, device=device))
        u = r_i / eps_safe
        
        if potential_type == 'cosh':
            # Target: mean(sinh(u)) - 1 = 0
            val = torch.mean(torch.sinh(u), dim=reduction_dims, keepdim=True) - 1.0
            grad = torch.mean(torch.cosh(u) * (-u / eps_safe), dim=reduction_dims, keepdim=True)
            
        elif potential_type == 'superexp':
            # Target: mean(u * exp(u^2)) - 1 = 0
            exp_u2 = torch.exp(u**2)
            term = u * exp_u2
            
            val = torch.mean(term, dim=reduction_dims, keepdim=True) - 1.0
            
            # Derivative: -(1/eps) * term * (1 + 2u^2)
            d_term = -(1.0 / eps_safe) * term * (1.0 + 2.0 * u**2)
            grad = torch.mean(d_term, dim=reduction_dims, keepdim=True)

        elif potential_type == 'logarithmic':
            # Target: mean(log(u + 1)) - 1 = 0
            # This enforces the Shifted Entropy constraint
            val = torch.mean(torch.log(u + 1.0), dim=reduction_dims, keepdim=True) - 1.0
            
            # Derivative: d/deps [log(r/eps + 1)] = -1/eps * u/(u+1)
            d_term = -(1.0 / eps_safe) * u / (u + 1.0)
            grad = torch.mean(d_term, dim=reduction_dims, keepdim=True)

        else:
            val, grad = 0.0, 1.0

        # Update (Standard Newton Step)
        eps = eps - val / (grad - eps_floor)

    # Final Output Clamp
    return torch.maximum(eps, torch.tensor(eps_floor, device=device))


def update_spatial_weights(R, Lambda_batch, Par, it):
    """
    PyTorch vRBA spatial weight update.
    """
    # Extract Hyperparameters
    gamma = Par['gamma']
    eta = Par['eta']
    phi = Par.get('phi', 1.0)
    c_log = Par.get('c_log', 1.0)
    potential_type = Par.get('weighting_potential', 'exponential')
    p_val = Par.get('p_val', 4.0)

    # Detach & Abs
    r_i = torch.abs(R).detach()
    # reduction_dims example: if r_i is (Batch, T, X), dims are (1, 2)
    reduction_dims = tuple(range(1, r_i.ndim))
    
    device = r_i.device
    eps_floor = 1e-6
    
    # 2. Compute Unnormalized Weights (q_it)
    if potential_type == 'exponential':
        r_max = torch.amax(r_i, dim=reduction_dims, keepdim=True)
        log_k = torch.log(torch.tensor(it + 2.0, device=device))
        epsilon_q = c_log * r_max / log_k
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = beta_it * torch.exp(beta_it * r_i)
        
    elif potential_type == 'cosh':
        epsilon_q = get_exact_epsilon_torch(r_i, reduction_dims, 'cosh', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = torch.sinh(beta_it * r_i)

    elif potential_type == 'superexp':
        epsilon_q = get_exact_epsilon_torch(r_i, reduction_dims, 'superexp', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        u = beta_it * r_i
        q_it = u * torch.exp(u**2)

    elif potential_type == 'logarithmic':
        # NEW: Exact Newton Solver for Logarithmic
        # Solves E[ln(r/eps + 1)] = 1
        epsilon_q = get_exact_epsilon_torch(r_i, reduction_dims, 'logarithmic', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        # Safe Weight: log(beta * r + 1)
        # This guarantees q_it >= 0 (No Radon-Nikodym violation)
        q_it = torch.log(beta_it * r_i + 1.0)

    elif potential_type == 'logarithmic_simple':
        # Legacy/Fast Heuristic
        r_mean = torch.mean(r_i, dim=reduction_dims, keepdim=True)
        epsilon_q = torch.maximum(r_mean, torch.tensor(eps_floor, device=device)) / (2.71828 - 1.0)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = torch.log(beta_it * r_i + 1.0)

    elif potential_type == 'lp':
        q_it = torch.pow(r_i, p_val - 1.0)
        
    elif potential_type == 'quadratic': 
        q_it = r_i

    elif potential_type == 'sublinear':
        q_it = torch.pow(r_i, 0.5)

    elif potential_type == 'linear': 
        q_it = torch.ones_like(r_i)

    else:
        print('Error! Potential not available')
        q_it = torch.ones_like(r_i)

    # 3. Normalize (by Max) and Mix
    q_max = torch.amax(q_it, dim=reduction_dims, keepdim=True)
    q_normalized = q_it / (q_max + 1e-20)
    lambda_it = phi * q_normalized + (1.0 - phi)
    
    # 4. Update
    new_Lambda = gamma * Lambda_batch + eta * lambda_it
    return new_Lambda


def update_function_pdf(Lambda_global, Par, it):
    """
    PyTorch vRBA function sampling PDF.
    """
    phi = Par.get('phi', 1.0)
    c_log = Par.get('c_log', 1.0)
    potential_type = Par.get('sampling_potential', 'exponential')
    p_val = Par.get('p_val', 4.0)

    # Sum spatial weights -> scalar score per sample
    # Result is (BatchSize, )
    reduction_dims_input = tuple(range(1, Lambda_global.ndim))
    lambda_scores = torch.sum(Lambda_global, dim=reduction_dims_input)
    
    # Reshape to (1, N_samples) so we can treat samples as "spatial points" for the solver
    r_i = lambda_scores.unsqueeze(0) 
    solver_dims = (1,) # Reduce over dimension 1 (samples)
    
    device = r_i.device
    eps_floor = 1e-6

    # 2. Compute Unnormalized Weights
    if potential_type == 'exponential':
        r_max = torch.amax(r_i, dim=solver_dims, keepdim=True)
        log_k = torch.log(torch.tensor(it + 2.0, device=device))
        epsilon_q = c_log * r_max / log_k
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = beta_it * torch.exp(beta_it * r_i)
        
    elif potential_type == 'cosh':
        epsilon_q = get_exact_epsilon_torch(r_i, solver_dims, 'cosh', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = torch.sinh(beta_it * r_i)

    elif potential_type == 'superexp':
        epsilon_q = get_exact_epsilon_torch(r_i, solver_dims, 'superexp', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        u = beta_it * r_i
        q_it = u * torch.exp(u**2)

    elif potential_type == 'logarithmic':
        # NEW: Exact Newton Solver for Logarithmic
        epsilon_q = get_exact_epsilon_torch(r_i, solver_dims, 'logarithmic', n_newton=20)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = torch.log(beta_it * r_i + 1.0)
        
    elif potential_type == 'logarithmic_simple':
        r_mean = torch.mean(r_i, dim=solver_dims, keepdim=True)
        epsilon_q = torch.maximum(r_mean, torch.tensor(eps_floor, device=device)) / (2.71828 - 1.0)
        beta_it = 1.0 / (epsilon_q + eps_floor)
        q_it = torch.log(beta_it * r_i + 1.0)

    elif potential_type == 'lp':
        q_it = torch.pow(r_i, p_val - 1.0)
        
    elif potential_type == 'quadratic': 
        q_it = r_i
        
    elif potential_type == 'linear': 
        q_it = torch.ones_like(r_i)

    else: 
        print('Error! Potential not available')
        q_it = torch.ones_like(r_i)

    # 3. Normalize to PDF
    q_it = q_it.squeeze(0) # Remove dummy batch dim
    q_max = torch.amax(q_it)
    q_normalized = q_it / (q_max + 1e-20)
    lambda_it = phi * q_normalized + (1.0 - phi)
    q_pdf = lambda_it / torch.sum(lambda_it)
    
    return q_pdf

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, Par, Lambda=None):
        # Implement your custom loss calculation here
        if Lambda is not None:
            r_i =torch.absolute(y_true - y_pred)
            updated_Lambda = update_spatial_weights(r_i, Lambda, Par, Par['it'])
            loss = torch.mean(torch.square(updated_Lambda*r_i)) 
            return loss, updated_Lambda
        else:
            loss = torch.mean(torch.square(y_true - y_pred)) 
            return loss, None

def preprocess(X_func, X_loc, y, Par):
    X_func = (X_func - Par['X_func_shift'])/Par['X_func_scale']
    X_loc  = (X_loc - Par['X_loc_shift'])/Par['X_loc_scale']
    y      = (y - Par['out_shift'])/(Par['out_scale'])
    print('X_func: ', X_func.shape)
    print('X_loc : ', X_loc.shape)
    print('y     : ', y.shape)

    return X_func, X_loc, y

    
def data_prep(dataset, m, npoints_output):
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m)) #[1500, m] 
    X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y      = R(np.ravel(X_loc)) #[1500, npoints_output] 

    return X_func, X_loc, y


def get_flat_gradients(param_tensors):
    grad_list = []
    for p in param_tensors:
        if p.grad is not None:
            grad_list.append(p.grad.view(-1))
    flat_gradients = torch.cat(grad_list)
    return flat_gradients

def get_snr(L_theta_ls):
    L_theta = np.vstack(L_theta_ls)
    L_theta = np.nan_to_num(L_theta, nan=0.0, posinf=1e12, neginf=-1e12)
    mu  = np.mean(L_theta, axis=0)
    sig = np.std(L_theta, axis=0)
    NUM = np.linalg.norm(mu, ord=2)
    DEN = np.linalg.norm(sig, ord=2)
    snr = NUM/DEN

    if np.isnan(L_theta).any():
        print(f"Warning: NaN detected in gradients at L_theta")
    if np.isinf(L_theta).any():
        print(f"Warning: inf detected in gradients at L_theta")

    return snr
def get_gamma(eta, max_RBA):
    gamma_it = 1-eta/max_RBA
    return gamma_it

    

# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################

debug = False

dataset = np.load('../data/0.1/res_1000.npz')

m = 200
npoints_output = 500

X_func, X_loc, y = data_prep(dataset, m, npoints_output) 

idx1 = int(0.8*X_func.shape[0])
idx2 = int(0.9*X_func.shape[0])

X_func_train = X_func[:idx1]
X_func_val   = X_func[idx1:idx2]
X_func_test  = X_func[idx2:]

X_loc_train = X_loc
X_loc_val   = X_loc 
X_loc_test  = X_loc

y_train = y[:idx1]
y_val   = y[idx1:idx2]
y_test  = y[idx2:]

Par = {
       'bn_res'        : X_func_train.shape[1],
       'tn_res'        : X_loc_train.shape[1],
       'ld'            : 100,   
       'X_func_shift'  : np.mean(X_func_train),
       'X_func_scale'  : np.std(X_func_train),
       'X_loc_shift'   : np.min(X_loc_train),
       'X_loc_scale'   : np.max(X_loc_train)-np.min(X_loc_train),
       'out_shift'     : np.mean(y_train),
       'out_scale'     : np.std(y_train),
       'eta'           : 0.1,
       'gamma'         : 0.99,
       'get_snr'       : True,
        'phi': 0.9,
        'c_log': 1.0,
        'p_val': 4.0,
        'weighting_potential' : args.weighting_potential,
        'sampling_potential'  : args.sampling_potential
       }

Par['Lambda_max'] = Par['eta']/(1 - Par['gamma'])


if debug:
    Par['num_epochs']  = 5
else:
    Par['num_epochs']  = 10000

print('\nTrain Dataset')
X_func_train, X_loc_train, y_train = preprocess(X_func_train, X_loc_train, y_train, Par)
Lambda = np.ones((y_train.shape[0], y_train.shape[1]), dtype=np.float32)*Par['Lambda_max']/2.0
print("Lambda: ", Lambda.shape)

print('\nValidation Dataset')
X_func_val, X_loc_val, y_val = preprocess(X_func_val, X_loc_val, y_val, Par)
print('\nTest Dataset')
X_func_test, X_loc_test, y_test = preprocess(X_func_test, X_loc_test, y_test, Par)

print('Par:\n', Par)

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
X_func_train_tensor = torch.tensor(X_func_train, dtype=torch.float32)
X_loc_train_tensor = torch.tensor(X_loc_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Lambda_tensor = torch.tensor(Lambda, dtype=torch.float32)

X_func_val_tensor = torch.tensor(X_func_val, dtype=torch.float32)
X_loc_val_tensor = torch.tensor(X_loc_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

X_func_test_tensor = torch.tensor(X_func_test, dtype=torch.float32)
X_loc_test_tensor = torch.tensor(X_loc_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

# Define data loaders
train_batch_size = 50
val_batch_size   = 50
test_batch_size  = 50

# Initialize your DeepONet model
model = DeepONet(Par).to(device).to(torch.float32)
summary(model, input_size=((1,)+X_func_train.shape[1:], X_loc_train.shape)  )

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * int(y_train.shape[0]/train_batch_size) )  # Adjust T_max as needed

# Training loop

history = {
    'train_loss': [],
    'val_loss': [],
    'snr': [],
    'var_residual': [],
    'L_var_residual': [],
    'L_infty_residual': [], 
    'wd': [],
    'lr': [],
    'epochs': []
}

os.makedirs('results', exist_ok=True)

max_RBA0=15
cap_RBA=25
gamma_it=get_gamma(Par['eta'], max_RBA0)
max_RBA=max_RBA0
num_epochs = Par['num_epochs']

step_RBA=(cap_RBA-max_RBA0)/(num_epochs/10-1)

total_iterations=0
best_val_loss = float('inf')
best_model_id = 0
all_KL=[]
all_SNR_g=[]
Par['it']=total_iterations
os.makedirs('models', exist_ok=True)
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0
    counter=0
    L_theta = []
    max_RBA=max_RBA0+step_RBA*epoch//10
    Par['gamma']=get_gamma(Par['eta'], max_RBA)
    if Par['get_snr']:
                # Initialize lists for all diagnostics
                residual_ls = []
                L_residual_ls = []
                wd_ls = []
                L_theta = []
                
                wd_sample_size = 10000 # Sample size for WD calculation

                for start in range(0, X_func_train.shape[0]-1, train_batch_size):
                    end = start + train_batch_size
                    x = X_func_train_tensor[start:end]
                    y_true = y_train_tensor[start:end]
                    
                    # Perform diagnostic calculations that don't need gradients
                    with torch.no_grad():
                        Lambda = Lambda_tensor[start:end]
                        y_pred_diag = model(x.to(device), X_loc_train_tensor.to(device))
                        
                        # --- Variance Calculation ---
                        residual = y_true.to(device) - y_pred_diag
                        residual_ls.append(residual.cpu().numpy())
                        
                        weighted_residual = Lambda.to(device) * residual
                        L_residual_ls.append(weighted_residual.cpu().numpy())

                        # --- MODIFIED: Wasserstein Distance using SciPy ---
                        # Flatten on CPU and sample
                        Lambda_flat = Lambda.cpu().numpy().flatten()
                        if len(Lambda_flat) > wd_sample_size:
                            q_res_sampled = np.random.choice(Lambda_flat, size=wd_sample_size, replace=False)
                        else:
                            q_res_sampled = Lambda_flat
                        
                        # Create the two probability mass functions (PMFs)
                        p_pmf = np.ones_like(q_res_sampled) / len(q_res_sampled)
                        
                        q_sum = q_res_sampled.sum()
                        q_pmf = (q_res_sampled / q_sum) if q_sum > 0 else p_pmf
                        
                        # Define the common support (values) for the distributions
                        value_range = np.arange(len(p_pmf))
                        
                        # Correctly call the SciPy function with values and weights
                        wd_ls.append(wasserstein_distance(value_range, value_range, u_weights=p_pmf, v_weights=q_pmf))

                    # --- UNCHANGED: Original code for SNR calculation ---
                    optimizer.zero_grad()
                    y_pred = model(x.to(device), X_loc_train_tensor.to(device))
                    loss, _ = criterion(y_pred, y_true.to(device), Par)
                    loss.backward()
                    raw_gradient_gpu = get_flat_gradients(model.parameters())
                    norm = torch.linalg.norm(raw_gradient_gpu)
                    normalized_gradient_gpu = raw_gradient_gpu / norm
                    L_theta.append(normalized_gradient_gpu.cpu().numpy())

                # --- Final computation of all diagnostics ---
                snr = get_snr(L_theta)
                var_residual = np.var(np.concatenate(residual_ls))
                L_var_residual = np.var(np.concatenate(L_residual_ls))
                L_infty=np.max(residual_ls)
                wd = np.mean(wd_ls)

    else:
                snr = 0
                var_residual = 0
                L_var_residual = 0
                wd = 0

    for _ in range(0, X_func_train.shape[0] // train_batch_size):
            # CORRECT: Define the sampling distribution from the entire Lambda_tensor
            q_pdf_tensor = update_function_pdf(Lambda_tensor, Par, total_iterations)
            q_pdf = q_pdf_tensor.cpu().numpy()
            
            # Sample a batch of indices
            idx_train = np.random.choice(
                a=X_func_train.shape[0],
                size=train_batch_size,
                replace=False,
                p=q_pdf
            )
            # Create the batch using the sampled indices
            x = X_func_train_tensor[idx_train]
            y_true = y_train_tensor[idx_train]
            lambda_batch = Lambda_tensor[idx_train]

            # --- Standard training step ---
            optimizer.zero_grad()
            y_pred = model(x.to(device), X_loc_train_tensor.to(device))
            
            loss, temp_Lambda = criterion(y_pred, y_true.to(device), Par, lambda_batch.to(device))

            # CORRECT: Update Lambda_tensor at the sampled indices
            Lambda_tensor[idx_train] = temp_Lambda.detach().cpu()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            counter += 1
            total_iterations += 1
            scheduler.step()

    train_loss /= counter


    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for start in range(0, X_func_val.shape[0]-1, val_batch_size):
            end = start + val_batch_size
            x = X_func_val_tensor[start:end]
            y_true = y_val_tensor[start:end]  
            y_pred = model(x.to(device), X_loc_val_tensor.to(device))
            loss = torch.norm(y_true.to(device)-y_pred, p=2)/torch.norm(y_true.to(device), p=2)
            loss = loss.item()
            val_loss += loss
    val_loss /= int(y_val.shape[0]/val_batch_size)

        # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        model_name = f'models/best_model_{args.weighting_potential}_{args.sampling_potential}.pt'        
        torch.save(model.state_dict(), model_name)
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, ,it:{total_iterations} ,Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}, snr: {snr:.4e}, var(R): {var_residual:.4e}, var(L*R): {L_var_residual:.4e}, WD: {wd:.4e}, L_infty: {L_infty:.4e}')
    # --- SAVE METRICS TO HISTORY ---
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['snr'].append(snr)
    history['var_residual'].append(var_residual)
    history['L_var_residual'].append(L_var_residual)
    history['L_infty_residual'].append(L_infty)
    history['wd'].append(wd)
    history['lr'].append(scheduler.get_last_lr()[0])
    history['epochs'].append(epoch + 1)
print('Training finished.')

# --- SAVE METRICS TO NPZ ---
metric_filename = f'results/metrics_{args.weighting_potential}_{args.sampling_potential}.npz'

np.savez(metric_filename, 
         train_loss=np.array(history['train_loss']),
         val_loss=np.array(history['val_loss']),
         snr=np.array(history['snr']),
         var_residual=np.array(history['var_residual']),
         L_var_residual=np.array(history['L_var_residual']),
         L_infty_residual=np.array(history['L_infty_residual']),
         wd=np.array(history['wd']),
         lr=np.array(history['lr']),
         epochs=np.array(history['epochs']),
         best_model_id=best_model_id
         )

print(f"Metrics saved to {metric_filename}")
# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for start in range(0, X_func_test.shape[0]-1, test_batch_size):
        end = start + test_batch_size
        x = X_func_test_tensor[start:end]
        y_true = y_test_tensor[start:end]  

        y_pred = model(x.to(device), X_loc_test_tensor.to(device))
        loss, _ = criterion(y_pred, y_true.to(device), Par)
        loss = loss.item()
        test_loss += loss 
test_loss /= int(y_test.shape[0]/test_batch_size)
print(f'Test Loss: {test_loss:.4e}')
