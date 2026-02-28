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
from neuralop.models import TFNO3d
from scipy.stats import wasserstein_distance
import scipy


# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import pickle

torch.manual_seed(23)

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, Par, Lambda=None):
        # Implement your custom loss calculation here
        if Lambda is not None:
            r_i =torch.absolute(y_true - y_pred)
            with torch.no_grad():
                log_it = torch.log(torch.tensor(Par['it'] + 2.0, device=device))
                mu_r=Par['c_log']*torch.amax(r_i,dim=(2,3),keepdims=True)/log_it
                beta_it=1/(mu_r+1e-16)
                q_it = torch.exp(beta_it * r_i)
                lambda_it=Par['phi']*q_it/torch.amax(q_it,dim=(2,3),keepdims=True)+(1-Par['phi'])
                Lambda = Par['gamma']*Lambda + Par['eta']*lambda_it
            loss = torch.mean(torch.square(r_i)) 
        else:
            loss = torch.mean(torch.square(y_true - y_pred)) 
        return loss, Lambda

class YourDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample

class Normalizer():
    def __init__(self, shift, scale):
        self.shift = torch.tensor(shift, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
    
    def normalize(self, x):
        return (x - self.shift)/self.scale
    
    def renormalize(self, x):
        return x*self.scale + self.shift

def get_xyt_grid(nx=None, ny=None, nt=None, bot=[0, 0, 0], top=[1, 1, 1], dtype='float32',
                 x_arr=None, y_arr=None, t_arr=None, dt=0.):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (n_x, n_y, n_t, 3) array of grid points
    '''
    if x_arr is None:
        x_arr = np.linspace(bot[0], top[0], num=nx, endpoint=True)

    if y_arr is None:
        y_arr = np.linspace(bot[1], top[1], num=ny, endpoint=True)

    if t_arr is None:
        if dt is None:
            dt = (top[2] - bot[2]) / nt
        t_arr = np.linspace(bot[2] + dt, top[2], num=nt)

    x_grid, y_grid, t_grid = np.meshgrid(x_arr, y_arr, t_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    t_axis = np.ravel(t_grid)
    grid = np.stack([x_axis, y_axis, t_axis], axis=0).T

    x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    grid_space = np.stack([x_axis, y_axis], axis=0).T

    grids_dict = {'grid_x': x_arr, 
                  'grid_y': y_arr,
                  'grid_t': t_arr, 
                  'grid_space': grid_space, 
                  'grid': grid}
    for key in grids_dict.keys():
        grids_dict[key] = torch.tensor(grids_dict[key], dtype=eval('torch.' + dtype), device='cpu')
    return grids_dict



def _preprocess_fno(data_dict, Par):
    Grid_train = get_xyt_grid(Par['nx'], Par['ny'], Par['lf'], bot=[0, 0, 0], top=[1, 1, 1], dtype='float32')
    Grid_test  = get_xyt_grid(Par['nx'], Par['ny'], Par['lf'], bot=[0, 0, 0], top=[1, 1, 1], dtype='float32')
    
    data_dim = len(data_dict['x_train'].shape) - 2
    if data_dim != len(data_dict['x_test'].shape) - 2:
        raise ValueError('Data dimension mismatch between train and test sets.' +
                            f'Train: {data_dim}, Test: {len(data_dict["x_test"].shape)}')

    time_steps_train = Par['lf'] #self.configs.time_steps_train
    time_steps_val = Par['lf'] #self.configs.time_steps_inference if self.configs.scenario == 'hypersonics' \
    time_steps_test = Par['lf'] #self.configs.time_steps_inference
    repeat_shape_train = [1, time_steps_train] + [1] * data_dim
    repeat_shape_val = [1, time_steps_val] + [1] * data_dim
    repeat_shape_test = [1, time_steps_test] + [1] * data_dim

    data_dict['x_train'] = data_dict['x_train'].repeat(repeat_shape_train)
    data_dict['x_val'] = data_dict['x_val'].repeat(repeat_shape_val)
    data_dict['x_test'] = data_dict['x_test'].repeat(repeat_shape_test)

  

    for dataset in ['x_train', 'x_val', 'x_test']:
        n_samples = data_dict[dataset].shape[0]
        if True:
            grid = Grid_test if dataset == 'x_test' else Grid_train
            time_steps = time_steps_test if dataset == 'x_test' else time_steps_train

        grid_t = grid['grid_t'].reshape([1, time_steps] + [1] * data_dim)
        grid_t = grid_t.repeat(
            [n_samples, 1] + list(data_dict[dataset].shape[2:]))
        if data_dim == 1:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3])
            grid_x = grid['grid_x'].reshape(
                x_shape).repeat(n_samples, time_steps, 1)
            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_t], dim=1)
        elif data_dim == 2:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1]
            y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:])

            grid_x = grid['grid_x'].reshape(x_shape).repeat(
                n_samples, time_steps, 1, y_shape[-1])
            grid_y = grid['grid_y'].reshape(y_shape).repeat(
                n_samples, time_steps, x_shape[-2], 1)

            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_y, grid_t], dim=1)
        elif data_dim == 3:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1, 1]
            y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:4]) + [1]
            z_shape = [1, 1, 1, 1] + list(data_dict[dataset].shape[4:])
            grid_x = grid['grid_x'].reshape(x_shape)
            grid_y = grid['grid_y'].reshape(y_shape)
            grid_z = grid['grid_z'].reshape(z_shape)

            grid_x = grid_x.repeat(
                n_samples, time_steps, 1, y_shape[-2], z_shape[-1])
            grid_y = grid_y.repeat(
                n_samples, time_steps, x_shape[-3], 1, z_shape[-1])
            grid_z = grid_z.repeat(
                n_samples, time_steps, x_shape[-3], y_shape[-2], 1)

            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_y, grid_z, grid_t], dim=1)
    return data_dict


def preprocess(traj, Par):
    x = sliding_window_view(traj[:,:-(Par['lf']-1),:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lb'],Par['nx'], Par['ny'])
    y = sliding_window_view(traj[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    t = np.linspace(0,1,Par['lf']).reshape(-1,1)

    nt = y.shape[1]
    n_samples = y.shape[0]

    print('x: ', x.shape)
    print('y: ', y.shape)
    print('t: ', t.shape)
    print()
    return x,y,t


def get_flat_gradients(param_tensors):
    grad_list = []
    for p in param_tensors:
        if p.grad is not None:
            grad_list.append(p.grad.view(-1))
    flat_gradients = torch.cat(grad_list)
    return flat_gradients


def get_snr(L_theta_ls):
    L_theta = np.concatenate(L_theta_ls, axis=0) #[NB, W]
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


def KL_divergence_torch(p, q):
    """
    Computes the Kullback-Leibler divergence between two probability distributions.

    Args:
        p (torch.Tensor): The first probability distribution (target).
        q (torch.Tensor): The second probability distribution (approximation).
                          Must have the same shape as p.

    Returns:
        torch.Tensor: The KL divergence (a scalar).
    """
    # Ensure both tensors have the same shape
    if p.shape != q.shape:
        raise ValueError(f"Input tensors p and q must have the same shape. Got p: {p.shape}, q: {q.shape}")

    # Avoid log of zero by adding a small epsilon
    epsilon = 1e-16
    p = p + epsilon
    q = q + epsilon

    term1 = p * np.log(p)
    term2 = p * np.log(q)
    return np.sum(term1 - term2)

#########################
x = np.load('../data/x.npy')  #[_, 64, 64]
t = np.load('../data/t.npy')  #[_, 200]
y = np.load('../data/y.npy')  #[_, 64, 64, 200]

traj = np.append( np.expand_dims(x, axis=-1), y, axis=-1 ).transpose(0,3,1,2) #[_, 64, 64, 201]

traj_train = traj[:800, ::4]
traj_val   = traj[800:900, ::4]
traj_test  = traj[900:, ::4]

print()

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_train.shape[2]
Par['ny'] = traj_train.shape[3]
Par['nf'] = 1
Par['lb'] = 1
Par['lf'] = 51 
# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2

Par['num_epochs'] = 500

print('\nTrain Dataset')
x_train, y_train, t_train = preprocess(traj_train, Par)
print('\nValidation Dataset')
x_val, y_val, t_val  = preprocess(traj_val, Par)
print('\nTest Dataset')
x_test, y_test, t_test  = preprocess(traj_test, Par)

t_min = np.min(t_train)
t_max = np.max(t_train)

Par['inp_shift'] = np.mean(x_train) 
Par['inp_scale'] = np.std(x_train)
Par['out_shift'] = np.mean(y_train)
Par['out_scale'] = np.std(y_train)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min

Par['eta']   = 0.1
Par['gamma'] = 0.99

Par['do_rba']  = True
Par['get_snr'] = True

Par['Lambda_max'] = Par['eta']/(1 - Par['gamma'])

Lambda = np.ones(y_train.shape, dtype=np.float32)*Par['Lambda_max']/2.0
print("Lambda: ", Lambda.shape)

inp_normalizer = Normalizer(Par['inp_shift'], Par['inp_scale'])
out_normalizer = Normalizer(Par['out_shift'], Par['out_scale'])

Par['model'] = 'FNO'

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

#########################

# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Lambda_tensor = torch.tensor(Lambda, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
t_val_tensor   = torch.tensor(t_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
t_test_tensor  = torch.tensor(t_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

data_dict = {'x_train':x_train_tensor, 'x_val':x_val_tensor, 'x_test':x_test_tensor}
data_dict = _preprocess_fno(data_dict, Par)

x_train_tensor = torch.tensor(data_dict['x_train'].cpu().numpy(), dtype=torch.float32)
x_val_tensor   = torch.tensor(data_dict['x_val'].cpu().numpy()  , dtype=torch.float32)
x_test_tensor  = torch.tensor(data_dict['x_test'].cpu().numpy() , dtype=torch.float32)

# Define data loaders
train_batch_size = 10
val_batch_size   = 10
test_batch_size  = 10

# Initialize your Unet2D model
if Par['model']=='FNO':
    model = TFNO3d(12, 12, 12, hidden_channels=20, in_channels=4, out_channels=1).cuda()
print('Created '+Par['model'])


# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * int(y_train.shape[0]/train_batch_size) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

snr = 0
wd = 0
history = {
    'train_loss': [], 
    'val_loss': [], 
    'snr': [], 
    'var_residual': [], 
    'L_var_residual': [], 
    'L_infty_residual': [], 
    'wd': [], 
    'max_rba': [],
    'lr': [], 
    'epochs': []
}
os.makedirs('results', exist_ok=True)

os.makedirs('models', exist_ok=True)




all_errors=[]
all_its=[]
all_Var_q=[]
all_Var_p=[]
all_KL=[]
all_SNR_g=[]
Par['c_log']=1.0
Par['phi']=0.9
total_iterations=0
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0
    counter = 0
    L_theta = []

    for start in range(0, x_train.shape[0]-1, train_batch_size):
        end = start + train_batch_size
        x = x_train_tensor[start:end]
        y_true = y_train_tensor[start:end]  
        Lambda = Lambda_tensor[start:end]
        optimizer.zero_grad()
        #with autocast():
        x = inp_normalizer.normalize(x)
        y_pred  = out_normalizer.renormalize(model(x.to(device))[:,0])
        Par['it']=total_iterations
        # Update cap_RBA
        if Par['do_rba']:         
            loss, temp_Lambda = criterion(y_pred, y_true.to(device), Par, Lambda.to(device))
        else: 
            loss, _ = criterion(y_pred, y_true.to(device), Par)

        if Par['do_rba']:
            Lambda_tensor[start:end] = temp_Lambda.detach()
        loss.backward()#scaler.scale(loss).backward()
        optimizer.step()#scaler.step(optimizer)
        train_loss += loss.item()
        counter += 1
        total_iterations=  total_iterations+1

        # Update learning rate
        scheduler.step()

    train_loss /= counter

    if Par['get_snr']:
            residual_ls = []
            L_residual_ls = []
            L_theta = []
            
            # --- MODIFICATION: Initialize list for averaged WD ---
            wd_ls = []
            
            # --- MODIFICATION: Set a sample size for performance ---
            wd_sample_size = 10000 

            for start in range(0, x_train.shape[0]-1, train_batch_size):
                end = start + train_batch_size
                x = x_train_tensor[start:end]
                y_true = y_train_tensor[start:end]
                optimizer.zero_grad()
                
                Lambda = Lambda_tensor[start:end]
                
                # --- MODIFICATION: Fast, Sub-sampled WD Calculation ---
                Lambda_flat = Lambda.flatten()
                if len(Lambda_flat) > wd_sample_size:
                    indices = torch.randperm(len(Lambda_flat))[:wd_sample_size]
                    q_res_sampled = Lambda_flat[indices].detach().cpu().numpy()
                else:
                    q_res_sampled = Lambda_flat.detach().cpu().numpy()

                p_res_sampled = np.ones_like(q_res_sampled)
                p_res_sampled /= p_res_sampled.sum()
                q_sum = q_res_sampled.sum()
                if q_sum > 0:
                    q_res_sampled /= q_sum
                else:
                    q_res_sampled = p_res_sampled
                wd_ls.append(wasserstein_distance(p_res_sampled, q_res_sampled))
                # --- END MODIFICATION ---

                # --- UNCHANGED ORIGINAL LOGIC (Now Correct) ---
                sum_lambda = torch.sum(Lambda, dim=(2, 3), keepdims=True)
                q_it = Lambda.shape[2]*Lambda.shape[3]*(Lambda/(sum_lambda + 1e-16))
                q_it = q_it.flatten().detach().cpu().numpy()
                
                x = inp_normalizer.normalize(x) # Your correct normalization
                y_pred  = out_normalizer.renormalize(model(x.to(device))[:,0])

                residual = (((y_true.detach().cpu().numpy() - y_pred.detach().cpu().numpy()))).reshape(-1,)

                residual_ls.append((residual)**2)
                L_residual_ls.append((q_it*residual)**2)

                loss, _ = criterion(y_pred, y_true.to(device), Par, Lambda.to(device))
                loss.backward()
                flat_gradients = get_flat_gradients(model.parameters())
                L_theta.append(flat_gradients.cpu().numpy().reshape(1,-1))

            snr = get_snr(L_theta)
            residual_ls = np.concatenate(residual_ls)
            var_residual = np.var(residual_ls)
            
            # --- MODIFICATION: Final WD is the mean of batch WDs ---
            wd = np.mean(wd_ls)
            L_infty=np.max(residual_ls)
            L_residual_ls = np.concatenate(L_residual_ls)
            L_var_residual = np.var(L_residual_ls)

    else:
            snr=0
            wd=0

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for start in range(0, x_val.shape[0]-1, val_batch_size):
            end = start + val_batch_size
            x = x_val_tensor[start:end]
            y_true = y_val_tensor[start:end]  
            x = inp_normalizer.normalize(x)
            y_pred = out_normalizer.renormalize(model(x.to(device))[:,0]) 
            loss= torch.norm(y_true.to(device)-y_pred, p=2)/torch.norm(y_true.to(device), p=2)
            val_loss += loss.item()

    val_loss /= int(y_val.shape[0]/val_batch_size)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs},it:{total_iterations} Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}, snr: {snr:.4e}, var(R): {var_residual:.4e}, var(L*R): {L_var_residual:.4e}, WD: {wd:.4e}, L_infty: {L_infty:.4e}')
    #Save iteration metrics
    all_errors.append(val_loss)
    all_its.append(epoch+1)
    all_Var_q.append(var_residual)
    all_Var_p.append(L_var_residual)
    all_KL.append(wd)
    all_SNR_g.append(snr)
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

model.eval()
test_loss = 0.0
with torch.no_grad():
    for start in range(0, x_test.shape[0]-1, test_batch_size):
        end = start + test_batch_size
        x = x_test_tensor[start:end]
        y_true = y_test_tensor[start:end]
        x = inp_normalizer.normalize(x)
        y_pred = out_normalizer.renormalize(model(x.to(device))[:,0]) 
        loss= torch.norm(y_true.to(device)-y_pred, p=2)/torch.norm(y_true.to(device), p=2)
        test_loss += loss.item()

test_loss /= int(y_test.shape[0]/test_batch_size)
print(f'Test Loss: {test_loss:.4e}')

results_dict = {
    'all_errors': all_errors,
    'all_its': all_its,
    'all_Var_q': all_Var_q,
    'all_Var_p': all_Var_p,
    'all_KL': all_KL,
    'all_SNR_g': all_SNR_g,
}

# Save dictionary as a .mat file
scipy.io.savemat('Log_files.mat', results_dict)

metric_filename = f'results/metrics_Vanilla.npz'

np.savez(metric_filename, 
         train_loss=np.array(history['train_loss']),
         val_loss=np.array(history['val_loss']),
         snr=np.array(history['snr']),
         var_residual=np.array(history['var_residual']),
         L_var_residual=np.array(history['L_var_residual']),
         L_infty_residual=np.array(history['L_infty_residual']),
         wd=np.array(history['wd']),
         lr=np.array(history['lr']),
         epochs=np.array(history['epochs'])
         )

print(f"All printed metrics saved to {metric_filename}")