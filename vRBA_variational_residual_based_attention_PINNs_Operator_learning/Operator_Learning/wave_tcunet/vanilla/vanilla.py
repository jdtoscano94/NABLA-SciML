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
from tcunet import Unet2D
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from scipy.stats import wasserstein_distance
import scipy

torch.manual_seed(23)

import pickle

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your custom loss function here
# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()

#     def forward(self, y_pred, y_true):
#         # Implement your custom loss calculation here
#         # loss = torch.mean((y_pred - y_true) ** 2)  # Example: Mean Squared Error
#         loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
#         # loss = torch.mean(torch.square(y_true - y_pred))
#         return loss

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, Par, Lambda=None):
        # Implement your custom loss calculation here
        if Lambda is not None:
            r_i =torch.absolute(y_true - y_pred)
            log_it = torch.log(torch.tensor(Par['it'] + 2.0, device=device))
            mu_r=Par['c_log']*torch.amax(r_i,dim=(2,3),keepdims=True)/log_it
            beta_it=1/(mu_r+1e-16)
            q_it = torch.exp(beta_it * r_i)
            lambda_it=Par['phi']*q_it/torch.amax(q_it,dim=(2,3),keepdims=True)+(1-Par['phi'])
            Lambda = Par['gamma']*Lambda + Par['eta']*lambda_it
            #loss = torch.mean(torch.square(Lambda*r_i)) 
            loss = torch.norm(r_i, p=2)/torch.norm(y_true, p=2)
        
        else:
            loss = torch.mean(torch.square(y_true - y_pred)) 
            # loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)

        return loss, Lambda

class YourDataset(Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)

        return x_sample, t_sample, y_sample

class YourDataset_L(Dataset):
    def __init__(self, x, t, y, Lambda, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.Lambda = Lambda
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]
        Lambda_sample = self.Lambda[idx]

        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)

        return x_sample, t_sample, y_sample, Lambda_sample, idx


def preprocess(traj, Par):
    x = sliding_window_view(traj[:,:-(Par['lf']-1),:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lb'],Par['nx'], Par['ny'])
    y = sliding_window_view(traj[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    t = np.linspace(0,1,Par['lf']).reshape(-1,1)

    nt = y.shape[1]
    n_samples = y.shape[0]

    t = np.tile(t, [n_samples,1]).reshape(-1,)                                                     #[_*nt, ]
    x = np.repeat(x,nt, axis=0)                                   #[_*nt, 1, 64, 64]
    y = y.reshape(y.shape[0]*y.shape[1],1,y.shape[2],y.shape[3])  #[_*nt, 64, 64]


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
    mu  = np.mean(L_theta, axis=0) #[W,]
    sig = np.std(L_theta, axis=0)  #[W,]
    NUM = np.linalg.norm(mu, ord=2)
    DEN = np.linalg.norm(sig, ord=2)
    snr = NUM/DEN

    # Save MU and STD as well!
    # Do not use GradScaler for calculating SNR
    # Set gamma as gamma = 1 - eta

    if np.isnan(L_theta).any():
        print(f"Warning: NaN detected in gradients at L_theta")
    if np.isinf(L_theta).any():
        print(f"Warning: inf detected in gradients at L_theta")

    return snr
def get_gamma(eta, max_RBA):
    gamma_it = 1-eta/max_RBA
    return gamma_it

debug = False
# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
x = np.load('../data/x.npy')  #[_, 64, 64]
t = np.load('../data/t.npy')  #[_, 200]
y = np.load('../data/y.npy')  #[_, 64, 64, 200]

if debug:
    x = x[:100]
    y = y[:100]

idx1 = int(0.8 * x.shape[0])
idx2 = int(0.9 * x.shape[0])

traj = np.append( np.expand_dims(x, axis=-1), y, axis=-1 ).transpose(0,3,1,2) #[_, 64, 64, 201]

traj_train = traj[:idx1, ::4]
traj_val   = traj[idx1:idx2, ::4]
traj_test  = traj[idx2:, ::4]

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_train.shape[2]
Par['ny'] = traj_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 1
Par['lf'] = 51 
# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2
Par['num_epochs'] = 500
if debug:
    Par['num_epochs'] = 5 

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

# Par['eta']   = 0.1 #use 0.01 so that eta = 1-gamma
Par['eta']   = 0.1
Par['gamma'] = 0.99

Par['do_rba']  = True
Par['get_snr'] = True

Par['Lambda_max'] = Par['eta']/(1 - Par['gamma'])

Lambda = np.ones(y_train.shape, dtype=np.float32)*Par['Lambda_max']/2.0
print("Lambda: ", Lambda.shape)

print("Par: \n", Par)

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
Lambda_tensor  = torch.tensor(Lambda, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
t_val_tensor   = torch.tensor(t_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
t_test_tensor  = torch.tensor(t_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = YourDataset_L(x_train_tensor, t_train_tensor, y_train_tensor, Lambda_tensor)
val_dataset = YourDataset(x_val_tensor, t_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, t_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 100
val_batch_size   = 100
test_batch_size  = 100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = Unet2D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8)).to(device).to(torch.float32)
summary(model, input_size=((1,)+x_train.shape[1:], (1,))  )

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

# Training loop
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


num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)

max_RBA0=15
cap_RBA=25
gamma_it=get_gamma(Par['eta'], max_RBA0)
max_RBA=max_RBA0

step_RBA=(cap_RBA-max_RBA0)/(num_epochs/10-1)



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
    L_theta = []
    max_RBA=max_RBA0+step_RBA*epoch//10
    Par['gamma']=get_gamma(Par['eta'], max_RBA)
    for x, t, y_true, Lambda, indices in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        Par['it']=total_iterations
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            if Par['do_rba']:         
                loss, temp_Lambda = criterion(y_pred, y_true.to(device), Par, Lambda.to(device))
            else: 
                loss, _ = criterion(y_pred, y_true.to(device), Par)

        if Par['do_rba']:
            # print('og: ', train_loader.dataset.Lambda.device)
            # print('te: ', temp_Lambda.device)
            train_loader.dataset.Lambda[indices] = temp_Lambda.cpu().detach()#.numpy()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        total_iterations=  total_iterations+1
        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)
    if Par['get_snr']:
        residual_ls = []
        L_residual_ls = []
        L_theta = []
        wd_ls = [] # List for batch-wise Wasserstein distances

        # --- Define a sample size for the WD calculation ---
        wd_sample_size = 10000 

        for x, t, y_true, Lambda, indices in train_loader:
            optimizer.zero_grad()
            
            # 1. Flatten all weights in the batch into a single vector
            Lambda_flat = Lambda.flatten()
            
            # 2. Take one random sub-sample from this flattened vector
            if len(Lambda_flat) > wd_sample_size:
                indices_wd = torch.randperm(len(Lambda_flat))[:wd_sample_size]
                q_res_sampled = Lambda_flat[indices_wd].detach().cpu().numpy()
            else:
                q_res_sampled = Lambda_flat.detach().cpu().numpy()

            # 3. Create a corresponding uniform distribution
            p_res_sampled = np.ones_like(q_res_sampled)
            p_res_sampled /= p_res_sampled.sum()
            q_sum = q_res_sampled.sum()
            if q_sum > 0:
                q_res_sampled /= q_sum
            else:
                q_res_sampled = p_res_sampled
            
            # 4. Compute distance for the batch and append
            wd_ls.append(wasserstein_distance(p_res_sampled, q_res_sampled))
            # --- END OF FNO LOGIC ---

            # The rest of the diagnostic calculations remain the same
            sum_lambda = torch.sum(Lambda, dim=(2, 3), keepdims=True)
            q_it = Lambda.shape[2]*Lambda.shape[3]*(Lambda/(sum_lambda + 1e-16))
            q_it = q_it.flatten().detach().cpu().numpy()

            y_pred = model(x.to(device), t.to(device))
            
            # Variance calculation
            full_residual = (((y_true.detach().cpu().numpy() - y_pred.detach().cpu().numpy()))).reshape(-1,)
            residual_ls.append((full_residual)**2)
            L_residual_ls.append((q_it*full_residual)**2)            
            
            # SNR calculation
            loss, _ = criterion(y_pred, y_true.to(device), Par)
            loss.backward()
            
            flat_gradients = get_flat_gradients(model.parameters())
            L_theta.append(flat_gradients.cpu().numpy().reshape(1,-1))

        snr = get_snr(L_theta)
        residual_ls = np.concatenate(residual_ls)
        var_residual = np.var(residual_ls)
        
        # Final metric is the mean of batch Wasserstein distances
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
        for x, t, y_true in val_loader:
            with autocast():
                y_pred = model(x.to(device), t.to(device))
                loss= torch.norm(y_true.to(device)-y_pred, p=2)/torch.norm(y_true.to(device), p=2)
            val_loss += loss.item()

    val_loss /= len(val_loader)

        # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs},it:{total_iterations}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}, snr: {snr:.4e}, var(R): {var_residual:.4e}, var(L*R): {L_var_residual:.4e}, WD: {wd:.4e}, L_infty: {L_infty:.4e}')
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
    history['epochs'].append(epoch + 1)
print('Training finished.')

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, t, y_true in test_loader:
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            loss= torch.norm(y_true.to(device)-y_pred, p=2)/torch.norm(y_true.to(device), p=2)
        test_loss += loss.item()

test_loss /= len(test_loader)
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

# Save using your specific format
np.savez(metric_filename, 
         train_loss=np.array(history['train_loss']),
         val_loss=np.array(history['val_loss']),
         snr=np.array(history['snr']),
         var_residual=np.array(history['var_residual']),
         L_var_residual=np.array(history['L_var_residual']),
         L_infty_residual=np.array(history['L_infty_residual']),
         wd=np.array(history['wd']),
         max_rba=np.array(history['max_rba']),
         epochs=np.array(history['epochs'])
         )
