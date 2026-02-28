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
            log_it = torch.log(torch.tensor(Par['it'] + 2.0, device=device))
            mu_r=Par['c_log']*torch.amax(r_i,dim=(1),keepdims=True)/log_it
            beta_it=1/(mu_r+1e-16)
            q_it = torch.exp(beta_it * r_i)
            lambda_it=Par['phi']*q_it/torch.amax(q_it,dim=(1),keepdims=True)+(1-Par['phi'])
            Lambda = Par['gamma']*Lambda + Par['eta']*lambda_it
            loss = torch.mean(torch.square(r_i)) 
        
        else:
            loss = torch.mean(torch.square(y_true - y_pred)) 

        return loss, Lambda

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
       'get_snr'       : True
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
Par['c_log']=1.0
Par['phi']=0.9
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

    for start in range(0, X_func_train.shape[0]-1, train_batch_size):
        end = start + train_batch_size
        x = X_func_train_tensor[start:end]
        y_true = y_train_tensor[start:end]  
        Lambda = Lambda_tensor[start:end]   

        optimizer.zero_grad()
        y_pred = model(x.to(device), X_loc_train_tensor.to(device))
        Par['it']=total_iterations
        loss, temp_Lambda   = criterion(y_pred, y_true.to(device), Par, Lambda.to(device))
        # print(temp_Lambda.detach())
        Lambda_tensor[start:end] = temp_Lambda.detach()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        counter += 1            
        total_iterations=  total_iterations+1
        # Update learning rate
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
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, ,it:{total_iterations} ,Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}, snr: {snr:.4e}, var(R): {var_residual:.4e}, var(L*R): {L_var_residual:.4e}, WD: {wd:.4e},L_infty: {L_infty:.4e}')

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
