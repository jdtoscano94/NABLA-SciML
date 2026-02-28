# Libraries
import numpy as np
# from Instant_AIV.manage.plots import *
from Crunch.Auxiliary.metrics import  *
import cv2
import os
import tqdm
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.tri as tri
import jax.numpy as jnp
import jax

def save_list(Loss,path,name='loss-'):
    filename=path+name+".npy"
    np.save(filename, np.array(Loss))
    
def create_save_name(Eqn, Mode, use_RBA=False, Mod_MLP=False, Adaptive_AF=False, Weight_Norm=False,resample=False,batch_size=1):
    save_name = f'{Eqn}:{Mode}'
    if use_RBA:
        save_name += 'RBA'
    if Mod_MLP:
        save_name += 'mMLP'
    if Adaptive_AF:
        save_name += 'AF'
    if Weight_Norm:
        save_name += 'WN'
    if resample:
        save_name += f'_BS:{batch_size}'
    return save_name

def create_save_nameSA(Eqn, Mode, use_RBA=False, Mod_MLP=False, Adaptive_AF=False, Weight_Norm=False,resample=False,batch_size=1):
    save_name = f'{Eqn}:{Mode}'
    if use_RBA:
        save_name += 'SA'
    if Mod_MLP:
        save_name += 'mMLP'
    if Adaptive_AF:
        save_name += 'AF'
    if Weight_Norm:
        save_name += 'WN'
    if resample:
        save_name += f'_BS:{batch_size}'
    return save_name
    
    
def create_and_return_directories(save_path, dataset_name, subdirectories):
    # Base directory
    result_path = os.path.join(save_path, dataset_name)

    # Creating subdirectories and storing their paths
    paths = {}
    for subdir in subdirectories:
        path = os.path.join(result_path, subdir+'/')
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path

        # Printing the paths
        print(f"The {subdir.lower().replace('_', ' ')} path is: {path}")

    return paths



#Dataloader
class Dataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, sample_function) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_function = sample_function

    def __iter__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]
        return iter(data)
    def __getitem__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]

# Random Sampling
def random_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )
# Uniform Sampling
def get_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.arange(N),
        np.arange(batch_size, N, batch_size),
    )



#Rescale boundaries
def rescale_bcs(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    v_mbcs=mBCs[:,4,0:0+1].flatten()[:,None]
    u_mbcs=mBCs[:,5,0:0+1].flatten()[:,None]
    w_mbcs=mBCs[:,6,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        v_mbcs=mBCs[:,4,t_idx:t_idx+1].flatten()[:,None]
        u_mbcs=mBCs[:,5,t_idx:t_idx+1].flatten()[:,None]
        w_mbcs=mBCs[:,6,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points,u_mbcs,v_mbcs,w_mbcs))
    return mbcs_f,[dx,dy,dz]

#Rescale boundaries
def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def normalizeZ(X,X_mean,X_std):
    H = (X- X_mean)/X_std
    return H    

def normalize_between(X,X_min,X_max,lb=-1,ub=1):
    X = (ub-lb) * (X- X_min) / (X_max - X_min)+lb 
    return X    

def identity(X,X_min,X_max):
    return X


def make_video(image_folder, video_name, fps):
    video_name=image_folder+str(fps)+'fps-'+video_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort the images by name
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for i in tqdm.tqdm(range(len(images))):
        image=f'{image_folder}{images[i]}'
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def extract_arrays_from_params(params_test,Use_ResNet=False):
    # Extract arrays from the 'params' dictionary
    params_arrays = []
    if Use_ResNet:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))
    else:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))

    # Extract arrays from the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_arrays = [np.array(params_test['mMLP'][0][key]) for key in mMLP_keys if key in params_test['mMLP'][0]]

    # Extract arrays from the 'AdaptiveAF' dictionary
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    AdaptiveAF_arrays = []
    for adaptive_dict in params_test['AdaptiveAF']:
        for key in AdaptiveAF_keys:
            AdaptiveAF_arrays.append(np.array(adaptive_dict[key]))

    # Combine all extracted arrays
    all_arrays = params_arrays + mMLP_arrays + AdaptiveAF_arrays
    return all_arrays

def reconstruct_params(numpy_arrays_list, params_length, params_length_AF,Use_ResNet=False):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    # Reconstruct the 'params' dictionary
    params_dicts = []
    if Use_ResNet:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)
    else:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)

    # Reconstruct the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_dict = {}
    for key in mMLP_keys:
        if pointer < len(numpy_arrays_list):
            mMLP_dict[key] = numpy_arrays_list[pointer]
            pointer += 1

    # Reconstruct the 'AdaptiveAF' dictionary
    AdaptiveAF_dicts = []
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    for _ in range(params_length_AF):
        adaptive_dict = {}
        for key in AdaptiveAF_keys:
            if pointer < len(numpy_arrays_list):
                adaptive_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        AdaptiveAF_dicts.append(adaptive_dict)

    # Combine all reconstructed dictionaries
    reconstructed_params_test = {
        'AdaptiveAF': AdaptiveAF_dicts,
        'mMLP': [mMLP_dict],
        'params': params_dicts,
    }
    
    return reconstructed_params_test

def reconstruct_params_ResNet(numpy_arrays_list, params_length, params_length_AF):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    
    # Reconstruct the 'params' dictionary
    params_dicts = []
    for _ in range(params_length):
        param_dict = {}
        for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
            if pointer < len(numpy_arrays_list):
                param_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        params_dicts.append(param_dict)

    # Reconstruct the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_dict = {}
    for key in mMLP_keys:
        if pointer < len(numpy_arrays_list):
            mMLP_dict[key] = numpy_arrays_list[pointer]
            pointer += 1

    # Reconstruct the 'AdaptiveAF' dictionary
    AdaptiveAF_dicts = []
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    for _ in range(params_length_AF):
        adaptive_dict = {}
        for key in AdaptiveAF_keys:
            if pointer < len(numpy_arrays_list):
                adaptive_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        AdaptiveAF_dicts.append(adaptive_dict)

    # Combine all reconstructed dictionaries
    reconstructed_params_test = {
        'AdaptiveAF': AdaptiveAF_dicts,
        'mMLP': [mMLP_dict],
        'params': params_dicts,
    }
    
    return reconstructed_params_test

# Saving All params
def save_all_params(All_params,all_params_path):
    with h5py.File(all_params_path, 'w') as hf:
        for i, inner_list in tqdm.tqdm(enumerate(All_params)):
            group = hf.create_group(f"list_{i}")
            for j, arr in enumerate(inner_list):
                group.create_dataset(f"array_{j}", data=arr)


# Reading All params
def read_all_params(all_params_path):
    loaded_All_params = []
    with h5py.File(all_params_path, 'r') as hf:
        for key in tqdm.tqdm(sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))):  # Ensure keys are processed in order
            inner_list = []
            for sub_key in sorted(hf[key].keys(), key=lambda x: int(x.split('_')[1])):
                data = hf[key][sub_key]
                if data.shape == ():  # Check if scalar
                    inner_list.append(data[()])
                else:
                    inner_list.append(data[:])
            loaded_All_params.append(inner_list)
    return loaded_All_params


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

def sample_points(batch_sizes, dataset):
    """
    This function randomly samples indices based on batch_sizes.
    
    Args:
    - batch_sizes: Dictionary of batch sizes for different data types.

    Returns:
    - Dictionary of sampled indices for each data type.
    """
    return {key: np.random.choice(len(dataset[key]), batch_sizes[key]) for key in batch_sizes}



def sample_points_pdf(subkey, batch_sizes, dataset, lambdas,k=1,c=0.5):
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices

          
    
def sample_points_PDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices

    
def sample_points_jax(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],))
    return batch_indices
def sample_all(it, batch_sizes, dataset, lambdas=[],k=1,c=0.5):
    batch_indices = {}
    for key in batch_sizes:
        batch_indices[key] = jnp.arange(len(dataset[key]))
    return batch_indices


def sample_points_softPDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=jax.nn.softmax(lambdas_key))
    return batch_indices

def generate_random_matrix(m, d):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    B_standard = jax.random.normal(subkey, (m, d))
    return B_standard 
    
def Encode_Fourier(X,M,N):
    t=X[0]
    x=X[1]
    y=X[2]
    P_x=2
    P_y=2
    n_num = jnp.arange(1, N+1)
    m_num = jnp.arange(1, M+1)
    n, m = jnp.meshgrid(n_num, m_num)
    n=n.flatten()
    m=m.flatten()
    w_x = 2.0 * jnp.pi / P_x
    w_y = 2.0 * jnp.pi / P_y    

    out = jnp.hstack([t,
                      x,
                      y,
                      jnp.cos(n* w_x * x)  * jnp.cos(m * w_y * y),
                      jnp.cos(n * w_x * x) * jnp.sin(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.cos(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.sin(m * w_y * y)])
    return out
def Encode_Fourier2(X, B, s=10):
    B = s * B
    out = jnp.multiply(B, X).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier_time(X, B, s=10):
    out =  s *jnp.multiply(B, X[1:]).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier_time_1(X, M1,M2):
    X_min,X_max=M1
    B,s=M2
    X = 2 * (X- X_min) / (X_max - X_min)-1 
    out =  s *jnp.multiply(B, X[1:]).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier0(X,M,N):
    x=X[0]
    y=X[1]
    P_x=2
    P_y=2
    n_num = jnp.arange(1, N+1)
    m_num = jnp.arange(1, M+1)
    n, m = jnp.meshgrid(n_num, m_num)
    n=n.flatten()
    m=m.flatten()
    w_x = 2.0 * jnp.pi / P_x
    w_y = 2.0 * jnp.pi / P_y    

    out = jnp.hstack([jnp.cos(n* w_x * x)  * jnp.cos(m * w_y * y),
                      jnp.cos(n * w_x * x) * jnp.sin(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.cos(m * w_y * y),])
    return out



def Encode_Chebysev_5(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),
                        ])
    return out
def Encode_hybrid(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        jnp.sin(10*out),jnp.cos(10*out),
                        jnp.exp(out),jnp.exp(-out),
                        jnp.sin(10*out)*jnp.cos(out),
                        jnp.sin(out)*jnp.cos(10*out),
                        ])
    return out
def Encode_Chebysev_10(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        T7(out),T8(out),T9(out),
                        T10(out),
                        ])
    return out
def Encode_Chebysev_12(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        T7(out),T8(out),T9(out),
                        T10(out),T11(out),T12(out),
                        ])
    return out
def Encode_Chebysev_time(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    X=X.flatten()
    out =X[1:] 
    out = jnp.concatenate([
                        X,T2(out),T3(out),
                        T4(out),T5(out)
                        ])
    return out
# Activation functions mapping
ACTIVATION_FUNCTIONS = {
    'sin': jnp.sin,
    'tanh': jnp.tanh,
    'tanh_08': lambda x: 0.8*jnp.tanh(x),
    'swish': lambda x: x * jax.nn.sigmoid(x),
    'leaky_relu': lambda x: jnp.where(x > 0, x, 0.01 * x),
    'custom': lambda x: jnp.where(x < 0, -1, jnp.where(x > 2, 1, 0.5*(x)**2-1)),
    'sigmoid':lambda x: jax.nn.sigmoid(x),
    'sigmoid_11':lambda x: 2/(1+jnp.exp(-x*1.1**2))-1

}

# Normalization functions and metrics mapping
NORMALIZATION_FUNCTIONS = {
    'plusminus1': {
        'fn': normalize_between,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'normal': {
        'fn': normalizeZ,
        'metric1': lambda x: x.mean(0, keepdims=False),
        'metric2': lambda x: x.std(0, keepdims=False)
    },
    'identity': {
        'fn': identity,
        'metric1': lambda x: 0,
        'metric2': lambda x: 0
    },
    'fourier': {
        'fn': Encode_Fourier2,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourierfull': {
        'fn': Encode_Fourier0,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourier_time': {
        'fn': Encode_Fourier_time,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourier_time_1': {
        'fn': Encode_Fourier_time_1,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev5': {
        'fn': Encode_Chebysev_5,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev10': {
        'fn': Encode_Chebysev_10,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev12': {
        'fn': Encode_Chebysev_12,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev_time': {
        'fn': Encode_Chebysev_time,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'hybrid': {
        'fn': Encode_hybrid,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
}

# Loss metric mapping
ERROR_FUNCTIONS = {
    'l2': MSE,
    'l1': MAE
}



def filter_Magnitude(BCs_frame,row=7,T_max=0.7,T_min=0.49):
    T  =BCs_frame[:,row]
    upper_limit = T_max
    lower_limit = T_min
    idx1=np.argwhere(T<upper_limit)
    idx2=np.argwhere(T>lower_limit)
    idxT=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idxT]
    return BCs_framef
def filter_Magnitude_inverse(BCs_frame,row=7,T_max=0.7,T_min=0.49):
    T  =BCs_frame[:,row]
    upper_limit = T_max
    lower_limit = T_min
    idx1=np.argwhere(T>upper_limit)
    idx2=np.argwhere(T<lower_limit)
    idxT=np.union1d(idx1,idx2)
    BCs_framef=BCs_frame[idxT]
    return BCs_framef

def filter_Z(BCs_frame,row=7,permissibility=3):
    u  =BCs_frame[:,row]
    #Z score FILTERING
    #Filter u
    mean_u = np.nanmean(u)
    std_u  = np.nanstd(u)
    upper_limit = mean_u + permissibility*std_u
    lower_limit = mean_u - permissibility*std_u
    idx1=np.argwhere(u<upper_limit)
    idx2=np.argwhere(u>lower_limit)
    idx=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idx]
    return BCs_framef

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
