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



def sample_points_PDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices


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


static_options_SSBroyden = {
    'gtol': 1e-9,
    'update_method': "ssbroyden2",
    'initial_scale': True,
    'ls_normal_c1': 1e-4, 'ls_normal_c2': 0.9, 'ls_normal_maxiter': 15,
    'ls_fb_c1_try1': 1e-4, 'ls_fb_c2_try1': 0.8, 'ls_fb_maxiter_try1': 10,
    'ls_fb_c1_try2': 1e-4, 'ls_fb_c2_try2': 0.5, 'ls_fb_maxiter_try2': 25
}