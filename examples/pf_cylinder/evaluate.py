import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
plt.style.use('ggplot')


# delete if not needed
######
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
######

import args
from src.utils import get_args, set_torch_dtype, set_seed, ftype_torch #, set_torch_multiprocessing
from src.model_simple import Model
from src.plotter_examples import plot_predictions, plot_cylinder
from generate_data import generate_sobol

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)
dtype = ftype_torch[args.ftype]

# Set multiprocessing
# set_torch_multiprocessing()

# Set seed for reproducibility
set_seed(args.seed)

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    data = torch.from_numpy(data)
    return data

def coef_pd(model, circulation=0, U_inf=1, R=1):
    theta = np.linspace(0,2*np.pi,1000)
    x, y = R*np.cos(theta), R*np.sin(theta)
    X = np.column_stack([x,y])
    circ = np.full((X.shape[0], 1), circulation)
    X = np.column_stack([X, circ])
    X = torch.tensor(X, dtype=dtype)
    vel_pred = torch.norm(model.predict(X).detach(), dim=1)
    cp_pred = 1 - vel_pred**2/U_inf**2

    d_theta = np.diff(theta)[0]
    cl_pred = -.5 * torch.sum(cp_pred*np.sin(theta)*d_theta)
    cl  = -circulation/(U_inf*R)

    cd_pred = -.5 * torch.sum(cp_pred*np.cos(theta)*d_theta)
    cd = 0

    return cl, cl_pred, cd, cd_pred 

def evaluate(model, circulations):
    cl, cl_pred, cd, cd_pred = np.zeros((4,len(circulations)))
    for i,c in enumerate(circulations):
        cl[i], cl_pred[i], cd[i], cd_pred[i]  = coef_pd(model, circulation=c, U_inf=1, R=1)
    return cl, cl_pred, cd, cd_pred 

def load_dataset(idx, circ):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_ = {'train': torch.Tensor(), 'val': torch.Tensor(), 'test': torch.Tensor()}
    for mode in ['train', 'val']:
        file_path = os.path.join(dir_path, 'datasets', f'{mode}_circ{circ:.2f}_set{idx+1}')
        data = load_pickle(file_path)
        dataset_[f'{mode}'] = torch.cat((dataset_[f'{mode}'], data), dim=0)
    file_path = os.path.join(dir_path, 'datasets', f'test_circ{circ:.2f}_set')
    data = load_pickle(file_path)
    dataset_['test'] = torch.cat((dataset_['test'], data), dim=0)
    return dataset_

def load_model_from_checkpoint(checkpoint, dataset, args):
    fname = f"src/training/{checkpoint}.tar"
    args.kind = checkpoint.split('_')[0]
    args.x_vars = ["x", "y", "circulation"]
    return Model.load_checkpoint(fname, dataset=dataset, args=args)

def plot_lift_and_drag(checkpoint, fname=None):
    circulations = np.linspace(-15,15,100)
    dataset = load_dataset(1, 0)
    model = load_model_from_checkpoint(checkpoint, dataset, args)
    cl, cl_pred, cd, cd_pred = evaluate(model, circulations)
    trained = generate_sobol(4, -5, 5)

    plot_predictions(circulations, cl_pred, cl,  trained=trained, quantity='lift', extrapolate=True, fname=fname)
    plot_predictions(circulations, cd_pred, cd, trained=trained, quantity='drag', extrapolate=True, fname=fname)
    return 

def plot_field(checkpoint, circulation, fname=None):
    dataset = load_dataset(1, 0)
    model = load_model_from_checkpoint(checkpoint, dataset, args)
    plot_cylinder(model, bounds=args.sampling_box, mode='norm', other_args=[circulation], plot_stream=True, kind=checkpoint.split('_')[0], fname=None)
    plot_cylinder(model, bounds=args.sampling_box, mode='error', other_args=[circulation], plot_stream=True, kind=checkpoint.split('_')[0], fname=None)
    return

def main():
    base = 'baseline_7KB4NV_checkpoint_117' # idx=1
    inc = 'incompressible_DQNOF8_checkpoint_97' # idx=1, limited training
    checkpoint = base
    plot_lift_and_drag(checkpoint)
    plot_field(checkpoint, 0)
    return

if __name__ == '__main__':
    main()