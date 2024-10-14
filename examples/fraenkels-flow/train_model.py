import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
from scipy.linalg import null_space
import matplotlib.gridspec as gridspec
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
from src.utils import get_args, set_torch_dtype, set_seed, load_pickle, load_data #, set_torch_multiprocessing
from src.model import Model, TRANSFORM_INPUT
import src.config

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set multiprocessing
# set_torch_multiprocessing()

# Set seed for reproducibility
set_seed(args.seed)


def train_model(dataset, args):
    # Create model
    model = Model(dataset, args)
    print(model.nn.layers)
    
    # Train model
    start = timer()
    model.train()
    end = timer()
    print(f'Training time: {timedelta(seconds=end-start)}')

    return model

def change_parameters(args, data_size=5e2):
    args.kind = 'incompressible'
    args.layers = [16]*2
    args.n_epochs = 2_000
    args.n_epochs_adam = 1_000
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = int(2**np.floor(np.log2(data_size)))
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "vort", "r"]

    if args.kind == 'incompressible':
        args.normalize_inputs = False
        args.reduce_inputs = True
        args.transform_output = True
    elif args.kind in ['baseline','soft incompressible']:
        args.normalize_inputs = True
        args.reduce_inputs = False
        args.transform_output = False

    args.sampling_box = [[-2.5,2.5],[-2.5,2.5],[.1,3],[.5,1.5]] # x,y,vort,R

    args.phi = [[1,0,0,-1],[0,1,0,-1]] # x/R, y/R
    
    args.phi_output = [0,0,1,2] # vort * R**2

    return

def main():
    # Set data size
    n_train, n_val, n_test  = 1e3, 1e3, 1
    idx = 0

    # Change parameters
    change_parameters(args, data_size=n_train)

    # Load data
    name = f'fraenkels-flow/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'
    dataset = load_data(name)

    # Train model
    train_model(dataset, args)


if __name__ == '__main__':
    main()