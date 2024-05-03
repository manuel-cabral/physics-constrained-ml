import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
from scipy.linalg import null_space
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
from src.model import Model

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
    # args.kind = 'baseline'
    args.layers = [16]*2
    args.n_epochs = 2_000
    args.n_epochs_adam = 1_000
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = int(2**np.floor(np.log2(data_size)))
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = False                         
    args.x_vars = ["x", "y", "vort", "r"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True
    # args.sampling_box = [[-2.5,2.5],[0,2.5],[0,10],[.5,5],[.1,1.5]] # x,y,c,U,r

    args.phi = [[1,0,0,-1],[0,1,0,-1]]

    args.phi_output = [0,0,1,2] # U_inf * y

    return


def main():
    n_train = 1e3
    n_val = 5e3
    n_test = 1
    idx = 0
    change_parameters(args, data_size=n_train)

    name = f'fraenkels_flow/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'
    dataset = load_data(name)

    train_model(dataset, args)


if __name__ == '__main__':
    main()