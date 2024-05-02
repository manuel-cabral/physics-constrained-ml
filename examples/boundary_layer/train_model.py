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
from src.utils import get_args, set_seed, set_torch_dtype #, set_torch_multiprocessing
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
    # print(model.nn.layers)

    # Train model
    start = timer()
    model.train()
    end = timer()
    print(f'Training time: {timedelta(seconds=end-start)}')

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f: data = pickle.load(f)
    return data

def load_data(name, folder='datasets'):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, folder)
    dataset = load_pickle(os.path.join(folder_path,name))
    return dataset

def change_parameters(args):
    args.kind = 'incompressible'
    # args.kind = 'baseline'
    args.layers = [3]*1
    args.n_epochs = 700
    args.n_epochs_adam = 500
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = 256
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "visc", "U_infty"]

    args.normalize_inputs = False
    args.reduce_inputs = not args.normalize_inputs
    args.transform_output = True

    # args.sampling_box = [(5e-2, 1), (0, 5e-2), (1/7e4,1/3e4), (0,5)]

    args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
    # args.phi = [[-.5,1,-.5,.5]]
    # args.phi_output = [0,1,0,1] # U_inf*y
    # args.phi_output = [1,0,0,1] # U_inf*x
    # args.phi_output = [1,-1,1,0] # x*visc/y
    # args.phi_output = [-2,0,1,-2] # visc/(x*U)^2

    return


def main():
    change_parameters(args)

    def phi_base(D):
        return null_space(D)
    def phi_zero(D):
        phi = phi_base(D)

        phi[:,-1] /= -phi[-1,-1]
        
        phi[:,0] += phi[-1,0]*phi[:,-1]
        phi[:,1] += phi[-1,1]*phi[:,-1]
        # k = 1
        # phi[:,-1] += ((1-phi[-2,-1])/phi[-2,k])*phi[:,k]

        return phi


    D = [[1,1,2,1, 2],[0,0,-1,-1, -1]] # x,y,visc,U_inf,psi
    phi = phi_zero(D).T
    # print(phi)
    *inputs, outputs = phi
    args.phi = [input[:-1] for input in inputs]
    # args.phi_output = outputs[:-1]

    # D = [[1,1,2,1],[0,0,-1,-1]] # x,y,visc,U_inf
    # phi = phi_base(D).T
    # # print(phi)
    # args.phi = phi

    # args.phi_output = [1,0,0,1]
    # args.phi_output = [0,1,0,1]
    args.phi_output = [.5,0,.5,.5]
    # args.phi_output = [0,0,0,0]

    # confirmation:
    # phi = np.concatenate((args.phi, [args.phi_output])).T
    # phi = phi.T
    # print(phi)
    # # print(f'\n{np.linalg.norm(D@phi.T):.2e}')
    # print(np.matrix.round(phi@phi.T,2))
    

    # print(args.phi)
    # print(args.phi_output)

    # args.phi = [[-1,1,0,0],[-1,0,1,-1]]
    # args.phi_output = [1,0,0,1]
    # args.phi = [[1,-1,0,0],[0,-1,1,-1]]
    # args.phi_output = [0,1,0,1]
    
    # a = np.concatenate((args.phi, [args.phi_output])).T
    # print(a.T@a)

    dataset = load_data('data_5e2_1e3_1_idx0')
    train_model(dataset, args)


if __name__ == '__main__':
    main()