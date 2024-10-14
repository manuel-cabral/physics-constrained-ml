import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.signal import argrelextrema
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
from src.utils import get_args, set_torch_dtype, set_seed, load_data, generate_sobol, ftype_torch #, set_torch_multiprocessing
from src.model import Model
import src.config

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)
dtype = ftype_torch[args.ftype]

# Set multiprocessing
# set_torch_multiprocessing()

# Set seed for reproducibility
set_seed(args.seed)

def coef_pd(model, circulation=0, U_inf=1, R=1):
    theta = np.linspace(0,2*np.pi,1000)
    x, y = R*np.cos(theta), R*np.sin(theta)
    X = np.column_stack([x,y])
    circ = np.full((X.shape[0], 1), circulation)
    vel = np.full((X.shape[0], 1), U_inf)
    rad = np.full((X.shape[0], 1), R)
    X = np.column_stack([X, circ, vel, rad])
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

def load_model(checkpoint, name, args, pasta='datasets', folder='best_models', initial_optm='lbfgs'):
    file_path = os.path.join(os.getcwd(), 'src', folder, f'{checkpoint}')
    dataset = load_data(name, pasta)
    model = Model.load_checkpoint(file_path, dataset=dataset, args=args, initial_optm=initial_optm)

    return model

def plot_surface_velocity(checkpoint, data_file, circulation, fname=None):
    fig, ax = plt.subplots(figsize=(10,8))
    theta = np.linspace(0,2*np.pi,1000)
    x, y = np.cos(theta), np.sin(theta)
    X = np.column_stack([x,y])
    circ = np.full((X.shape[0], 1), circulation)
    vel = np.full((X.shape[0], 1), 1)
    rad = np.full((X.shape[0], 1), 1)
    X = np.column_stack([X, circ, vel, rad])
    X = torch.tensor(X, dtype=dtype)

    model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')
    vel_pred = torch.norm(model.predict(X).detach(), dim=1)

    ax.plot(theta, vel_pred, color='black', linewidth=2)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$||\mathbf{u}||$')
    ax.set_title('Surface velocity')

    ax.hlines(0, 0, 2*np.pi, color='black', linestyle='--')

    plt.show()
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=256)
    return

def plot_angles(checkpoint, data_file, circulations, fname=None):
    theta = np.linspace(0,2*np.pi,1000)
    x, y = np.cos(theta), np.sin(theta)
    diff = np.zeros_like(circulations)
    for i,circulation in enumerate(circulations):
        X = np.column_stack([x,y])
        vel = np.full((X.shape[0], 1), 1)
        rad = np.full((X.shape[0], 1), 1)
        circ = np.full((X.shape[0], 1), circulation)
        X = np.column_stack([X, circ, vel, rad])
        X = torch.tensor(X, dtype=dtype)

        model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')
        vel_pred = torch.norm(model.predict(X).detach(), dim=1)

        thetas = theta[argrelextrema(vel_pred.detach().numpy(), np.less)]
        min_vel = vel_pred[argrelextrema(vel_pred.detach().numpy(), np.less)]
        
        if len(thetas) == 1:
            theta1, theta2 = 0, thetas
        else:
            min1, min2 = np.inf, np.inf
            for ind1 in range(len(thetas)):
                if min_vel[ind1]<min1:
                    theta1 = thetas[ind1]
                    min1 = min_vel[ind1]
            for ind2 in range(len(thetas)):
                if min_vel[ind2]<min2 and min_vel[ind2]>min1:
                    theta2 = thetas[ind2]
                    min2 = min_vel[ind2]

        if circulation<0:
            diff[i] = 2*np.pi-np.abs(theta2-theta1)
        else:
            diff[i] = np.abs(theta2-theta1)
        # diff[i] = theta2-theta1

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(circulations, diff)
    ratio = circulations/(4*np.pi*1*1)
    l = []
    for i in ratio:
        if np.abs(i)<1:
            l.append(np.pi-2*np.arcsin(i))
        elif i>1:
            l.append(0)
        else:
            l.append(2*np.pi)

    ax.plot(circulations, l, 'k--')
    ax.hlines(np.pi, circulations[0], circulations[-1], color='gray', alpha=.5) 
    plt.xlabel('$\Gamma$')
    plt.ylabel('$\Delta \\theta$')
    plt.title('Angle between stagnation points')

    ax.set_xticks(np.arange(-5,6,1))
    ax.set_yticks([np.pi/2, np.pi, 3*np.pi/2])
    ax.set_yticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
    plt.show()

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=256)
    return


def change_parameters(args, data_size=5e2):
    args.kind = 'incompressible'
    args.layers = [6]*1 # [n_neurons]*n_layers
    args.n_epochs = 2_000 # n_epochs_adam + n_epochs_lbfgs
    args.n_epochs_adam = 1_000
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None 
    args.batch_size = int(2**np.floor(np.log2(data_size)))
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "circ", "U_infty", "r"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True

    # args.normalize_inputs = True
    # args.reduce_inputs = False
    # args.transform_output = False

    args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    args.phi = [[1,0,0,0,-1],[0,1,0,0,-1],[0,0,1,-1,-1]]

    # # args.phi_output = [0,0,0,0] # 1
    args.phi_output = [0,0,0,1,1] # U_inf * R


    return

def main():
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    data_file = f'cylinder-with-circulation/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    checkpoint = '' # input the checkpoint name here

    fname = None
    # plot_surface_velocity(checkpoint, data_file, circulation=-2, fname=fname)
    plot_angles(checkpoint, data_file, circulations=np.linspace(-5,5,100), fname=fname)
    return

if __name__ == '__main__':
    main()