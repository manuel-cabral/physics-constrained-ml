import matplotlib.pyplot as plt
import numpy as np
import torch
from generate_data import quantities
from matplotlib.animation import FuncAnimation, writers
import tqdm


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
from src.utils import get_args, set_seed, set_torch_dtype, load_data, ftype_torch #, set_torch_multiprocessing
from src.plotter import plot_quantity
from src.model import Model
import src.config

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set seed for reproducibility
set_seed(args.seed)

def load_model(checkpoint, name, args, pasta='datasets', folder='best_models', initial_optm='lbfgs'):
    file_path = os.path.join(os.getcwd(), 'src', folder, f'{checkpoint}')
    dataset = load_data(name, pasta)
    model = Model.load_checkpoint(file_path, dataset=dataset, args=args, initial_optm=initial_optm)

    return model

def change_parameters(args, data_size=5e2):
    # args.kind = 'incompressible'
    args.kind = 'baseline'
    # args.layers = [6]*1 # [n_neurons]*n_layers
    args.layers = [16]*2 # [n_neurons]*n_layers
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

    # args.normalize_inputs = False
    # args.reduce_inputs = True
    # args.transform_output = True

    args.normalize_inputs = True
    args.reduce_inputs = False
    args.transform_output = False


    args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    args.phi = [[1,0,0,0,-1],[0,1,0,0,-1],[0,0,1,-1,-1]]

    # args.phi_output = [0,0,0,0] # 1
    args.phi_output = [0,0,0,1,1] # U_inf * R


    return

dtype = ftype_torch[args.ftype]
from scipy.signal import argrelextrema
def get_stagnation_points(model, circulation):
    theta = np.linspace(0,2*np.pi,1000)
    x, y = np.cos(theta), np.sin(theta)
    X = np.column_stack([x,y])
    vel = np.full((X.shape[0], 1), 1)
    rad = np.full((X.shape[0], 1), 1)
    circ = np.full((X.shape[0], 1), circulation)
    X = np.column_stack([X, circ, vel, rad])
    X = torch.tensor(X, dtype=dtype)

    vel_pred = torch.norm(model.predict(X).detach(), dim=1)

    thetas = theta[argrelextrema(vel_pred.detach().numpy(), np.less)]
    min_vel = vel_pred[argrelextrema(vel_pred.detach().numpy(), np.less)]

    if len(thetas) == 1:
        theta1, theta2 = 0, thetas
        min1, min2 = vel_pred[0], min_vel[0]
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
    return theta1, theta2, min1, min2, np.abs(theta2-theta1) if np.abs(theta2-theta1)<np.pi else 2*np.pi-np.abs(theta2-theta1)


def plot_surface_velocity(ax, model, circulation, label='predicted'):
    # fig, ax = plt.subplots(figsize=(10,8))
    theta = np.linspace(0,2*np.pi,500)
    x, y = np.cos(theta), np.sin(theta)
    X = np.column_stack([x,y])
    circ = np.full((X.shape[0], 1), circulation)
    vel = np.full((X.shape[0], 1), 1)
    rad = np.full((X.shape[0], 1), 1)
    X = np.column_stack([X, circ, vel, rad])
    X = torch.tensor(X, dtype=dtype)
    vel_pred = torch.norm(model.predict(X).detach(), dim=1)

    ax.plot(theta, vel_pred, color='black', linewidth=2, label=label)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$||\mathbf{u}||$')
    ax.set_title('Norm of velocity along the surface')
    ax.set_aspect(2*np.pi/2.5, adjustable='box')

    ax.hlines(0, 0, 2*np.pi, color='black', linestyle='--')
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2.5)

    # plt.show()

    return ax

def plot_stagnation_points(model, N=200, bounds=[[-3,3],[-3,3]], circulation=0, U_inf=1, R=1):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)

    circ_ = np.ones_like(xx)*circulation
    U_inf_ = np.ones_like(xx)*U_inf
    R_ = np.ones_like(xx)*R

    points = np.column_stack((xx.flatten(),yy.flatten(),circ_.flatten(),U_inf_.flatten(), R_.flatten()))
    points = torch.from_numpy(points).to(torch.float32)
    u,v = (model.predict(points)).detach().numpy().T
    u,v = u.reshape(N,N), v.reshape(N,N)
    # psi = model.psi(points).detach().numpy().reshape(N,N)

    mask = xx**2 + yy**2 <= R**2
    u[mask], v[mask], = [np.nan]*2
    # u[mask], v[mask], psi[mask] = [np.nan]*3

    vel_norm = np.sqrt(u**2+v**2)
    arg_min1 = np.unravel_index(np.nanargmin(vel_norm), vel_norm.shape)
    vel_min1 = vel_norm[arg_min1]
    r_n = 1e-1
    neighborhood = (xx-x[arg_min1[1]])**2 + (yy-y[arg_min1[0]])**2 > r_n**2
    # arg_min2 = np.unravel_index(np.nanargmin(vel_norm[neighborhood]), vel_norm[neighborhood].shape)
    vel_outside_min = vel_norm.copy()
    vel_outside_min[~neighborhood] = np.nan
    arg_min2 = np.unravel_index(np.nanargmin(vel_outside_min), vel_outside_min.shape)
    vel_min2 = vel_norm[arg_min1]
    # print(f'Minimum velocity: {vel_min1} at x={x[arg_min1[1]]}, y={y[arg_min1[0]]}')
    # print(f'Minimum velocity: {vel_min2} at x={x[arg_min2[1]]}, y={y[arg_min2[0]]}')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    
    axes[0] = plot_quantity(vel_norm, x, y, label=r'$||\mathbf{u}||$', ax=axes[0])[1]
    


    circle = plt.Circle((0,0), R, color='firebrick', fill=True, alpha=.3)
    axes[0].add_artist(circle)

    # theta1, theta2, min_1, min_2, delta_theta = get_stagnation_points(model, circulation)
    # axes[0].scatter([R*np.cos(theta1), R*np.cos(theta2)], [R*np.sin(theta1), R*np.sin(theta2)], marker='o', color='white', linewidths=1, edgecolors='black', s=100)

    if np.abs(circulation) <= 4*np.pi*U_inf*R:
        theta1, theta2 = np.arcsin(circulation/(4*np.pi*U_inf*R)), np.pi-np.arcsin(circulation/(4*np.pi*U_inf*R))
        R_ = R
    else:
        theta1, theta2 = np.pi/2, np.pi/2
        t = circulation/(2*np.pi*U_inf*R)
        R_ = np.sign(t)*R*np.sqrt(np.abs(t) - 1 )

    axes[0].scatter(x[arg_min1[1]], y[arg_min1[0]], marker='X', color='white', linewidths=1, edgecolors='black', s=125, alpha =.85, zorder=10)
    axes[0].scatter(x[arg_min2[1]], y[arg_min2[0]], marker='X', color='white', linewidths=1, edgecolors='black', s=125, alpha =.85, label='predicted', zorder=10)
    axes[0].scatter([R_*np.cos(theta1), R_*np.cos(theta2)], [R_*np.sin(theta1), R_*np.sin(theta2)], marker='o', color='white', linewidths=1, edgecolors='black', s=75, alpha =.75, label='exact', zorder=5)
    

    axes[1] = plot_surface_velocity(axes[1], model, circulation)
    thetas = np.linspace(0,2*np.pi,500)
    axes[1].plot(thetas, np.abs(-2*U_inf*np.sin(thetas)+circulation/(2*np.pi*U_inf*R)), color='black', linestyle='--', alpha=.5, zorder=5, label='exact')
    # axes[1].scatter([theta1, theta2], [min_1, min_2], marker='o', color='white', linewidths=1, edgecolors='black', s=100, zorder=10)
    # axes[1].vlines([np.arcsin(circulation/(4*np.pi*U_inf*R)),np.pi-np.arcsin(circulation/(4*np.pi*U_inf*R))], 0, 2.5, color='gray', linestyle='-', zorder=5)
    
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='upper right')

    return fig, axes

def main():
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    data_file = f'cylinder-with-circulation/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    # checkpoint = 'incompressible_K18GGI_checkpoint_1519' # [6]*1, out=[0,0,0,1,1]
    # checkpoint = 'baseline_8KCAEL_checkpoint_1682' # [6]*1, baseline

    checkpoint = 'baseline_KD7E0P_checkpoint_1901' # [16]*2, baseline

    model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')

    bounds = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    for i,c in enumerate(np.linspace(-5,5,150)):
        fig, axes = plot_stagnation_points(model, N=500, bounds=bounds[:2], circulation=c, U_inf=1, R=1)
        fig.savefig(f'src/frames/frame_{i}.png', bbox_inches='tight')
        plt.close()
    
    # fig, axes = plot_stagnation_points(model, N=500, bounds=bounds[:2], circulation=15, U_inf=1, R=1)
    # plt.show()

if __name__ == '__main__':
    main()