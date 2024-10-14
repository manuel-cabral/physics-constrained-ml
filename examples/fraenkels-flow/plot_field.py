import matplotlib.pyplot as plt
import numpy as np
import torch
from generate_data import quantities

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
    args.kind = 'incompressible'
    args.layers = [16]*2 # [n_neurons]*n_layers
    args.n_epochs = 2_000 # n_epochs_adam + n_epochs_lbfgs
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

    args.sampling_box = [[-2.5,2.5],[-2.5,2.5],[.1,3],[.5,1.5]] # x,y,vort,R

    args.phi = [[1,0,0,-1],[0,1,0,-1]]

    args.phi_output = [0,0,1,2] # vort * R**2

    return

def plot_fields(model, N=200, bounds=[[-3,3],[-3,3]], vort=1, R=1, add_noise=False):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)

    vort_ = np.ones_like(xx)*vort
    R_ = np.ones_like(xx)*R

    points = np.column_stack((xx.flatten(),yy.flatten(),vort_.flatten(),R_.flatten()))
    points = torch.from_numpy(points).to(torch.float32)
    u,v = (model.predict(points)).detach().numpy().T
    u,v = u.reshape(N,N), v.reshape(N,N)
    psi = model.psi(points).detach().numpy().reshape(N,N)

    u_teo, v_teo, psi_teo = quantities(xx, yy, vort, R,)

    mask = xx**2 + yy**2 <= R**2
    u_teo[mask], v_teo[mask], psi_teo[mask] = [np.nan]*3
    u[mask], v[mask], psi[mask] = [np.nan]*3

    print(f'Error u: {100*np.nanmean(np.linalg.norm(u-u_teo, axis=0)/np.linalg.norm(u_teo, axis=0)):.2f}%')
    print(f'Error v: {100*np.nanmean(np.linalg.norm(v-v_teo, axis=0)/np.linalg.norm(v_teo, axis=0)):.2f}%')
    # print(f'Error psi: {100*np.linalg.norm(psi-psi_teo)/np.linalg.norm(psi_teo-U_inf*yy):.2f}%')

    if add_noise: 
        for q in [u,v,psi]: q = add_noise(q)

    labels = ['$u$', '$u_{target}$', '$|u_{target}-u|$', '$v$', '$v_{target}$', '$|v_{target}-v|$', '$\psi$', '$\psi_{target}$', '$|\psi_{target}-\psi|$']
    names = ['u', 'u_target', 'u_error', 'v', 'v_target', 'v_error', 'psi', 'psi_target', 'psi_error']
    for q in [u, u_teo, u_teo-u, v, v_teo, v_teo-v, psi, psi_teo, psi_teo-psi]:
        fig, ax = plot_quantity(q, x, y, label=labels.pop(0))
        circle = plt.Circle((0,0), R, color='firebrick', fill=True, alpha=.3)
        ax.add_artist(circle)
        # plt.savefig(f'{names.pop(0)}.png', bbox_inches='tight', dpi=256)
        plt.show()

def main():
    n_train, n_val, n_test  = 1e3, 5e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    name = f'fraenkels-flow/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}' 
    checkpoint = '' # input checkpoint name

    model = load_model(f'{checkpoint}.tar', name=name, args=args, initial_optm='lbfgs')

    bounds = [[-2.5,2.5],[-2.5,2.5],[.1,3],[.5,1.5]] # x,y,vort,R

    vort = 1
    R = .5

    plot_fields(model, N=500, bounds=bounds[:2], vort=vort, R=R, add_noise=False)

if __name__ == '__main__':
    main()