import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import matplotlib.colors as colors


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
from src.utils import get_args, set_seed, set_torch_dtype, load_data, load_pickle, ftype_torch #, set_torch_multiprocessing
from src.model import Model
import src.config

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set multiprocessing
# set_torch_multiprocessing()

# Set seed for reproducibility
set_seed(args.seed)

def load_model(checkpoint, name, args, pasta='datasets', folder='best_models', initial_optm='lbfgs'):
    file_path = os.path.join(os.getcwd(), 'src', folder, f'{checkpoint}')
    dataset = load_data(name, pasta)
    model = Model.load_checkpoint(file_path, dataset=dataset, args=args, initial_optm=initial_optm)

    return model

args.kind = 'incompressible'
# args.kind = 'baseline'
args.layers = [6]*1
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

args.sampling_box = [(5e-2, 1), (0, 5e-2), (1/7e4,1/3e4), (0,5)]

# args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
args.phi = [[-.5,1,-.5,.5]]
args.phi_output = [0,1,0,1]

checkpoint = 'incompressible_IIUKC2_checkpoint_690'

model = load_model(f'{checkpoint}.tar', name='data_5e2_1e3_1_idx0', args=args, initial_optm='lbfgs')

def plot_field(var, bounds, N, ratio=False):

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow((var).reshape(N,N), extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], origin='lower', cmap='RdBu',) \
                # norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=250, base=10))
    ax.set_aspect(20)
    cbar = plt.colorbar(im)
    cbar.set_label('Velocity magnitude')
    cbar.formatter.set_powerlimits((0, 0))
    if ratio: im.set_clim(0, 250)
    plt.show()


bounds = [(5e-2, 1), (1e-7, 5e-2), (1/7e4,1/3e4),(0,5)]
N = 500
x = np.linspace(bounds[0][0], bounds[0][1], N)
y = np.linspace(bounds[1][0], bounds[1][1], N)
X, Y = np.meshgrid(x, y)
visc = np.ones_like(X)*1/5e4
U_inf = np.ones_like(X)*1
points = np.column_stack((X.flatten(),Y.flatten(),visc.flatten(),U_inf.flatten()))
points = torch.from_numpy(points).to(torch.float32)
u,v = (model.predict(points, kind='incompressible')).detach().numpy().T
psi = model.psi(points).detach().numpy().reshape(N,N)

plot_field(u, bounds, N)
plot_field(v, bounds, N)
plot_field(psi, bounds, N)
plot_field(u/v, bounds, N, ratio=True)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(2*X/Y, extent=[5e-2, 1, 0, 5e-2], origin='lower', cmap='RdBu',)  \
               # norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=250, base=10))
ax.set_aspect(20)
cbar = plt.colorbar(im)
cbar.set_label('Velocity magnitude')
im.set_clim(0, 250)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
# print('ratio: \n', 2*X/Y)
# print('\nu/v: \n', (u/v).reshape(N,N))
# print('\ndifference: \n', 2*X/Y-(u/v).reshape(N,N))
im = ax.imshow(2*X/Y/(u/v).reshape(N,N), extent=[5e-2, 1, 0, 5e-2], origin='lower', cmap='RdBu',) 
ax.set_aspect(20)
cbar = plt.colorbar(im)
cbar.set_label('Velocity magnitude')
# im.set_clim(-.1, .1)
plt.show()