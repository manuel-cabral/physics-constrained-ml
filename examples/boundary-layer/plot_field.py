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
    args.x_vars = ["x", "y", "visc", "U_infty"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True

    args.sampling_box = [(5e-2, 1), (0, 5e-2), (1/7e4,1/3e4), (0,5)]

    args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]

    # args.phi_output = [0,0,0,0] # 1
    # args.phi_output = [.5,0,.5,.5] # sqrt(U_inf * visc * x)
    args.phi_output = [0,1,0,1] # U_inf * y
    
    return

def bl_thickness(x, visc, U_inf):
    return 5*np.sqrt(x*visc/U_inf)

def plot_fields(model, N=200, bounds=[[5e-2, 1],[0, .05]], Re=5e4, U_inf=1, add_noise=False):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)

    visc = np.ones_like(xx)*1/Re
    U_inf = np.ones_like(yy)*U_inf

    points = np.column_stack((xx.flatten(),yy.flatten(),visc.flatten(),U_inf.flatten()))
    points = torch.from_numpy(points).to(torch.float32)
    u,v = (model.predict(points)).detach().numpy().T
    u,v = u.reshape(N,N), v.reshape(N,N)
    psi = model.psi(points).detach().numpy().reshape(N,N)

    u_teo,v_teo,psi_teo = quantities(xx, yy, 1/Re, U_inf[0])
    print(f'Error u: {100*np.linalg.norm(u-u_teo)/np.linalg.norm(u_teo):.2f}%')
    print(f'Error v: {100*np.linalg.norm(v-v_teo)/np.linalg.norm(v_teo):.2f}%')
    # print(f'Error psi: {100*np.linalg.norm(psi-psi_teo)/np.linalg.norm(psi_teo-U_inf*yy):.2f}%')

    if add_noise: 
        for q in [u,v,psi]: q = add_noise(q)

    labels = ['$u$', '$v$', '$\Tilde{\psi}$']
    for q in [u,v,psi-U_inf*yy]:
        fig, ax = plot_quantity(q, x, y, label=labels.pop(0))
        ax.plot(x, bl_thickness(x, visc[0], U_inf[0]), 'k--', label='BL thickness')
        plt.show()

def main():
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    data_file = f'boundary-layer/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    # checkpoint = 'incompressible_3SHWCO_checkpoint_1783' # [3]*1, w/ sched,

    # checkpoint = 'incompressible_G2KLXA_checkpoint_1810' # [6]*1, out=[0,0,0,0]
    # checkpoint = 'incompressible_KP4QVS_checkpoint_1602' # [6]*1, out=[.5,0,.5,.5]
    checkpoint = 'incompressible_78DSCZ_checkpoint_1880' # [6]*1, out=[0,1,0,1]


    model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')

    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,5]] # x,y,visc,U_inf

    plot_fields(model, N=250, bounds=bounds[:2], Re=5e4, U_inf=1, add_noise=False)

if __name__ == '__main__':
    main()