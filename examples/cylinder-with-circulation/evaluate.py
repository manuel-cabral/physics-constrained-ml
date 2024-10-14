import numpy as np
import matplotlib.pyplot as plt
import os
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

def plot_predictions(circulations, pred, true, labels, quantity='lift', extrapolate=False, fname=None):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(circulations, true, 'k--', label='exact')
    colors = ['firebrick', 'forestgreen', 'royalblue', 'darkorange', 'purple']

    # axins = inset_axes(ax, width='45%', height='35%')
    # ip = InsetPosition(ax,[0.075, 0.075, 0.35, 0.3])
    # axins.set_axes_locator(ip)
    for i,pred_ in enumerate(pred):
        ax.plot(circulations, pred_, colors[i], label=labels[i])
    #     axins.plot(circulations, pred_-true, c=colors[i])
    # axins.hlines(0, -5, 5, color='gray', alpha=.5)
    # l = []
    # for i in circulations:
    #     if i>=-5 and i<=5:
    #         l.append(True)
    #     else:
    #         l.append(False)
    # print(pred_[l]-true[l])
    # b = np.max(np.abs(pred_[l]-true[l]))
    # b += b/10
    # axins.set_xlim(-5, 5)
    # axins.set_ylim(-b, b)
    # ax.plot(circulations, pred, label='$U_{\infty} R$ transformation')
    ax.fill_betweenx([min(np.concatenate((np.array(pred).flatten(),true))), max(np.concatenate((np.array(pred).flatten(),true)))], -5, 5, color='gray', alpha=.2)
    ax.set_xlabel('$\Gamma$')
    ax.set_ylabel(quantity)
    # ax.set_title(f'{quantity} vs Circulation')
    ax.legend()


    if extrapolate:
        ax.set_xlim(-15,15)
    plt.show()
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=256)
    return

def plot_lift_and_drag(checkpoints, data_file, labels, fname=None):
    circulations = np.linspace(-15,15,250)
    # trained = generate_sobol(4, -5, 5, seed=args.seed)
    cl_preds, cd_preds = [], []
    phi_output = [[0,0,0,0,0],[0,0,0,1,1]]
    for i,checkpoint in enumerate(checkpoints):
        args.phi_output = phi_output[1]
        if i == 1:
            args.normalize_inputs = True
            args.reduce_inputs = False
            args.transform_output = False
            args.kind = 'baseline'
            # args.layers = [16]*2 
        model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')
        cl, cl_pred, cd, cd_pred = evaluate(model, circulations)
        cl_preds.append(cl_pred)
        cd_preds.append(cd_pred)
        
    plot_predictions(circulations, cl_preds, cl, labels=labels, quantity='$C_l$', extrapolate=True, fname=f'{fname}_cl.png')
    plot_predictions(circulations, cd_preds, cd, labels=labels, quantity='$C_d$', extrapolate=True, fname=f'{fname}_cd.png')
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

    args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    args.phi = [[1,0,0,0,-1],[0,1,0,0,-1],[0,0,1,-1,-1]]

    # args.phi_output = [0,0,0,0,0] # 1
    # args.phi_output = [0,0,0,1,1] # U_inf * R


    return

def main():
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    data_file = f'cylinder-with-circulation/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'
    
    checkpoint_base = '' # input the baseline checkpoint name here
    checkpoint_inc = '' # input the incompressible checkpoint name here

    fname = None
    plot_lift_and_drag([checkpoint_inc, checkpoint_base], data_file, labels=['input-output physics', 'no physics'], fname=fname)
    
    return

if __name__ == '__main__':
    main()