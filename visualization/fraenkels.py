import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider
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
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
######

import args
from src.utils import get_args, set_torch_dtype, set_seed #, set_torch_multiprocessing
from src.model import Model, REDUCED_VARS, NORMALIZE

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set seed for reproducibility
set_seed(args.seed)

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    # data = torch.from_numpy(data)
    return data

def load_data(name, folder='datasets'):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, folder)
    dataset = load_pickle(os.path.join(folder_path,name))
    return dataset

def change_parameters(args, dataset, layers=[6]*1):
    args.kind = 'incompressible'
    args.layers = layers
    args.n_epochs = 700
    args.n_epochs_adam = 500
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = 256
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "circ", "U_infty", "r"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True
    args.phi = [[ 1, 0, 0, 0,-1],
                [ 0, 1, 0, 0,-1],
                [ 0, 0,1,-1, -1]]
    args.phi_output = [0,0,0,1,1]

    if args.reduce_inputs==True:
        model = Model(dataset, args)
        pi = REDUCED_VARS.forward(model, torch.tensor(dataset['train'][:,:5])).T
        pi_np = pi.detach().numpy()
        args.sampling_box = [[np.min(pi_np[0]),np.max(pi_np[0])],[np.min(pi_np[1]),np.max(pi_np[1])],[np.min(pi_np[2]),np.max(pi_np[2])]] # pi1,pi2,pi3
    else:
        args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    return



def load_model_fr(data_size=100, layers=[6]*1):
    if data_size == 100:
        dataset = load_data('fraenkels/fraenkels_1e2_5e3_1_ypos_fixedR_fulldomain')
    elif data_size == 1_000:
        dataset = load_data('fraenkels/fraenkels_1e3_5e3_1_ypos_fixedR_fulldomain')
    elif data_size == 10_000:
        dataset = load_data('fraenkels/fraenkels_1e4_5e3_1_ypos_fixedR_fulldomain')
    else:
        raise ValueError('Invalid data size. Choose from 100, 1_000, or 10_000')
    
    change_parameters(args, dataset, layers=layers)
    
    if layers == [16]*2 and data_size == 100:
        checkpoint = 'src/best_models/incompressible_FWL9TJ_checkpoint_693.tar' 
    elif layers == [16]*2 and data_size == 1_000:
        checkpoint = 'src/best_models/incompressible_YNLZRU_checkpoint_583.tar' 
    elif layers == [16]*2 and data_size == 10_000:
        checkpoint = 'src/best_models/incompressible_KS7YQW_checkpoint_669.tar' 
    elif layers == [32]*4 and data_size == 100:
        checkpoint = 'src/best_models/incompressible_96USCV_checkpoint_666.tar'
    elif layers == [32]*4 and data_size == 1_000:
        checkpoint = 'src/best_models/incompressible_XVODPW_checkpoint_521.tar' 
    elif layers == [32]*4 and data_size == 10_000:
        checkpoint = 'src/best_models/incompressible_5W3P5R_checkpoint_669.tar' 
    else:
        raise ValueError('Invalid data size and layers combination')
    model = Model.load_checkpoint(checkpoint, dataset=dataset, args=args, initial_optm='lbfgs')
    return model

# model = load_model(data_size=data_size, layers=layers)

def data(xmin, xmax, ymin, ymax, v, U, r, model, N=250):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    X_, Y_ = np.meshgrid(x,y)
    X, Y = X_.flatten(), Y_.flatten()
    V = np.ones_like(X)*v
    U = np.ones_like(X)*U
    R = np.ones_like(X)*r

    inputs = np.column_stack((X,Y,C,U,R))
    inputs = torch.from_numpy(inputs).to(torch.float32)
    outputs = model.predict(inputs).detach().numpy()
    Ux = outputs[:,0].reshape(N,N).T
    Uy = outputs[:,1].reshape(N,N).T
    psi = model.psi(inputs).detach().numpy().reshape(N,N)

    return inputs, psi, Ux, Uy

def streamfunction(x, y, vort, U_inf, R=1):
    y = np.abs(y)
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    psi = vort*y**2/2 + vort/(2*np.pi)*((1-1/r**2)*y + (x*y*(r**4-1))/(2*r**4)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
                +1/2*((1+1/r**4)*(x**2-y**2)-2)*np.arctan(2*y/(r**2-1))) + U_inf*(1-1/r**2)*y 
    psi *= R**2
    return psi

def u(x,y,vort,U_inf,R=1):
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    return R**2*(vort*y + vort/(2*np.pi)*(2*y**2/r**4+(1-1/r**2) + x/2*(1-1/r**4+4*y**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 4*x**2*y**2*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1))-(2*y*(x**2-y**2)/r**6+(1+1/r**4)*y)*np.arctan(2*y/(r**2-1)) \
            + ((1+1/r**4)*(x**2-y**2)-2)*(x**2-y**2-1)/((r**2-1)**2+4*y**2)) \
            + U_inf*(1-(x**2-y**2)/r**4))
def v(x,y,vort,U_inf,R=1):
    r = np.sqrt(x**2 + y**2)/R
    x, y = x/R, y/R
    U_inf /= R
    return R**2*(-vort/(2*np.pi)*(2*x*y/r**4+y/2*(1-1/r**4+4*x**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 2*x*y*(x**2-y**2-1)*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1)) \
            - (2*x*(x**2-y**2)/r**6-(1+1/r**4)*x)*np.arctan(2*y/(r**2-1)) \
            - 2*x*y*((1+1/r**4)*(x**2-y**2)-2)/((r**2-1)**2+4*y**2)) \
            - U_inf*2*x*y/r**4)
def velocity(x, y, vort, U_inf, R=1):
    # U_inf *= -1
    if y<0: return -u(x,-y,vort,U_inf,R), v(x,-y,vort,U_inf,R)
    else: return u(x,y,vort,U_inf,R), v(x,y,vort,U_inf,R)

velocity = np.vectorize(velocity)

def error(X, Y, v, u, r, model):
    C = np.ones_like(X)*v
    U = np.ones_like(X)*u
    R = np.ones_like(X)*r
    mask = np.sqrt((X)**2 + (Y)**2) > r
    Xm = X[mask]
    Ym = Y[mask] 
    C = C[mask]
    V = V[mask]
    R = R[mask]
    target = velocity(Xm,Ym,v, u, r).T
    inputs = np.column_stack((Xm,Ym,V,U,R))
    inputs = torch.from_numpy(inputs).to(torch.float32)
    outputs = model.predict(inputs).detach().numpy()
    error = np.linalg.norm(target-outputs, axis=1)/np.linalg.norm(target, axis=1)*100
    error = np.mean(np.nan_to_num(error))
    return error

def plot(inputs, psi, Ux, Uy, c, U, r, model, N=250):
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_aspect('equal')
    inputs = inputs.detach().numpy()
    X, Y = inputs[:,0].reshape(N,N), inputs[:,1].reshape(N,N)
    im = ax.pcolor(X, Y, (np.sqrt(Ux**2+Uy**2)).T,shading='auto',cmap='viridis',vmin=0,vmax=2*1)
    ct = ax.contour(X,Y, psi+U*Y, levels=30,linewidths=.9, colors='firebrick')
    ax.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
    circle = plt.Circle((0,0),1,color='k',alpha=0.35)
    ax.add_artist(circle)
    cbar = plt.colorbar(im, ax=ax, label='Velocity magnitude', aspect=30, shrink=1, format='%.1f')
    err = error(X, Y, c, U, r, model)
    ax.text(2., 2.4, f'Error (rel l2)= {err:.2f}%', fontsize=11, ha='center', va='center', bbox=dict(facecolor='w', alpha=.9, edgecolor='black', boxstyle='round,pad=1'))
    return fig, ax

if __name__=='__main__':    
    # Define initial parameters
    init_circ = 0
    init_u = 1
    init_radius = 1

    model = load_model(data_size=100, layers=[6]*1)
    inputs, psi, Ux, Uy = data(-3, 3, -3, 3, init_circ, init_u, init_radius, model=model, N=250)
    plot(inputs, psi, Ux, Uy, c=init_circ, U=init_u, r=init_radius, model=model, N=250)
    plt.show()