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
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
######

import args
from src.utils import get_args, set_torch_dtype, set_seed #, set_torch_multiprocessing
from src.model import Model

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

def change_parameters(args):
    args.kind = 'incompressible'
    args.layers = [6]*1
    # args.layers = [16]*2
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

    args.normalize_inputs = True
    args.reduce_inputs = True
    args.transform_output = True

    if args.reduce_inputs==True:
        from src.model import REDUCED_VARS, NORMALIZE
        model = Model(dataset, args)
        pi = REDUCED_VARS.forward(model, torch.tensor(dataset['train'][:,:5])).T
        pi_np = pi.detach().numpy()
        args.sampling_box = [[np.min(pi_np[0]),np.max(pi_np[0])],[np.min(pi_np[1]),np.max(pi_np[1])],[np.min(pi_np[2]),np.max(pi_np[2])]] # pi1,pi2,pi3
    else:
        args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    return

args.phi = [[ 1, 0, 0, 0,-1],
            [ 0, 1, 0, 0,-1],
            [ 0, 0,1,-1, -1]]
args.phi_output = [0,0,0,1,1]

dataset = load_data('data_1e2_5e3_1_idx0')
# dataset = load_data('data_1e3_5e3_1_idx0')
# dataset = load_data('data_1e4_5e3_1_idx0')

change_parameters(args)
checkpoint = 'src/best_models/incompressible_QAKI7B_checkpoint_675.tar' # reduced - 'data_1e2_5e3_1_idx0', [6]*1, 1.87% val error
# checkpoint = 'src/best_models/incompressible_QK6AZQ_checkpoint_699.tar' # reduced - 'data_1e4_5e3_1_idx0', [16]*2, 0.54% val error

model = Model.load_checkpoint(checkpoint, dataset=dataset, args=args, initial_optm='lbfgs')

fig, ax = plt.subplots(figsize=(14,14))
N = 250
x = np.linspace(-3,3,N)
y = np.linspace(-3,3,N)
X_, Y_ = np.meshgrid(x,y)
X = X_.flatten()
Y = Y_.flatten()
# Define initial parameters
init_circ = 0
init_u = 1
init_radius = 1
ax.set_xlabel("$x$", fontsize=16)
ax.set_ylabel("$y$", fontsize=16)
ax.set_aspect('equal')
C = np.ones_like(X)*0
U = np.ones_like(X)*1
R = np.ones_like(X)*1
inputs = np.column_stack((X,Y,C,U,R))
inputs = torch.from_numpy(inputs).to(torch.float32)
outputs = model.predict(inputs).detach().numpy()
Ux = outputs[:,0].reshape(N,N).T
Uy = outputs[:,1].reshape(N,N).T
im = ax.pcolor(x, y, (np.sqrt(Ux**2+Uy**2)).T,shading='auto',cmap='viridis',vmin=0,vmax=2*1)
psi = model.psi(inputs).detach().numpy().reshape(N,N)
ct = ax.contour(x,y, psi+1*Y_,levels=30,linewidths=.9, colors='firebrick')
ax.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
circle = plt.Circle((0,0),1,color='k',alpha=0.35)
ax.add_artist(circle)
cbar = plt.colorbar(im, ax=ax, label='Velocity magnitude', aspect=30, shrink=1, format='%.1f')

def velocity(r,theta, circulation=0, U_inf=1, R=1):
    u = (1-np.cos(2*theta)*R**2/r**2)*U_inf - circulation*np.sin(theta)/(2*np.pi*r)
    v = -R**2/r**2*np.sin(2*theta)*U_inf + circulation*np.cos(theta)/(2*np.pi*r)
    return np.array((u, v))
def cart2polar(x,y):
    return (np.sqrt(x**2 + y**2),np.arctan2(y, x))

def error(circ, u, r):
    C = np.ones_like(X_)*circ
    U = np.ones_like(X_)*u
    R = np.ones_like(X_)*r
    mask = np.sqrt((X_)**2 + (Y_)**2) > r
    Xm = X_[mask]
    Ym = Y_[mask] 
    C = C[mask]
    U = U[mask]
    R = R[mask]
    coord = cart2polar(Xm,Ym)
    target = velocity(*coord,circ, u, r).T
    inputs = np.column_stack((Xm,Ym,C,U,R))
    inputs = torch.from_numpy(inputs).to(torch.float32)
    outputs = model.predict(inputs).detach().numpy()
    error = np.linalg.norm(target-outputs, axis=1)/np.linalg.norm(target, axis=1)*100
    error = np.mean(np.nan_to_num(error))
    return error

def coef_pd(c, u, r):
    theta = np.linspace(0,2*np.pi,1000)
    x, y = r*np.cos(theta), r*np.sin(theta)
    C = np.ones_like(x)*c
    U = np.ones_like(x)*u
    R = np.ones_like(x)*r
    X = np.column_stack((x,y,C,U,R))
    X = torch.tensor(X, dtype=torch.float32)
    vel_pred = torch.norm(model.predict(X).detach(), dim=1)
    cp_pred = 1 - vel_pred**2/u**2

    d_theta = np.diff(theta)[0]
    cl_pred = -.5 * torch.sum(cp_pred*np.sin(theta)*d_theta)
    cl  = -c/(u*r)

    cd_pred = -.5 * torch.sum(cp_pred*np.cos(theta)*d_theta)
    cd = 0

    return cl, cl_pred, cd, cd_pred 
err = error(init_circ, init_u, init_radius)
cl, cl_pred, cd, cd_pred  = coef_pd(init_circ, init_u, init_radius)
cl_err =  np.linalg.norm(cl-cl_pred)
cd_err =  np.linalg.norm(cd-cd_pred)
ax.text(2., 2.4, f'Error (rel l2)= {err:.2f}%\n$C_l$ error (l2) = {cl_err:.2f}\n$C_d$ error (l2) = {cd_err:.2f}', fontsize=11, ha='center', va='center', bbox=dict(facecolor='w', alpha=.9, edgecolor='black', boxstyle='round,pad=1'))

def plot(ax, c,u,r):
    C = np.ones_like(X)*c
    U = np.ones_like(X)*u
    R = np.ones_like(X)*r
    inputs = np.column_stack((X,Y,C,U,R))
    inputs = torch.from_numpy(inputs).to(torch.float32)
    outputs = model.predict(inputs).detach().numpy()
    Ux = outputs[:,0].reshape(N,N).T
    Uy = outputs[:,1].reshape(N,N).T
    im = ax.pcolor(x, y, (np.sqrt(Ux**2+Uy**2)).T,shading='auto',cmap='viridis',vmin=0,vmax=2*u,)
    psi = model.psi(inputs).detach().numpy().reshape(N,N)
    ct = ax.contour(x,y, psi+u*Y_,levels=30,linewidths=.9, colors='firebrick')
    ax.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
    circle = plt.Circle((0,0),r,color='k',alpha=0.35)
    ax.add_artist(circle)
    err = error(c, u, r)
    cl, cl_pred, cd, cd_pred  = coef_pd(c,u,r)
    cl_err =  np.linalg.norm(cl-cl_pred)
    cd_err =  np.linalg.norm(cd-cd_pred)
    ax.text(2., 2.4, f'Error (rel l2)= {err:.2f}%\n$C_l$ error (l2) = {cl_err:.2f}\n$C_d$ error (l2) = {cd_err:.2f}', fontsize=11, ha='center', va='center', bbox=dict(facecolor='w', alpha=.9, edgecolor='black', boxstyle='round,pad=1'))
    plt.show()

    return im


# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

axcirc = fig.add_axes([0.25, 0.16, 0.6, 0.03])
circ_slider = Slider(
    ax=axcirc,
    label='$\Gamma$',
    valmin=-5,
    valmax=5,
    valinit=init_circ,
)

axu = fig.add_axes([0.25, 0.12, 0.6, 0.03])
u_slider = Slider(
    ax=axu,
    label="$U_{\infty}$",
    valmin=.5,
    valmax=5,
    valinit=init_u,
)

axrad = fig.add_axes([0.25, 0.085, 0.6, 0.03])
rad_slider = Slider(
    ax=axrad,
    label="$R$",
    valmin=.1,
    valmax=3,
    valinit=init_radius,
)

# The function to be called anytime a slider's value changes
def update(val):
    ax.clear()
    im = plot(ax, circ_slider.val, u_slider.val, rad_slider.val)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_aspect('equal')
    im.set_clim([0, 2*u_slider.val])
    cbar.update_normal(im)
    
    fig.canvas.draw_idle()


# register the update function with each slider
circ_slider.on_changed(update)
u_slider.on_changed(update)
rad_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    circ_slider.reset()
    u_slider.reset()
    rad_slider.reset()
button.on_clicked(reset)

plt.show()