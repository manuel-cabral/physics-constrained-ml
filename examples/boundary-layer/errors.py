#! Importing libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from generate_data import quantities
from scipy.stats import median_abs_deviation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

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
from src.model import Model
import src.config

#! Set parameters
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

    # args.phi_output = [.5,0,.5,.5] # sqrt(U_inf * visc * x)
    args.phi_output = [0,1,0,1] # U_inf * y

    return

def displacement_errors(model, Re=5e4, U_inf=1, chord=.9, N=1000):
    displacement_u = 1.7207874472823652
    displacement_v = 2.5370438217529157
    d_acc = np.zeros((2,), dtype=np.float32) # {u,v}
    ymax = 10./np.sqrt(U_inf*Re/chord) # 2*BL_thickness
    y = torch.linspace(1e-5, ymax, N)
    x = chord*torch.ones_like(y)
    eta = y*np.sqrt(U_inf*Re/x)

    u_,v_ = model.predict(torch.column_stack((x,y,1/Re*torch.ones_like(x),U_inf*torch.ones_like(x)))).T
    u_ = u_.detach().numpy()
    v_ = v_.detach().numpy()
    beta_u = np.trapz((1-u_/u_[-1]), eta)
    d_acc[0] = 100*(np.abs(displacement_u - np.nan_to_num(beta_u))/displacement_u)
    beta_v = np.trapz((1-v_/v_[-1]), eta)
    d_acc[1] = 100*(np.abs(displacement_v - np.nan_to_num(beta_v))/displacement_v)

    u_exact, v_exact, _ = quantities(x, y, 1/Re*torch.ones_like(x), U_inf*torch.ones_like(x))  
    d_sq = np.zeros((2,), dtype=np.float32) # {u,v}
    d_sq[0] = np.linalg.norm((u_ - u_exact)/u_exact[-1], 2)
    d_sq[1] = np.linalg.norm((v_ - v_exact)/v_exact[-1], 2)
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot(u_, eta, c='firebrick', label='u')
    # ax.plot(u_exact, eta, 'k--', label='u exact')
    # ax.legend(loc='lower right', fontsize=14)
    # ax.set_xlabel('$u$', fontsize=16)
    # ax.set_ylabel('$\eta$', fontsize=16)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axins = inset_axes(ax, width='45%', height='35%')
    # ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    # axins.set_axes_locator(ip)
    # axins.plot(u_-u_exact, eta, c='steelblue')
    # axins.vlines(0, 0, 10, color='forestgreen', alpha=.5)
    # b = np.max(np.abs(u_-u_exact))
    # b += b/10
    # axins.set_xlim(-b, b)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # fname = f'u_profile_0000_idx3.png'
    # # plt.savefig(fname, bbox_inches='tight', dpi=256)
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot(v_, eta, c='firebrick', label='v')
    # ax.plot(v_exact, eta, 'k--', label='v exact')
    # plt.legend(loc='lower right',  fontsize=14)
    # ax.set_xlabel('$v$', fontsize=16)
    # ax.set_ylabel('$\eta$', fontsize=16)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axins = inset_axes(ax, width='45%', height='35%')
    # ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    # axins.set_axes_locator(ip)
    # axins.plot(v_-v_exact, eta, c='steelblue')
    # axins.vlines(0, 0, 10, color='forestgreen', alpha=.5)
    # b = np.max(np.abs(v_-v_exact))
    # b += b/10
    # axins.set_xlim(-b, b)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # fname = f'v_profile_0000_idx3.png'
    # # plt.savefig(fname, bbox_inches='tight', dpi=256)
    # plt.show()

    return d_acc[0], d_acc[1], d_sq[0], d_sq[1]

def errors(model, Re=5e4, U_inf=1, Nx=100, Ny=100):
    x = torch.linspace(5e-2, 1, Nx)
    y = torch.linspace(1e-7, 5e-2, Ny)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    visc = 1/Re*torch.ones_like(X)
    U_inf = U_inf*torch.ones_like(X)
    points = torch.column_stack((X.flatten(),Y.flatten(),visc.flatten(),U_inf.flatten()))
    points = points.to(torch.float32)
    vel = (model.predict(points, kind='incompressible')).detach().numpy().T
    psi = model.psi(points).detach().numpy().reshape(Nx,Ny)
    u = vel[0].reshape(Nx,Ny)
    v = vel[1].reshape(Nx,Ny)
    psi = psi.reshape(Nx,Ny)
    *vel_exact, psi_exact = quantities(X, Y, visc, U_inf)
    u_exact = vel_exact[0].reshape(Nx,Ny)
    v_exact = vel_exact[1].reshape(Nx,Ny)
    psi_exact = psi_exact.reshape(Nx,Ny) - (U_inf*Y).numpy()
    u_error = 100*np.linalg.norm(u_exact - u)/np.linalg.norm(u_exact)
    v_error = 100*np.linalg.norm(v_exact - v)/np.linalg.norm(v_exact)
    psi_error = 100*np.linalg.norm(psi_exact - psi)/np.linalg.norm(psi_exact)

    gl_error = 100*np.linalg.norm(vel.reshape((2,Nx,Ny))-np.array(vel_exact))/np.linalg.norm(np.array(vel_exact))

    return u_error, v_error, gl_error, psi_error


def displacement_u(model, ax=None, axins=None, Re=5e4, U_inf=1, chord=.9, N=1000, label='u', color='firebrick', plot_target=False, save=False):
    displacement_u = 1.7207874472823652
    displacement_v = 2.5370438217529157
    d_acc = np.zeros((2,), dtype=np.float32) # {u,v}
    ymax = 10./np.sqrt(U_inf*Re/chord) # 2*BL_thickness
    y = torch.linspace(1e-5, ymax, N)
    x = chord*torch.ones_like(y)
    eta = y*np.sqrt(U_inf*Re/x)

    u_,v_ = model.predict(torch.column_stack((x,y,1/Re*torch.ones_like(x),U_inf*torch.ones_like(x)))).T
    u_ = u_.detach().numpy()
    v_ = v_.detach().numpy()
    beta_u = np.trapz((1-u_/u_[-1]), eta)
    d_acc[0] = 100*(np.abs(displacement_u - np.nan_to_num(beta_u))/displacement_u)
    beta_v = np.trapz((1-v_/v_[-1]), eta)
    d_acc[1] = 100*(np.abs(displacement_v - np.nan_to_num(beta_v))/displacement_v)

    u_exact, v_exact, _ = quantities(x, y, 1/Re*torch.ones_like(x), U_inf*torch.ones_like(x))  
    
    fig, ax = plt.subplots(figsize=(10, 10)) if ax==None else ax
    if plot_target:
        ax.plot(u_exact, eta, 'k--', label='exact')
    ax.plot(u_, eta, c=color, label=label)
    ax.legend(loc='lower right', fontsize=16)
    ax.set_xlabel('$u$', fontsize=20)
    ax.set_ylabel('$\eta$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axins = inset_axes(ax, width='45%', height='35%') if axins==None else axins
    ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    axins.set_axes_locator(ip)
    axins.plot(u_-u_exact, eta, c=color)
    axins.vlines(0, 0, 10, color='gray', alpha=.5)
    b = np.max(np.abs(u_-u_exact))
    b += b/10
    axins.set_xlim(-b, b)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if save: 
        fname = f'imgs/u_profile_comparison.png'
        plt.savefig(fname, bbox_inches='tight', dpi=256)

    return fig, ax, axins

def displacement_v(model, ax=None, axins=None, Re=5e4, U_inf=1, chord=.9, N=1000, label='v', color='firebrick', plot_target=False, save=False):
    displacement_u = 1.7207874472823652
    displacement_v = 2.5370438217529157
    d_acc = np.zeros((2,), dtype=np.float32) # {u,v}
    ymax = 10./np.sqrt(U_inf*Re/chord) # 2*BL_thickness
    y = torch.linspace(1e-5, ymax, N)
    x = chord*torch.ones_like(y)
    eta = y*np.sqrt(U_inf*Re/x)

    u_,v_ = model.predict(torch.column_stack((x,y,1/Re*torch.ones_like(x),U_inf*torch.ones_like(x)))).T
    u_ = u_.detach().numpy()
    v_ = v_.detach().numpy()
    beta_u = np.trapz((1-u_/u_[-1]), eta)
    d_acc[0] = 100*(np.abs(displacement_u - np.nan_to_num(beta_u))/displacement_u)
    beta_v = np.trapz((1-v_/v_[-1]), eta)
    d_acc[1] = 100*(np.abs(displacement_v - np.nan_to_num(beta_v))/displacement_v)

    u_exact, v_exact, _ = quantities(x, y, 1/Re*torch.ones_like(x), U_inf*torch.ones_like(x))  

    fig, ax = plt.subplots(figsize=(10, 10)) if ax==None else ax
    if plot_target:
        ax.plot(v_exact, eta, 'k--', label='exact')
    ax.plot(v_, eta, c=color, label=label)
    ax.legend(loc='lower right', fontsize=16)
    ax.set_xlabel('$v$', fontsize=20)
    ax.set_ylabel('$\eta$', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axins = inset_axes(ax, width='45%', height='35%') if axins==None else axins
    ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    axins.set_axes_locator(ip)
    axins.plot(v_-v_exact, eta, c=color)
    axins.vlines(0, 0, 10, color='gray', alpha=.5)
    b = np.max(np.abs(v_-v_exact))
    b += b/10
    axins.set_xlim(-b, b)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if save:
        fname = f'imgs/v_profile_comparison.png'
        plt.savefig(fname, bbox_inches='tight', dpi=256)

    return fig, ax, axins


def main():
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    Re = 5e4
    U_inf = 1

    x_component = True

    change_parameters(args, data_size=n_train)

    data_file = f'boundary-layer/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    checkpoint = '' # input checkpoint name
    args.phi_output = [0,1,0,1] # U_inf * y
    
    model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')
    if x_component: 
        # fig, ax, axins = displacement_u(model, (fig,ax), axins, Re=Re, U_inf=U_inf, label='$U_{\infty} y$ transformation', color='firebrick', save=False)
        fig, ax, axins = displacement_u(model, Re=Re, U_inf=U_inf, label='input-output physics', color='firebrick', plot_target=True, save=False)
    else:
        # fig, ax, axins = displacement_v(model, (fig,ax), axins, Re=Re, U_inf=U_inf, label='$U_{\infty} y$ transformation', color='firebrick', save=False)
        fig, ax, axins = displacement_v(model, Re=Re, U_inf=U_inf, label='input-output physics', color='firebrick', plot_target=True, save=False)

    # plt.savefig('u_profile_comparison.png', bbox_inches='tight', dpi=256)
    plt.show()




if __name__=='__main__':
    main()