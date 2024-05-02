#! Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
from scipy.linalg import null_space
from scipy.stats import median_abs_deviation
import torch
import os
import glob
import pickle
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
from src.utils import get_args, set_seed, set_torch_dtype  #, set_torch_multiprocessing
from src.model import Model
import src.config

#! Set parameters
# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set seed for reproducibility
set_seed(args.seed)

#! Solving ODE
def blasius_eq(f,z):
    return f[1], f[2], -f[0]*f[2]/2

def find_c0(f2_0):
    f = odeint(blasius_eq, (0,0,f2_0), [0, 10])
    return 1.-f[-1, 1]

f0_0, f1_0 = 0, 0
f2_0 = fsolve(find_c0, .5)

def solve_pointwise(z, density=250):
    initial = f0_0, f1_0, f2_0
    z_span = np.linspace(0,z,int(z*density))
    f, f_z, f_zz = odeint(blasius_eq, initial, z_span).T
    return f, f_z, f_zz, z_span

def quantities(x, y, visc, U_inf):
    pi1 = np.sqrt(U_inf * x / visc)
    pi2 = y * np.sqrt(U_inf / (visc * x))

    f, f_z,*_ = solve_pointwise(pi2, density=250)
    f, f_z = f[-1] if len(f) else 0, f_z[-1] if len(f_z) else 0
    u = U_inf * f_z     # u = U_inf * f'
    v = U_inf/(2*pi1)*(pi2 * f_z - f)   # v = 1/(2*pi1) * (pi2 * f' - f)

    psi = U_inf * y * f / pi2 if y!=0 else 0

    return u, v, psi

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def load_data(name=None, pasta='datasets'):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, pasta)
    dataset = load_pickle(os.path.join(folder_path,name))
    return dataset

def load_model(checkpoint, name, args, pasta='datasets', folder='best_models', initial_optm='lbfgs'):
    file_path = os.path.join(os.getcwd(), 'src', folder, f'{checkpoint}')
    dataset = load_data(name, pasta)
    model = Model.load_checkpoint(file_path, dataset=dataset, args=args, initial_optm=initial_optm)

    return model

# make quantities a vectorized function
quantities = np.vectorize(quantities, otypes=[args.ftype, args.ftype, args.ftype])

def change_parameters(args):
    args.kind = 'incompressible'
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

    return

def phi_base(D):
    return null_space(D)

def phi_base(D):
    return null_space(D)
def phi_zero(D):
    phi = phi_base(D)

    phi[:,-1] /= -phi[-1,-1]
    
    phi[:,0] += phi[-1,0]*phi[:,-1]
    phi[:,1] += phi[-1,1]*phi[:,-1]

    return phi

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
    # ax.plot(range(len(u_)), [(np.trapz((1-u_[:i]/u_[i]),eta[:i])-displacement_u)**2 for i in range(len(u_))], label='u')
    # ax.set_yscale('symlog', linthresh=1e-5)
    # plt.show()
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(u_, eta, c='firebrick', label='u')
    ax.plot(u_exact, eta, 'k--', label='u exact')
    ax.legend(loc='lower right', fontsize=14)
    ax.set_xlabel('$u$', fontsize=16)
    ax.set_ylabel('$\eta$', fontsize=16)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axins = inset_axes(ax, width='45%', height='35%')
    ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    axins.set_axes_locator(ip)
    axins.plot(u_-u_exact, eta, c='steelblue')
    axins.vlines(0, 0, 10, color='forestgreen', alpha=.5)
    b = np.max(np.abs(u_-u_exact))
    b += b/10
    axins.set_xlim(-b, b)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fname = f'u_profile_0000_idx3.png'
    # plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(v_, eta, c='firebrick', label='v')
    ax.plot(v_exact, eta, 'k--', label='v exact')
    plt.legend(loc='lower right',  fontsize=14)
    ax.set_xlabel('$v$', fontsize=16)
    ax.set_ylabel('$\eta$', fontsize=16)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axins = inset_axes(ax, width='45%', height='35%')
    ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    axins.set_axes_locator(ip)
    axins.plot(v_-v_exact, eta, c='steelblue')
    axins.vlines(0, 0, 10, color='forestgreen', alpha=.5)
    b = np.max(np.abs(v_-v_exact))
    b += b/10
    axins.set_xlim(-b, b)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fname = f'v_profile_0000_idx3.png'
    # plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

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

    # fig, ax = plt.subplots(figsize=(6, 6))
    # im = ax.imshow((u).reshape(Nx,Ny), extent=[5e-2, 1, 1e-7, 5e-2,], origin='lower', cmap='RdBu',)
    # ax.set_aspect(20)
    # cbar = plt.colorbar(im)
    # cbar.set_label('Velocity magnitude')
    # cbar.formatter.set_powerlimits((0, 0))
    # plt.show()

    # fig, ax = plt.subplots(figsize=(6, 6))
    # im = ax.imshow((v).reshape(Nx,Ny), extent=[5e-2, 1, 1e-7, 5e-2,], origin='lower', cmap='RdBu',)
    # ax.set_aspect(20)
    # cbar = plt.colorbar(im)
    # cbar.set_label('Velocity magnitude')
    # cbar.formatter.set_powerlimits((0, 0))
    # plt.show()

    return u_error, v_error, gl_error, psi_error

def main():
    change_parameters(args)

    D = [[1,1,2,1, 2],[0,0,-1,-1, -1]] # x,y,visc,U_inf,psi
    phi = phi_zero(D).T
    *inputs, outputs = phi
    args.phi = [input[:-1] for input in inputs]
    args.phi_output = [0,1,0,1]

    # checkpoint = 'incompressible_TAD95M_checkpoint_677'
    # args.transform_output = False
    # args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
    # args.phi_output = [0,1,0,1]

    # args.phi_output = [0,0,0,0]

    dir = os.path.join(os.getcwd(), 'src', 'best_models')
    outs = [[0,0,0,0],[0,1,0,1],[1,0,0,1],[0,0,1,0],[1,-1,1,0]]
    pis = 'pi1pi2'
    # args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
    # pis = 'pi2'
    # args.phi = [[-.5,1,-.5,.5]]
    k = 1
    args.phi_output = outs[k]
    out = ''.join([str(i) for i in outs[k]])
    # args.normalize_inputs = True
    # args.reduce_inputs = not args.normalize_inputs
    checkpoints = glob.glob(f'{dir}/{args.kind}_BASIS_transform{out}_idx*_*.tar')
    checkpoints = [check.split('/')[-1] for check in checkpoints]

    U_inf = 2
    Re = 5e4
    du, dv, squ, sqv, eu, ev, eg = np.zeros((7,len(checkpoints)), dtype=np.float32)
    for i,checkpoint in enumerate(checkpoints[:]):
        print(checkpoint)
        model = load_model(checkpoint, name='data_5e2_1e3_1_idx0', args=args, initial_optm='lbfgs')
        du[i], dv[i], squ[i], sqv[i] = displacement_errors(model, Re=Re, U_inf=U_inf)
        # eu[i], ev[i], eg[i] = errors(model, Re=Re, U_inf=U_inf)[:3]
    # # print(eg)
    # print(f"Global error: ${np.nanmedian(eg):.2f}\pm{median_abs_deviation(eg, nan_policy='omit'):.2f}$")
    # print(f"Error u: ${np.nanmedian(eu):.2f}\pm{median_abs_deviation(eu, nan_policy='omit'):.2f}$")
    # print(f"Error v: ${np.nanmedian(ev):.2f}\pm{median_abs_deviation(ev, nan_policy='omit'):.2f}$")
    # print(f"displacement u: ${np.nanmedian(du):.2f}\pm{median_abs_deviation(du, nan_policy='omit'):.2f}$")
    # print(f"displacement v: ${np.nanmedian(dv):.2f}\pm{median_abs_deviation(dv, nan_policy='omit'):.2f}$")
    # print(f"displacement u (l2): ${np.nanmedian(squ):.2f}\pm{median_abs_deviation(squ, nan_policy='omit'):.2f}$")
    # print(f"displacement v (l2): ${np.nanmedian(sqv):.2f}\pm{median_abs_deviation(sqv, nan_policy='omit'):.2f}$")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
def displacement_u(model, ax=None, axins=None, Re=5e4, U_inf=1, chord=.9, N=1000, label='u', color='firebrick', save=False):
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
    if color=='firebrick':
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

def displacement_v(model, ax=None, axins=None, Re=5e4, U_inf=1, chord=.9, N=1000, label='v', color='firebrick', save=False):
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

    # fig, ax = plt.subplots(figsize=(10, 10)) if ax==None else ax
    # if color=='firebrick':
    #     ax.plot(v_exact, eta, 'k--', label='v exact')
    # ax.plot(v_, eta, c=color, label=label)
    # plt.legend(loc='lower right',  fontsize=16)
    # ax.set_xlabel('$v$', fontsize=20)
    # ax.set_ylabel('$\eta$', fontsize=20)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # axins = inset_axes(ax, width='45%', height='35%') if axins==None else axins
    # ip = InsetPosition(ax,[0.15, 0.6, 0.45, 0.35])
    # axins.set_axes_locator(ip)
    # axins.plot(v_-v_exact, eta, c=color)
    # axins.vlines(0, 0, 10, color='gray', alpha=.5)
    # b = np.max(np.abs(v_-v_exact))
    # b += b/10
    # axins.set_xlim(-b, b)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig, ax = plt.subplots(figsize=(10, 10)) if ax==None else ax
    if color=='firebrick':
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


def main2():
    change_parameters(args)

    D = [[1,1,2,1, 2],[0,0,-1,-1, -1]] # x,y,visc,U_inf,psi
    phi = phi_zero(D).T
    *inputs, outputs = phi
    args.phi = [input[:-1] for input in inputs]
    # args.phi_output = [0,1,0,1]

    # checkpoint = 'incompressible_TAD95M_checkpoint_677'
    # args.transform_output = False
    # args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
    # args.phi_output = [0,1,0,1]

    # args.phi_output = [0,0,0,0]

    dir = os.path.join(os.getcwd(), 'src', 'best_models')
    outs = [[0,0,0,0],[0,1,0,1],[1,0,0,1],[0,0,1,0],[1,-1,1,0]]
    pis = 'pi1pi2'
    # args.phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]]
    # pis = 'pi2'
    # args.phi = [[-.5,1,-.5,.5]]
    
    U_inf = 2
    Re = 5e4

    k = 1
    args.phi_output = outs[k]
    out = ''.join([str(i) for i in outs[k]])
    idx = 9
    checkpoints = glob.glob(f'{dir}/{args.kind}_BASIS_transform{out}_idx{idx}_*.tar')
    checkpoint = checkpoints[0]
    model_0101 = load_model(checkpoint, name='data_5e2_1e3_1_idx0', args=args, initial_optm='lbfgs')
    fig, ax, axins = displacement_u(model_0101, Re=Re, U_inf=U_inf, label='Uy transformation', color='firebrick')

    k = 0
    args.phi_output = outs[k]
    out = ''.join([str(i) for i in outs[k]])
    idx = 9 
    checkpoints = glob.glob(f'{dir}/{args.kind}_BASIS_transform{out}_idx{idx}_*.tar')
    checkpoint = checkpoints[0]
    model_0000 = load_model(checkpoint, name='data_5e2_1e3_1_idx0', args=args, initial_optm='lbfgs')
    fig, ax, axins = displacement_u(model_0000, (fig,ax), axins, Re=Re, U_inf=U_inf, label='no transformation', color='forestgreen', save=True)


    plt.show()
    
    # ax, axins = displacement_v(model, Re=Re, U_inf=U_inf)
    # plt.show()



if __name__=='__main__':
    # main()
    main2()