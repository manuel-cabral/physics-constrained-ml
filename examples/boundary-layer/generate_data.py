
#! Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import os

# change directory to the root
######
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
######

import args
from src.utils import get_args, set_seed, set_torch_dtype, save_datasets, save_to_pickle, load_pickle, add_noise
from src.plotter import plot_quantity
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

# make quantities a vectorized function
quantities = np.vectorize(quantities, otypes=[args.ftype]*3)

#! Sample points
def sample_points(N_points, bounds):
    return np.array([np.random.uniform(*bound, size=N_points) for bound in bounds]).T

#! Plot targets 
z_max = 10
f, f_z, _, z_span = solve_pointwise(z_max)
beta = np.trapz((1-f_z), z_span)
v_norm = (z_span*f_z-f)/beta

def bl_thickness(x, visc, U_inf):
    return 5*np.sqrt(x*visc/U_inf)

def plot_fields(N=200, bounds=[[5e-2, 1],[0, .05]], Re=5e4, U_inf=1, add_noise=False):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)

    visc = 1/Re 
    u,v,psi = quantities(xx, yy, visc, U_inf)
    
    if add_noise: 
        for q in [u,v,psi]: q = add_noise(q)

    labels = ['$u$', '$v$', '$\Tilde{\psi}$']
    for q in [u,v,psi-U_inf*yy]:
        fig, ax = plot_quantity(q, x, y, label=labels.pop(0))
        ax.plot(x, bl_thickness(x, visc, U_inf), 'k--', label='BL thickness')
        if q is u: 
            plt.savefig('fp_u_teo.png', bbox_inches='tight', dpi=256)
        elif q is v: 
            plt.savefig('fp_v_teo.png', bbox_inches='tight', dpi=256)
        else:
            plt.savefig('fp_psi_teo.png', bbox_inches='tight', dpi=256)
        plt.show()

def plot_profile(f_z, v_norm, z_span, z_max=10, capped=True, fname=None):
    u_norm = f_z
    _, ax = plt.subplots(figsize=(11,9))
    ax.plot(u_norm, z_span, color='firebrick', label="$u_{norm}$")
    ax.plot(v_norm, z_span, color='steelblue', label="$v_{norm}$")
    ax.vlines(1,0,z_max, color='k', ls='--', zorder=1, alpha=.5)
    if capped:
        mask = u_norm<.95
        ax.fill_between(u_norm[mask], z_span[mask], color='firebrick', alpha=0.2)
        disp_thick = np.trapz((1-u_norm[mask]), z_span[mask])
        mom_thick = np.trapz((1-u_norm[mask])*u_norm[mask], z_span[mask])
        print(f"Displacement thickness: {disp_thick:.2f}")
        print(f"Momentum thickness: {mom_thick:.2f}")
        mask = v_norm<.95
        ax.fill_between(v_norm[mask], z_span[mask], color='steelblue', alpha=0.2)
        ax.vlines(.95,0,z_max, color='gray', ls='-', zorder=1, alpha=.5)
        disp_thick_v = np.trapz((1-v_norm[mask]), z_span[mask])
        mom_thick_v = np.trapz((1-v_norm[mask])*v_norm[mask], z_span[mask])
        print(f"Displacement thickness: {disp_thick_v:.2f}")
        print(f"Momentum thickness: {mom_thick_v:.2f}")

    # ax.fill_between(u_norm, z_span, color='firebrick', alpha=0.2)
    # ax.fill_between(v_norm, z_span, color='steelblue', alpha=0.2)
    ax.set_ylabel('$\eta$')
    ax.legend()
    if fname: plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

#! FLAGS
PLOT_PROFILE = True
PLOT_FIELDS = True
ADD_NOISE = False
SAVE_DATA = False

#! Save datasets
def main():
    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,5]] # x,y,visc,U_inf

    if PLOT_FIELDS: plot_fields(N=250, bounds=bounds[:2], Re=5e4, U_inf=1, add_noise=ADD_NOISE)
    if PLOT_PROFILE: plot_profile(f_z, v_norm, z_span, z_max=10, fname='fp_profile_teo_placeholder.png')

    n_train = 5e2
    n_val = 1e3
    n_test = 1
    idx = 0
    name = f'boundary_layer/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'
    
    if SAVE_DATA:
        save_datasets(sample_points, quantities, name, bounds, N_train=int(n_train), N_val=int(n_val), N_test=int(n_test))
    
    return
    
if __name__=='__main__':
    main()