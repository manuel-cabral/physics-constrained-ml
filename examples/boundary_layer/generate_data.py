
#! Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
import os
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
import src.config

#! Set parameters
# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set seed for reproducibility
set_seed(args.seed)

#! FLAGS
PRINT_DISPLACEMENT = False
PLOT_PROFILE = False
ADD_NOISE = False
PLOT_QUANTITIES = True
PLOT_POINTS = False
PRINT_VOLUME = False

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
    v = 1/(2*pi1)*(pi2 * f_z - f)   # v = 1/(2*pi1) * (pi2 * f' - f)

    psi = U_inf * y * f / pi2 if y!=0 else 0

    return u, v, psi

# make quantities a vectorized function
quantities = np.vectorize(quantities, otypes=[args.ftype, args.ftype, args.ftype])

if PRINT_DISPLACEMENT:
    z_max = 10
    f, f_z, _, z_span = solve_pointwise(z_max)
    # print('momentum_u = ', np.trapz((u_norm*(1-u_norm)), z_span))
    beta = np.trapz((1-f_z), z_span)
    print(f'displacement_u = {beta:.5f}')

    v_norm = (z_span*f_z-f)/beta
    # print('momentum_u = ', np.trapz((v_norm*(1-v_norm)), z_span))
    print(f'displacement_v = {np.trapz((1-v_norm), z_span):.5f}')

if PLOT_PROFILE:
    u_norm = f_z
    fig, ax = plt.subplots(figsize=(11,9))
    ax.plot(u_norm, z_span, color='firebrick', label="$u_{norm}$")
    ax.plot(v_norm, z_span, color='steelblue', label="$v_{norm}$")
    ax.vlines(1,0,z_max, color='k', ls='--', zorder=1, alpha=.5)
    ax.fill_between(u_norm, z_span, color='firebrick', alpha=0.2)
    ax.fill_between(v_norm, z_span, color='steelblue', alpha=0.2)
    ax.set_ylabel('$\eta$')
    ax.legend()
    fname = f'imgs/profile_u&v.png'
    plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

def plot_quantity(q, label=None, bl=False, fname=None):
    fig, ax = plt.subplots(figsize=(11,9))
    im = ax.imshow(q, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='RdBu')
    ax.set_aspect(20)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    cbar = plt.colorbar(im)
    cbar.formatter.set_powerlimits((0, 0))
    if bl:
        ax.plot(x, 5*np.sqrt(x*visc/U_inf), 'k--', label='BL thickness')
    if label is not None:
        x0, y0 = (x.max()-x.min())*.9+x.min(),(y.max()-y.min())*.9+y.min()
        ax.text(x0, y0,label,fontsize=30, ha='center', va='center', color='white')#, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

def add_noise(quantity, percentage=1e-2):
    quantity += np.random.normal(0, percentage*np.mean(np.abs(quantity)), quantity.shape)
    return quantity

if PLOT_QUANTITIES:
    N = 250
    x = np.linspace(5e-2, 1, N)
    y = np.linspace(0, .05, N)
    xx,yy = np.meshgrid(x, y)

    U_inf = 1
    Re = 5e4
    visc = 1/Re 
    u,v,psi = quantities(xx, yy, visc*np.ones_like(xx), U_inf*np.ones_like(xx))
    if ADD_NOISE:
        for q in [u,v,psi]: q = add_noise(q)
    plot_quantity(u, label='$u$')
    plot_quantity(v, label='$v$')
    plot_quantity(psi-U_inf*yy, label='$\Tilde{\psi}$')

def sample_points(N_points, bounds, n_dims=4):
    points = np.zeros((N_points,n_dims))
    for i,bound in enumerate(bounds):
        points[:,i] = np.random.uniform(*bound,size=(N_points,))
    return points

def bl_thickness(x, visc, U_inf):
    return 5*np.sqrt(x*visc/U_inf)

def sample_points_inside_bl(N_points, bounds, n_dims=4):
    points = []
    bounds = np.array(bounds)
    while len(points)<N_points:
        pt = np.random.uniform(bounds.T[0],bounds.T[1])
        if pt[1]<bl_thickness(pt[0], pt[2], pt[3]): points.append(list(pt))
    return np.array(points)

ONLY_INSIDE = True
if PLOT_POINTS:
    N_points = int(1e3)
    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,2]]
    if ONLY_INSIDE: points = sample_points_inside_bl(N_points, bounds)
    else: points = sample_points(N_points, bounds)

    fig, ax = plt.subplots(figsize=(13,9))
    color, size = points[:,2], 5*points[:,3] 
    print(np.sum([0 if point[1]>bl_thickness(point[0], point[2], point[3]) else 1 for point in points])/len(points))
    color = ['firebrick' if point[1]>bl_thickness(point[0], point[2], point[3]) else 'forestgreen' for point in points]
    size = 50
    ax.scatter(points[:,0], points[:,1], c=color, s=size, alpha=.7,) #points)
    ax.vlines(bounds[0][0],*bounds[1], color='k', ls='--', zorder=1, alpha=.5)
    ax.vlines(bounds[0][1],*bounds[1], color='k', ls='--', zorder=1, alpha=.5)
    ax.hlines(bounds[1][0],*bounds[0], color='k', ls='--', zorder=1, alpha=.5)
    ax.hlines(bounds[1][1],*bounds[0], color='k', ls='--', zorder=1, alpha=.5)
    ax.set_aspect(20)
    x_gap = (bounds[0][1]-bounds[0][0])/20
    ax.set_xlim(bounds[0][0]-x_gap, bounds[0][1]+x_gap)
    y_gap = (bounds[1][1]-bounds[1][0])/20
    ax.set_ylim(bounds[1][0]-y_gap, bounds[1][1]+y_gap)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    custom = [Line2D([], [], marker='.', color='firebrick', linestyle='None', markersize=15, alpha=.7, markeredgecolor=None),
          Line2D([], [], marker='.', color='forestgreen', linestyle='None', markersize=15, alpha=.7, markeredgecolor=None)]
    plt.legend(handles = custom, labels=['$\\text{outside BL}$', '$\\text{inside BL}$'], bbox_to_anchor= (1.025, 0.5), loc= "lower left")
    fname = f'imgs/bl_sampling_4d_normal_inside.png'
    # plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

    fig, ax = plt.subplots(figsize=(16,9))
    color, size = points[:,2], 5*points[:,3] 
    color = ['firebrick' if point[1]>bl_thickness(point[0], point[2], point[3]) else 'forestgreen' for point in points]
    size = 50
    pi1 = np.sqrt(points[:,3] * points[:,0] / points[:,2])
    pi2 = points[:,1] * np.sqrt(points[:,3] / (points[:,2] * points[:,0]))
    ax.scatter(pi1, pi2, c=color, s=size, alpha=.7,) #points)
    x = np.linspace(*bounds[0], 100)
    y = np.linspace(*bounds[1], 100)
    pi1_ = np.linspace(pi1.min(), pi1.max(), 100)
    pi2_ = y*np.sqrt(1/(5e4*x))
    ax.plot(pi1_, 5*np.ones_like(pi1_), 'k--', label='BL thickness')
    # ax.vlines(bounds[0][0],*bounds[1], color='k', ls='--', zorder=1, alpha=.5)
    # ax.vlines(bounds[0][1],*bounds[1], color='k', ls='--', zorder=1, alpha=.5)
    # ax.hlines(bounds[1][0],*bounds[0], color='k', ls='--', zorder=1, alpha=.5)
    # ax.hlines(bounds[1][1],*bounds[0], color='k', ls='--', zorder=1, alpha=.5)
    # ax.set_aspect(20)
    # x_gap = (bounds[0][1]-bounds[0][0])/20
    # ax.set_xlim(bounds[0][0]-x_gap, bounds[0][1]+x_gap)
    # y_gap = (bounds[1][1]-bounds[1][0])/20
    # ax.set_ylim(bounds[1][0]-y_gap, bounds[1][1]+y_gap)
    ax.set_xlabel('$\\pi_1$')
    ax.set_ylabel('$\\pi_2$')
    custom = [Line2D([], [], marker='.', color='firebrick', linestyle='None', markersize=15, alpha=.7, markeredgecolor=None),
          Line2D([], [], marker='.', color='forestgreen', linestyle='None', markersize=15, alpha=.7, markeredgecolor=None),
          Line2D([], [], ls='--', color='k', linestyle='None', markersize=15, alpha=.7, markeredgecolor=None)]
    plt.legend(handles = custom, labels=['$\\text{outside BL}$', '$\\text{inside BL}$', '$\\text{boundary layer } (\pi_2=5)$'], bbox_to_anchor= (.7, .8), loc= "lower left")
    fname = f'imgs/bl_sampling_4d_pi_inside.png'
    # plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(5e-2, 1, 100)
    v = np.linspace(1/7e4, 1/3e4, 100)
    xx,vv = np.meshgrid(x, v)
    U = .2
    yy = bl_thickness(xx, vv, U)
    ax.plot_surface(xx, vv, np.array([y if y<5e-2 else 5e-2 for y in yy.flatten()]).reshape(yy.shape), alpha=.5, color='k')
    N_points = int(1e3)
    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[U,U]]
    if ONLY_INSIDE: points = sample_points_inside_bl(N_points, bounds)
    else: points = sample_points(N_points, bounds)
    color = ['firebrick' if point[1]>bl_thickness(point[0], point[2], point[3]) else 'forestgreen' for point in points]
    ax.scatter(points[:,0], points[:,2], points[:,1], c=color, s=18)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\nu$')
    ax.set_zlabel('$y$')
    ax.view_init(elev=8, azim=113)
    fname = f'imgs/bl_thickness_3d_inside.png'
    # plt.savefig(fname, bbox_inches='tight', dpi=256)
    plt.show()
    

def save_to_pickle(folder, fname, data):
    data = np.nan_to_num(data)
    file_path = os.path.join(folder, fname)
    with open(f'{file_path}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_datasets(name, bounds, folder='datasets', N_train=int(5e2), N_val=int(1e3), N_test=1):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(dir_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    N_points = N_train + N_val + N_test
    points = sample_points(N_points, bounds)

    u,v,psi = quantities(*points.T)
    complete = np.column_stack((points, u, v, psi))
    dataset = {}
    dataset['train'], dataset['val'], dataset['test'] = complete[:N_train], complete[N_train:N_train+N_val], complete[N_train+N_val:]

    save_to_pickle(folder_path, name, dataset)
    return

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,5]]
    save_datasets('data_5e2_1e3_1_idx0', bounds, N_train=int(5e2), N_val=int(1e3), N_test=1)
    return
    
if __name__=='__main__':
    main()