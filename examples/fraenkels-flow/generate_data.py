
#! Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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

#! Get fields
def streamfunction(x, y, vort, R=1):
    x, y = x/R, y/R
    r = np.sqrt(x**2 + y**2)
    psi = vort*y**2/2 + vort/(2*np.pi)*((1-1/r**2)*y + (x*y*(r**4-1))/(2*r**4)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
                +1/2*((1+1/r**4)*(x**2-y**2)-2)*np.arctan(2*y/(r**2-1))) 
    psi *= R**2
    return psi

def u(x,y,vort,R=1):
    x, y = x/R, y/R
    r = np.sqrt(x**2 + y**2)
    u = vort*y + vort/(2*np.pi)*(2*y**2/r**4+(1-1/r**2) + x/2*(1-1/r**4+4*y**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 4*x**2*y**2*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1))-(2*y*(x**2-y**2)/r**6+(1+1/r**4)*y)*np.arctan(2*y/(r**2-1)) \
            + ((1+1/r**4)*(x**2-y**2)-2)*(x**2-y**2-1)/((r**2-1)**2+4*y**2))
    u *= R
    return u

def v(x,y,vort,R=1):
    x, y = x/R, y/R
    r = np.sqrt(x**2 + y**2)
    v = -vort/(2*np.pi)*(2*x*y/r**4+y/2*(1-1/r**4+4*x**2/r**6)*np.log((r**2-2*x+1)/(r**2+2*x+1)) \
            + 2*x*y*(x**2-y**2-1)*(1-1/r**4)/((r**2-2*x+1)*(r**2+2*x+1)) \
            - (2*x*(x**2-y**2)/r**6-(1+1/r**4)*x)*np.arctan(2*y/(r**2-1)) \
            - 2*x*y*((1+1/r**4)*(x**2-y**2)-2)/((r**2-1)**2+4*y**2))
    v *= R
    return v

def quantities(x, y, vort, R=1):
    # the velocity field is only defined in the upper plane; we make it anti-symmetric with respect to the x-axis
    if y<0: return -u(x,-y,vort,R), v(x,-y,vort,R), streamfunction(x,-y,vort,R)
    else: return u(x,y,vort,R), v(x,y,vort,R), streamfunction(x,y,vort,R)

quantities = np.vectorize(quantities)

def plot_fields(N=200, bounds=[[5e-2, 1],[0, .05]], vort=3,  R=1, add_noise=False):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)

    u,v,psi = quantities(xx, yy, vort, R)
    
    if add_noise: 
        for q in [u,v,psi]: q = add_noise(q)

    psi_min, psi_max = np.min(psi), np.max(psi)
    labels = ['$u$', '$v$', '$\psi$']
    names = ['u', 'v', 'psi']
    for q in [u,v,psi]:
        mask = np.sqrt(xx**2 + yy**2) < R
        u[mask], v[mask] = np.nan, np.nan              
        psi[mask] = np.nan
        fig, ax = plot_quantity(q, x, y, label=labels.pop(0))
        circle = plt.Circle((0,0), R, color='firebrick', fill=True, alpha=.5)
        ax.add_artist(circle)
        # ax.streamplot(x, y, u, v, color='k', linewidth=1, density=1.5)
        a = 3
        lim = psi_max**(1/a)
        levels = np.linspace(-lim, lim,30)**a
        ax.contour(xx, yy, psi, levels=levels, colors='black', linewidths=1)
        # plt.savefig(f'{names.pop(0)}_vort2_Rsqrt2.png', bbox_inches='tight', dpi=256)
        plt.show()

def sample_points(N_points, bounds):
    points = np.zeros((N_points, len(bounds)))
    j = 0
    for i,bound in enumerate(bounds[2:]):
        points[:,i+2] = np.random.uniform(*bound,size=(N_points,))
    while j < N_points:
        x, y = np.random.uniform(*bounds[0],size=(1,)), np.random.uniform(*bounds[1],size=(1,))
        distance = np.sqrt(x**2 + y**2)
             
        if distance - points[j,3] >= 0: # points[j,3] is the radius
            points[j,0], points[j,1] = x,y
            j += 1

    return points

#! FLAGS
PLOT_FIELDS = True
ADD_NOISE = False
SAVE_DATA = False

#! Save datasets
def main():
    bounds = [[-2.5,2.5],[-2.5,2.5],[.1,3],[.5,1.5]] # x,y,vort,R
    
    if PLOT_FIELDS: 
        vort = 2
        R = 1/np.sqrt(2)

        plot_fields(N=300, bounds=bounds[:2], vort=vort, R=R, add_noise=ADD_NOISE)

    n_train = 1e3
    n_val = 5e3
    n_test = 1
    idx = 0
    
    name = f'fraenkels-flow/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    if SAVE_DATA:
        save_datasets(sample_points, quantities, name, bounds, N_train=int(n_train), N_val=int(n_val), N_test=int(n_test))

if __name__=='__main__':
    main()