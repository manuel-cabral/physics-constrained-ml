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
def velocity(r,theta, circulation=0, U_inf=1, R=1):
    u = (1-np.cos(2*theta)*R**2/r**2)*U_inf - circulation*np.sin(theta)/(2*np.pi*r)
    v = -R**2/r**2*np.sin(2*theta)*U_inf + circulation*np.cos(theta)/(2*np.pi*r)
    return np.array((u, v))

def streamfunc(r,theta, circulation=0, U_inf=1, R=1):
    return (1-R**2/r**2)*r*U_inf*np.sin(theta) - circulation/(2*np.pi)*np.log(r/R)

def pressure(u, rho=1, p_inf=0, U_inf=1):
    return .5*rho*(U_inf**2-u**2)+p_inf

def stagnation_pts(circulation, U_inf=1, R=1):
    theta = 180/np.pi*np.arcsin(circulation/(4*np.pi*U_inf*R))
    return (theta, 180-theta)

def cart2polar(x,y):
    return (np.sqrt(x**2 + y**2),np.arctan2(y, x))

def quantities(x, y, circulation=0, U_inf=1, R=1):
    r, theta = cart2polar(x,y)
    u, v = velocity(r, theta, circulation, U_inf, R)
    psi = streamfunc(r, theta, circulation, U_inf, R)
    return u, v, psi

# make quantities a vectorized function
quantities = np.vectorize(quantities, otypes=[args.ftype]*3)


#! Sample points
def sample_points(N_points, bounds):
    points = np.zeros((N_points, len(bounds)))
    j = 0
    for i,bound in enumerate(bounds[2:]):
        points[:,i+2] = np.random.uniform(*bound,size=(N_points,))
    while j < N_points:
        x, y = np.random.uniform(*bounds[0],size=(1,)), np.random.uniform(*bounds[1],size=(1,))
        distance = np.sqrt(x**2 + y**2)
             
        if distance - points[j,4] >= 0: # points[j,4] is the radius
            points[j,0], points[j,1] = x,y
            j += 1

    return points

#! Plot targets 
def plot_fields(N=200, bounds=[[-3,3],[-3,3]], circulation=0, U_inf=1, R=1, add_noise=False):
    x = np.linspace(*bounds[0], N)
    y = np.linspace(*bounds[1], N)
    xx,yy = np.meshgrid(x, y)
    u,v,psi = quantities(xx, yy, circulation, U_inf, R)
    
    if add_noise: 
        for q in [u,v,psi]: q = add_noise(q)

    mask = xx**2 + yy**2 <= R**2

    labels = ['$u$', '$v$', '$\Tilde{\psi}$']
    for q in [u,v,psi-U_inf*yy]:
        q_ = np.ma.masked_array(q, mask)
        fig, ax = plot_quantity(q_, x, y, label=labels.pop(0))
        circle = plt.Circle((0,0), R, color='firebrick', fill=True, alpha=.3)
        ax.add_artist(circle)
        plt.show()

#! FLAGS
PLOT_FIELDS = True
ADD_NOISE = False
SAVE_DATA = True


#! Save datasets
def main():
    bounds = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    if PLOT_FIELDS: plot_fields(N=500, bounds=bounds[:2], circulation=4*np.pi, U_inf=1, R=1, add_noise=ADD_NOISE)

    n_train = 5e2
    n_val = 1e3
    n_test = 1
    idx = 0
    name = f'data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    if SAVE_DATA:
        save_datasets(sample_points, quantities, name, bounds, folder='datasets/cylinder-with-circulation/', N_train=int(n_train), N_val=int(n_val), N_test=int(n_test))
    
    return

if __name__=='__main__':
    main()