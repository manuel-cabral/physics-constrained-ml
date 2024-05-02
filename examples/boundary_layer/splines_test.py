import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

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

np.random.seed(1994)

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

def sample_points(N_points, bounds, n_dims=4):
    points = np.zeros((N_points,n_dims))
    for i,bound in enumerate(bounds):
        points[:,i] = np.random.uniform(*bound,size=(N_points,))
    return points

N_points = int(1e3)
bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,2]]
x, y, visc, U_inf = sample_points(N_points,bounds).T
u,v,psi = quantities(x, y, visc, U_inf)
speed = np.sqrt(u**2 + v**2)

interp = Rbf(x, y, visc, U_inf, speed, function='cubic')


N = 200
xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
yi, xi = np.mgrid[ymin:ymax:(N * 1j), xmin:xmax:(N * 1j)]

visc = 1/5e4
U_inf = 2
ki = interp(xi, yi, np.full_like(xi, visc), np.full_like(xi, U_inf))

fig, ax = plt.subplots(figsize=(10,10))
# plt.plot(x, y, 'ko', alpha=)
im = ax.imshow(ki, extent=[xmin, xmax, ymin, ymax], cmap='RdBu')
plt.colorbar(im)
ax.set_aspect(20)
plt.show()