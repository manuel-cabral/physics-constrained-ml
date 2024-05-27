import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
from scipy.linalg import null_space
import matplotlib.gridspec as gridspec
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
from src.utils import get_args, set_torch_dtype, set_seed, load_pickle, load_data #, set_torch_multiprocessing
from src.model import Model, TRANSFORM_INPUT
import src.config

# Get args
args = get_args(args)

# Set float precision
set_torch_dtype(args.ftype)

# Set multiprocessing
# set_torch_multiprocessing()

# Set seed for reproducibility
set_seed(args.seed)


def train_model(dataset, args):
    # Create model
    model = Model(dataset, args)
    print(model.nn.layers)
    
    # Train model
    start = timer()
    model.train()
    end = timer()
    print(f'Training time: {timedelta(seconds=end-start)}')

    return model

def plot_1D(ax, parameter, bounds, label=None, fontsize=20):
    ax.hlines(1,*bounds, colors='k', lw=1)
    for lim in bounds:
        ax.text(lim, 0.7, f'${lim:.0f}$', ha='center',)
    ax.set_ylim(0.9,1.15)

    ax.axis('off')
    ax.vlines(0, .9, 1.1, colors='k', alpha=1, lw=1, zorder=4)
    ax.vlines(1, .9, 1.1, colors='k', alpha=1, lw=1, zorder=4)
    ax.vlines(parameter, 1, 1.1, colors='indigo', alpha=1, lw=2, zorder=5)
    ax.text(parameter, 1.15, f'${parameter:.1f}$', ha='center',)
    ax.text(np.mean(bounds), .7, label, ha='center',)
    return ax

def plot_sampling(x, y, vort, R, psi, args, beta=0, gamma=0):

    args.phi = [[1,0,0,-beta],[0,1,0,-beta]]           
    input = torch.tensor(np.array([x,y,vort,R]), dtype=torch.float32).T
    transform_input = TRANSFORM_INPUT(args)
    x,y = transform_input(input).numpy().T

    # fig, ax = plt.subplots(figsize=(10,10))

    fig = plt.figure(figsize=(10, 12))  

    gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[11, .5, .5])  

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    # print(np.min(psi)/(3*1.5**2), np.max(psi)/(.1*.5**2))
    # print(np.min(psi/(vort*R**2)), np.max(psi/(vort*R**2)))
    # vmin, vmax = -1, 1
    # vmin, vmax = -5, 5
    vmin, vmax = 0, 10
    # vmin, vmax = 0, 1
    im = ax1.scatter(x, y, c=psi/(vort*R**2)**gamma, cmap='coolwarm' ,s=60, vmin=vmin, vmax=vmax)
    ax1.set_aspect('equal')
    draw_circle = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--', linewidth=2)
    ax1.add_artist(draw_circle)
    for r in [.5,1.5]:
        draw_circle = plt.Circle((0, 0), r**(1-beta), fill=False, color='firebrick', linestyle='--', alpha=.75, linewidth=2)
        ax1.add_artist(draw_circle)
    R_min = .5
    l = 2.5/R_min**beta
    ax1.hlines(l,-l,l, color='k', linestyle='--', linewidth=1)
    ax1.hlines(-l,-l,l, color='k', linestyle='--', alpha=.5, linewidth=1)
    ax1.vlines(l,-l,l, color='k', linestyle='--', alpha=.5, linewidth=1)
    ax1.vlines(-l,-l,l, color='k', linestyle='--', alpha=.5, linewidth=1)
    ax1.set_xlabel('$\\frac{x}{R^{\\beta}}$')
    ax1.set_ylabel('$\\frac{y}{R^{\\beta}}$')
    ax1.set_xlim(-2*R_min,2*R_min)
    ax1.set_ylim(-2*R_min,2*R_min)
    ax1.set_xticks(np.arange(-5,6,1))
    ax1.set_yticks(np.arange(-5,6,1))
    label='$\\frac{\psi}{(\omega R^2)^{\gamma}}$'
    cbar = plt.colorbar(im, label=label, fraction=0.046, pad=0.04, boundaries=np.arange(vmin,vmax+1e-3*(vmax-vmin),1e-3*(vmax-vmin)))
    cbar.set_ticks(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))

    ax2 = plot_1D(ax2, beta, [0,1], label='$\\beta$', fontsize=20)
    ax3 = plot_1D(ax3, gamma, [0,1], label='$\\gamma$', fontsize=20)
    # plt.show()

    return fig, ax1, ax2, ax3

def streamfunction(x_, y_):
    r_ = np.sqrt(x_**2 + y_**2)
    if y_<0: y_ = -y_
    psi = y_**2/2 + 1/(2*np.pi)*((1-1/r_**2)*y_ + (x_*y_*(r_**4-1))/(2*r_**4)*np.log((r_**2-2*x_+1)/(r_**2+2*x_+1)) \
                +1/2*((1+1/r_**4)*(x_**2-y_**2)-2)*np.arctan(2*y_/(r_**2-1)))
    return psi

streamfunction = np.vectorize(streamfunction)

def plot_3d_sampling(x, y, vort, R, psi, args, beta=0, gamma=0, difference=False):
    
        args.phi = [[1,0,0,-1],[0,1,0,-1]]           
        input = torch.tensor(np.array([x,y,vort,R]), dtype=torch.float32).T
        transform_input = TRANSFORM_INPUT(args)
        x,y = transform_input(input).numpy().T

        #plot a single 3d scatter plot
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        ax1 = plt.subplot(gs[0], projection='3d')

        xx = np.linspace(-5,5,100)
        yy = np.linspace(-5,5,100)
        X, Y = np.meshgrid(xx, yy)

        if difference:
            im = ax1.scatter(x, y, psi/(vort*R**2)-y**2/2, s=60, c='firebrick', alpha=.5)
            ax1.plot_surface(X, Y, np.zeros_like(X), alpha=.3, color='steelblue')
        else:
            im = ax1.scatter(x, y, psi/(vort*R**2), s=60, c='firebrick', alpha=.5)
            ax1.plot_surface(X, Y, Y**2/2, alpha=.3, color='steelblue')

        ax1.view_init(elev=8, azim=185)

        # r = np.sqrt(X**2 + Y**2)
        # ax1.plot_surface(X, Y, Y**2/2 + 1/(2*np.pi)*((1-1/r**2)*Y + (X*Y*(r**4-1))/(2*r**4)*np.log((r**2-2*X+1)/(r**2+2*X+1)) \
        #         +1/2*((1+1/r**4)*(X**2-Y**2)-2)*np.arctan(2*Y/(r**2-1))) , alpha=.3, color='forestgreen')

        # ax1.plot_surface(X, Y, streamfunction(X,Y), alpha=.3, color='forestgreen')


        # print(np.linalg.norm(psi/(vort*R**2)-y**2/2,2)/np.linalg.norm(psi/(vort*R**2),2)*100)
        # print(np.linalg.norm(psi/(vort*R**2)- streamfunction(x,y) ,2)/np.linalg.norm(psi/(vort*R**2),2)*100)

        # print('\nPositive y')
        # mask = y>0
        # x_,y_,vort_,R_,psi_ = x[mask],y[mask],vort[mask],R[mask],psi[mask]
        # print(np.linalg.norm(psi_/(vort_*R_**2)-y_**2/2,2)/np.linalg.norm(psi_/(vort_*R_**2),2)*100)
        # print(np.linalg.norm(psi_/(vort_*R_**2)- streamfunction(x_, y_) ,2)/np.linalg.norm(psi_/(vort_*R_**2),2)*100)

        # print('\nNegative y')
        # mask = y<0
        # x_,y_,vort_,R_,psi_ = x[mask],y[mask],vort[mask],R[mask],psi[mask]
        # print(np.linalg.norm(psi_/(vort_*R_**2)-y_**2/2,2)/np.linalg.norm(psi_/(vort_*R_**2),2)*100)
        # print(np.linalg.norm(psi_/(vort_*R_**2)- streamfunction(x_, y_) ,2)/np.linalg.norm(psi_/(vort_*R_**2),2)*100)


        # print(np.linalg.norm(psi[mask]/(vort[mask]*R[mask]**2)- (y[mask]**2/2 + 1/(2*np.pi)*((1-1/r[mask]**2)*y[mask] + (x[mask]*y[mask]*(r[mask]**4-1))/(2*r[mask]**4)*np.log((r[mask]**2-2*x[mask]+1)/(r[mask]**2+2*x[mask]+1)) \
        #         +1/2*((1+1/r[mask]**4)*(x[mask]**2-y[mask]**2)-2)*np.arctan(2*y[mask]/(r[mask]**2-1))))  ,2)/np.linalg.norm(psi[mask]/(vort[mask]*R[mask]**2),2)*100)
        # mask = y<0
        # print(np.linalg.norm(psi[mask]/(vort[mask]*R[mask]**2)- (y[mask]**2/2 + 1/(2*np.pi)*((1-1/r[mask]**2)*y[mask] + (x[mask]*y[mask]*(r[mask]**4-1))/(2*r[mask]**4)*np.log((r[mask]**2-2*x[mask]+1)/(r[mask]**2+2*x[mask]+1)) \
        #         +1/2*((1+1/r[mask]**4)*(x[mask]**2-y[mask]**2)-2)*np.arctan(2*y[mask]/(r[mask]**2-1))))  ,2)/np.linalg.norm(psi[mask]/(vort[mask]*R[mask]**2),2)*100)


        R_min = .5
        ax1.set_xlabel('$\pi_1$')
        ax1.set_ylabel('$\pi_2$')
        ax1.set_zlabel('$\chi$')
        ax1.set_xlim(-2*R_min,2*R_min)
        ax1.set_ylim(-2*R_min,2*R_min)
        ax1.set_xticks(np.arange(-5,6,1))
        ax1.set_yticks(np.arange(-5,6,1))
        # plt.show()

        return fig, ax1

def change_parameters(args, data_size=5e2):
    args.kind = 'incompressible'
    args.layers = [16]*2
    args.n_epochs = 2_000
    args.n_epochs_adam = 1_000
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = int(2**np.floor(np.log2(data_size)))
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "vort", "r"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True

    # args.sampling_box = [[-2.5,2.5],[-2.5,2.5],[.1,3],[.5,1.5]] # x,y,vort,R

    args.phi = [[1,0,0,-1],[0,1,0,-1]] # x/R, y/R
    
    args.phi_output = [0,0,1,2] # vort * R**2

    return


def main():
    n_train = 1e3 # 1e3 or 1e5
    n_val = 5e3
    n_test = 1
    idx = 0
    change_parameters(args, data_size=n_train)

    name = f'fraenkels-flow/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'


    dataset = load_data(name)

    x,y,vort,R,u,v,psi = dataset['train'].T

    # plot_sampling(x, y, vort, R, psi, args, beta=1, gamma=1)
    # # plt.savefig('src/sampling_transformation.png', bbox_inches='tight', dpi=156)
    # plt.show()

    # plot_3d_sampling(x, y, vort, R, psi, args, beta=1, gamma=1)
    # # plt.savefig('src/sampling_transformation_3d_withSurface_true.png', bbox_inches='tight', dpi=156)
    # plt.show()     

    # plot_3d_sampling(x, y, vort, R, psi, args, beta=1, gamma=1, difference=True)
    # # plt.savefig('src/sampling_transformation_3d_withSurface_true.png', bbox_inches='tight', dpi=156)
    # plt.show()        


    # psi=v
    SAVE_FRAMES = False
    if SAVE_FRAMES:
        # dpi = 156
        dpi = 64
        files = glob.glob('src/frames/*')
        for f in files:
            os.remove(f)
        idx = 0
        factor = 10
        lengths = [2*factor,5*factor,2*factor,5*factor,2*factor]
        print(sum(lengths))
        for i in range(lengths[0]):
            plot_sampling(x, y, vort, R, psi, args, beta=0, gamma=0)
            plt.savefig(f'src/frames/frame_{idx:03d}.png', bbox_inches='tight', dpi=dpi)
            idx += 1
            plt.close()
        for beta in np.linspace(0,1,lengths[1]):
            plot_sampling(x, y, vort, R, psi, args, beta=beta, gamma=0)
            plt.savefig(f'src/frames/frame_{idx:03d}.png', bbox_inches='tight', dpi=dpi)
            idx += 1
            plt.close()
        for i in range(lengths[2]):
            plot_sampling(x, y, vort, R, psi, args, beta=1, gamma=0)
            plt.savefig(f'src/frames/frame_{idx:03d}.png', bbox_inches='tight', dpi=dpi)
            idx += 1
            plt.close()
        for gamma in np.linspace(0,1,lengths[3]):
            plot_sampling(x, y, vort, R, psi, args, beta=1, gamma=gamma)
            plt.savefig(f'src/frames/frame_{idx:03d}.png', bbox_inches='tight', dpi=dpi)
            idx += 1
            plt.close()
        for i in range(lengths[4]):
            plot_sampling(x, y, vort, R, psi, args, beta=1, gamma=1)
            plt.savefig(f'src/frames/frame_{idx:03d}.png', bbox_inches='tight', dpi=dpi)
            idx += 1
            plt.close()

    train_model(dataset, args)


if __name__ == '__main__':
    main()