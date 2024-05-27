import tqdm
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection
import matplotlib as mpl

# change directory to the root
######
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
######


from src.streamlines import Streamlines

import args
from src.utils import get_args, set_seed, set_torch_dtype, load_data, ftype_torch #, set_torch_multiprocessing
from src.plotter import plot_quantity
from src.model import Model
from generate_data import quantities, bl_thickness
import src.config

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

    # args.phi_output = [0,0,0,0] # 1
    # args.phi_output = [.5,0,.5,.5] # sqrt(U_inf * visc * x)
    args.phi_output = [0,1,0,1] # U_inf * y
    
    return

def main():
    bounds = [[5e-2, 1],[0, 5e-2],[1/7e4, 1/3e4],[0,5]] # x,y,visc,U_inf 
    N = int(2e2)
    x = np.linspace(*bounds[0],N)
    y = np.linspace(*bounds[1],N)
    X,Y = np.meshgrid(x,y)
    visc = 1/5e4
    U_inf = 1

    # model
    n_train, n_val, n_test  = 5e2, 1e3, 1
    idx = 0
    
    change_parameters(args, data_size=n_train)

    data_file = f'boundary-layer/data_{n_train:.1e}_{n_val:.1e}_{n_test:.1e}_idx{idx}'

    checkpoint = '' 


    model = load_model(f'{checkpoint}.tar', name=data_file, args=args, initial_optm='lbfgs')


    visc = np.ones_like(X)*visc
    U_inf = np.ones_like(X)*U_inf

    points = np.column_stack((X.flatten(),Y.flatten(),visc.flatten(),U_inf.flatten()))
    points = torch.from_numpy(points).to(torch.float32)
    u,v = (model.predict(points)).detach().numpy().T
    U,V = u.reshape(N,N), v.reshape(N,N)
    U,V = -U, -V
    psi = model.psi(points).detach().numpy().reshape(N,N)

    # u_teo, v_teo, psi_teo = quantities(X, Y, visc, U_inf)
    # U, V, psi = -u_teo, -v_teo, psi_teo

    DO_GIF = True
    if DO_GIF:


        fig = plt.figure(figsize=(10,10))
        # fig = plt.figure(figsize=(19.8,9))
        ax = plt.subplot(1, 1, 1, aspect=1)



        lengths = []
        colors = []
        lines = []

        s = Streamlines(X, Y, U, V, res=0.1, spacing=3, maxLen=1500, detectLoops=False)

        for streamline in s.streamlines:
            x, y = streamline
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n = len(segments)
            
            D = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=-1))
            L = D.cumsum().reshape(n,1) + np.random.normal()
            C = np.zeros((n,3))
            C[:] = (L*1.5) % 1

            #linewidths = np.zeros(n)
            #linewidths[:] = 1.5 - ((L.reshape(n)*1.5) % 1)

            # line = LineCollection(segments, color=colors, linewidth=linewidths)
            line = LineCollection(segments, color=C, linewidth=.5)
            lengths.append(L)
            colors.append(C)
            lines.append(line)
            
            ax.add_collection(line)
        circulations = np.repeat(-1, 27)
        def update(frame_no):
            for i in range(len(lines)):
                lengths[i] += 0.05
                # for j in range(len(lengths[i])):
                #     pt = lines[i].get_segments()[j][0]
                #     sp = np.linalg.norm(s._interp(pt[0], pt[1]))
                #     # print(sp)
                #     lengths[i][j] += 0.05*sp
                colors[i][:] = (lengths[i]*1.5) % 1
                lines[i].set_color(colors[i])
            # cir1 = plt.Circle((0, 0), 1, color='rosybrown', edgecolor=None, alpha=1, zorder=2)
            # ax.add_patch(cir1)
            # #clear previous wedge
            # for p in reversed(ax.patches):
            #     if isinstance(p, Wedge):
            #         p.remove()
            # wedge = Wedge((0, 0), 1, (circulations[frame_no]*frame_no)*360/27, (circulations[frame_no]*frame_no)*360/27+180, color='gray', edgecolor=None, alpha=1, zorder=3)
            # ax.add_patch(wedge)
            pbar.update()

        scale = 1.5
        ax.set_xlim(*bounds[0]), ax.set_xticks([])
        # ax.set_xlim(-d+U[-1,0]/scale,d), ax.set_xticks([])
        ax.set_aspect(18)
        ax.set_ylim(*bounds[1]), ax.set_yticks([])
        plt.tight_layout(pad=2)
        ax.set_xlabel(r'$x$', fontsize=25)
        ax.set_ylabel(r'$y$', fontsize=25)
        # plot boundary layer thickness
        x = np.linspace(*bounds[0],100)
        y = bl_thickness(x, visc[0,0], U_inf[0,0])
        # ax.plot(x,y, color='steelblue', linewidth=2, label='BL thickness')
        # shade boundary layer
        ax.fill_between(x,0,y, color='steelblue', alpha=.2)


        y = np.linspace(*bounds[1],25)
        x = np.full_like(y, bounds[0][0]) 
        u = -U[::8,0]
        v = -V[::8,0]
        ax.quiver(x,y,u,v*18, color='forestgreen', scale=scale, width=3e-3, scale_units='inches', alpha=.5, zorder=10)
        u = U_inf[::8,0]
        v = np.zeros_like(u)
        ax.quiver(x,y,u,v, color='firebrick', scale=scale, width=3e-3, scale_units='inches', alpha=.5, zorder=10)

        n = 27
        # animation = FuncAnimation(fig, update, interval=10)
        animation = FuncAnimation(fig, update, frames=n, interval=20)
        pbar = tqdm.tqdm(total=n)
        # animation.save('wind.mp4', writer='ffmpeg', fps=60)
        fname = 'imgs/streamlines_model.gif'
        animation.save(fname, writer='imagemagick', fps=15, savefig_kwargs={"transparent": True})
        pbar.close()
        plt.show()

if __name__=='__main__':
    main()