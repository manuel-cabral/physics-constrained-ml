import numpy as np
import matplotlib.pyplot as plt
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
from src.model_simple import Model

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
    # args.kind = 'baseline'
    args.layers = [16]*2
    # args.layers = [32]*4
    args.n_epochs = 700
    args.n_epochs_adam = 500
    args.learning_rate_adam = 1e-2
    args.learning_rate_lbfgs = 1e-2
    args.patience = None
    args.batch_size = 256
    args.scheduler = False  
    args.norm_layer = True
    args.subtract_uniform_flow = True                         
    args.x_vars = ["x", "y", "vort", "U_infty", "r"]

    args.normalize_inputs = False
    args.reduce_inputs = True
    args.transform_output = True
    # args.sampling_box = [[-2.5,2.5],[0,2.5],[0,10],[.5,5],[.1,1.5]] # x,y,c,U,r

    return


def main():
    change_parameters(args)

    # args.phi = [[ 1, 0, 0, 0,-1], # x/R
    #             [ 0, 1, 0, 0,-1], # y/R
    #             [ 0, 0,1,-1, 1]] # U/U_inf
    
    args.phi = [[ 1, 0, 0, 0,-1], # x/R
                [ 0, 1, 0, 0,-1]] # y/R
    # args.phi_output = [0,0,0,1,1]
    args.phi_output = [0,0,1,0,2]


    # dataset = load_data('fraenkels_1e2_5e3_1_ypos_fixedR_fulldomain')
    # dataset = load_data('fraenkels_1e3_5e3_1_ypos_fixedR_fulldomain')
    # dataset = load_data('fraenkels_1e4_5e3_1_ypos_fixedR_fulldomain')

    # dataset = load_data('fraenkels_1e3_5e3_1_ypos_fulldomain')

    dataset = load_data('fraenkels_1e3_5e3_1_uzero')
    print(dataset['train'][0])
    print(dataset['train'].shape)

    # dataset = load_data('fraenkels_1e3_5e3_1_ypos_fixedR_zoomedin')
    

    from src.model_simple import REDUCED_VARS, NORMALIZE
    if args.reduce_inputs==True:
        model = Model(dataset, args)
        pi = REDUCED_VARS.forward(model, torch.tensor(dataset['train'][:,:5])).T
        pi_np = pi.detach().numpy()
        # args.sampling_box = [[np.min(pi_np[0]),np.max(pi_np[0])],[np.min(pi_np[1]),np.max(pi_np[1])],[np.min(pi_np[2]),np.max(pi_np[2])]] # pi1,pi2,pi3
        args.sampling_box = [[np.min(pi_np[0]),np.max(pi_np[0])],[np.min(pi_np[1]),np.max(pi_np[1])]] # pi1,pi2,pi3
        
        # pi1_, pi2_, pi3_ =  pi
        pi1_, pi2_ =  pi
        if args.normalize_inputs==True:
            pi1, pi2, pi3 = NORMALIZE.forward(model, pi)
            pi1, pi2, pi3 = pi1.detach().numpy(), pi2.detach().numpy(), pi3.detach().numpy()
        else:
            # pi1, pi2, pi3 = pi
            pi1, pi2 = pi
        # pi1, pi2, pi3 = NORMALIZE.forward(model, pi)
        # pi1, pi2, pi3 = pi1.detach().numpy(), pi2.detach().numpy(), pi3.detach().numpy()

    PLOT_TRANSFORMED = False
    if PLOT_TRANSFORMED:
        from mpl_toolkits.mplot3d import Axes3D
        norm = np.sqrt(dataset['train'][:,5]**2 + dataset['train'][:,6]**2)
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pi1, pi2, pi3, c=norm, s=30, cmap='viridis')
        ax.set_xlabel('$\pi_1$', fontsize=16)
        ax.set_ylabel('$\pi_2$', fontsize=16)
        ax.set_zlabel('$\pi_3$', fontsize=16)
        plt.colorbar(sc, ax=ax, label='$\chi$')
        ax.view_init(elev=90, azim=0) # top view
        # ax.view_init(elev=0, azim=0) # side view
        # fig.savefig('imgs/cylinder/cyl_wrong_transf_topview_norm_exp.png', dpi=256, bbox_inches='tight')
        plt.show()



    PLOT = False
    if PLOT:
        norm = dataset['train'][:,5]**2 + dataset['train'][:,6]**2
        fig, ax = plt.subplots(figsize=(10,10))
        alpha = ((dataset['train'][:,2]-np.min(dataset['train'][:,2]))/(np.max(dataset['train'][:,2])-np.min(dataset['train'][:,2])))**.4
        factor = 10 #25
        sc = ax.scatter(dataset['train'][:,0], dataset['train'][:,1], c=norm, s=factor*dataset['train'][:,3], alpha=alpha)
        # color = norm, size = U_inf, alpha = circulation
        # plt.colorbar(sc)
        ax.set_aspect('equal')
        center = (0,0)
        # center = (6,6)
        circle = plt.Circle(center,2,color='forestgreen',alpha=0.15)
        ax.add_artist(circle)
        circle = plt.Circle(center,.1,color='firebrick',alpha=0.2)
        ax.add_artist(circle)
        ax.set_xlabel('$x$', fontsize=16)
        ax.set_ylabel('$y$', fontsize=16)
        # ax.set_xlim(0,3)
        # ax.set_ylim(0,3)
        # plt.savefig('imgs/cylinder/cyl_sampling_centered.png', dpi=256, bbox_inches='tight')
        plt.show()

    train_model(dataset, args)

    # checkpoint = 'src/best_models/incompressible_FWL9TJ_checkpoint_693.tar' # fraenkels_1e2_5e3_1_ypos_fixedR_fulldomain, [16]*2, 4.36% val error
    checkpoint = 'src/best_models/incompressible_96USCV_checkpoint_666.tar' # fraenkels_1e2_5e3_1_ypos_fixedR_fulldomain, [32]*4, 3.64% val error
   
    # checkpoint = 'src/best_models/incompressible_YNLZRU_checkpoint_583.tar' # fraenkels_1e3_5e3_1_ypos_fixedR_fulldomain, [16]*2, 4.36% val error
    checkpoint = 'src/best_models/incompressible_XVODPW_checkpoint_521.tar' # fraenkels_1e3_5e3_1_ypos_fixedR_fulldomain, [32]*4, 3.64% val error

    # checkpoint = 'src/best_models/incompressible_KS7YQW_checkpoint_669.tar' # fraenkels_1e4_5e3_1_ypos_fixedR_fulldomain, [16]*2, 0.81% val error
    checkpoint = 'src/best_models/incompressible_5W3P5R_checkpoint_669.tar' # fraenkels_1e4_5e3_1_ypos_fixedR_fulldomain, [32]*4, 0.84% val error


    # checkpoint = 'src/best_models/incompressible_ND4VD4_checkpoint_693.tar' # fraenkels_1e3_5e3_1_ypos_fixedR_zoomedin, [32]*4, 3.64% val error
    
    # checkpoint = 'src/best_models/incompressible_XNHSY9_checkpoint_681.tar' # fraenkels_1e3_5e3_1_ypos_fulldomain, [16]*2, 23.22% val error

    model = Model.load_checkpoint(checkpoint, dataset=dataset, args=args, initial_optm='lbfgs')
    # # print(model.nn.layers)

    PLOT_MODEL = True
    if PLOT_MODEL:
        N = 500
        x = np.linspace(-1.75,-.85,N)
        y = np.linspace(0,.35,N)
        # x = np.linspace(-2,2,N)
        # y = np.linspace(0,1.5,N)
        X_, Y_ = np.meshgrid(x,y)
        X = X_.flatten()
        Y = Y_.flatten()
        vort, u, r = 5, 1, 1
        V = np.ones_like(X)*vort
        U = np.ones_like(X)*u
        R = np.ones_like(X)*r
        inputs = np.column_stack((X,Y,V,U,R))
        print(inputs.shape)
        inputs = torch.from_numpy(inputs).to(torch.float32)
        outputs = model.predict(inputs).detach().numpy()
        # print(outputs)
        Ux = outputs[:,0].reshape(N,N).T
        Uy = outputs[:,1].reshape(N,N).T
        psi = model.psi(inputs).detach().numpy().reshape(N,N)
        plt.figure(figsize=(18, 10))
        
        # im = plt.pcolor(x, y, (np.sqrt(Ux**2+Uy**2)).T,shading='auto',cmap='RdBu',vmin=-.5, vmax=.5)
        # im = plt.pcolor(x, y, Ux.T,shading='auto',cmap='RdBu',vmin=-.5, vmax=.5)
        im = plt.pcolor(x, y, Uy.T,shading='auto',cmap='RdBu',vmin=-.3, vmax=.3)
        ct = plt.contour(x,y, psi+u*Y_,levels=30,linewidths=.9, colors='firebrick')
        plt.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
        circle = plt.Circle((0,0),r,color='k',alpha=0.35)
        plt.gca().add_artist(circle)
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$y$", fontsize=16)
        plt.colorbar(im, shrink=.6, label='v')
        plt.gca().set_aspect('equal')
        # save figure tight layout
        plt.savefig('imgs/fraenkel/velocity_field_model_v.png', dpi=256, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()