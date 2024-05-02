import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
from scipy.linalg import null_space

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
    # args.layers = [6]*1
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
    args.x_vars = ["x", "y", "circ", "U_infty", "r"]

    args.normalize_inputs = True
    args.reduce_inputs = True
    args.transform_output = True
    # args.reduce_inputs = False
    # args.transform_output = False

    args.sampling_box = [[-3,3],[-3,3],[-5,5],[.5,5],[.1,3]] # x,y,c,U,r

    return


def main():
    change_parameters(args)

    def phi_base(D):
        return null_space(D)
    def phi_zero(D):
        phi = phi_base(D)

        phi[:,-1] /= -phi[-1,-1]
        
        phi[:,0] += phi[-1,0]*phi[:,-1]
        phi[:,1] += phi[-1,1]*phi[:,-1]
        phi[:,2] += phi[-1,2]*phi[:,-1]
        # k = 1
        # phi[:,-1] += ((1-phi[-2,-1])/phi[-2,k])*phi[:,k]

        return phi


    # D = [[1,1,2,1,1,2],[0,0,-1,-1,0,-1]] # x,y,c,U,r|psi
    # phi = phi_zero(D).T
    # # print(phi)
    # *inputs, outputs = phi
    # print(phi)
    # args.phi = [input[:-1] for input in inputs]
    # args.phi_output = outputs[:-1]

    D = [[1,1,2,1,1],[0,0,-1,-1,0]] # x,y,c,U,r
    phi = phi_base(D).T
    args.phi = phi
    # print(args.phi)

    # args.phi = [[1,-1,0,0,0],[1,0,-1,1,0],[0,1,0,0,-1]]

    # args.phi_output = [1,0,0,1,0]
    args.phi_output = [0,1,0,1,0]
    # args.phi_output = [0,0,0,0,0]

    # args.phi = [[1,-1, 0, 0, 0],
    #             [0, 1, 0, 0,-1],
    #             [1, 0,-1, 1, 0]]
    # args.phi = [[1/6,-1/6, 0, 0, 0],
    #             [0, 1/2, 0, 0,-1/2],
    #             [1/6, 0,-1/6, 1/6, 0]]

    # args.sampling_box = [[.5,1.6],[0,7],[0,6]] # pi1, pi2, pi3

    # args.phi_output = [0,0,1,0,0]

    # args.phi = [[ 1,-1, 0, 0, 0],
    #             [ 0, 1, 0, 0,-1],
    #             [-1, 0,1,-1, 0]]

    # args.phi = [[ 1,-1, 0, 0, 0],
    #             [ 0,1, 0, 0,-1],
    #             [ 0,1,-1,1, 0]]
    # args.phi = [[ 1, 0, 0, 0,-1],
    #             [ 0, 1, 0, 0,-1],
    #             [ 0,-1,1,-1, 0]]
    args.phi = [[ 1, 0, 0, 0,-1],
                [ 0, 1, 0, 0,-1],
                [ 0, 0,1,-1, -1]]
    # args.phi_output = [0,1,0,1,0]
    args.phi_output = [0,0,0,1,1]
    # args.phi_output = [0,0,0,0,0]


    # args.phi = [[1/6,-1/6, 0, 0, 0],
    #             [0, 1/4, 0, 0,-1/4],
    #             [0,-1/6,1/6,-1/6, 0]]
    # args.phi_output = [1,0,0,1,0]

    # # confirmation:
    # phi = np.concatenate((args.phi, [args.phi_output])).T
    # phi = phi.T
    # print(phi)
    # # print(f'\n{np.linalg.norm(D@phi.T):.2e}')
    # print(np.matrix.round(phi@phi.T,2))
    

    # print(args.phi)
    # print(args.phi_output)

    # args.phi = [[-1,1,0,0],[-1,0,1,-1]]
    # args.phi_output = [1,0,0,1]
    # args.phi = [[1,-1,0,0],[0,-1,1,-1]]
    # args.phi_output = [0,1,0,1]
    
    # a = np.concatenate((args.phi, [args.phi_output])).T
    # print(a.T@a)

    # dataset = load_data('data_5e2_1e3_1_idx0')
    # dataset = load_data('data_1e3_1e3_1_idx0_centered')
    # dataset = load_data('data_1e3_1e3_1_idx0_shifted')

    # dataset = load_data('data_1e3_1e3_1_idx0')
    # dataset = load_data('data_1e4_1e3_1_idx0_shifted_onlyposcirc')
    # dataset = load_data('data_1e3_1e3_1_idx0_shifted_onlyposcirc')

    # dataset = load_data('data_1e2_5e3_1_idx0')
    # dataset = load_data('data_1e3_5e3_1_idx0')
    dataset = load_data('data_1e4_5e3_1_idx0')


    from src.model_simple import REDUCED_VARS, NORMALIZE
    if args.reduce_inputs==True:
        model = Model(dataset, args)
        pi = REDUCED_VARS.forward(model, torch.tensor(dataset['train'][:,:5])).T
        pi_np = pi.detach().numpy()
        args.sampling_box = [[np.min(pi_np[0]),np.max(pi_np[0])],[np.min(pi_np[1]),np.max(pi_np[1])],[np.min(pi_np[2]),np.max(pi_np[2])]] # pi1,pi2,pi3
        
        pi1_, pi2_, pi3_ =  pi
        if args.normalize_inputs==True:
            pi1, pi2, pi3 = NORMALIZE.forward(model, pi)
            pi1, pi2, pi3 = pi1.detach().numpy(), pi2.detach().numpy(), pi3.detach().numpy()
        else:
            pi1, pi2, pi3 = pi
        # pi1, pi2, pi3 = NORMALIZE.forward(model, pi)
        # pi1, pi2, pi3 = pi1.detach().numpy(), pi2.detach().numpy(), pi3.detach().numpy()

    PLOT_TRANSFORMED = True
    if PLOT_TRANSFORMED:
        stream_red = pi2_*(-1/(pi1_**2+pi2_**2)+pi3_/pi2_*np.log(pi1_**2+pi2_**2)/(4*np.pi))
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pi1, pi2, pi3, c=stream_red, s=30, cmap='viridis')
        ax.set_xlabel('$\pi_1$', fontsize=16)
        ax.set_ylabel('$\pi_2$', fontsize=16)
        ax.set_zlabel('$\pi_3$', fontsize=16)
        plt.colorbar(sc, ax=ax, label='$\chi$')
        # top view
        ax.view_init(elev=90, azim=0)
        # side view
        # ax.view_init(elev=0, azim=0)
        # fig.savefig('imgs/cylinder/cyl_wrong_transf_topview_norm_exp.png', dpi=256, bbox_inches='tight')
        plt.show()

        # fig, ax = plt.subplots(figsize=(10,10))
        # sc = ax.scatter(dataset['train'][:,0], dataset['train'][:,1], c=stream_red)
        # ax.set_aspect('equal')
        # center = (0,0)
        # # circle = plt.Circle(center,2,color='forestgreen',alpha=0.15)
        # # ax.add_artist(circle)
        # circle = plt.Circle(center,1,color='firebrick',alpha=0.2)
        # ax.add_artist(circle)
        # ax.set_xlabel('$x$', fontsize=16)
        # ax.set_ylabel('$y$', fontsize=16)
        # # ax.set_xlim(0,3)
        # # ax.set_ylim(0,3)
        # # plt.savefig('imgs/cylinder/cyl_sampling_centered.png', dpi=256, bbox_inches='tight')
        # plt.show()



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
        center = (3,3)
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

    # train_model(dataset, args)

    checkpoint = 'src/best_models/incompressible_QAKI7B_checkpoint_675.tar' # reduced - 'data_1e2_5e3_1_idx0', [6]*1, 1.87% val error
    checkpoint = 'src/best_models/incompressible_D413B2_checkpoint_697.tar' # reduced - 'data_1e3_5e3_1_idx0', [6]*1, 2.71% val error
    checkpoint = 'src/best_models/incompressible_LKCM1D_checkpoint_639.tar' # reduced - 'data_1e4_5e3_1_idx0', [6]*1, 1.56% val error

    checkpoint = 'src/best_models/incompressible_WTNYV6_checkpoint_699.tar' # reduced - 'data_1e2_5e3_1_idx0', [16]*2, 1.73% val error
    checkpoint = 'src/best_models/incompressible_9GPPSO_checkpoint_695.tar' # reduced - 'data_1e3_5e3_1_idx0', [16]*2, 0.89% val error
    checkpoint = 'src/best_models/incompressible_QK6AZQ_checkpoint_699.tar' # reduced - 'data_1e4_5e3_1_idx0', [16]*2, 0.54% val error

    # checkpoint = 'incompressible_3RTFFT_checkpoint_522' # normalization - 'data_1e2_5e3_1_idx0', [6]*1, 16.27% val error
    # checkpoint = 'incompressible_QIXG3Y_checkpoint_601' # normalization - 'data_1e3_5e3_1_idx0', [6]*1, 10.81% val error
    # checkpoint = 'incompressible_B8TKZO_checkpoint_528' # normalization - 'data_1e4_5e3_1_idx0', [6]*1, 10.56% val error

    # checkpoint = 'incompressible_XSHOWV_checkpoint_171' # normalization - 'data_1e2_5e3_1_idx0', [16]*2, 33.87% val error (0% train error)
    # checkpoint = 'incompressible_31M124_checkpoint_682' # normalization - 'data_1e3_5e3_1_idx0', [16]*2, 3.11% val error (1.97% train error)
    # checkpoint = 'incompressible_S1PC5I_checkpoint_675' # normalization - 'data_1e4_5e3_1_idx0', [16]*2, 1.51% val error

    model = Model.load_checkpoint(checkpoint, dataset=dataset, args=args, initial_optm='lbfgs')
    # print(model.nn.layers)

    PLOT_MODEL = True
    if PLOT_MODEL:
        N = 500
        x = np.linspace(-3,3,N)
        y = np.linspace(-3,3,N)
        X_, Y_ = np.meshgrid(x,y)
        # mask = np.sqrt((X-3)**2 + (Y-3)**2) <= 1.5
        X = X_.flatten()
        Y = Y_.flatten()
        c, u, r = 0, 1, 1
        C = np.ones_like(X)*c
        U = np.ones_like(X)*u
        R = np.ones_like(X)*r
        inputs = np.column_stack((X,Y,C,U,R))
        print(inputs.shape)
        inputs = torch.from_numpy(inputs).to(torch.float32)
        outputs = model.predict(inputs).detach().numpy()
        Ux = outputs[:,0].reshape(N,N).T
        Uy = outputs[:,1].reshape(N,N).T
        # psi = outputs[:,2].reshape(100,100)
        psi = model.psi(inputs).detach().numpy().reshape(N,N)
        fig = plt.figure(figsize=(18, 10))
        im = plt.pcolor(x, y, (np.sqrt(Ux**2+Uy**2)).T,shading='auto',cmap='viridis',vmin=0,vmax=2*u)
        ct = plt.contour(x,y, psi+u*Y_,levels=30,linewidths=.9, colors='firebrick')
        plt.clabel(ct, inline=True, fontsize=8, fmt='%1.1e')
        circle = plt.Circle((0,0),r,color='k',alpha=0.35)
        plt.gca().add_artist(circle)
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$y$", fontsize=16)
        plt.colorbar(im)
        plt.gca().set_aspect('equal')
        fig.savefig('imgs/cylinder/cyl_velocity_streamlines.png', dpi=256, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()