import argparse
import numpy as np
import torch
from scipy.stats.qmc import Sobol

def get_device(device):
    if device in ['cpu','cuda']:
        return torch.device(device)
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_torch_multiprocessing():
    torch.multiprocessing.set_start_method('spawn')

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_args(params):
    """
    This arguments can be passed with "$ python train.py -<arg_name> <arg_value>",
    otherwise their default value is given by the params.py file.
    """
    parser = argparse.ArgumentParser(description='args for the flow field prediction with mlp + pflow')
    parser.add_argument('-dataset_path', default=params.dataset_path, help='dataset path for both CFD sims and sampled dataset objects')
    parser.add_argument('-cfd_dir', default=params.cfd_dir, help='directory containing the RANS simulations')
    parser.add_argument('-cfd_sims', nargs='+', default=params.cfd_sims, help='RANS simulations to sample from')
    parser.add_argument('-dataset', default=params.dataset, help='numpy file containing the dataset')
    parser.add_argument('-dataset_maxsize', type=int, default=params.dataset_maxsize, help='maximum dataset size')
    parser.add_argument('-dataset_sampling_size', type=int, default=params.dataset_sampling_size, help='maximum points per case')
    parser.add_argument('-sampling_mode', default=params.sampling_mode, help='sampling mode (uniform or mesh_density)')
    parser.add_argument('-sampling_weight', default=params.sampling_weight, help='sampling weight given to uniform distribution if in mixed mode, [0,1]')
    parser.add_argument('-sampling_distance', nargs='+', type=float, default=params.sampling_distance, help='distance from the body when sampling')
    parser.add_argument('-sampling_box', nargs='+', type=float, default=params.sampling_box, help='box around the body when sampling')
    parser.add_argument('-panels_maxsize', type=int, default=params.panels_maxsize, help='number of points defining the panels of a case')
    parser.add_argument('-panels_minlength', type=float, default=params.panels_minlength, help='panels minimum length')
    parser.add_argument('-panels_dist', default=params.panels_dist, help="distribution of panels; 'cfd' or 'cos'; 'cfd' by default")
    parser.add_argument('-panels_eccentricity', type=float, default=params.panels_eccentricity, help='eccentricity of elliptical distribution')
    parser.add_argument('-panels_skewness', type=float, default=params.panels_skewness, help='skewness of elliptical distribution')
    parser.add_argument('-panels_skewness_type', default=params.panels_skewness_type, help="type of skewness; 'linear', 'exp' or 'inverse'; 'linear' by default")
    parser.add_argument('-symm_on', nargs='+', type=int, default=params.symm_on, help='symmetry axis')
    parser.add_argument('-modelmode', default=params.modelmode, type=int, help='0: nothing (only creates trainer class). 1: train. 2: train and test. 3: test')
    parser.add_argument('-x_vars', nargs='+', default=params.x_vars, help='input features')
    parser.add_argument('-y_vars', nargs='+', default=params.y_vars, help='target features')
    parser.add_argument('-layers', nargs='+', type=int, default=params.layers, help='DNN hidden layers list')
    parser.add_argument('-act', default=params.act, help='network activation function')
    parser.add_argument('-kind', default=params.kind, help='choose models architecture')
    parser.add_argument('-scheduler', default=params.scheduler, help='activate scheduler of learning rate for chosen optimizer (bool)')
    parser.add_argument('-normalize_inputs', default=params.normalize_inputs, help='normalize inputs of NN (bool)')
    parser.add_argument('-reduce_inputs', default=params.reduce_inputs, help='reduce inputs of NN (bool)')
    parser.add_argument('-transform_output', default=params.transform_output, help='transform output of NN (bool)')
    parser.add_argument('-phi', default=params.phi, help='exponents for inputs in reduce transformation (list of lists)')
    parser.add_argument('-phi_output', default=params.phi_output, help='exponents for output in reduce transformation (list)')
    parser.add_argument('-norm_layer', default=params.norm_layer, help='add LayerNorm layer after every linear layer (bool)')
    parser.add_argument('-subtract_uniform_flow', default=params.subtract_uniform_flow, help='subtract uniform flow from the output of the NN (as a way to normalize them) (bool)')
    parser.add_argument('-soft_weight', default=params.soft_weight, help="weight given to the divergence penalty, when kind='soft incompressible'")
    parser.add_argument('-data_split', nargs='+', type=float, default=params.data_split, help='split for dataset: train/test and (within train) train/val')
    parser.add_argument('-n_epochs', type=int, default=params.n_epochs, help='number of epochs')
    parser.add_argument('-n_epochs_adam', type=int, default=params.n_epochs_adam, help='number of epochs of ADAM (n_epochs-n_epochs_adam epochs of L-BFGS)')
    parser.add_argument('-batch_size', type=int, default=params.batch_size, help='batch size')
    parser.add_argument('-shuffle_dataset', type=int, default=params.shuffle_dataset, help='shuffle dataset before splitting train/val/test indices')
    parser.add_argument('-shuffle_sampler', type=int, default=params.shuffle_sampler, help='make subdataset sampler random or not')
    parser.add_argument('-learning_rate', type=float, default=params.learning_rate, help='learning rate')
    parser.add_argument('-learning_rate_adam', type=float, default=params.learning_rate_adam, help='learning rate for ADAM')
    parser.add_argument('-learning_rate_lbfgs', type=float, default=params.learning_rate_lbfgs, help='learning rate for L-BFGS')
    parser.add_argument('-training_loss_function', default=params.training_loss_function, help='loss function for node regression: mse_loss, l1_loss, smooth_l1_loss')
    parser.add_argument('-training_loss_reduction',default=params.training_loss_reduction, help='loss function reduction method: mean or sum, so that we can compute sum sq error or mean abs error.')
    parser.add_argument('-patience', type=int, default=params.patience, help='epochs for validation loss not improving so early stopping is triggered')
    parser.add_argument('-checkpoint', type=int, default=params.checkpoint, help='checkpoint where to load the model from (useful for test or continued training)')
    parser.add_argument('-training_path', default=params.training_path, help='Training path for storing model checkpoints')
    parser.add_argument('-plots_path', default=params.plots_path, help='plotting path for storing training history')
    parser.add_argument('-history_plot_name', default=params.history_plot_name, help='File name for the training history plot')
    parser.add_argument('-device', default=params.device, help='cpu or cuda for training/inference')
    parser.add_argument('-pin_memory', type=str_to_bool, nargs='?', const=True, default=params.pin_memory, help='pin batches to memory when training (torch.DataLoader option)')
    parser.add_argument('-ftype', default=params.ftype, help='float precision')
    parser.add_argument('-seed', type=int, default=params.seed, help='seed')
    parser.add_argument('-verbose', type=int, default=params.verbose, help='verbose level (0,1,2). 0: almost none. 1: text. 2: text and plots')
    args, _ = parser.parse_known_args()
    return args

ftype_numpy = {'float16': np.float16,
               'float32': np.float32,
               'float64': np.float64}
ftype_torch = {'float16': torch.float16,
               'float32': torch.float32,
               'float64': torch.float64}

def set_torch_dtype(ftype_str):
    return torch.set_default_dtype(ftype_torch[ftype_str])

def set_seed(seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def cross_flip_inplace(a):
    """ swaps indices 0 & 1, and negates the new value at index 1 """
    a[..., [0, 1]] = a[..., [1, 0]]
    a[..., 1] *= -1
    return a

def cross_flip_stack(a):
    """ swaps indices 0 & 1, and negates the new value at index 1 """
    return torch.vstack([a[...,1],-a[...,0]]+[a[...,i] for i in range(2,a.shape[-1])]).T

# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
def compare_torch_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def circle(N, a=1, x0=0, y0=0, t0=0):
    theta = torch.linspace(0, 2*torch.pi, N+1)[:-1]-t0
    return torch.stack([a*torch.cos(theta)+x0, a*torch.sin(theta)+y0,theta],dim=1)

def generate_sobol(num_pow, xmin, xmax, seed, add_zero=True, add_lims=True):
    sobol = Sobol(d=1, seed=seed)
    samples = xmin + (xmax - xmin) * sobol.random_base2(num_pow)
    samples = samples.flatten()
    if add_zero: samples = np.concatenate(([0], samples))
    if add_lims: samples = np.concatenate(([xmin, xmax], samples))
    samples.sort()
    return samples