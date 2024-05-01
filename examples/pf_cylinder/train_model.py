import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import torch
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
from generate_data import generate_sobol

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

    # Train model
    start = timer()
    model.train()
    end = timer()
    print(f'Training time: {timedelta(seconds=end-start)}')

    return model

def load_pickle(path):
    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)
    data = torch.from_numpy(data)
    return data

def load_dataset(circulations, idx=1):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset = {'train': torch.Tensor(), 'val': torch.Tensor(), 'test': torch.Tensor()}
    for circ in circulations:
        for mode in ['train', 'val']:
            file_path = os.path.join(dir_path, 'datasets', f'{mode}_circ{circ:.2f}_set{idx+1}')
            data = load_pickle(file_path)
            dataset[f'{mode}'] = torch.cat((dataset[f'{mode}'], data), dim=0)
        file_path = os.path.join(dir_path, 'datasets', f'test_circ{circ:.2f}_set')
        data = load_pickle(file_path)
        dataset['test'] = torch.cat((dataset['test'], data), dim=0)
    return dataset

def main():
    # change some parameters
    args.kind = 'incompressible' 
    args.n_epochs = 120
    args.n_epochs_adam = 10_000
    args.learning_rate_adam = 1e-3
    args.learning_rate_lbfgs = 1e-2
    args.patience = 50
    args.batch_size = 200
    args.scheduler = False

    # for generalization
    args.x_vars = ["x", "y", "circulation"]

    circulations = generate_sobol(4, -5, 5)

    # load data and train model
    dataset = load_dataset(circulations, idx=1)
    model = train_model(dataset, args)

if __name__ == '__main__':
    main()