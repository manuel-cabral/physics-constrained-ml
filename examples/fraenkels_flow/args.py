# Mode
modelmode = 0 # 0: nothing (only creates trainer class). 1: train. 2: train and test. 3: test
checkpoint = None # if modelmode = 2, specify the checkpoint to load the trained model from, else None

# Dataset
dataset_path = "dataset/" # Dataset path
cfd_dir = "dataset/RANS" # CFD simulations path
cfd_sims=  ["airFoil2D_SST_31.577_2.75_2.604_4.088_17.312"] # airFoil2D_SST_90.415_14.684_4.356_6.209_11.299, airFoil2D_SST_33.816_-1.984_6.59_0.0_7.983, airFoil2D_SST_31.577_2.75_2.604_4.088_17.312
dataset = None # "dataset/test.npy" # dataset file (same file for panels `_panels.npy` expected). Use `None` to generate a new datafile from `cfd_dir`
dataset_maxsize = None # crop dataset if necessary (use `None` otherwise)
dataset_sampling_size = 20_000 # points sampled per case (use `None` otherwise)
sampling_mode = "mesh_density" # choose from "uniform", "mesh_density" or "mix"
sampling_weight = 0 # sampling weight given to uniform distribution if in mix mode, [0,1]
sampling_distance = None # (0.0015, 2.5) # distance from the body min/max sampling values
sampling_box = [(5e-2, 1), (0, 5e-2), (1/7e4,1/3e4),(0,5)] # None # xmin, xmax, ymin, ymax

# Panels
panels_maxsize = 100
panels_minlength = None # 0.001
panels_dist = 'cos' # choose 'cfd' or 'cos'
panels_eccentricity = 0.95 # in case of 'cos' sampling
panels_skewness = 0 # in case of 'cos' sampling
panels_skewness_type = 'linear' # in case of 'cos' sampling; choose 'linear', 'exp' or 'inverse'.
symm_on = ()

# Model
x_vars = ["x", "y", "visc", "U_inf"] # MLP inputs (x, y; parameters...)
y_vars = ["u", "v"] # target output
layers = [6]*1
act = "Tanh" # activation function: Tanh, ReLU
kind = 'incompressible' # choose NN architecture from 'model', 'baseline', 'incompressible' or 'soft incompressible'
scheduler = False
normalize_inputs = False
reduce_inputs = False
transform_output = False
phi = [[.5,0,-.5,.5],[-.5,1,-.5,.5]] # if reduce_inputs==True, choose initial phi
phi_output = [0,1,0,1]
subtract_uniform_flow = True
norm_layer = True
soft_weight = 0 # if kind = 'soft incompressible', choose weight given to divergence penalty

# Training
device = "cpu" # perform training in GPU ("cuda") or CPU ("cpu")
training_path = "training/" # directory to store trained models
plots_path = "plots/" # directory to store history plots
shuffle_dataset = 1 # shuffle dataset before splitting it
shuffle_sampler = 1 # shuffle sampler (once the dataset has been split, shuffle samples in each sampler)
data_split = [0.8, 0.8] # ratio for train/test & (within the train split) train/val
training_loss_function = "mse_loss" # loss function. Can try: mse_loss, l1_loss, smooth_l1_loss, etc (see torch.nn.functional)
training_loss_reduction = "sum" # how the loss is reduced across batch samples ("sum" or "mean")
n_epochs = 300 # number of epochs
n_epochs_adam = 200 # number of epochs of ADAM (n_epochs-n_epochs_adam epochs of L-BFGS)
learning_rate = 1e-2 # starting learning rate
learning_rate_adam = 1e-2 # learning rate for ADAM optimizer
learning_rate_lbfgs = 1e-2 # learning rate for L-BFGS optimizer
batch_size = 512 # batch size
patience = None # number of epochs of patience before training is stopped when validation loss is not reduced
history_plot_name = "history.pdf"

# Others
ftype = "float32"
pin_memory = False # only for CPU usage
seed = 101 # random seed
verbose = 1 # verbose level. 0: None. 1: Text. 2: Text + plots