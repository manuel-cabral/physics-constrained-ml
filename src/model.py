import os
import random
from argparse import Namespace
import numpy as np
import torch
import torch.utils.data
from functorch import vmap, jacrev, hessian
import src.utils as utils

class NORMALIZE(torch.nn.Module):
    """    
    Normalizes input tensors to the range [-3, 3] using min/max values from `args.sampling_box`. 

    Args:
        args: Contains:
            - sampling_box: List of (min, max) pairs for each feature.
            - kind: String that determines the axis of normalization.

    Forward:
        input (Tensor): Input tensor to normalize.
        Returns: Normalized tensor.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        sampling_box = torch.tensor(args.sampling_box)  # convert to tensor
        self.offset = sampling_box[:, 0]  # min values
        self.range = sampling_box[:, 1] - sampling_box[:, 0]  # max - min

    def forward(self, input):
        dims = (1,) if self.args.kind == 'incompressible' else (0,) # broadcast to all dimensions
        offset_m = self.offset.view(-1, *dims).expand_as(input)
        range_m = self.range.view(-1, *dims).expand_as(input)

        return 6 * (input - offset_m) / range_m - 3


class TRANSFORM_INPUT(torch.nn.Module):
    """Transform input variables to new variables. This is done as a layer as it needs to be differentiable.

    Args:
        args (Namespace): An argparse.Namespace object containing the following attributes:
            - phi (list): List of tuples specifying the powers of each input variable
            - x_vars (list): List of input variable names.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, input):
        if len(input.shape)==1:
            new_vars = torch.ones_like(input[:len(self.args.phi)])
            for i,l in enumerate(self.args.phi):
                for j,k in enumerate(l):
                    new_vars[i] *= torch.pow(input[j],k)
        else:
            new_vars = torch.ones_like(input[:,:len(self.args.phi)])
            for i,l in enumerate(self.args.phi):
                for j,k in enumerate(l):
                    new_vars[:,i] *= torch.pow(input[:,j],k)
        return new_vars
    
class TRANSFORM_INPUT(torch.nn.Module):
    """
    Applies transformations to input variables based on predefined powers (from `phi`), making the process differentiable.

    Args:
        args: Contains:
            - phi: List of tuples specifying powers for the corresponding input variables.
            - x_vars: List of input variable names.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.phi = torch.tensor(args.phi)  # Convert `phi` to a tensor for easy operations later

    def forward(self, input):
        # Precompute shape to avoid conditionals and handle 1D or 2D inputs
        batch_dim = input.shape[0] if input.ndim > 1 else 1
        new_vars = torch.ones(batch_dim, len(self.phi), dtype=input.dtype, device=input.device)

        # Apply transformations
        for i, powers in enumerate(self.phi):
            new_vars[..., i] = torch.prod(torch.pow(input[..., :len(powers)], powers), dim=-1)

        return new_vars


class NN(torch.nn.Module):
    """Create a feed-forward neural network, based on the provided arguments.

    Args:
        args (Namespace): An argparse.Namespace object containing the following attributes:
            - kind (str): The kind of neural network ('baseline', 'soft_incompressible' or 'incompressible').
            - x_vars (list): List of input variable names.
            - act (str): The name of the activation function to use.
            - layers (list): List of integers specifying the number of neurons in each hidden layer.
    """
    def __init__(self, args):
        super(NN, self).__init__()
        input_dim = len(args.x_vars)
        output_dim = 1 if args.kind=='incompressible' else 2
        act = getattr(torch.nn, args.act)() # activation function defined in args.act

        layers = torch.nn.ModuleList()

        if args.reduce_inputs:
            layers.append(TRANSFORM_INPUT(args))
            input_dim = len(args.phi)

        if args.normalize_inputs:
            layers.append(NORMALIZE(args))
            # assert len(args.x_vars)==len(args.sampling_box), 'Number of input variables must match the number of sampling boxes.'

        layers.append(torch.nn.Linear(input_dim, args.layers[0])) # input layer
        if args.norm_layer: layers.append(torch.nn.LayerNorm(args.layers[0]))
        layers.append(act)
        for i in range(len(args.layers)-1): # hidden layers
            layers.append(torch.nn.Linear(args.layers[i], args.layers[i+1]))
            if args.norm_layer: layers.append(torch.nn.LayerNorm(args.layers[0]))
            layers.append(act)
        layers.append(torch.nn.Linear(args.layers[-1], output_dim, bias=True)) # output layer

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)
    
class Model(object):
    """Create the model around the NN.

    This class encapsulates the training and evaluation of the chosen model kind
    on a given dataset. It provides methods to train the model, save and load its
    state, track training history, as well as computing some other relevant quantities.
   
    Args:
        dataset (Dataset): The dataset used for training and evaluation.
        args (argparse.Namespace): An argparse.Namespace object containing various
            configuration parameters for the model.
        model_id (str, optional): An optional identifier for the model. If not provided,
            a random ID will be generated.
        model_state_dict (dict, optional): The state dictionary of the model's parameters.
        optimizer_state_dict (dict, optional): The state dictionary of the optimizer.
        epoch (int, optional): The current epoch of training (-1 if not specified).
        history (dict, optional): A dictionary to track training history (if not specified,
            it will be initialized with empty lists).
        training_path (list, optional): A list to store the training path.
    """
    def __init__(self, dataset, args,
        model_id=None, model_state_dict=None, initial_optm='adam', optimizer_state_dict=None, epoch=-1, history=None, training_path=None):

        # Get device
        self.device = utils.get_device(args.device)

        # Store args
        self.args = args

        # Set seed for reproducibility
        utils.set_seed(args.seed)

        # Generate new ID
        self.model_id = model_id if model_id is not None else ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(6))

        # Define NN
        self.nn = NN(args)
        self.nn = self.nn.to(args.device)
        if model_state_dict is not None: self.nn.load_state_dict(model_state_dict)

        # Define dataset, and split it
        self.dataset = dataset
        self.data_loaders = self._split_dataset(dataset)

        # Define optimizers
        self.optimizer_adam = torch.optim.Adam(
            self.nn.parameters(), 
            lr=args.learning_rate_adam
            )
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.nn.parameters(), 
            lr=args.learning_rate_lbfgs,
            line_search_fn = "strong_wolfe"
            )
        if initial_optm=='adam': self.optimizer = self.optimizer_adam
        elif initial_optm=='lbfgs': self.optimizer = self.optimizer_lbfgs
        if optimizer_state_dict is not None: self.optimizer.load_state_dict(optimizer_state_dict)

        if args.scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-4, factor=.5, patience=50)
        self.optimizer = self.optimizer_adam

        # Training parameters
        self.dtype = utils.ftype_torch[self.args.ftype]
        self.loss_function = getattr(torch.nn.functional, args.training_loss_function)

        # Set history tracking
        self.epoch = epoch
        self.training_path = training_path or []
        self.history = history if history is not None else {
            'train_loss': [],
            'train_error': [],
            'val_loss': [],
            'val_error': [],
        }

    def forward(self, X):
        # returns the predicted velocity field, regardless of 'kind'

        def background(X):
            # returns the predicted velocity field, regardless of 'kind'
            predictions = self.curl(X) if self.args.kind=='incompressible' else self.nn(X)

            if self.args.subtract_uniform_flow: 
                predictions[:,0] += X[:,3] # assumes fourth variable of the input is the background uniform flow
                # predictions[:,0] += X[:,1]*X[:,2] # vort*y - only for frankels example!

            return predictions

        return background(X)

    def loss(self, output, target, X):
        loss = self.loss_function(output, target, reduction=self.args.training_loss_reduction)
        
        # modify loss function, if using soft contraints
        if self.args.kind=='soft_incompressible':
            X.requires_grad = True
            loss += self.args.soft_weight * torch.norm(self.divergence(X, kind='soft_incompressible'))/X.shape[0]
        return loss

    def error(self, output, target):
        return torch.norm(output-target,2)/torch.norm(target,2)

    def train(self, extra=0):
        print(f'\n Training starting on {next(self.nn.parameters()).device}...\n')

        if len(self.history['val_loss']): # resume training
            for k,v in self.history.items():
                self.history[k] = v[:self.epoch+1]
            best_val_loss = self.history['val_loss'][self.epoch]
            best_model_state_dict = self.nn.state_dict()
            best_optimizer_state_dict = self.optimizer.state_dict()
        else: # init training
            best_val_loss = np.inf
            best_model_state_dict = None
            best_optimizer_state_dict = None

        running_patience = 0
        for e_id in range(self.epoch + 1, self.args.n_epochs): # epochs iteration
            running_loss = {'train': 0.0, 'val': 0.0}
            running_error = {'train': 0.0, 'val': 0.0}

            for mode in ['train', 'val']: # train/val split iteration
                for batch in self.data_loaders[mode]: # dataset iteration
                    if mode == 'train':
                        self.nn.train() # set NN in training mode
                    else:
                        self.nn.eval() # set NN in evaluation mode

                    X = batch[:,:len(self.args.x_vars)]
                    uv = batch[:,len(self.args.x_vars):len(self.args.x_vars)+len(self.args.y_vars)].to(self.dtype)

                    if next(self.nn.parameters()).device.type == "cuda":
                        X = X.cuda(non_blocking=self.args.pin_memory)
                        uv = uv.cuda(non_blocking=self.args.pin_memory)
                    
                    X = X.to(self.dtype)
                    uv_nn = self.forward(X) 

                    loss = self.loss(uv_nn, uv, X)
                    error = self.error(uv_nn, uv)
                    running_loss[mode] += loss.cpu().detach()
                    running_error[mode] += error.cpu().detach()

                    def closure():
                        if torch.is_grad_enabled(): 
                            self.optimizer.zero_grad()
                        outputs = self.forward(X)
                        loss = self.loss(outputs, uv, X)
                        if loss.requires_grad:
                            loss.backward()
                        return loss

                    if mode == 'train':
                        self.optimizer.step(closure)

                # Save training path
                if mode == 'train': 
                    epoch_params = [p.detach().numpy().copy() for p in self.nn.parameters()]
                    self.training_path.append(epoch_params)


            # Save epoch in self.history & output epoch train/val performance
            train_loss = running_loss['train'] / self.dataset['train'].shape[0] #!
            train_error = running_error['train'] / len(self.data_loaders['train'])
            val_loss = running_loss['val'] / (len(self.data_loaders['val']) * self.data_loaders['val'].batch_size)
            val_error = running_error['val'] / len(self.data_loaders['val'])
            self.history['train_loss'].append(train_loss)
            self.history['train_error'].append(train_error)
            self.history['val_loss'].append(val_loss)
            self.history['val_error'].append(val_error)

            if self.args.scheduler:
                self.scheduler.step(val_loss)

            if e_id%10==0:
                print(f"{self.optimizer.__class__.__name__} - lr = {self.optimizer.param_groups[0]['lr']: .2e}")
                print(f'Epoch {e_id:03d}: train_loss* = {train_loss:.4e} | train_err = {train_error:05.2%} | val_loss* = {val_loss:.4e} | val_err = {val_error:05.2%}')

            # Early-stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = self.nn.state_dict()
                best_optimizer_state_dict = self.optimizer.state_dict()
                self.epoch = e_id
                running_patience = 0
            else:
                running_patience += 1
                if self.args.patience is not None and running_patience == self.args.patience:
                    if self.optimizer==self.optimizer_adam:
                        print('\nEarly-stopping criteria for ADAM triggered at epoch {}, after {} epochs of validation loss not improving. Switched to L-BFGS optimizer.'
                            .format(e_id, self.args.patience))
                        self.optimizer = self.optimizer_lbfgs
                        running_patience = 0                          
                    else:
                        print(f'\nEarly-stopping criteria triggered at epoch {e_id}, after {self.args.patience} epochs of validation loss not improving.')
                        self.save_checkpoint(best_model_state_dict, best_optimizer_state_dict)
                        return self.epoch
            
            if self.optimizer==self.optimizer_adam and e_id>=self.args.n_epochs_adam-1:
                print('\nChanged optimizer from ADAM to LBFGS')
                # Save model at the end of the end of ADAM.
                self.save_checkpoint(self.nn.state_dict(), self.optimizer.state_dict(), training_file='partial_models/', name=f'{self.args.kind}_ADAM_{self.args.n_epochs_adam}_set{extra}_{self.model_id}')
                self.optimizer = self.optimizer_lbfgs

        
        # Save also the last model to continue training (for testing purposes).
        self.save_checkpoint(self.nn.state_dict(), self.optimizer.state_dict(), training_file='partial_models/', name=f'{self.args.kind}_LBFGS_{self.args.n_epochs_adam}ad_{self.args.n_epochs-self.args.n_epochs_adam}lb_set{extra}_{self.model_id}')

        # Training finished with max number of epochs. Save best model.
        self.save_checkpoint(best_model_state_dict, best_optimizer_state_dict, training_file='best_models/')
        print(f'Epoch {e_id:03d}: train_loss* = {train_loss:.4e} | train_err = {train_error:05.2%} | val_loss* = {val_loss:.4e} | val_err = {val_error:05.2%}')
        print(f'\nTraining finished at epoch {self.args.n_epochs - 1}, with best model at epoch {self.epoch}.')

        return self.epoch

    def predict(self, X, kind=None):
        """Predict the output of the model for input data.

        Args:
            X (Tensor or array-like): Input data to make predictions on. If not a Tensor,
                it will be converted to a Tensor with gradient tracking enabled.

        Returns:
            torch.Tensor: Predicted output of the model.

        Note:
            If the model is on a GPU (CUDA), the input data will be transferred to the GPU
            for prediction.
        """
        # check if X is torch tensors, otherwise convert
        if not torch.is_tensor(X):
            X = torch.tensor(X, requires_grad=True)
        elif not X.requires_grad:
            X.requires_grad_(True)
        # check if model is in CPU or GPU
        if next(self.nn.parameters()).device.type == "cuda":
            X = X.to("cuda")

        return self.forward(X)
    
    def transform_output(self, X):
        X = X.to(self.dtype)
        out = self.nn(X)

        if self.args.transform_output:
            for i, exp in enumerate(self.args.phi_output):
                out *= torch.pow(X[i],exp)
        return out
    
    def psi(self, X):
        """
        Stream-function.
        To apply symmetry conditions specify the dimensions on symm_on=(1,) or symm_on=(0,1)
        to apply it on the dim=1 (y) or dim=(0,1) (x) respectively.

        Args:
            X (torch.Tensor): The input tensor for which the curl is computed.
    
        Returns:
            torch.Tensor: A tensor representing the output of the neural network's output with
                respect to the input, after the symmetric conditions have been applied (stream-function).
                The resulting tensor has shape (batch_size,).
        """

        return vmap(self.transform_output)(X)

    def curl(self, X):
        """
        Compute the curl of the neural network's output with respect to the input.

        Args:
            X (torch.Tensor): The input tensor for which the curl is computed.
    
        Returns:
            torch.Tensor: A tensor representing the curl of the neural network's output with
                respect to the input. The resulting tensor has shape (batch_size, 2).
        """
        J = vmap(jacrev(self.transform_output))(X) # J.shape = (batch_size, len(Y), len(X))
        J = torch.squeeze(J) # single output
        J = utils.cross_flip_stack(J) # swaps indices 0 & 1, and negates the new value at 1
        return J[..., :2] 

    def vorticity(self, X):
        """
        Compute the vorticity of the neural network's output with respect to the input.

        Args:
            X (torch.Tensor): The input tensor for which the vorticity is computed.

        Returns:
            torch.Tensor: A tensor representing the vorticity of the neural network's output with
                respect to the input. The resulting tensor has shape (batch_size,).
        """
        # H = vmap(hessian(self.nn))(X) # H.shape = (batch_size, len(Y), len(X), len(X))
        # H = hessian(self.transform_output)(X) # H.shape = (batch_size, len(Y), len(X), len(X))
        H = vmap(hessian(self.transform_output))(X) # H.shape = (batch_size, len(Y), len(X), len(X))
        H = torch.squeeze(H) # single output
        return H[..., 0, 0] + H[..., 1, 1] # vort = laplacian(psi)
    
    def divergence(self, X):
        """
        Compute the divergence of the neural network's output vector field.

        Args:
            X (torch.Tensor): The input tensor for which the divergence is computed.

        Returns:
            torch.Tensor: A tensor representing the divergence of the neural network's output vector field
                with respect to the input. The resulting tensor has shape (batch_size,).
        """
        y = self.predict(X)
        ones_tensor = torch.ones_like(y[:,0])
        grad_x = torch.autograd.grad(y[:,0], X, ones_tensor, create_graph=True)[0]
        grad_y = torch.autograd.grad(y[:,1], X, ones_tensor, create_graph=True)[0]

        return grad_x[:, 0] + grad_y[:, 1]

    def save_checkpoint(self, model_state_dict, optimizer_state_dict, training_file=None, name=None):
        # Save model
        dir_path = os.path.dirname(os.path.abspath(__file__))
        if training_file is None: training_file = self.args.training_path
        folder_path = os.path.join(dir_path, training_file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        name = f'{self.args.kind}_{self.model_id}_checkpoint_{self.epoch}.tar' if name is None else name
        model_fname = os.path.join(folder_path, name)
        torch.save({
            'model_id': self.model_id,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'epoch': self.epoch,
            'history': self.history,
            'args': vars(self.args),
            'training_path': self.training_path
        }, model_fname)
        print(f"Model has been saved in {model_fname}")

    @classmethod
    def load_checkpoint(cls, fname, dataset=None, args=None, initial_optm='adam'):
        checkpoint = torch.load(fname)
        args = Namespace(**checkpoint['args']) if args is None else args
        model_id = checkpoint['model_id']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        history = checkpoint['history']
        training_path = checkpoint['training_path']
        dataset = dataset
        return cls(dataset, args, model_id, model_state_dict, initial_optm, optimizer_state_dict, epoch, history, training_path)
    
    def _split_dataset(self, dataset):

        train_set = dataset['train']
        val_set = dataset['val']
        test_set = dataset['test']
        
        train_len = len(train_set)
        val_len = len(val_set)
        test_len = len(test_set)
        if self.args.shuffle_dataset:
            train_indices = np.random.choice(len(train_set), train_len, replace=False)
            val_indices = np.random.choice(len(val_set), val_len, replace=False)
        else:
            train_indices = list(range(train_len))
            val_indices = list(range(val_len))

        test_indices = list(range(test_len))

        batch_size = self.args.batch_size if self.args.batch_size is not None else len(train_indices)
        pin_memory = self.args.pin_memory if self.args.device == "cpu" else False
        
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices) if self.args.shuffle_sampler else train_indices
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices) if self.args.shuffle_sampler else val_indices
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_indices), sampler=val_sampler, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_indices), sampler=test_sampler, pin_memory=pin_memory)

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}