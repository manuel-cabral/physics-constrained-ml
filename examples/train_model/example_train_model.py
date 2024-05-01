from timeit import default_timer as timer
from datetime import timedelta
from examples.train_model import args
from src.utils import get_args, set_torch_dtype #, set_torch_multiprocessing
from src.dataset import Dataset
from src.model import Model
from src.plotter import plot_dataset_points


if __name__ == '__main__':
    # Get args
    args = get_args(args)

    # Set float precision
    set_torch_dtype(args.ftype)

    # Set multiprocessing
    # set_torch_multiprocessing()

    # Create dataset
    dataset = Dataset(args)
    plot_dataset_points(dataset, 0)

    # Create model
    model = Model(dataset, args)

    # Train model
    start = timer()
    model.train()
    end = timer()
    print("Training time: {}".format(timedelta(seconds=end-start)))
