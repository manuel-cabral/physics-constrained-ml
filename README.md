# Phy-Informed-Aerofoil-Flow

Mixing potential flow with ML for fast flow field interpolation.

## Get environment working

The main dependencies are: `torch`, `shapely`, and `pyvista`.
To automatically create a `conda` environment with a working setup run:
```
conda env create -f environment.yml
```
This will create a `conda` environment named `ml` (first line in `environment.yml`) containing all the required packages.

To create and install the packages manually, do:

1. Create a `conda` virtual environment and activate it:
```
conda create --name ml python=3.10
conda activate ml
```

2. Install dependencies:

```
pip install torch shapely pyparsing six airfrans
conda install -c conda-forge pyvista
```