# Phisics-Constrained-ML in flow prediction

Enforcing incompressibility and non-dimensionalization on neural networks for fast and accurate flow predictions. 

This code accompanies the paper titled 'On Physical Constraints in Scientific Machine Learning for Flow Field Predictions' (ECCOMAS, 2024).

## Get environment working

The only special dependency is `torch`.

For version control, to automatically create a `conda` environment with a working setup run:
```
conda env create -f environment.yml
```
This will create a `conda` environment named `ml` (first line in `environment.yml`) containing all the required packages.