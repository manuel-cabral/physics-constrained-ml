# Physics-Constrained-ML in flow prediction

![scheme_git](https://github.com/user-attachments/assets/672192e8-58d4-4f6c-bd88-a1807a0d063c)

Enforcing incompressibility and non-dimensionalization on neural networks for fast and accurate flow predictions. 

This code accompanies the paper titled 'On Physical Constraints in Scientific Machine Learning for Flow Field Predictions' (ECCOMAS, 2024).

### Running the examples

1. Select a test case: `boundary-layer`, `cylinder-with-circulation`, or `fraenkels-flow` (refer to the paper for more details).
2. Define the global parameters for the problem, either by modifying `args.py`, specifying them in the run file, or passing them via the command line.
3. Set the sizes for the training, validation, and test datasets. Generate and save the datasets using `generate_data.py`.
4. Choose a model architecture (`baseline`, `incompressible`, or `soft incompressible`), configure the number of parameters and optimization settings, and train the model using `train_model.py`.
5. Visualize the fields or analyze physical parameters using the provided additional scripts.

### Get environment working

This project uses no special packages. 

Regardless, for version control purposes, the same virtual environment can be created with the following command:
```
conda env create -f environment.yml
```
This will create a `conda` environment named `pcml` (first line in `environment.yml`).
