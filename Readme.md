In this work, machine learning modeling is conducted using Python, and the relevant software version numbers are as follows:

```
Python 3.12.1
scikit-learn 1.4.0
Pytorch 2.3.0.dev20240128+cu121
xgboost 2.0.3
Optuna 3.5.0
```

To run the `main.py` script, load the `dataset.xlsx` file from the 'Dataset' folder, and initiate the training of the models. The dataset is split into an 80/20 ratio, where 80% serves as the training set. Five-fold cross-validation is performed on the training set, and Optuna is employed for hyperparameter optimization. The performance metrics of each model are printed in the terminal. All algorithms for the models are located in the 'Model' folder, and the `Model_ensemble.py` script is used to invoke these algorithms. 

The results of the trained models are stored in the 'Result' folder