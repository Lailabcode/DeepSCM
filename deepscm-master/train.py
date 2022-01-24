# Fix random seed number for reproducibility
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0) 

# Import libraries
import numpy as np

# Import machine learning models
from Conv1D import Conv1D_regression

# Import custom functions
from utils import one_hot_encoder, load_input_data


if __name__ == "__main__":

    # ----------------------
    # Load input data
    # ----------------------

    name_list, seq_list, SCM_list = load_input_data('DeepSCM_ML_data_final.txt')

    # ----------------------
    # Run regression
    # ----------------------

    Conv1D_model = Conv1D_regression(seq_list, SCM_list)

