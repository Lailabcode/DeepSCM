# Import libraries
import numpy as np

# Import machine learning libraries
import keras
from keras.layers import BatchNormalization

def load_input_data(filename):
    name_list=[]
    seq_list=[]
    SCM_list=[]
    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
            SCM_list.append(float(line[2]))
    return name_list, seq_list, SCM_list

def load_pred_data(filename):
    name_list=[]
    seq_list=[]
    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
    return name_list, seq_list

def one_hot_encoder(s):
    d = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}

    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1

    return x

def create_conv1D(input_shape):
    model = keras.Sequential(name="model_conv1D")
    
    model.add(keras.layers.Input(shape=input_shape))
    
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation = 'relu', name="Conv1D_1"))
    model.add(BatchNormalization())
    
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation = 'relu', name="Conv1D_2"))
    model.add(BatchNormalization())
    
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation = 'relu', name="Conv1D_3"))
    model.add(BatchNormalization())
  
    model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(32, activation = 'relu', name="Dense_1"))
    
    model.add(keras.layers.Dense(1, name="Dense_2"))

    print(model.summary())
    return model
